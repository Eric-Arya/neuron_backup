from __future__ import annotations

import ast
import csv
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StopStringCriteria,
    StoppingCriteriaList,
)

from .common import CHOICES


DEFAULT_LLAMA3 = Path("/workspace/xcy/models/Meta-Llama-3-8B-Instruct")
DEFAULT_LLAMA3_SFT_ADAPTER = Path(
    "/workspace/xcy/models/Meta-Llama-3-8B-Instruct-SFT-IA3-SNRawDot256-E20"
)
DEFAULT_LLAMA3_SFT_PATCH_RANKING = Path(
    "/workspace/xcy/safety_repro/neurips_neuron/output/change_scores/"
    "llama3_instruct_vs_sft_snrawdot256_alpha3_snheldout_seed42_n200_raw_completion.pt"
)
DEFAULT_SN_ALPHA8 = Path(
    "/workspace/xcy/safety_repro/iclr_neuron_expanded_kv/neuron_enhancement/outputs/"
    "sn_delta_scale/exact100_200_cap25_docs256_ep20_alpha8"
)
DEFAULT_SN_DIRECT_NEURONS = Path(
    "/workspace/xcy/safety_repro/iclr_neuron_expanded_kv/neuron_detection/"
    "output_neurons/Meta-Llama-3-8B-Instruct_zou_train_attn100_ffn200_"
    "kvexpanded_ranked_raw_rank_sweep_200.txt"
)
DEFAULT_GRAD_RANKING = Path(
    "/workspace/xcy/safety_repro/grad_neuron/results/gradients/"
    "raw_refusal_advbench_rows100_299/top_neurons.csv"
)
DEFAULT_NEURIPS_REPO = Path("/workspace/xcy/safety_repro/neurips_neuron")


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def scale_ia3_displacement_(model: torch.nn.Module, alpha: float) -> int:
    """Scale learned IA3 gates around their identity initialization of one."""
    if not isinstance(alpha, (int, float)) or not torch.isfinite(torch.tensor(alpha)):
        raise ValueError("IA3 displacement alpha must be finite")
    if alpha < 0:
        raise ValueError("IA3 displacement alpha must be nonnegative")
    scaled_parameters = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".ia3_l." not in f".{name}.":
                continue
            scaled = 1.0 + float(alpha) * (parameter.float() - 1.0)
            parameter.copy_(scaled.to(dtype=parameter.dtype))
            scaled_parameters += parameter.numel()
    if scaled_parameters == 0:
        raise ValueError("No IA3 gate parameters found in loaded adapter")
    return scaled_parameters


def scale_hooked_ia3_displacement_(model: torch.nn.Module, alpha: float) -> int:
    """Scale IA3 tensors loaded by the paper's custom HookedLlama implementation."""
    if not isinstance(alpha, (int, float)) or not torch.isfinite(torch.tensor(alpha)):
        raise ValueError("IA3 displacement alpha must be finite")
    if alpha < 0:
        raise ValueError("IA3 displacement alpha must be nonnegative")
    scaled_parameters = 0
    if alpha == 1:
        return scaled_parameters
    with torch.no_grad():
        for module in model.modules():
            tensors = getattr(module, "ia3_l", None)
            if not isinstance(tensors, list):
                continue
            for index, tensor in enumerate(tensors):
                scaled = 1.0 + float(alpha) * (tensor.float() - 1.0)
                tensors[index] = scaled.to(device=tensor.device, dtype=tensor.dtype)
                scaled_parameters += tensor.numel()
    if scaled_parameters == 0:
        raise ValueError("No IA3 tensors found in hooked guide model")
    return scaled_parameters


def _trim_generated(ids: list[int], eos_ids: set[int], pad_id: int | None) -> list[int]:
    trimmed: list[int] = []
    for token in ids:
        if pad_id is not None and token == pad_id and trimmed:
            break
        trimmed.append(token)
        if token in eos_ids:
            break
    return trimmed


class Method:
    name: str
    prompt_style: str

    @property
    def tokenizer(self):
        raise NotImplementedError

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        stop_strings: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def option_logits(self, prompts: Sequence[str], option_ids: Sequence[int]) -> torch.Tensor:
        raise NotImplementedError

    def option_token_ids(self, prefix_space: bool = False) -> list[int]:
        ids: list[int] = []
        for choice in CHOICES:
            answer = " " + choice if prefix_space else choice
            encoded = self.tokenizer.encode(answer, add_special_tokens=False)
            if prefix_space and encoded:
                encoded = encoded[-1:]
            if len(encoded) != 1:
                raise ValueError(f"MMLU option {answer!r} is not one token: {encoded}")
            ids.append(encoded[0])
        return ids

    def close(self) -> None:
        clear_memory()


class StandardMethod(Method):
    def __init__(
        self,
        name: str,
        model_path: Path,
        device: str,
        dtype_name: str,
        prompt_style: str,
        adapter_path: Path | None = None,
        ia3_displacement_alpha: float | None = None,
    ) -> None:
        self.name = name
        self.prompt_style = prompt_style
        self.model_path = model_path.resolve()
        self.device = device
        self.dtype_name = dtype_name
        self.adapter_path = adapter_path.resolve() if adapter_path is not None else None
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            device_map={"": device},
            torch_dtype=DTYPES[dtype_name],
            low_cpu_mem_usage=True,
        )
        self.model = (
            PeftModel.from_pretrained(model, self.adapter_path).eval()
            if self.adapter_path is not None
            else model.eval()
        )
        self.ia3_scaled_parameters = None
        if ia3_displacement_alpha is not None:
            self.ia3_scaled_parameters = scale_ia3_displacement_(
                self.model, ia3_displacement_alpha
            )
        self.model.requires_grad_(False)
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None

    @property
    def tokenizer(self):
        return self._tokenizer

    def _encode(self, prompts: Sequence[str]):
        # Rendered Llama-3 chat strings already include BOS; raw prompts need it.
        chat_bos = "<|begin_of_text|>"
        add_special_tokens = not all(prompt.startswith(chat_bos) for prompt in prompts)
        return self.tokenizer(
            list(prompts),
            padding=True,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        ).to(self.model.get_input_embeddings().weight.device)

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        stop_strings: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        encoded = self._encode(prompts)
        input_width = encoded.input_ids.shape[1]
        stopping_criteria = None
        if stop_strings:
            stopping_criteria = StoppingCriteriaList(
                [StopStringCriteria(self.tokenizer, list(stop_strings))]
            )
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=stopping_criteria,
            )
        eos = self.model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        rows: list[dict[str, Any]] = []
        for index in range(output.shape[0]):
            prompt_ids = encoded.input_ids[index][encoded.attention_mask[index].bool()].tolist()
            generated_ids = _trim_generated(
                output[index, input_width:].tolist(), eos_ids, self.tokenizer.pad_token_id
            )
            rows.append(
                {
                    "response": self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                    "score_text": self.tokenizer.decode(
                        prompt_ids + generated_ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    ),
                    "generated_token_count": len(generated_ids),
                }
            )
        return rows

    def option_logits(self, prompts: Sequence[str], option_ids: Sequence[int]) -> torch.Tensor:
        encoded = self._encode(prompts)
        with torch.inference_mode():
            logits = self.model(**encoded).logits[:, -1, list(option_ids)]
        return logits.float().cpu()

    def close(self) -> None:
        self.model = None
        self._tokenizer = None
        clear_memory()


class GradMethod(StandardMethod):
    def __init__(
        self,
        model_path: Path,
        ranking_path: Path,
        top_k: int,
        strength: float,
        scope: str,
        direction: str,
        max_batch_size: int,
        device: str,
        dtype_name: str,
        scale_path: Path | None = None,
    ) -> None:
        super().__init__("grad", model_path, device, dtype_name, "raw")
        if (
            top_k <= 0
            or strength < 0
            or scope not in {"last", "all"}
            or direction not in {"signed", "positive-only"}
        ):
            raise ValueError("Invalid gradient-controller configuration")
        with ranking_path.open(newline="", encoding="utf-8") as handle:
            ranking = list(csv.DictReader(handle))
        if ranking and "rank" in ranking[0]:
            ranking.sort(key=lambda row: int(row["rank"]))
        else:
            ranking.sort(key=lambda row: float(row["abs_mean_g"]), reverse=True)
        if direction == "positive-only":
            ranking = [row for row in ranking if float(row["mean_g"]) > 0]
        if top_k > len(ranking):
            raise ValueError(f"Requested top-{top_k}, ranking has {len(ranking)} rows")
        self.ranking_path = ranking_path.resolve()
        self.top_k = top_k
        self.strength = strength
        self.scope = scope
        self.direction = direction
        self.scale_path = scale_path.resolve() if scale_path is not None else None
        self.scale_mode = None
        scale_rows = None
        if self.scale_path is not None:
            payload = json.loads(self.scale_path.read_text(encoding="utf-8"))
            if (
                payload.get("schema") != "fisher_grad_scales_v1"
                or payload.get("top_k") != top_k
                or payload.get("direction") != direction
                or payload.get("scope") != scope
            ):
                raise ValueError("Fisher Grad scale artifact is incompatible with the run")
            scale_rows = payload.get("rows")
            if not isinstance(scale_rows, list) or len(scale_rows) != top_k:
                raise ValueError("Fisher Grad scale artifact has the wrong row count")
            for ranking_row, scale_row in zip(ranking[:top_k], scale_rows):
                if (
                    int(ranking_row["layer"]) != int(scale_row["layer"])
                    or int(ranking_row["neuron"]) != int(scale_row["neuron"])
                ):
                    raise ValueError("Fisher Grad scale artifact does not match ranking order")
                multiplier = float(scale_row["multiplier"])
                if not math.isfinite(multiplier) or multiplier < 1.0:
                    raise ValueError(
                        "Positive-only Fisher Grad multipliers must be finite and >= 1"
                    )
            self.scale_mode = str(payload.get("mode"))
        self.masks: list[torch.Tensor] = []
        self.handles: list[Any] = []
        for layer in self.model.model.layers:
            mask = torch.ones(
                max_batch_size,
                layer.mlp.down_proj.in_features,
                device=layer.mlp.down_proj.weight.device,
                dtype=layer.mlp.down_proj.weight.dtype,
            )

            def scale_input(_module, inputs, current_mask=mask):
                activation = inputs[0]
                batch_mask = current_mask[: activation.shape[0]].unsqueeze(1)
                if self.scope == "all" or activation.shape[1] == 1:
                    scaled = activation * batch_mask
                else:
                    scaled = torch.cat(
                        (activation[:, :-1], activation[:, -1:] * batch_mask), dim=1
                    )
                return (scaled, *inputs[1:])

            self.masks.append(mask)
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(scale_input))
        for mask in self.masks:
            mask.fill_(1)
        for index, row in enumerate(ranking[:top_k]):
            if scale_rows is None:
                row_direction = 1.0 if float(row["mean_g"]) > 0 else -1.0
                multiplier = max(0.0, 1.0 + strength * row_direction)
            else:
                multiplier = float(scale_rows[index]["multiplier"])
            self.masks[int(row["layer"])][:, int(row["neuron"])] = multiplier

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.masks.clear()
        super().close()


SN_STRUCTURES = ("fwd_up", "fwd_down", "q", "k", "v")


def load_sn_direct_selection(
    path: Path,
    *,
    num_layers: int,
    intermediate_size: int,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    cap: int,
) -> tuple[list[dict[int, tuple[int, ...]]], str]:
    """Load the raw ranked SN-Tune mask using its file-order/head-aware rules."""
    if cap <= 0:
        raise ValueError("SN direct cap must be positive")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(SN_STRUCTURES):
        raise ValueError(f"Expected five neuron dictionaries in {path}, found {len(lines)}")
    head_dim = hidden_size // num_attention_heads
    kv_repeat = num_attention_heads // num_key_value_heads
    physical_kv_size = num_key_value_heads * head_dim
    limits = (intermediate_size, intermediate_size, hidden_size, hidden_size, hidden_size)
    selected: list[dict[int, tuple[int, ...]]] = []
    expected_layers = set(range(num_layers))
    for position, (name, line, limit) in enumerate(zip(SN_STRUCTURES, lines, limits)):
        parsed = ast.literal_eval(line)
        if not isinstance(parsed, dict) or set(parsed) != expected_layers:
            raise ValueError(f"{name} must contain exactly layers 0 through {num_layers - 1}")
        layers: dict[int, tuple[int, ...]] = {}
        for layer in range(num_layers):
            values = parsed[layer]
            if not isinstance(values, list):
                raise ValueError(f"{name} layer {layer} must be an ordered list")
            ordered = list(dict.fromkeys(int(index) for index in values))
            if any(index < 0 or index >= limit for index in ordered):
                raise ValueError(f"{name} layer {layer} has an index outside [0, {limit})")
            if position in (3, 4):
                # The detector expands GQA K/V to query-head space. SN-Tune maps back to
                # physical projection rows before applying the per-layer cap.
                ordered = list(dict.fromkeys(
                    ((index // head_dim) // kv_repeat) * head_dim + index % head_dim
                    for index in ordered
                ))
                if any(index >= physical_kv_size for index in ordered):
                    raise ValueError(f"Mapped {name} index exceeds physical K/V width")
            layers[layer] = tuple(sorted(ordered[:cap]))
        selected.append(layers)
    serializable = [
        {str(layer): list(indices) for layer, indices in sorted(layers.items())}
        for layers in selected
    ]
    payload = json.dumps(serializable, sort_keys=True, separators=(",", ":"))
    return selected, hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SnDirectMethod(StandardMethod):
    """Scale raw-detected SN-Tune activation dimensions without fine-tuning."""

    def __init__(
        self,
        model_path: Path,
        neuron_path: Path,
        cap: int,
        strength: float,
        device: str,
        dtype_name: str,
    ) -> None:
        super().__init__("sn_direct", model_path, device, dtype_name, "raw")
        if not torch.isfinite(torch.tensor(strength)) or strength <= -1:
            raise ValueError("SN direct strength must be finite and greater than -1")
        config = self.model.config
        self.neuron_path = neuron_path.resolve()
        self.cap = cap
        self.strength = strength
        self.multiplier = 1.0 + strength
        self.selection, self.selection_sha256 = load_sn_direct_selection(
            neuron_path,
            num_layers=int(config.num_hidden_layers),
            intermediate_size=int(config.intermediate_size),
            hidden_size=int(config.hidden_size),
            num_attention_heads=int(config.num_attention_heads),
            num_key_value_heads=int(config.num_key_value_heads),
            cap=cap,
        )
        self.selection_counts = {
            name: sum(len(indices) for indices in layers.values())
            for name, layers in zip(SN_STRUCTURES, self.selection)
        }
        self.handles: list[Any] = []

        def output_scaler(indices: tuple[int, ...]):
            def scale(_module, _inputs, output):
                if not indices or self.multiplier == 1.0:
                    return output
                multiplier = output.new_ones(output.shape[-1])
                multiplier[list(indices)] = self.multiplier
                return output * multiplier
            return scale

        def input_scaler(indices: tuple[int, ...]):
            def scale(_module, inputs):
                if not indices or self.multiplier == 1.0:
                    return inputs
                activation = inputs[0]
                multiplier = activation.new_ones(activation.shape[-1])
                multiplier[list(indices)] = self.multiplier
                return (activation * multiplier, *inputs[1:])
            return scale

        for layer_index, layer in enumerate(self.model.model.layers):
            self.handles.extend((
                layer.mlp.up_proj.register_forward_hook(
                    output_scaler(self.selection[0][layer_index])
                ),
                layer.mlp.down_proj.register_forward_pre_hook(
                    input_scaler(self.selection[1][layer_index])
                ),
                layer.self_attn.q_proj.register_forward_hook(
                    output_scaler(self.selection[2][layer_index])
                ),
                layer.self_attn.k_proj.register_forward_hook(
                    output_scaler(self.selection[3][layer_index])
                ),
                layer.self_attn.v_proj.register_forward_hook(
                    output_scaler(self.selection[4][layer_index])
                ),
            ))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        super().close()


def _import_neurips_eval(repo: Path):
    repo = repo.resolve()
    for path in (repo, repo / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from eval import table1_harmbench as module

    return module


class GuidePatchMethod(Method):
    """Shared inference implementation for same-family activation patching."""

    @property
    def tokenizer(self):
        return self._tokenizer

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        stop_strings: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        generated = self.module.greedy_generate(
            self.base,
            self.tokenizer,
            prompts,
            max_new_tokens,
            guide_model=self.guide,
            neurons_by_layer=self.neurons_by_layer,
        )
        return [
            {
                "response": row["completion"],
                "score_text": row["score_text"],
                "generated_token_count": row["generated_token_count"],
            }
            for row in generated
        ]

    def option_logits(self, prompts: Sequence[str], option_ids: Sequence[int]) -> torch.Tensor:
        base_device = self.module.model_device(self.base)
        guide_device = self.module.model_device(self.guide)
        encoded = self.tokenizer(
            list(prompts), padding=True, add_special_tokens=True, return_tensors="pt"
        )
        base_inputs = {key: value.to(base_device) for key, value in encoded.items()}
        guide_inputs = {key: value.to(guide_device) for key, value in encoded.items()}
        with torch.inference_mode(), self.module.activation_bridge(
            self.base, self.guide, self.neurons_by_layer
        ) as captured:
            captured.clear()
            self.guide(**guide_inputs, use_cache=False, return_dict=True)
            if len(captured) != len(self.neurons_by_layer):
                raise RuntimeError("Not all NeurIPS guide activations were captured")
            output = self.base(**base_inputs, use_cache=False, return_dict=True)
            logits = output.logits[:, -1, list(option_ids)]
        return logits.float().cpu()

    def close(self) -> None:
        self.base = None
        self.guide = None
        self._tokenizer = None
        clear_memory()


class Llama3GuidePatchMethod(GuidePatchMethod):
    """Patch ranked post-MLP activations from a Llama-3 PEFT guide."""

    prompt_style = "raw"

    def __init__(
        self,
        name: str,
        guide_label: str,
        repo: Path,
        model_path: Path,
        guide_adapter: Path,
        ranking_path: Path,
        top_k: int,
        base_device: str,
        guide_device: str,
        dtype_name: str,
        ia3_displacement_alpha: float = 1.0,
    ) -> None:
        self.name = name
        self.guide_label = guide_label
        self.repo = repo.resolve()
        self.model_path = model_path.resolve()
        self.guide_adapter = guide_adapter.resolve()
        self.ranking_path = ranking_path.resolve()
        self.top_k = top_k
        self.module = _import_neurips_eval(repo)
        self.base, self._tokenizer = self.module.load_hooked_model(
            self.model_path, self.model_path, [], base_device, dtype_name
        )
        self.guide, guide_tokenizer = self.module.load_hooked_model(
            self.model_path,
            self.model_path,
            [self.guide_adapter],
            guide_device,
            dtype_name,
        )
        self.ia3_displacement_alpha = ia3_displacement_alpha
        self.ia3_scaled_parameters = scale_hooked_ia3_displacement_(
            self.guide, ia3_displacement_alpha
        )
        if self._tokenizer.get_vocab() != guide_tokenizer.get_vocab():
            raise ValueError(f"Llama-3 base and {guide_label} tokenizers differ")
        selected, self.ranking_count = self.module.load_ranked_neurons(
            self.ranking_path, top_k, self.base
        )
        self.neurons_by_layer = self.module.group_neurons(selected)

    def option_logits(self, prompts: Sequence[str], option_ids: Sequence[int]) -> torch.Tensor:
        base_device = self.module.model_device(self.base)
        guide_device = self.module.model_device(self.guide)
        bos = self.tokenizer.bos_token or ""
        add_special_tokens = not (bos and all(prompt.startswith(bos) for prompt in prompts))
        encoded = self.tokenizer(
            list(prompts), padding=True, add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )
        base_inputs = {key: value.to(base_device) for key, value in encoded.items()}
        guide_inputs = {key: value.to(guide_device) for key, value in encoded.items()}
        with torch.inference_mode(), self.module.activation_bridge(
            self.base, self.guide, self.neurons_by_layer
        ) as captured:
            captured.clear()
            self.guide(**guide_inputs, use_cache=False, return_dict=True)
            if len(captured) != len(self.neurons_by_layer):
                raise RuntimeError(
                    f"Not all Llama-3 {self.guide_label} activations were captured"
                )
            output = self.base(**base_inputs, use_cache=False, return_dict=True)
            logits = output.logits[:, -1, list(option_ids)]
        return logits.float().cpu()


class NeuripsDirectMethod(Method):
    """Scale a ranked set of Llama-3 post-MLP activation dimensions."""

    name = "neurips_direct"
    prompt_style = "raw"

    def __init__(
        self,
        repo: Path,
        model_path: Path,
        ranking_path: Path,
        top_k: int,
        multiplier: float,
        device: str,
        dtype_name: str,
    ) -> None:
        self.repo = repo.resolve()
        self.model_path = model_path.resolve()
        self.module = _import_neurips_eval(repo)
        self.model, self._tokenizer = self.module.load_hooked_model(
            self.model_path, self.model_path, [], device, dtype_name
        )
        if not torch.isfinite(torch.tensor(multiplier)) or multiplier <= 0:
            raise ValueError("NeurIPS direct multiplier must be finite and positive")
        self.ranking_path = ranking_path.resolve()
        self.top_k = top_k
        self.multiplier = float(multiplier)
        selected, self.ranking_count = self.module.load_ranked_neurons(
            self.ranking_path, top_k, self.model
        )
        self.neurons_by_layer = self.module.group_neurons(selected)
        self.handles: list[Any] = []

        def scale_hook(indices: torch.Tensor):
            def scale(_module, _inputs, output):
                if self.multiplier == 1.0:
                    return output
                scaled = output.clone()
                scaled[..., indices] *= self.multiplier
                return scaled

            return scale

        model_device = self.module.model_device(self.model)
        for layer, neuron_list in self.neurons_by_layer.items():
            indices = torch.tensor(neuron_list, dtype=torch.long, device=model_device)
            self.handles.append(
                self.model.model.layers[layer].mlp.hook_post.register_forward_hook(
                    scale_hook(indices)
                )
            )

    @property
    def tokenizer(self):
        return self._tokenizer

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        stop_strings: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        generated = self.module.greedy_generate(
            self.model, self.tokenizer, prompts, max_new_tokens
        )
        return [
            {
                "response": row["completion"],
                "score_text": row["score_text"],
                "generated_token_count": row["generated_token_count"],
            }
            for row in generated
        ]

    def option_logits(self, prompts: Sequence[str], option_ids: Sequence[int]) -> torch.Tensor:
        device = self.module.model_device(self.model)
        encoded = self.tokenizer(
            list(prompts), padding=True, add_special_tokens=True, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            logits = self.model(**encoded, use_cache=False, return_dict=True).logits[
                :, -1, list(option_ids)
            ]
        return logits.float().cpu()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.model = None
        self._tokenizer = None
        clear_memory()


def build_method(args, max_batch_size: int) -> Method:
    if args.method == "llama3_base":
        return StandardMethod(
            "llama3_base", args.llama3_model, args.device,
            args.llama3_base_dtype, "raw",
        )
    if args.method == "llama3_sft":
        return StandardMethod(
            "llama3_sft", args.llama3_model, args.device,
            args.llama3_sft_dtype, "raw", args.llama3_sft_adapter,
            args.llama3_sft_ia3_alpha,
        )
    if args.method == "llama3_sft_patch":
        return Llama3GuidePatchMethod(
            "llama3_sft_patch",
            "IA3-SFT-guide",
            args.neurips_repo,
            args.llama3_model,
            args.llama3_sft_adapter,
            args.llama3_sft_patch_ranking,
            args.llama3_sft_patch_top_k,
            args.base_device,
            args.guide_device,
            args.llama3_sft_patch_dtype,
            args.llama3_sft_ia3_alpha,
        )
    if args.method == "grad":
        return GradMethod(
            args.llama3_model, args.grad_ranking, args.grad_top_k,
            args.grad_strength, args.grad_scope, args.grad_direction, max_batch_size,
            args.device, args.grad_dtype, args.grad_scale_file,
        )
    if args.method == "sn":
        return StandardMethod(
            "sn", args.sn_model, args.device, args.sn_dtype, "raw"
        )
    if args.method == "sn_direct":
        return SnDirectMethod(
            args.llama3_model, args.sn_direct_neurons, args.sn_direct_cap,
            args.sn_direct_strength, args.device, args.sn_direct_dtype,
        )
    if args.method == "neurips_direct":
        return NeuripsDirectMethod(
            args.neurips_repo,
            args.llama3_model,
            args.neurips_direct_ranking,
            args.neurips_top_k,
            args.neurips_direct_multiplier,
            args.device,
            args.neurips_dtype,
        )
    raise ValueError(args.method)
