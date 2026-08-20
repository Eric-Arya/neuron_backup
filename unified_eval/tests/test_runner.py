from __future__ import annotations

import pytest

from unified_eval.runner import (
    build_parser,
    configure_float32_execution,
    floating_point_protocol,
    render_harm_prompt,
    render_math500_prompt,
    resolve_method_defaults,
    score_math500_response,
    truncate_bbh_response,
)


@pytest.mark.parametrize(
    "method",
    [
        "llama3_base",
        "llama3_sft",
        "llama3_sft_patch",
        "grad",
        "sn",
        "sn_direct",
        "neurips_direct",
    ],
)
def test_gsm8k_token_default_is_unified(method: str) -> None:
    args = build_parser().parse_args(["run", "--method", method])
    resolve_method_defaults(args)
    assert args.gsm8k_max_new_tokens == 256


def test_explicit_gsm8k_token_limit_overrides_default() -> None:
    args = build_parser().parse_args(
        ["run", "--method", "grad", "--gsm8k-max-new-tokens", "768"]
    )
    resolve_method_defaults(args)
    assert args.gsm8k_max_new_tokens == 768


def test_llama3_base_defaults_to_fresh_capability_evaluation() -> None:
    args = build_parser().parse_args(["run", "--method", "llama3_base"])
    assert args.llama3_base_capability_source == "fresh"
    assert args.tasks == ["harmbench"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "--method", "llama2_base"],
        ["run", "--method", "llama3_dpo"],
        ["run", "--method", "llama3_base", "--tasks", "beavertails"],
    ],
)
def test_retired_experiments_are_not_cli_choices(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_grad_direction_defaults_to_positive_only() -> None:
    args = build_parser().parse_args(["run", "--method", "grad"])
    assert args.grad_direction == "positive-only"


def test_bbh_defaults_to_official_harness_generation_limit() -> None:
    args = build_parser().parse_args(
        ["run", "--method", "llama3_base", "--tasks", "bbh"]
    )
    assert args.bbh_max_new_tokens == 1024
    resolve_method_defaults(args)
    assert args.bbh_batch_size == 8


def test_bbh_guide_patch_uses_benchmarked_batch_size() -> None:
    args = build_parser().parse_args(
        ["run", "--method", "llama3_sft_patch", "--tasks", "bbh"]
    )
    resolve_method_defaults(args)
    assert args.bbh_batch_size == 16


def test_math500_defaults_to_full_benchmark_generation_limit() -> None:
    args = build_parser().parse_args(
        ["run", "--method", "llama3_base", "--tasks", "math500"]
    )
    resolve_method_defaults(args)
    assert args.math500_max_new_tokens == 1024
    assert args.math500_batch_size == 16


def test_math500_guide_patch_uses_benchmarked_batch_size() -> None:
    args = build_parser().parse_args(
        ["run", "--method", "llama3_sft_patch", "--tasks", "math500"]
    )
    resolve_method_defaults(args)
    assert args.math500_batch_size == 32


def test_ifeval_method_defaults_reuse_benchmarks() -> None:
    direct = build_parser().parse_args(
        ["run", "--method", "sn_direct", "--tasks", "ifeval"]
    )
    grad = build_parser().parse_args(
        ["run", "--method", "grad", "--tasks", "ifeval"]
    )
    ranking_direct = build_parser().parse_args(
        ["run", "--method", "neurips_direct", "--tasks", "ifeval"]
    )
    resolve_method_defaults(direct)
    resolve_method_defaults(grad)
    resolve_method_defaults(ranking_direct)
    assert direct.ifeval_batch_size == 32
    assert grad.ifeval_batch_size == 8
    assert ranking_direct.ifeval_batch_size == 32


def test_direct_overlays_reuse_harmbench_batch_benchmarks() -> None:
    for method in ("sn_direct", "neurips_direct"):
        args = build_parser().parse_args(
            ["run", "--method", method, "--tasks", "harmbench"]
        )
        resolve_method_defaults(args)
        assert args.harmbench_batch_size == 32


def test_direct_multiplier_arguments_accept_attenuation() -> None:
    sn_args = build_parser().parse_args(
        ["run", "--method", "sn_direct", "--sn-direct-strength", "-0.1"]
    )
    ranking_args = build_parser().parse_args(
        [
            "run",
            "--method",
            "neurips_direct",
            "--neurips-direct-multiplier",
            "0.8",
        ]
    )
    assert 1 + sn_args.sn_direct_strength == pytest.approx(0.9)
    assert ranking_args.neurips_direct_multiplier == pytest.approx(0.8)


def test_neurips_direct_defaults_to_sncorpus_sft_ranking() -> None:
    args = build_parser().parse_args(["run", "--method", "neurips_direct"])
    assert "vs_sft_snrawdot256" in args.neurips_direct_ranking.name


def test_bbh_harness_stop_strings_use_earliest_match() -> None:
    response, stop = truncate_bbh_response(
        "Reasoning. So the answer is (B).\n\nQ: an unwanted continuation"
    )
    assert response == "Reasoning. So the answer is (B)."
    assert stop == "\n\n"


def test_llama3_sft_ia3_alpha_defaults_to_trained_adapter() -> None:
    args = build_parser().parse_args(["run", "--method", "llama3_sft"])
    assert args.llama3_sft_ia3_alpha == 1.0
    assert args.llama3_sft_training_format == "raw"
    assert "SNRawDot256" in args.llama3_sft_adapter.name


def test_llama3_sft_patch_accepts_alpha_override() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--method",
            "llama3_sft_patch",
            "--llama3-sft-ia3-alpha",
            "3",
        ]
    )
    assert args.llama3_sft_ia3_alpha == 3.0


def test_signed_grad_direction_remains_explicitly_available() -> None:
    args = build_parser().parse_args(
        ["run", "--method", "grad", "--grad-direction", "signed"]
    )
    assert args.grad_direction == "signed"


def test_all_unified_evaluator_dtypes_default_to_float32() -> None:
    args = build_parser().parse_args(["run", "--method", "grad"])
    dtype_fields = (
        "llama3_base_dtype",
        "llama3_sft_dtype",
        "llama3_sft_patch_dtype",
        "grad_dtype",
        "sn_dtype",
        "sn_direct_dtype",
        "neurips_dtype",
    )
    assert {getattr(args, field) for field in dtype_fields} == {"float32"}


def test_float32_execution_protocol_disables_tf32() -> None:
    configure_float32_execution()
    assert floating_point_protocol() == {
        "default_model_dtype": "float32",
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return f"<chat>{messages[0]['content']}<assistant>"


class _FakeLlama3Method:
    name = "llama3_base"
    tokenizer = _FakeTokenizer()


def test_math500_uses_publisher_prompt_in_native_chat() -> None:
    prompt = render_math500_prompt(_FakeLlama3Method(), "Compute $1+1$.")
    assert prompt.startswith("<chat>Solve the following math problem")
    assert "\\boxed{ANSWER}" in prompt
    assert prompt.endswith("Compute $1+1$.<assistant>")


def test_math500_symbolic_evaluator_accepts_equivalence_and_rejects_invalid() -> None:
    equivalent = score_math500_response(
        r"Thus the answer is $\boxed{\frac{14}{3}}$.",
        r"Therefore, the final answer is: $\boxed{28/6}$.",
    )
    invalid = score_math500_response(
        r"Thus the answer is $\boxed{9}$.", "I cannot solve this."
    )
    assert equivalent == {
        "correct": True,
        "answer_extracted": True,
        "scoring_error": None,
    }
    assert invalid == {
        "correct": False,
        "answer_extracted": False,
        "scoring_error": None,
    }


def test_llama3_harm_prompt_format_can_use_native_chat() -> None:
    raw_args = build_parser().parse_args(["run", "--method", "llama3_base"])
    chat_args = build_parser().parse_args(
        ["run", "--method", "llama3_base", "--llama3-harm-prompt-format", "chat"]
    )
    method = _FakeLlama3Method()
    assert render_harm_prompt(raw_args, method, "behavior") == "behavior"
    assert (
        render_harm_prompt(chat_args, method, "behavior")
        == "<chat>behavior<assistant>"
    )
