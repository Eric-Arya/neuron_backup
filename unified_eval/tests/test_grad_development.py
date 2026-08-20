from __future__ import annotations

from pathlib import Path

import torch

from unified_eval.grad_development import (
    attach_gradient_alphas,
    build_parser,
    build_complement_split,
    choose_gate_threshold,
    choose_config,
    configure_masks,
    first_refusal_cue,
    first_refusal_sentence,
    generated_corpus_record,
    overlapping_token_indices,
    rank_gradients,
    render_prompt,
    select_corpus_target_examples,
)


def test_sweep_direction_defaults_to_positive_only() -> None:
    args = build_parser().parse_args(["sweep"])
    assert args.direction == "positive-only"


def test_gradient_extraction_defaults_to_tail_and_accepts_global() -> None:
    assert build_parser().parse_args(["extract-corpus"]).alpha_scope == "tail"
    args = build_parser().parse_args(
        ["extract-corpus", "--alpha-scope", "global"]
    )
    assert args.alpha_scope == "global"


def test_global_alpha_gradient_sums_activation_products_over_all_positions() -> None:
    class Mlp(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down_proj = torch.nn.Linear(2, 1, bias=False)

    class Layer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = Mlp()

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer()])

    model = Model()
    model.model.layers[0].mlp.down_proj.weight.data.fill_(1.0)
    activation = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    alphas, handles, _ = attach_gradient_alphas(model, "global")
    try:
        model.model.layers[0].mlp.down_proj(activation).sum().backward()
        assert alphas[0].grad.tolist() == [9.0, 12.0]
    finally:
        for handle in handles:
            handle.remove()


def test_first_refusal_sentence_keeps_generated_prefix_and_first_sentence() -> None:
    response = ".\nEchoed request.\nI cannot provide that information. Can I help instead?"
    assert first_refusal_sentence(response) == (
        ".\nEchoed request.\n",
        "I cannot provide that information.",
        18,
        52,
    )
    assert first_refusal_sentence("This is not a refusal.") is None


def test_on_policy_parser_accepts_generic_generated_response_file() -> None:
    args = build_parser().parse_args(
        [
            "extract-on-policy-refusals",
            "--responses-path",
            "generated.jsonl",
            "--response-column",
            "model_response",
            "--no-safe-only",
        ]
    )
    assert args.responses_path == Path("generated.jsonl")
    assert args.response_column == "model_response"
    assert args.safe_only is False


def test_on_policy_parser_accepts_first_refusal_cue_tail_mode() -> None:
    args = build_parser().parse_args(
        ["extract-on-policy-refusals", "--target-mode", "first-refusal-cue"]
    )
    assert args.target_mode == "first-refusal-cue"
    assert args.alpha_scope == "tail"


def test_first_refusal_cue_uses_earliest_completed_lexicon_match() -> None:
    response = "Preface: this is illegal and unethical. I cannot help."
    prefix, cue, start, end = first_refusal_cue(response)
    assert cue == "illegal"
    assert response[start:end] == cue
    assert prefix + cue == response[:end]
    assert first_refusal_cue("I cannot create that.")[1] == "I cannot"
    assert first_refusal_cue("There is no refusal cue here.") is None


def test_target_alignment_includes_token_spanning_cue_boundary() -> None:
    offsets = [(0, 0), (0, 4), (4, 8), (8, 15)]
    assert overlapping_token_indices(offsets, 5, 15) == [2, 3]


def test_safe_corpus_generation_defaults_to_benchmarked_configuration() -> None:
    args = build_parser().parse_args(["generate-safe-corpus"])
    assert args.target_safe_count == 256
    assert args.batch_size == 32
    assert args.max_new_tokens == 256
    assert args.prompt_format == "raw"
    assert args.dtype == "bfloat16"


def test_generated_corpus_record_scores_safety_and_preserves_source() -> None:
    record = generated_corpus_record(
        7,
        {"prompt": "harmful prompt", "llama3_output": "teacher refusal"},
        "I cannot assist with that.",
        6,
        "model_response",
    )
    assert record["source_index"] == 7
    assert record["completion"] == "teacher refusal"
    assert record["jailbroken"] is False
    assert "I cannot" in record["matched_refusal_prefixes"]


def test_positive_only_masks_skip_negative_gradient_neurons() -> None:
    masks = [torch.ones(1, 4)]
    ranking = [
        {"layer": "0", "neuron": "0", "mean_g": "-2"},
        {"layer": "0", "neuron": "1", "mean_g": "1"},
        {"layer": "0", "neuron": "2", "mean_g": "0.5"},
    ]
    configure_masks(masks, ranking, k=2, strength=2.0, direction="positive-only")
    assert masks[0].tolist() == [[1.0, 3.0, 3.0, 1.0]]


def test_complement_split_is_disjoint_deterministic_and_complete() -> None:
    all_rows = [
        {"id": f"harmbench_{index}", "prompt": f"p{index}"} for index in range(400)
    ]
    test_rows = all_rows[::2]
    development, selection, tuning, excluded = build_complement_split(
        all_rows, test_rows, seed=112, selection_count=150
    )
    assert len(development) == 200
    assert len(selection) == 150
    assert len(tuning) == 50
    assert excluded == []
    assert {row["id"] for row in test_rows}.isdisjoint(
        {row["id"] for row in development}
    )
    repeated = build_complement_split(all_rows, test_rows, 112, 150)
    assert selection == repeated[1]
    assert tuning == repeated[2]


def test_complement_split_excludes_exact_test_prompt_duplicates() -> None:
    all_rows = [
        {"id": f"harmbench_{index}", "prompt": f"p{index}"} for index in range(400)
    ]
    test_rows = all_rows[::2]
    all_rows[1]["prompt"] = test_rows[0]["prompt"]
    development, selection, tuning, excluded = build_complement_split(
        all_rows, test_rows, seed=112, selection_count=150
    )
    assert [row["id"] for row in excluded] == ["harmbench_1"]
    assert len(development) == 199
    assert len(selection) == 150
    assert len(tuning) == 49


def test_stable_ranking_stays_within_abs_mean_candidate_pool() -> None:
    gradients = torch.tensor(
        [
            [[10.0, 2.0, 1.0]],
            [[-8.0, 2.0, 1.1]],
            [[10.0, 2.0, 0.9]],
            [[-8.0, 2.0, 1.0]],
            [[10.0, 2.0, 1.1]],
            [[-8.0, 2.0, 0.9]],
        ]
    )
    _, stable, absolute = rank_gradients(gradients, top_k=2, candidate_pool=2, folds=3)
    assert absolute[0]["neuron"] == 1
    assert stable[0]["neuron"] == 1
    assert {row["neuron"] for row in stable} <= {0, 1}


def test_corpus_target_selection_requires_prefix_and_excludes_test_overlap() -> None:
    records = [
        {"prompt": "harm one", "llama3_output": "I cannot help."},
        {"prompt": "harm two", "llama3_output": "Sorry, I cannot help."},
        {"prompt": "Harm, three!", "llama3_output": "I cannot assist."},
    ]
    selected, counts = select_corpus_target_examples(
        records, 0, 3, "I cannot", {"harm three"}
    )
    assert [row["source_index"] for row in selected] == [0]
    assert counts == {
        "slice_records": 3,
        "target_prefix_matches": 2,
        "target_prefix_misses": 1,
        "excluded_test_prompt_overlaps": 1,
        "selected_records": 1,
    }


def test_render_prompt_uses_native_chat_template() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is True
            return f"<user>{messages[0]['content']}<assistant>"

    tokenizer = Tokenizer()
    assert render_prompt(tokenizer, "hello", "raw") == "hello"
    assert render_prompt(tokenizer, "hello", "chat") == "<user>hello<assistant>"


def test_config_selection_penalizes_regressions_and_respects_gsm_constraint() -> None:
    summaries = [
        {"experiment": "baseline"},
        {
            "experiment": "unsafe_low_gsm",
            "selection_score": 8,
            "attack_success_rate": 20,
            "gsm8k_accuracy": 20,
            "gsm_eligible": False,
            "top_k": 50,
            "strength": 1.0,
        },
        {
            "experiment": "stable",
            "selection_score": 5,
            "attack_success_rate": 30,
            "gsm8k_accuracy": 65,
            "gsm_eligible": True,
            "top_k": 25,
            "strength": 0.75,
        },
    ]
    assert choose_config(summaries)["experiment"] == "stable"


def test_gate_threshold_preserves_high_confidence_baseline_refusal() -> None:
    ids = ["a", "b", "c"]
    confidence = {"a": -1.0, "b": -5.0, "c": -6.0}
    baseline = {
        "a": {"jailbroken": False},
        "b": {"jailbroken": True},
        "c": {"jailbroken": True},
    }
    controller = {
        "a": {"jailbroken": True},
        "b": {"jailbroken": False},
        "c": {"jailbroken": True},
    }
    selected, _ = choose_gate_threshold(
        ids, confidence, baseline, controller, regression_penalty=2.0
    )
    assert selected["attack_success_count"] == 1
    assert selected["unsafe_to_safe"] == 1
    assert selected["safe_to_unsafe"] == 0
