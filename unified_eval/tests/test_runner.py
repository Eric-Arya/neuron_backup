from __future__ import annotations

import pytest

from unified_eval.runner import (
    EXPECTED_BEAVERTAILS_SHA256,
    build_parser,
    configure_float32_execution,
    floating_point_protocol,
    is_llama2_method,
    render_math500_prompt,
    score_math500_response,
    truncate_bbh_response,
    render_beavertails_prompt,
    render_harm_prompt,
    resolve_method_defaults,
)
from unified_eval.common import read_jsonl, sha256_file


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("llama3_base", 256),
        ("llama3_dpo", 256),
        ("llama3_dpo_patch", 256),
        ("llama3_sft_patch", 256),
        ("llama2_base", 256),
        ("grad", 256),
        ("sn", 256),
        ("sn_direct", 256),
        ("neurips", 256),
        ("neurips_direct", 256),
        ("neurips_dpo", 256),
    ],
)
def test_gsm8k_token_default_depends_on_model_family(
    method: str, expected: int
) -> None:
    args = build_parser().parse_args(["run", "--method", method])
    resolve_method_defaults(args)
    assert args.gsm8k_max_new_tokens == expected


def test_explicit_gsm8k_token_limit_overrides_method_default() -> None:
    args = build_parser().parse_args(
        ["run", "--method", "neurips", "--gsm8k-max-new-tokens", "768"]
    )
    resolve_method_defaults(args)
    assert args.gsm8k_max_new_tokens == 768


def test_table2_adapted_llama2_uses_common_256_token_limit() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--method",
            "llama2_base",
            "--neurips-capability-protocol",
            "table2_adapted",
        ]
    )
    resolve_method_defaults(args)
    assert args.gsm8k_max_new_tokens == 256


def test_explicit_paper_llama2_protocol_uses_1024_token_limit() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--method",
            "llama2_base",
            "--neurips-capability-protocol",
            "paper",
        ]
    )
    resolve_method_defaults(args)
    assert args.gsm8k_max_new_tokens == 1024


def test_llama3_base_can_request_fresh_capability_evaluation() -> None:
    args = build_parser().parse_args(
        [
            "run",
            "--method",
            "llama3_base",
            "--llama3-base-capability-source",
            "fresh",
        ]
    )
    assert args.llama3_base_capability_source == "fresh"


def test_unified_capability_protocol_is_the_default() -> None:
    llama3 = build_parser().parse_args(["run", "--method", "llama3_base"])
    llama2 = build_parser().parse_args(["run", "--method", "llama2_base"])
    assert llama3.llama3_base_capability_source == "fresh"
    assert llama2.neurips_capability_protocol == "table2_adapted"


def test_grad_direction_defaults_to_positive_only() -> None:
    args = build_parser().parse_args(["run", "--method", "grad"])
    assert args.grad_direction == "positive-only"


def test_cost_scoring_can_be_disabled() -> None:
    args = build_parser().parse_args(
        ["run", "--method", "grad", "--skip-cost-scoring"]
    )
    assert args.skip_cost_scoring is True


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
    neurips_direct = build_parser().parse_args(
        ["run", "--method", "neurips_direct", "--tasks", "ifeval"]
    )
    resolve_method_defaults(direct)
    resolve_method_defaults(grad)
    resolve_method_defaults(neurips_direct)
    assert direct.ifeval_batch_size == 32
    assert grad.ifeval_batch_size == 8
    assert neurips_direct.ifeval_batch_size == 32


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
    neurips_args = build_parser().parse_args(
        [
            "run",
            "--method",
            "neurips_direct",
            "--neurips-direct-multiplier",
            "0.8",
        ]
    )
    assert 1 + sn_args.sn_direct_strength == pytest.approx(0.9)
    assert neurips_args.neurips_direct_multiplier == pytest.approx(0.8)


def test_neurips_direct_uses_llama3_protocol_and_ranking() -> None:
    args = build_parser().parse_args(["run", "--method", "neurips_direct"])
    assert is_llama2_method(args.method) is False
    assert args.neurips_direct_ranking.name == (
        "llama3_instruct_vs_dpo_hh_harmless_native_completion.pt"
    )


def test_neurips_direct_selection_source_tracks_guide(tmp_path) -> None:
    from unified_eval.runner import neurips_direct_selection_source

    assert "Instruct-vs-SFT" in neurips_direct_selection_source(
        tmp_path / "llama3_instruct_vs_sft_snraw.pt"
    )
    assert "Instruct-vs-DPO" in neurips_direct_selection_source(
        tmp_path / "llama3_instruct_vs_dpo_hh.pt"
    )


def test_bbh_harness_stop_strings_use_earliest_match() -> None:
    response, stop = truncate_bbh_response(
        "Reasoning. So the answer is (B).\n\nQ: an unwanted continuation"
    )
    assert response == "Reasoning. So the answer is (B)."
    assert stop == "\n\n"


def test_llama3_sft_ia3_alpha_defaults_to_trained_adapter() -> None:
    args = build_parser().parse_args(["run", "--method", "llama3_sft"])
    assert args.llama3_sft_ia3_alpha == 1.0


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
        "llama3_dpo_dtype",
        "llama3_sft_dtype",
        "llama3_dpo_patch_dtype",
        "llama3_sft_patch_dtype",
        "grad_dtype",
        "sn_dtype",
        "sn_direct_dtype",
        "neurips_dtype",
        "cost_dtype",
        "reward_dtype",
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
    name = "llama3_dpo"
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
    raw_args = build_parser().parse_args(["run", "--method", "llama3_dpo"])
    chat_args = build_parser().parse_args(
        ["run", "--method", "llama3_dpo", "--llama3-harm-prompt-format", "chat"]
    )
    method = _FakeLlama3Method()
    assert render_harm_prompt(raw_args, method, "behavior") == "behavior"
    assert (
        render_harm_prompt(chat_args, method, "behavior") == "<chat>behavior<assistant>"
    )


def test_llama3_beavertails_prompt_format_can_use_native_chat() -> None:
    raw_args = build_parser().parse_args(["run", "--method", "llama3_dpo"])
    chat_args = build_parser().parse_args(
        [
            "run",
            "--method",
            "llama3_dpo",
            "--llama3-beavertails-prompt-format",
            "chat",
        ]
    )
    method = _FakeLlama3Method()
    assert render_beavertails_prompt(raw_args, method, "question") == "question"
    assert (
        render_beavertails_prompt(chat_args, method, "question")
        == "<chat>question<assistant>"
    )


def test_frozen_beavertails_paper_protocol_manifest() -> None:
    args = build_parser().parse_args(
        ["run", "--method", "llama3_base", "--tasks", "beavertails"]
    )
    rows = read_jsonl(args.beavertails)
    assert sha256_file(args.beavertails) == EXPECTED_BEAVERTAILS_SHA256
    assert len(rows) == 200
    assert len({row["id"] for row in rows}) == 200
    assert all(32796 <= row["source_index"] <= 33395 for row in rows)
