from __future__ import annotations

from unified_eval.common import (
    extract_gsm_answer,
    extract_gsm_flexible_answer,
    extract_gsm_strict_answer,
    score_asr,
    tulu_chat,
)


def test_asr_is_case_sensitive_and_matches_iclr_rule() -> None:
    assert score_asr("I cannot help with that.")[0] is False
    assert score_asr("i cannot help with that.")[0] is True
    assert score_asr("Here are the requested steps.")[0] is True


def test_gsm_extraction_matches_table2_reproduction() -> None:
    assert extract_gsm_answer("work\n#### 1,250") == "1250"
    assert extract_gsm_answer("first 7, final -2.50") == "-2.5"
    assert extract_gsm_answer("no numeric answer") is None


def test_strict_gsm_extraction_requires_and_uses_final_hash_delimiter() -> None:
    assert extract_gsm_strict_answer("work 16 = 288 over 16 days") is None
    assert extract_gsm_strict_answer("work\n#### $1,250\ntrailing 16") == "1250"
    assert extract_gsm_strict_answer("#### 18\nrevision\n#### -2.50") == "-2.5"
    assert extract_gsm_strict_answer("work\n#### 18\nQuestion: hallucinated\n#### 9") == "18"


def test_flexible_gsm_extraction_priority() -> None:
    assert extract_gsm_flexible_answer("work = 288 over 16 days\n#### 18") == (
        "18", "hash_delimiter"
    )
    assert extract_gsm_flexible_answer("Therefore, the answer is $1,250.") == (
        "1250", "answer_phrase"
    )
    assert extract_gsm_flexible_answer("The answer is 50 / 20 = 2.5 cups.") == (
        "2.5", "answer_phrase_equation_rhs"
    )
    assert extract_gsm_flexible_answer("In total, $18 * 16 = $288 over 16 days.") == (
        "288", "final_equation_rhs"
    )
    assert extract_gsm_flexible_answer("No conclusion; values 3 and 7.") == (
        "7", "last_number_fallback"
    )
    assert extract_gsm_flexible_answer("No numeric claim.") == (None, None)


def test_tulu_multi_turn_ends_at_assistant_header() -> None:
    prompt = tulu_chat(
        [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "Q2"},
        ]
    )
    assert prompt == (
        "<|user|>\nQ1\n<|assistant|>\nA</s>\n<|user|>\nQ2\n<|assistant|>\n"
    )
