from pathlib import Path
import sys

from transformers import AutoTokenizer
from trl import DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.dpo import format_hh_preference


LLAMA3 = Path("/workspace/xcy/models/Meta-Llama-3-8B-Instruct")


def test_native_llama3_hh_format_has_exact_generation_boundary():
    tokenizer = AutoTokenizer.from_pretrained(LLAMA3, local_files_only=True)
    sample = {
        "chosen": "\n\nHuman: First question\n\nAssistant: First answer"
        "\n\nHuman: Final question\n\nAssistant: Preferred answer",
        "rejected": "\n\nHuman: First question\n\nAssistant: First answer"
        "\n\nHuman: Final question\n\nAssistant: Rejected answer",
    }

    row = format_hh_preference(sample, tokenizer, "native")

    assert row["prompt"].count("<|begin_of_text|>") == 1
    assert row["prompt"].endswith(
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    assert "First answer<|eot_id|>" in row["prompt"]
    assert "Final question<|eot_id|>" in row["prompt"]
    assert row["chosen"] == "Preferred answer"
    assert row["rejected"] == "Rejected answer"
    assert "<|eot_id|>" not in row["chosen"]

    tokenized = DPOTrainer.tokenize_row(
        row,
        tokenizer,
        max_prompt_length=2048,
        max_completion_length=2048,
        add_special_tokens=False,
    )
    assert tokenized["prompt_input_ids"][0] == tokenizer.bos_token_id
    assert tokenized["chosen_input_ids"][-1] == tokenizer.eos_token_id
    assert tokenized["chosen_input_ids"][-2] != tokenizer.eos_token_id


def test_tulu_hh_format_is_preserved():
    sample = {
        "chosen": "\n\nHuman: Question\n\nAssistant: Preferred",
        "rejected": "\n\nHuman: Question\n\nAssistant: Rejected",
    }

    row = format_hh_preference(sample, tokenizer=None, chat_format="tulu")

    assert row == {
        "prompt": "<|user|>\nQuestion\n<|assistant|>\n",
        "chosen": "Preferred\n",
        "rejected": "Rejected\n",
    }


def test_native_format_handles_rejected_multiturn_continuation():
    tokenizer = AutoTokenizer.from_pretrained(LLAMA3, local_files_only=True)
    sample = {
        "chosen": "\n\nHuman: Question\n\nAssistant: Preferred",
        "rejected": "\n\nHuman: Question\n\nAssistant: Rejected"
        "\n\nHuman: Follow-up\n\nAssistant: More rejected text",
    }

    row = format_hh_preference(sample, tokenizer, "native")

    assert row["chosen"] == "Preferred"
    assert row["rejected"].startswith("Rejected<|eot_id|>")
    assert "<|start_header_id|>user<|end_header_id|>\n\nFollow-up" in row["rejected"]
    assert row["rejected"].endswith("More rejected text")
