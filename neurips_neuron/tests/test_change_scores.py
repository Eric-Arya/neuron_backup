from types import SimpleNamespace

from src import change_scores


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<bos><user>{messages[0]['content']}<assistant>"


def test_native_change_score_prompts_use_tokenizer_template(monkeypatch):
    monkeypatch.setattr(
        change_scores.AutoTokenizer,
        "from_pretrained",
        lambda _path: FakeTokenizer(),
    )
    dataset = [{"chosen": "\n\nHuman: hello\n\nAssistant: response"}]
    args = SimpleNamespace(
        chat_format="native",
        tokenizer_name_or_path="unused",
        generation_startswith="",
    )
    assert change_scores.format_prompts(dataset, args) == [
        "<bos><user>hello<assistant>"
    ]


def test_tulu_change_score_prompts_remain_available():
    dataset = [{"prompt": "hello"}]
    args = SimpleNamespace(
        chat_format="tulu",
        tokenizer_name_or_path="unused",
        generation_startswith="Answer:",
    )
    assert change_scores.format_prompts(dataset, args) == [
        "<|user|>\nhello\n<|assistant|>\nAnswer:"
    ]


def test_raw_change_score_prompts_preserve_sn_training_boundary():
    dataset = [{"prompt": "hello?"}]
    args = SimpleNamespace(
        chat_format="raw",
        tokenizer_name_or_path="unused",
        generation_startswith=".",
    )
    assert change_scores.format_prompts(dataset, args) == ["hello?."]
