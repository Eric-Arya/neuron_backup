import tempfile
import unittest
from pathlib import Path

from neuron_deactivate.run_table1_capability import gsm8k_prompts, resolve_dataset_path


class _Split(list):
    def select(self, indices):
        return _Split(self[index] for index in indices)


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.assertions = (messages, tokenize, add_generation_prompt)
        return f"<chat>{messages[0]['content']}<assistant>"


class Gsm8kPromptTest(unittest.TestCase):
    def setUp(self):
        self.dataset = {
            "train": _Split([{"question": "Demo?", "answer": "Reasoning. #### 1"}]),
            "test": _Split([{"question": "What is 2 + 2?"}]),
        }

    def test_chat_zero_shot_uses_template_and_cot_trigger(self):
        tokenizer = _Tokenizer()
        prompts = gsm8k_prompts(self.dataset, tokenizer, 0, "chat")
        self.assertEqual(
            prompts,
            [
                "<chat>Question: What is 2 + 2?\n"
                "Answer: Let's think step by step.<assistant>"
            ],
        )
        messages, tokenize, add_generation_prompt = tokenizer.assertions
        self.assertEqual(messages[0]["role"], "user")
        self.assertFalse(tokenize)
        self.assertTrue(add_generation_prompt)

    def test_legacy_raw_five_shot_shape_is_preserved(self):
        prompts = gsm8k_prompts(self.dataset, _Tokenizer(), 1, "raw")
        self.assertEqual(
            prompts,
            [
                "Question: Demo?\nAnswer: Reasoning. #### 1\n\n"
                "Question: What is 2 + 2?\nAnswer:"
            ],
        )

    def test_rejects_too_many_demonstrations(self):
        with self.assertRaisesRegex(ValueError, "num-fewshot"):
            gsm8k_prompts(self.dataset, _Tokenizer(), 2, "raw")

    def test_shared_dataset_layout_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "gsm8k" / "main"
            expected.mkdir(parents=True)
            self.assertEqual(resolve_dataset_path(root, "gsm8k"), expected)


if __name__ == "__main__":
    unittest.main()
