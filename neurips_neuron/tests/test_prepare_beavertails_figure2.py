import importlib.util
import json
import lzma
import random
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/data/prepare_beavertails_figure2.py"
SPEC = importlib.util.spec_from_file_location("prepare_beavertails_figure2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_extracts_prompts_without_labels_and_matches_released_sample(tmp_path: Path):
    source = tmp_path / "test.jsonl.xz"
    raw = tmp_path / "test.jsonl"
    examples = [
        {"prompt": f"prompt {i}", "response": f"response {i}", "is_safe": i % 2 == 0}
        for i in range(12)
    ]
    with lzma.open(source, "wt", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row) + "\n")

    records, counts = MODULE.decompress_and_extract(source, raw)
    selected = MODULE.released_sample(records, sample_size=2, seed=42, window_multiplier=3)
    expected = random.Random(42).sample(records[-6:], 2)

    assert counts == {"safe": 6, "unsafe": 6}
    assert selected == expected
    assert set(records[0]) == {"dataset", "id", "source_index", "prompt"}
    assert [json.loads(line) for line in raw.read_text().splitlines()] == examples
