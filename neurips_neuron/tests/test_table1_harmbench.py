import json
from pathlib import Path

import pytest

from eval.table1_harmbench import (
    aggregate_results,
    freeze_manifest,
    load_shards,
    tulu_prompt,
    write_batch_shard,
)


def _manifest(count=3):
    return [{"id": f"harmbench_{i}", "prompt": f"prompt {i}"} for i in range(count)]


def test_tulu_prompt_matches_released_template():
    assert tulu_prompt("hello") == "<|user|>\nhello\n<|assistant|>\n"


def test_manifest_is_frozen(tmp_path: Path):
    source = tmp_path / "source.jsonl"
    frozen = tmp_path / "run" / "prompt_manifest.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in _manifest()), encoding="utf-8")
    assert freeze_manifest(source, frozen, 3) == _manifest()
    source.write_text(json.dumps({"id": "changed", "prompt": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen manifest"):
        freeze_manifest(source, frozen, None)


def test_shards_reject_duplicate_ids(tmp_path: Path):
    directory = tmp_path / "shards"
    directory.mkdir()
    row = {"id": "same"}
    (directory / "batch_000000_000000.jsonl").write_text(json.dumps(row) + "\n")
    (directory / "batch_000001_000001.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Duplicate id"):
        load_shards(directory)


def test_write_shard_refuses_overwrite(tmp_path: Path):
    records = [{"id": "a"}, {"id": "b"}]
    positions = {"a": 0, "b": 1}
    path = write_batch_shard(tmp_path, positions, records)
    assert path.name == "batch_000000_000001.jsonl"
    with pytest.raises(FileExistsError):
        write_batch_shard(tmp_path, positions, records)


def test_aggregate_uses_paper_formula(tmp_path: Path):
    manifest = _manifest(2)
    values = {"base": 8.0, "patched": -3.9, "dpo": -11.0}
    positions = {row["id"]: i for i, row in enumerate(manifest)}
    for condition, cost in values.items():
        rows = [{"id": row["id"], "condition": condition, "cost": cost} for row in manifest]
        write_batch_shard(tmp_path / "costs" / condition, positions, rows)
    result = aggregate_results(manifest, tmp_path, top_k=20_000, ranking_count=341_248)
    assert result["causal_effect_percent"] == pytest.approx(62.6315789474)
    assert result["selected_fraction_percent"] == pytest.approx(5.8608403859)
    assert result["paper_reference"]["within_5_percentage_points"] is True
