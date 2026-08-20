#!/usr/bin/env python3
"""Run the stock/global Transformers Llama-3 baseline and score every output."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import transformers
from transformers.generation.utils import GenerationMixin

from table1_generation_common import add_common_arguments, run_generation


def verify_global_transformers() -> None:
    transformers_path = Path(transformers.__file__).resolve()
    if "/miniconda3/" in str(transformers_path):
        raise RuntimeError(
            f"Baseline must use the global Transformers package, found {transformers_path}. "
            "Run this script with /usr/bin/python3."
        )
    if "activate_keys_fwd_up_set" in inspect.signature(GenerationMixin.generate).parameters:
        raise RuntimeError("Baseline Transformers unexpectedly contains the deactivation overlay")


def baseline_factory(_config) -> tuple[dict[str, object], dict[str, object]]:
    return {}, {"deactivation": False}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "baseline_raw")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    verify_global_transformers()
    try:
        return run_generation(args, "baseline", baseline_factory)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
