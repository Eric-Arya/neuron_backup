#!/usr/bin/env python3
"""Generate an origin-model baseline with the same patched environment and prompt pipeline."""

from __future__ import annotations

import argparse
import os

try:
    from .run_table1_capability import no_op_factory
    from .run_table1_deactivated import verify_deactivation_transformers
    from .table1_generation_common import add_common_arguments, run_generation
except ImportError:
    from run_table1_capability import no_op_factory
    from run_table1_deactivated import verify_deactivation_transformers
    from table1_generation_common import add_common_arguments, run_generation


def main() -> int:
    verify_deactivation_transformers()
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "strongreject/origin")
    args = parser.parse_args()
    try:
        return run_generation(args, "origin", no_op_factory)
    except (OSError, RuntimeError, ValueError) as exc:
        if os.environ.get("NEURON_DEBUG") == "1":
            raise
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
