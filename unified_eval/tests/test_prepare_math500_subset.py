from __future__ import annotations

import pytest

from scripts.prepare_math500_subset import proportional_allocation


def test_math500_l1_l3_allocation_matches_source_ratios() -> None:
    assert proportional_allocation({1: 43, 2: 90, 3: 105}, 50) == {
        1: 9,
        2: 19,
        3: 22,
    }


def test_proportional_allocation_rejects_oversampling() -> None:
    with pytest.raises(ValueError, match="eligible pool"):
        proportional_allocation({1: 2, 2: 3}, 6)
