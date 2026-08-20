#!/usr/bin/env python3
"""Focused tests for expanded-K/V mapping and sparse SN-Tune updates."""

import unittest

import torch

try:
    from .train_neuron import SparseNeuronLinear, expanded_to_physical, select_indices
    from .scale_sn_tune_delta import scaled_tensor
except ImportError:
    from train_neuron import SparseNeuronLinear, expanded_to_physical, select_indices
    from scale_sn_tune_delta import scaled_tensor


class ExpandedKvMappingTest(unittest.TestCase):
    def test_head_aware_mapping(self):
        # Expanded head 4 is the first replica of physical K/V head 1.
        expanded_index = 4 * 128 + 7
        self.assertEqual(expanded_to_physical(expanded_index, 128, 4, "head-aware"), 128 + 7)
        self.assertEqual(
            expanded_to_physical(expanded_index, 128, 4, "paper-code-divide"),
            expanded_index // 4,
        )


class NeuronSelectionTest(unittest.TestCase):
    def test_file_order_preserves_detector_rank(self):
        self.assertEqual(select_indices([9, 3, 7], 2, "file-order"), {9, 3})

    def test_file_order_rejects_unordered_artifact(self):
        with self.assertRaises(ValueError):
            select_indices({9, 3, 7}, 2, "file-order")


class SparseNeuronLinearTest(unittest.TestCase):
    def test_empty_selection_does_not_create_trainable_parameter(self):
        base = torch.nn.Linear(5, 4, bias=False).to(dtype=torch.bfloat16)
        wrapped = SparseNeuronLinear(base, set(), "row")
        self.assertEqual(wrapped.delta.numel(), 0)
        self.assertFalse(wrapped.delta.requires_grad)

    def test_row_delta_only_changes_selected_rows(self):
        base = torch.nn.Linear(5, 4, bias=False).to(dtype=torch.bfloat16)
        original = base.weight.detach().float().clone()
        wrapped = SparseNeuronLinear(base, {1, 3}, "row")
        with torch.no_grad():
            wrapped.delta.fill_(1e-6)
        merged = wrapped.merge()
        difference = merged.weight.detach() - original
        self.assertTrue(torch.count_nonzero(difference[0]) == 0)
        self.assertTrue(torch.count_nonzero(difference[2]) == 0)
        self.assertTrue(torch.count_nonzero(difference[1]) > 0)
        self.assertTrue(torch.count_nonzero(difference[3]) > 0)

    def test_column_delta_only_changes_selected_columns(self):
        base = torch.nn.Linear(5, 4, bias=False).to(dtype=torch.bfloat16)
        original = base.weight.detach().float().clone()
        wrapped = SparseNeuronLinear(base, {0, 2, 4}, "column")
        with torch.no_grad():
            wrapped.delta.fill_(1e-6)
        merged = wrapped.merge()
        difference = merged.weight.detach() - original
        self.assertTrue(torch.count_nonzero(difference[:, 1]) == 0)
        self.assertTrue(torch.count_nonzero(difference[:, 3]) == 0)
        self.assertTrue(torch.count_nonzero(difference[:, 0]) > 0)
        self.assertTrue(torch.count_nonzero(difference[:, 2]) > 0)
        self.assertTrue(torch.count_nonzero(difference[:, 4]) > 0)


class DeltaScalingTest(unittest.TestCase):
    def test_scaled_tensor_amplifies_delta_from_base(self):
        base = torch.tensor([1.0, 2.0])
        tuned = torch.tensor([1.5, 1.0])
        result = scaled_tensor(base, tuned, 4.0)
        torch.testing.assert_close(result, torch.tensor([3.0, -2.0]))


if __name__ == "__main__":
    unittest.main()
