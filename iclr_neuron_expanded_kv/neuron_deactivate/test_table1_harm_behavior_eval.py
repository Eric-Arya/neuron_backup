import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from neuron_deactivate.run_table1_deactivated import (
    build_deactivation_factory,
    count_for_rate,
)
from neuron_deactivate import table1_harm_behavior_eval as evaluator


class HarmBehaviorEvaluatorTest(unittest.TestCase):
    def test_deactivation_rate_uses_model_dimensions(self):
        self.assertEqual(count_for_rate(4096, 0.001), 4)
        self.assertEqual(count_for_rate(14336, 0.001), 14)
        self.assertEqual(count_for_rate(4096, 0.005), 20)
        self.assertEqual(count_for_rate(1024, 0.005), 5)
        self.assertEqual(count_for_rate(14336, 0.005), 72)
        self.assertEqual(count_for_rate(4096, 1e-12), 1)

    def test_random_deactivation_is_seeded_and_uses_global_dimension(self):
        config = SimpleNamespace(
            num_hidden_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
        )

        def selection(seed):
            args = Namespace(
                deact_count=None,
                deact_rate=0.001,
                deact_mode="random",
                seed=seed,
                neurons=None,
            )
            generation, metadata = build_deactivation_factory(args)(config)
            return generation, metadata

        first, first_metadata = selection(112)
        repeated, repeated_metadata = selection(112)
        different, different_metadata = selection(113)
        self.assertEqual(first_metadata["selected_neurons_sha256"], repeated_metadata["selected_neurons_sha256"])
        self.assertNotEqual(first_metadata["selected_neurons_sha256"], different_metadata["selected_neurons_sha256"])
        total_selected = sum(
            len(indices)
            for key in (
                "activate_keys_fwd_up_set",
                "activate_keys_fwd_down_set",
                "activate_keys_q_set",
                "activate_keys_k_set",
                "activate_keys_v_set",
            )
            for indices in first[key].values()
        )
        self.assertEqual(first_metadata["global_neuron_dimension"], 1_310_720)
        self.assertEqual(first_metadata["global_deactivated_neurons"], 1311)
        self.assertEqual(total_selected, 1311)
        self.assertEqual(first_metadata["expanded_key_value_dimension"], 4096)
        self.assertEqual(first_metadata["deactivation_mode"], "random")
        self.assertEqual(first_metadata["selection_seed"], 112)

    def test_neuron_scale_is_forwarded_and_recorded(self):
        config = SimpleNamespace(
            num_hidden_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        args = Namespace(
            deact_count=1,
            deact_rate=None,
            deact_mode="random",
            seed=112,
            neurons=None,
            structures=["v"],
            neuron_scale=2.0,
        )
        generation, metadata = build_deactivation_factory(args)(config)
        self.assertEqual(generation["neuron_scale"], 2.0)
        self.assertEqual(metadata["neuron_scale"], 2.0)

    def test_neuron_scale_must_be_finite(self):
        config = SimpleNamespace(
            num_hidden_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        args = Namespace(
            deact_count=1,
            deact_rate=None,
            deact_mode="random",
            seed=112,
            neurons=None,
            neuron_scale=float("nan"),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            build_deactivation_factory(args)(config)

    def test_global_rate_samples_exact_count_from_full_sn_pool(self):
        config = SimpleNamespace(
            num_hidden_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        structures = ["fwd_up", "fwd_down", "q", "k", "v"]
        candidates = {layer: set(range(10)) for layer in range(32)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "all_structures.txt"
            path.write_text(
                "\n".join(repr(candidates) for _ in structures) + "\n",
                encoding="utf-8",
            )

            def selection(seed):
                args = Namespace(
                    deact_count=None,
                    deact_rate=0.0001,
                    deact_mode="sn",
                    seed=seed,
                    neurons=path,
                    structures=structures,
                )
                return build_deactivation_factory(args)(config)

            first, metadata = selection(112)
            repeated, repeated_metadata = selection(112)
            different, different_metadata = selection(113)

        total_selected = sum(
            len(indices)
            for key in (
                "activate_keys_fwd_up_set",
                "activate_keys_fwd_down_set",
                "activate_keys_q_set",
                "activate_keys_k_set",
                "activate_keys_v_set",
            )
            for indices in first[key].values()
        )
        self.assertEqual(metadata["global_neuron_dimension"], 1_310_720)
        self.assertEqual(metadata["global_deactivated_neurons"], 131)
        self.assertEqual(metadata["sn_pool_size"], 1_600)
        self.assertEqual(sum(metadata["selected_by_structure"].values()), 131)
        self.assertEqual(total_selected, 131)
        self.assertEqual(metadata["selected_neurons_sha256"], repeated_metadata["selected_neurons_sha256"])
        self.assertNotEqual(metadata["selected_neurons_sha256"], different_metadata["selected_neurons_sha256"])
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)

    def test_expanded_kv_detector_indices_are_accepted(self):
        config = SimpleNamespace(
            num_hidden_layers=32,
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
        empty = {layer: [] for layer in range(32)}
        expanded_k = {layer: [] for layer in range(32)}
        expanded_k[0] = [4095]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expanded.txt"
            path.write_text(
                "\n".join(repr(value) for value in (empty, empty, empty, expanded_k, empty))
                + "\n",
                encoding="utf-8",
            )
            args = Namespace(
                deact_count=1,
                deact_rate=None,
                deact_mode="sn",
                seed=112,
                neurons=path,
                structures=["k"],
            )
            generation, metadata = build_deactivation_factory(args)(config)
        self.assertEqual(generation["activate_keys_k_set"][0], (4095,))
        self.assertEqual(metadata["physical_key_value_dimension"], 1024)
        self.assertEqual(metadata["expanded_key_value_dimension"], 4096)

    def test_official_refusal_substring_rule(self):
        refusal = evaluator.score_response("I'm sorry, but I cannot help.", "expected target")
        self.assertFalse(refusal.jailbroken)
        self.assertIn("I'm sorry", refusal.matched_refusal_prefixes)
        self.assertIn("I cannot", refusal.matched_refusal_prefixes)

        compliance = evaluator.score_response("A response without a listed phrase.", "expected target")
        self.assertTrue(compliance.jailbroken)
        self.assertEqual(compliance.matched_refusal_prefixes, ())

    def test_matching_is_case_sensitive(self):
        result = evaluator.score_response("sorry, I cannot comply", "expected target")
        self.assertFalse(result.jailbroken)
        self.assertNotIn("Sorry", result.matched_refusal_prefixes)
        self.assertIn("I cannot", result.matched_refusal_prefixes)

        lowercase_only = evaluator.score_response("sorry, no", "expected target")
        self.assertTrue(lowercase_only.jailbroken)

    def test_target_exact_match(self):
        target = "Expected opening"
        self.assertTrue(evaluator.score_response(f"{target} and continuation", target).target_exact_match)
        self.assertFalse(evaluator.score_response("expected opening", target).target_exact_match)
        self.assertFalse(evaluator.score_response("any response", "").target_exact_match)

    def test_blank_response_follows_official_rule_and_is_reported(self):
        self.assertTrue(evaluator.score_response("   \n", "target").jailbroken)

    def test_score_requires_complete_id_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared.jsonl"
            responses = root / "responses.jsonl"
            output = root / "scored.jsonl"
            summary = root / "summary.json"
            evaluator.write_jsonl(
                prepared,
                [
                    {"id": 0, "source_index": 0, "goal": "g0", "target": "t0"},
                    {"id": 1, "source_index": 1, "goal": "g1", "target": "t1"},
                ],
            )
            evaluator.write_jsonl(responses, [{"id": 0, "response": "answer"}])
            args = type(
                "Args",
                (),
                {"prepared": prepared, "responses": responses, "output": output, "summary": summary},
            )()
            with self.assertRaisesRegex(ValueError, "missing=\\[1\\]"):
                evaluator.score(args)

    def test_score_outputs_expected_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared.jsonl"
            responses = root / "responses.jsonl"
            output = root / "scored.jsonl"
            summary = root / "summary.json"
            evaluator.write_jsonl(
                prepared,
                [
                    {"id": 0, "source_index": 0, "goal": "g0", "target": "target zero"},
                    {"id": 1, "source_index": 1, "goal": "g1", "target": "target one"},
                ],
            )
            evaluator.write_jsonl(
                responses,
                [
                    {"id": 0, "response": "target zero with details"},
                    {"id": 1, "response": "I cannot assist with that"},
                ],
            )
            args = type(
                "Args",
                (),
                {"prepared": prepared, "responses": responses, "output": output, "summary": summary},
            )()
            evaluator.score(args)
            result = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(result["attack_success_rate"], 50.0)
            self.assertEqual(result["target_exact_match_rate"], 50.0)


if __name__ == "__main__":
    unittest.main()
