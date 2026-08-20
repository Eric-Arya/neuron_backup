import unittest
from types import SimpleNamespace

import torch

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from transformers.models.llama.configuration_llama import LlamaConfig


class NeuronScalingTest(unittest.TestCase):
    def test_mlp_scaling_is_runtime_only_and_does_not_compound(self):
        config = SimpleNamespace(
            hidden_size=2,
            intermediate_size=2,
            mlp_bias=False,
            hidden_act="silu",
            pretraining_tp=1,
        )
        mlp = LlamaMLP(config)
        with torch.no_grad():
            mlp.gate_proj.weight.copy_(torch.eye(2))
            mlp.up_proj.weight.copy_(torch.eye(2))
            mlp.down_proj.weight.copy_(torch.eye(2))
        original_weights = {
            name: parameter.detach().clone() for name, parameter in mlp.named_parameters()
        }
        inputs = torch.tensor([[[1.0, 2.0]]])

        first = mlp(inputs, {0}, set(), 0.5)
        repeated = mlp(inputs, {0}, set(), 0.5)
        baseline = mlp(inputs, set(), set(), 0.0)

        self.assertTrue(torch.equal(first, repeated))
        self.assertTrue(torch.allclose(first[..., 0], baseline[..., 0] * 0.5))
        self.assertTrue(torch.equal(first[..., 1], baseline[..., 1]))
        for name, parameter in mlp.named_parameters():
            self.assertTrue(torch.equal(parameter, original_weights[name]))

    def test_expanded_kv_scaling_targets_one_replica_channel(self):
        config = LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=1,
        )
        attention = LlamaAttention(config, layer_idx=0)
        states = torch.ones(1, 2, 1, 4)
        scaled = attention._scale_channels(
            states, {5}, 2.0, "_test_expanded_multiplier", flattened=True
        )
        expected = states.clone()
        expected[:, 1, :, 1] = 2.0
        self.assertTrue(torch.equal(scaled, expected))

    def test_zero_scale_matches_hard_mask_and_one_is_no_op(self):
        config = LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=1,
        )
        attention = LlamaAttention(config, layer_idx=0)
        states = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
        zeroed = attention._scale_channels(states, {2, 6}, 0.0, "_test_zero")
        expected = states.clone()
        expected[..., [2, 6]] = 0
        self.assertTrue(torch.equal(zeroed, expected))
        self.assertIs(attention._scale_channels(states, {2}, 1.0, "_test_one"), states)


if __name__ == "__main__":
    unittest.main()
