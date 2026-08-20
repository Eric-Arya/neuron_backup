import torch

from unified_eval.methods import (
    DEFAULT_SN_DIRECT_NEURONS,
    SN_STRUCTURES,
    load_sn_direct_selection,
    scale_ia3_displacement_,
    scale_hooked_ia3_displacement_,
)


def test_ia3_scaling_extrapolates_from_identity() -> None:
    class TinyIa3(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ia3_l = torch.nn.ParameterDict(
                {"default": torch.nn.Parameter(torch.tensor([0.8, 1.0, 1.3]))}
            )

    model = TinyIa3()
    assert scale_ia3_displacement_(model, 2.0) == 3
    assert torch.allclose(
        model.ia3_l["default"], torch.tensor([0.6, 1.0, 1.6])
    )


def test_ia3_alpha_zero_recovers_identity() -> None:
    class TinyIa3(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ia3_l = torch.nn.ParameterDict(
                {"default": torch.nn.Parameter(torch.tensor([0.8, 1.3]))}
            )

    model = TinyIa3()
    scale_ia3_displacement_(model, 0.0)
    assert torch.equal(model.ia3_l["default"], torch.ones(2))


def test_hooked_ia3_scaling_extrapolates_plain_tensor_lists() -> None:
    class TinyHookedIa3(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.ia3_l = [torch.tensor([0.8, 1.0, 1.3])]

    model = TinyHookedIa3()
    assert scale_hooked_ia3_displacement_(model, 3.0) == 3
    assert torch.allclose(model.mlp.ia3_l[0], torch.tensor([0.4, 1.0, 1.9]))


def test_sn_direct_selection_matches_raw_sn_tune_cap25_manifest() -> None:
    selected, selection_hash = load_sn_direct_selection(
        DEFAULT_SN_DIRECT_NEURONS,
        num_layers=32,
        intermediate_size=14336,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        cap=25,
    )
    assert selection_hash == "6d4889c95431fa6e5a95726c486e811033b6d982e211cd9335be0481f27a725f"
    assert {
        name: sum(len(indices) for indices in layers.values())
        for name, layers in zip(SN_STRUCTURES, selected)
    } == {"fwd_up": 301, "fwd_down": 301, "q": 794, "k": 748, "v": 205}
