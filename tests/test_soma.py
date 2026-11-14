# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

import unittest
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import optuna
import torch
from optuna import Trial
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear
from torch import Tensor, nn

from heretic.config import Settings as HereticSettings
from heretic.modifier import Modifier
from heretic.modifiers.soma import (
    SOMA,
    DirectionWeightDistribution,
    Parameters,
    RowNormalization,
    Settings,
    SOMCalculator,
    compute_refusal_directions,
    decompose_delta,
)
from heretic.plugin import Context, load_plugin


def make_soma(
    overrides: dict[str, Any] | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> SOMA:
    """
    Builds a SOMA modifier instance with (possibly overridden) settings,
    without touching a model.
    """
    settings = Settings(**(overrides or {}), **kwargs)
    return SOMA(
        heretic_settings=cast(HereticSettings, SimpleNamespace(seed=seed)),
        settings=settings,
    )


class SOMCalculatorTests(unittest.TestCase):
    def test_deterministic_training(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(size=(100, 4))

        calc_a = SOMCalculator(3, 3, 200, 0.1, 1.0)
        calc_a.fit(data)

        calc_b = SOMCalculator(3, 3, 200, 0.1, 1.0)
        calc_b.fit(data)

        self.assertTrue(
            np.array_equal(
                calc_a.get_top_k_neuron_weights(3),
                calc_b.get_top_k_neuron_weights(3),
            )
        )

    def test_top_k_weights_shape(self) -> None:
        # Three well-separated clusters in 4D.
        rng = np.random.default_rng(1)
        data = np.vstack(
            [
                rng.normal(loc=(10, 0, 0, 0), scale=0.5, size=(50, 4)),
                rng.normal(loc=(0, 10, 0, 0), scale=0.5, size=(50, 4)),
                rng.normal(loc=(0, 0, 10, 0), scale=0.5, size=(50, 4)),
            ]
        )

        calc = SOMCalculator(4, 4, 500, 0.1, 1.0)
        calc.fit(data)
        weights = calc.get_top_k_neuron_weights(3)

        # At most 3 weights, each a 4D vector.
        self.assertLessEqual(weights.shape[0], 3)
        self.assertEqual(weights.shape[1], 4)

    def test_de_dups_close_neurons(self) -> None:
        # Zero-variance data: every neuron converges onto the same position,
        # so the winner-frequency de-duplication must return fewer distinct
        # neurons than k.
        data = np.ones((100, 4), dtype=np.float64) * 5

        calc = SOMCalculator(4, 4, 200, 0.1, 1.0)
        calc.fit(data)
        weights = calc.get_top_k_neuron_weights(4)

        self.assertLess(weights.shape[0], 4)

    def test_requires_2d_data(self) -> None:
        calc = SOMCalculator(2, 2, 10, 0.1, 1.0)
        with self.assertRaises(ValueError):
            calc.fit(np.zeros(10))

    def test_requires_training(self) -> None:
        calc = SOMCalculator(2, 2, 10, 0.1, 1.0)
        with self.assertRaises(RuntimeError):
            calc.get_top_k_neuron_weights(1)


class ObtainSomMeansTests(unittest.TestCase):
    def test_pads_de_duplicated_neurons_up_to_som_k(self) -> None:
        # A single tight cluster per layer: the SOM finds fewer than som_k
        # distinct neurons, which must be repeated back up to som_k.
        soma = make_soma(
            {
                "som_x": 3,
                "som_y": 3,
                "som_iterations": 200,
                "som_lr": 0.1,
                "som_sigma": 0.5,
                "som_k": 3,
            }
        )

        # Zero-variance residuals per layer: each layer's SOM finds a single
        # distinct neuron, which must be repeated back up to som_k.
        # (n_prompts, n_layers, hidden_dim)
        bad_residuals = torch.ones(50, 4, 8) * 5

        means = soma.obtain_som_means(bad_residuals)

        self.assertEqual(means.shape, (3, 4, 8))

    def test_multiple_layers(self) -> None:
        soma = make_soma(
            {
                "som_x": 2,
                "som_y": 2,
                "som_iterations": 50,
                "som_lr": 0.1,
                "som_sigma": 1.0,
                "som_k": 2,
            }
        )

        rng = torch.Generator().manual_seed(1)
        bad_residuals = torch.randn(30, 6, 5, generator=rng)

        means = soma.obtain_som_means(bad_residuals)

        self.assertEqual(means.shape, (2, 6, 5))


class DirectionComputationTests(unittest.TestCase):
    def test_shapes_and_unit_norms(self) -> None:
        good_means = torch.randn(5, 8)  # (n_layers, hidden_dim)
        bad_means = torch.randn(2, 5, 8)  # (n_directions, n_layers, hidden_dim)

        directions = compute_refusal_directions(
            good_means,
            bad_means,
            orthogonalize_direction=False,
        )

        self.assertEqual(directions.shape, (5, 2, 8))

        norms = torch.linalg.vector_norm(directions, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_single_direction_matches_classic_formula(self) -> None:
        good_means = torch.randn(5, 8)
        bad_means = torch.randn(5, 8).unsqueeze(0)  # (1, n_layers, hidden_dim)

        directions = compute_refusal_directions(
            good_means,
            bad_means,
            orthogonalize_direction=True,
        )

        # Classic projected-abliteration direction.
        expected = torch.nn.functional.normalize(bad_means[0] - good_means, p=2, dim=1)
        good_directions = torch.nn.functional.normalize(good_means, p=2, dim=1)
        projection = torch.sum(expected * good_directions, dim=1)
        expected = expected - projection.unsqueeze(1) * good_directions
        expected = torch.nn.functional.normalize(expected, p=2, dim=1)

        # directions has shape (n_layers, 1, hidden_dim).
        self.assertTrue(torch.allclose(directions[:, 0], expected, atol=1e-5))

    def test_orthogonalization_removes_good_component(self) -> None:
        good_means = torch.randn(5, 8)
        bad_means = torch.randn(3, 5, 8)

        directions = compute_refusal_directions(
            good_means,
            bad_means,
            orthogonalize_direction=True,
        )

        good_directions = torch.nn.functional.normalize(good_means, p=2, dim=1)

        for direction_index in range(3):
            for layer_index in range(5):
                dot = torch.sum(
                    directions[layer_index, direction_index]
                    * good_directions[layer_index]
                )
                self.assertLess(abs(dot.item()), 1e-4)


class DecomposeDeltaTests(unittest.TestCase):
    def test_recovers_rank1_delta(self) -> None:
        rng = torch.Generator().manual_seed(0)
        b = torch.randn(10, 1, generator=rng)
        a = torch.randn(1, 12, generator=rng)
        delta = b @ a

        lora_B, lora_A = decompose_delta(delta, rank=1, seed=42)

        self.assertTrue(torch.allclose(lora_B @ lora_A, delta, atol=1e-3))

    def test_recovers_multi_rank_delta(self) -> None:
        rng = torch.Generator().manual_seed(1)
        # A sum of 3 rank-1 matrices (the SOMA multi-direction case).
        delta = torch.zeros(16, 24)
        for _ in range(3):
            delta += torch.randn(16, 1, generator=rng) @ torch.randn(
                1, 24, generator=rng
            )

        lora_B, lora_A = decompose_delta(delta, rank=3, seed=42)

        error = torch.linalg.vector_norm(lora_B @ lora_A - delta)
        magnitude = torch.linalg.vector_norm(delta)
        self.assertLess(error / magnitude, 0.01)

    def test_deterministic_with_seed(self) -> None:
        rng = torch.Generator().manual_seed(2)
        delta = torch.randn(16, 24, generator=rng)

        b1, a1 = decompose_delta(delta, rank=4, seed=42)
        b2, a2 = decompose_delta(delta, rank=4, seed=42)

        self.assertTrue(torch.equal(b1, b2))
        self.assertTrue(torch.equal(a1, a2))

    def test_rank_exceeding_matrix_size(self) -> None:
        # The q = min(2r + 4, min(shape)) clamp must keep this working:
        # the effective rank is clamped to the matrix size.
        rng = torch.Generator().manual_seed(3)
        delta = torch.randn(4, 6, generator=rng)

        lora_B, lora_A = decompose_delta(delta, rank=100, seed=42)

        expected_rank = min(100, 4, 6)
        self.assertEqual(lora_B.shape, (4, expected_rank))
        self.assertEqual(lora_A.shape, (expected_rank, 6))


class LoraRankTests(unittest.TestCase):
    # SOMA is always multi-directional, so the som_k rank floor is always
    # active unless the user sets an explicit full_normalization_lora_rank.

    def test_default_full_normalization_floored(self) -> None:
        # Default full_normalization_lora_rank is 4, som_k is 4.
        soma = make_soma()
        self.assertEqual(soma.lora_rank, 4)

    def test_floor_at_som_k(self) -> None:
        soma = make_soma(som_k=5)
        self.assertEqual(soma.lora_rank, 5)

    def test_explicit_rank_wins(self) -> None:
        # An explicitly set rank takes precedence over the som_k lower bound.
        soma = make_soma(som_k=5, full_normalization_lora_rank=32)
        self.assertEqual(soma.lora_rank, 32)

    def test_non_full_normalization_floored(self) -> None:
        # Without renormalization the per-direction deltas are rank 1 each,
        # but the adapter is still floored at som_k to store all directions.
        soma = make_soma(row_normalization=RowNormalization.NONE)
        self.assertEqual(soma.lora_rank, 4)

    def test_explicit_rank_wins_over_som_k_floor(self) -> None:
        # An explicitly set rank takes precedence over the som_k lower bound,
        # even when it is below som_k.
        soma = make_soma(som_k=5, full_normalization_lora_rank=2)
        self.assertEqual(soma.lora_rank, 2)

    def test_non_full_ignores_explicit_full_rank(self) -> None:
        # full_normalization_lora_rank only applies to FULL; without it, the
        # rank is always som_k even if full_normalization_lora_rank is set.
        soma = make_soma(
            row_normalization=RowNormalization.NONE, full_normalization_lora_rank=32
        )
        self.assertEqual(soma.lora_rank, 4)


class ParametersTests(unittest.TestCase):
    def make_parameters(self) -> Parameters:
        return Parameters(
            direction_index=20.83,
            weight_distributions={
                "attn.o_proj": DirectionWeightDistribution(
                    max_weights=[1.28, 1.32, 1.14],
                    max_weight_position=23.56,
                    min_weights=[0.69, 0.13, 1.22],
                    min_weight_distance=12.4,
                ),
                "mlp.down_proj": DirectionWeightDistribution(
                    max_weights=[0.04, 0.53],
                    max_weight_position=43.41,
                    min_weights=[0.02, 0.42],
                    min_weight_distance=4.42,
                ),
            },
        )

    def test_dict_roundtrip(self) -> None:
        parameters = self.make_parameters()

        restored = Parameters.from_dict(parameters.to_dict())

        self.assertEqual(restored, parameters)

    def test_presentation_dict(self) -> None:
        parameters = self.make_parameters()

        presentation = parameters.to_presentation_dict()

        # direction_index comes first.
        self.assertEqual(next(iter(presentation)), "direction_index")
        self.assertEqual(presentation["direction_index"], "20.83")

        # Per-direction values are rendered as the value only (the index is
        # already carried by the key).
        self.assertEqual(presentation["attn.o_proj.max_weights.0"], "1.28")
        self.assertEqual(presentation["attn.o_proj.max_weights.1"], "1.32")
        self.assertEqual(presentation["attn.o_proj.max_weights.2"], "1.14")
        self.assertEqual(presentation["attn.o_proj.max_weight_position"], "23.56")
        self.assertEqual(presentation["attn.o_proj.min_weights.0"], "0.69")
        self.assertEqual(presentation["attn.o_proj.min_weight_distance"], "12.40")
        self.assertEqual(presentation["mlp.down_proj.max_weights.1"], "0.53")

    def test_presentation_dict_per_layer(self) -> None:
        parameters = Parameters(
            direction_index=None,
            weight_distributions={},
        )

        self.assertEqual(
            parameters.to_presentation_dict()["direction_index"], "per layer"
        )

    def test_render_trial_parameters(self) -> None:
        # The integration point used by main.py and the README generator.
        soma = make_soma()

        parameters = self.make_parameters()
        trial = cast(
            Trial, SimpleNamespace(user_attrs={"parameters": parameters.to_dict()})
        )

        rendered = soma.render_trial_parameters(trial)

        self.assertEqual(rendered["direction_index"], "20.83")
        self.assertEqual(rendered["mlp.down_proj.max_weight_position"], "43.41")


class PluginContractTests(unittest.TestCase):
    def test_loads_as_modifier_plugin(self) -> None:
        plugin_cls = load_plugin("heretic.modifiers.soma.SOMA", Modifier)
        self.assertIs(plugin_cls, SOMA)

    def test_validate_contract(self) -> None:
        SOMA.validate_contract()

    def test_settings_model(self) -> None:
        self.assertIs(SOMA.get_settings_model(), Settings)

    def test_settings_raw_filters_unknown_keys(self) -> None:
        model_extra = {
            "modifier": {
                "SOMA": {
                    "orthogonalize_direction": True,
                    "som_k": 8,
                    "unknown_key": "dropped",
                },
                "ARA": {
                    "lora_rank": 50,
                },
            }
        }

        raw = SOMA.get_settings_raw(model_extra, "modifier", None)

        self.assertEqual(raw["orthogonalize_direction"], True)
        self.assertEqual(raw["som_k"], 8)
        self.assertNotIn("unknown_key", raw)
        self.assertNotIn("lora_rank", raw)

    def test_instance_namespacing(self) -> None:
        model_extra = {
            "modifier": {
                "SOMA": {
                    "som_k": 4,
                },
                "SOMA_custom": {
                    "som_k": 8,
                },
            }
        }

        base_raw = SOMA.get_settings_raw(model_extra, "modifier", None)
        instance_raw = SOMA.get_settings_raw(model_extra, "modifier", "custom")

        self.assertEqual(base_raw["som_k"], 4)
        self.assertEqual(instance_raw["som_k"], 8)

    def test_modifier_names(self) -> None:
        cases = [
            (
                {
                    "orthogonalize_direction": True,
                    "row_normalization": "full",
                },
                "SOMPOA",
            ),
            (
                {
                    "orthogonalize_direction": True,
                    "row_normalization": "none",
                },
                "Projected SOMA",
            ),
            (
                {
                    "orthogonalize_direction": False,
                    "row_normalization": "none",
                },
                "SOMA",
            ),
        ]

        for overrides, expected_name in cases:
            with self.subTest(overrides=overrides):
                soma = make_soma(overrides)
                self.assertEqual(soma.modifier_name, expected_name)


class ModifyModuleTests(unittest.TestCase):
    # Wraps a small Linear module in PEFT, mimicking the module structure
    # that modify_model operates on.
    def make_peft_module(
        self,
        out_features: int,
        in_features: int,
        rank: int,
        seed: int = 0,
    ) -> tuple[Linear, Tensor]:
        rng = torch.Generator().manual_seed(seed)
        base_weight = torch.randn(out_features, in_features, generator=rng)

        class Base(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(in_features, out_features, bias=False)

            def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> dict:
                # Required by PEFT's CAUSAL_LM wrapper; unused in the tests.
                return {}

        base = Base()
        base.lin.weight.data = base_weight

        get_peft_model(
            base,
            LoraConfig(
                r=rank,
                target_modules=["lin"],
                lora_alpha=rank,
                lora_dropout=0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )

        # get_peft_model wraps base.lin in-place with a LoRA Linear,
        # with the base weight accessible via base_layer.
        module = cast(Linear, base.lin)
        return module, base_weight

    def test_none_mode_multi_direction_delta(self) -> None:
        soma = make_soma(row_normalization=RowNormalization.NONE)

        out_features, in_features = 8, 10
        n_directions = 3
        rank = n_directions

        module, W = self.make_peft_module(out_features, in_features, rank)

        rng = torch.Generator().manual_seed(1)
        directions = torch.nn.functional.normalize(
            torch.randn(n_directions, out_features, generator=rng),
            p=2,
            dim=-1,
        )
        layer_base_weight = 1.0
        max_weights = [1.0, 0.5, 2.0]
        params = DirectionWeightDistribution(
            max_weights=max_weights,
            max_weight_position=0.0,
            min_weights=[0.0, 0.0, 0.0],
            min_weight_distance=1.0,
        )

        soma._modify_module(
            module,
            directions,
            layer_base_weight,
            params,
            rank,
        )

        # Expected: sum of per-direction deltas with relative weights.
        expected_delta = sum(
            (-weight * v).view(-1, 1) @ (v @ W).view(1, -1)
            for weight, v in zip(max_weights, directions)
        )

        lora_A = cast(Tensor, module.lora_A["default"].weight)
        lora_B = cast(Tensor, module.lora_B["default"].weight)
        actual_delta = lora_B.float() @ lora_A.float()

        error = torch.linalg.vector_norm(actual_delta - expected_delta)
        magnitude = torch.linalg.vector_norm(expected_delta)
        self.assertLess(error / magnitude, 0.01)

    def test_full_mode_preserves_row_norms(self) -> None:
        soma = make_soma(
            row_normalization=RowNormalization.FULL,
            full_normalization_lora_rank=8,
        )

        out_features, in_features = 16, 16
        n_directions = 3
        rank = 8

        module, W = self.make_peft_module(out_features, in_features, rank)

        rng = torch.Generator().manual_seed(2)
        directions = torch.nn.functional.normalize(
            torch.randn(n_directions, out_features, generator=rng),
            p=2,
            dim=-1,
        )
        params = DirectionWeightDistribution(
            max_weights=[1.0, 1.0, 1.0],
            max_weight_position=0.0,
            min_weights=[0.0, 0.0, 0.0],
            min_weight_distance=1.0,
        )

        soma._modify_module(module, directions, 1.0, params, rank)

        lora_A = cast(Tensor, module.lora_A["default"].weight)
        lora_B = cast(Tensor, module.lora_B["default"].weight)
        effective_W = W + (lora_B.float() @ lora_A.float())

        original_norms = torch.linalg.vector_norm(W, dim=1)
        effective_norms = torch.linalg.vector_norm(effective_W, dim=1)

        relative_error = torch.linalg.vector_norm(
            effective_norms - original_norms
        ) / torch.linalg.vector_norm(original_norms)
        self.assertLess(relative_error.item(), 0.05)

    def test_full_mode_suppresses_directions(self) -> None:
        # Use a rank equal to the matrix size so the dense FULL delta is
        # preserved exactly and the suppression is unambiguous.
        soma = make_soma(
            row_normalization=RowNormalization.FULL,
            full_normalization_lora_rank=16,
        )

        out_features, in_features = 16, 16
        n_directions = 2
        rank = 16

        module, W = self.make_peft_module(out_features, in_features, rank)

        rng = torch.Generator().manual_seed(3)
        directions = torch.nn.functional.normalize(
            torch.randn(n_directions, out_features, generator=rng),
            p=2,
            dim=-1,
        )
        params = DirectionWeightDistribution(
            max_weights=[1.0, 1.0],
            max_weight_position=0.0,
            min_weights=[0.0, 0.0],
            min_weight_distance=1.0,
        )

        soma._modify_module(module, directions, 1.0, params, rank)

        lora_A = cast(Tensor, module.lora_A["default"].weight)
        lora_B = cast(Tensor, module.lora_B["default"].weight)
        effective_W = W + (lora_B.float() @ lora_A.float())

        # The weight's response to each ablated direction (W^T v, the
        # input-space component along v) must be reduced: in the classic
        # single-direction case it goes to zero, and MPOA preserves that
        # suppression while restoring row magnitudes.
        for v in directions:
            original = torch.linalg.vector_norm(W.T @ v)
            modified = torch.linalg.vector_norm(effective_W.T @ v)
            self.assertLess(modified, original)


class SuggestParametersTests(unittest.TestCase):
    def test_samples_per_direction_weights(self) -> None:
        soma = make_soma(som_k=3)

        # Stub the directions that init() would have computed.
        # (n_layers, n_directions, hidden_dim)
        soma.residual_directions = torch.randn(11, 3, 8)

        model = SimpleNamespace(
            get_layers=lambda: [None] * 10,
            get_modifiable_components=lambda: ["attn.o_proj", "mlp.down_proj"],
        )
        # The context is only used for get_model() here.
        ctx = cast(Context, SimpleNamespace(get_model=lambda: model))

        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))

        def objective(trial: Trial) -> float:
            parameters = soma.suggest_parameters(ctx, trial)
            self.assertIn("attn.o_proj", parameters.weight_distributions)
            self.assertIn("mlp.down_proj", parameters.weight_distributions)
            for component in ("attn.o_proj", "mlp.down_proj"):
                distribution = parameters.weight_distributions[component]
                self.assertEqual(len(distribution.max_weights), 3)
                self.assertEqual(len(distribution.min_weights), 3)
                for value in distribution.max_weights:
                    self.assertGreaterEqual(value, 0.0)
            trial.set_user_attr("parameters", parameters.to_dict())
            return 0.0

        study.optimize(objective, n_trials=2)

        trial = study.trials[0]
        for name in (
            "direction_scope",
            "direction_index",
            "attn.o_proj.max_weight.0",
            "attn.o_proj.max_weight.1",
            "attn.o_proj.max_weight.2",
            "mlp.down_proj.min_weight.2",
            "attn.o_proj.max_weight_position",
            "attn.o_proj.min_weight_distance",
        ):
            self.assertIn(name, trial.params)


if __name__ == "__main__":
    unittest.main()
