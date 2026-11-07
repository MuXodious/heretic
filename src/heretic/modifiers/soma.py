# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors

# Self-Organising Multidirectional Abliteration (SOMA).
#
# Extends directional abliteration with multiple refusal directions per layer,
# derived by training a Self-Organizing Map (SOM) on the bad prompt residuals 
# of each layer and using the weights of the most frequently winning (i.e. most 
# representative, de-duplicated) neurons as the "bad" centroids.
#
# Ported from https://github.com/p-e-w/heretic/pull/196 (kabachuha).

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, cast

import bitsandbytes.functional as BNB_F
import numpy as np
import torch
import torch.linalg as LA
import torch.nn.functional as F
from minisom import MiniSom
from optuna import Trial
from peft.tuners.lora.layer import Linear
from pydantic import (
    BaseModel,
    Field,
    PositiveInt,
)
from torch import Tensor

from heretic.config import DatasetSpecification, SingleDatasetSpecification
from heretic.modifier import Context, Modifier, Serializable
from heretic.modifiers.abliteration import RowNormalization
from heretic.system import empty_cache
from heretic.utils import format_dataset_specification, print


class SOMCalculator:
    """
    A simplified class to train a Self-Organizing Map (SOM) and extract neuron weights.
    """

    def __init__(
        self,
        som_x: int,
        som_y: int,
        iterations: int,
        lr: float,
        sigma: float,
    ) -> None:
        """
        Initializes the SOM calculator with training parameters.

        Args:
            som_x (int): Number of neurons in the x-axis of the SOM grid.
            som_y (int): Number of neurons in the y-axis of the SOM grid.
            iterations (int): Number of training iterations for the SOM.
            lr (float): Learning rate for the SOM.
            sigma (float): Radius of the neighborhood function.
        """
        self.som_x = som_x
        self.som_y = som_y
        self.iterations = iterations
        self.lr = lr
        self.sigma = sigma
        self.som: MiniSom | None = None

    def fit(self, data: np.ndarray) -> None:
        """
        Trains the SOM on the provided 2D data.

        Args:
            data (np.ndarray): A 2D NumPy array of shape (n_samples, n_features).
        """
        # Ensure data is 2D
        if len(data.shape) != 2:
            raise ValueError(f"Data must be a 2D array, but got shape {data.shape}")

        _, n_features = data.shape

        # Initialize and train the SOM using MiniSom
        self.som = MiniSom(
            self.som_x,
            self.som_y,
            n_features,
            sigma=self.sigma,
            learning_rate=self.lr,
            random_seed=0,  # For reproducibility
            activation_distance="euclidean",
            topology="hexagonal",
        )
        self.som.random_weights_init(data)
        self.som.train_random(data, self.iterations)

    def get_top_k_neuron_weights(self, k: int) -> np.ndarray:
        """
        Gets the weights of the top-k neurons based on their frequency of being winners.

        Every neuron is itself fed into the trained map, and its winner
        coordinate is counted. Neurons that are too close to each other
        map onto the same winner, so this de-duplicates them: the k most
        frequently winning (i.e. most representative) neurons are returned.
        If fewer than k distinct winners exist, fewer weights are returned;
        the caller is responsible for repeating weights up to k directions.

        Args:
            k (int): The number of top neurons to return.

        Returns:
            np.ndarray: A 2D array of shape (min(k, n_distinct_winners), n_features).
        """
        if self.som is None:
            raise RuntimeError("SOM has not been trained yet. Call `fit()` first.")

        winners = np.array(
            [
                self.som.winner(x)
                for x in self.som._weights.reshape(
                    -1,
                    self.som._weights.shape[2],
                )
            ]
        )
        counts = defaultdict(int)
        for w in winners:
            counts[tuple(w)] += 1

        # Sort neurons by their count (descending) and get the top-k
        sorted_neurons = sorted(counts.items(), key=lambda item: item[1], reverse=True)[
            :k
        ]

        # Get the coordinates of the top-k neurons
        top_k_coords = [coord for coord, _ in sorted_neurons]

        # Fetch the weights for these top-k neurons
        # self.som.get_weights() has shape (som_x, som_y, n_features)
        top_k_weights = np.array(
            [self.som.get_weights()[i, j] for i, j in top_k_coords]
        )

        return top_k_weights


@dataclass
class DirectionWeightDistribution:
    """
    The ablation weights of a single component, per refusal direction.

    `max_weights[i]` and `min_weights[i]` apply to refusal direction `i`.
    The layer-wise weight kernel is anchored at direction 0: the base weight
    of a layer is computed from `max_weights[0]` / `min_weights[0]`, and every
    other direction is scaled by the ratio of its `max_weights` to direction 0's.
    """

    max_weights: list[float]
    max_weight_position: float
    min_weights: list[float]
    min_weight_distance: float


@dataclass
class Parameters(Serializable):
    direction_index: float | None
    weight_distributions: dict[str, DirectionWeightDistribution]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_presentation_dict(self) -> dict[str, str]:
        parameters = {}

        parameters["direction_index"] = (
            "per layer"
            if (self.direction_index is None)
            else f"{self.direction_index:.2f}"
        )

        for component, weight_distribution in self.weight_distributions.items():
            for direction, value in enumerate(weight_distribution.max_weights):
                parameters[f"{component}.max_weights.{direction}"] = (
                    f"{direction}: {value:.2f}"
                )
            parameters[f"{component}.max_weight_position"] = (
                f"{weight_distribution.max_weight_position:.2f}"
            )
            for direction, value in enumerate(weight_distribution.min_weights):
                parameters[f"{component}.min_weights.{direction}"] = (
                    f"{direction}: {value:.2f}"
                )
            parameters[f"{component}.min_weight_distance"] = (
                f"{weight_distribution.min_weight_distance:.2f}"
            )

        return parameters

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        return Parameters(
            direction_index=data["direction_index"],
            weight_distributions={
                component: DirectionWeightDistribution(**weight_distribution)
                for component, weight_distribution in data[
                    "weight_distributions"
                ].items()
            },
        )


class Settings(BaseModel):
    good_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmless_alpaca",
            split="train[:400]",
            column="text",
        ),
        description="Dataset of prompts that tend to produce desirable responses.",
    )

    bad_prompts: DatasetSpecification = Field(
        default=SingleDatasetSpecification(
            dataset="mlabonne/harmful_behaviors",
            split="train[:400]",
            column="text",
        ),
        description="Dataset of prompts that tend to produce undesirable responses.",
    )

    multidirectional_som: bool = Field(
        default=False,
        description=(
            "Whether to derive multiple refusal directions per layer using a "
            "Self-Organizing Map (SOM) trained on the bad prompt residuals. "
            "When disabled, a single direction per layer is used (classic abliteration). "
            "See Piras et al. 2026 (arXiv:2511.08379) for the method, "
            "and https://github.com/p-e-w/heretic/pull/196 for the original "
            "Heretic implementation by kabachuha."
        ),
    )

    som_x: PositiveInt = Field(
        default=4,
        description="Number of SOM neurons in the x-axis.",
    )

    som_y: PositiveInt = Field(
        default=4,
        description="Number of SOM neurons in the y-axis.",
    )

    som_iterations: PositiveInt = Field(
        default=10000,
        description="Number of SOM training iterations.",
    )

    som_lr: float = Field(
        default=0.01,
        description="SOM learning rate (between 0 and 1).",
    )

    som_sigma: float = Field(
        default=0.5,
        description="SOM neighborhood radius.",
    )

    som_k: PositiveInt = Field(
        default=4,
        description=(
            "Number of top (most representative) SOM neurons to use as "
            "refusal directions. Also serves as a lower bound for the "
            "LoRA adapter rank when multidirectional_som is enabled "
            "(unless full_normalization_lora_rank is explicitly set)."
        ),
    )

    orthogonalize_direction: bool = Field(
        default=True,
        description=(
            "Whether to adjust the residual directions so that only the component that is "
            "orthogonal to the good direction is subtracted during abliteration."
        ),
    )

    row_normalization: RowNormalization = Field(
        default=RowNormalization.FULL,
        description=(
            "How to apply row normalization of the weights. Options: "
            '"none" (no normalization), '
            '"pre" (compute LoRA adapter relative to row-normalized weights), '
            '"full" (like "pre", but renormalizes to preserve original row magnitudes).'
        ),
    )

    full_normalization_lora_rank: PositiveInt = Field(
        default=3,
        description=(
            'The rank of the LoRA adapter to use when "full" row normalization is used. '
            "Row magnitude preservation is approximate due to non-linear effects, "
            "and this determines the rank of that approximation. Higher ranks produce "
            "larger output files and may slow down evaluation."
        ),
    )

    winsorization_quantile: float = Field(
        default=1.0,
        description=(
            "The symmetric winsorization to apply to the per-prompt, per-layer residual vectors, "
            "expressed as the quantile to clamp to (between 0 and 1). Disabled by default. "
            'This can tame so-called "massive activations" that occur in some models. '
            "Example: winsorization_quantile = 0.95 computes the 0.95-quantile of the absolute values "
            "of the components, then clamps the magnitudes of all components to that quantile."
        ),
    )


# Decomposes a delta matrix into LoRA factors (lora_B, lora_A) via a
# randomized low-rank SVD, splitting the singular values evenly between
# the two components to keep their norms balanced and avoid potential
# issues with numerical stability.
def decompose_delta(
    delta: Tensor,
    rank: int,
    seed: int | None = None,
) -> tuple[Tensor, Tensor]:
    if seed is not None:
        # svd_lowrank is randomized:
        # https://github.com/pytorch/pytorch/blob/20919052303c0b5ba87f8bf7e19237dc33ab09d3/torch/_lowrank.py#L108-L109
        # Reseed immediately before the call so restoring a trial is independent of RNG history.
        torch.manual_seed(seed)
        # "It's safe to call this function if CUDA is not available;
        # in that case, it is silently ignored."
        torch.cuda.manual_seed_all(seed)

    # Clamp q to the matrix size for small matrices (e.g. in tests).
    q = min(2 * rank + 4, min(delta.shape))

    U, S, Vh = torch.svd_lowrank(delta, q=q, niter=6)

    # Truncate it to the part we want to store in the LoRA adapter.
    # Note: svd_lowrank actually returns V, so transpose it to get Vh.
    U = U[:, :rank]
    S = S[:rank]
    Vh = Vh[:, :rank].T

    sqrt_S = torch.sqrt(S)
    lora_B = U @ torch.diag(sqrt_S)
    lora_A = torch.diag(sqrt_S) @ Vh

    return lora_B, lora_A


class SOMA(Modifier[Parameters]):
    settings: Settings

    @property
    def reproducible(self) -> bool:
        return True

    @property
    def modifier_name(self) -> str:
        if self.settings.multidirectional_som:
            if (
                self.settings.orthogonalize_direction
                and self.settings.row_normalization == RowNormalization.FULL
            ):
                return "SOMPOA"
            return "SOMA"
        elif (
            self.settings.orthogonalize_direction
            and self.settings.row_normalization == RowNormalization.FULL
        ):
            return "Magnitude-Preserving Orthogonal Ablation (MPOA)"
        elif self.settings.orthogonalize_direction:
            return "Projected Abliteration"
        else:
            return "Abliteration"

    @property
    def lora_rank(self) -> int:
        """
        The rank of the LoRA adapter to use.

        Without multidirectional SOM, rank 1 is sufficient to store the
        single-direction delta exactly (row normalization "full" is an
        approximation of rank `full_normalization_lora_rank`). With
        multidirectional SOM, the delta is the sum of k rank-1 direction
        deltas and has rank at most k, so truncating the adapter below
        k would drop entire directions: the rank is bounded below by
        `som_k`.

        An explicitly set `full_normalization_lora_rank` takes precedence
        over the `som_k` bound, which only applies to the built-in default.
        """
        if self.settings.row_normalization == RowNormalization.FULL:
            rank = self.settings.full_normalization_lora_rank
        else:
            # Rank 1 is sufficient for directional ablation without renormalization.
            rank = 1

        if self.settings.multidirectional_som:
            # The user's explicit choice takes precedence over the som_k bound.
            explicit_rank = (
                "full_normalization_lora_rank" in self.settings.model_fields_set
            )
            if not explicit_rank:
                rank = max(rank, self.settings.som_k)

        return rank

    def init(self, ctx: Context) -> None:
        model = ctx.get_model()

        print()
        print(
            f"Loading good prompts from [bold]{format_dataset_specification(self.settings.good_prompts)}[/]..."
        )
        good_prompts = ctx.load_prompts(self.settings.good_prompts)
        print(f"* [bold]{len(good_prompts)}[/] prompts loaded")

        print()
        print(
            f"Loading bad prompts from [bold]{format_dataset_specification(self.settings.bad_prompts)}[/]..."
        )
        bad_prompts = ctx.load_prompts(self.settings.bad_prompts)
        print(f"* [bold]{len(bad_prompts)}[/] prompts loaded")

        print()
        print("Calculating per-layer refusal directions...")

        print("* Obtaining residual mean for good prompts...")
        good_means = model.get_residuals_mean(
            good_prompts,
            winsorization_quantile=self.settings.winsorization_quantile,
        )

        if self.settings.multidirectional_som:
            # The SOM is trained on the distribution of per-prompt residuals,
            # not just their mean, so obtain the raw residuals.
            print("* Obtaining residuals for bad prompts...")
            bad_residuals = model.get_residuals_batched(
                bad_prompts,
                winsorization_quantile=self.settings.winsorization_quantile,
            )
            bad_means = self.obtain_som_means(bad_residuals)
            # We don't need the residuals after obtaining the SOM means.
            del bad_residuals
            empty_cache()
        else:
            print("* Obtaining residual mean for bad prompts...")
            bad_means = model.get_residuals_mean(
                bad_prompts,
                winsorization_quantile=self.settings.winsorization_quantile,
            ).unsqueeze(0)

        # bad_means has shape (n_directions, n_layers, hidden_dim),
        # where n_directions is 1 without multidirectional SOM.
        self.residual_directions = self.compute_refusal_directions(
            good_means,
            bad_means,
            orthogonalize_direction=self.settings.orthogonalize_direction,
        )

        # LoRA B matrices are initialized to zero by default in PEFT,
        # so we don't need to do anything manually.
        model.apply_lora(self.lora_rank)

    # Computes the per-layer refusal directions from the "good" and "bad"
    # residual means.
    @staticmethod
    def compute_refusal_directions(
        good_means: Tensor,
        bad_means: Tensor,
        orthogonalize_direction: bool,
    ) -> Tensor:
        """
        Computes the refusal directions for each "bad" mean.

        Args:
            good_means (Tensor): The "good" residual mean with shape
                (n_layers, hidden_dim).
            bad_means (Tensor): The "bad" residual means with shape
                (n_directions, n_layers, hidden_dim).
            orthogonalize_direction (bool): Whether to adjust the directions
                so that only the component orthogonal to the good direction
                is subtracted during abliteration.

        Returns:
            Tensor: The refusal directions with shape
            (n_layers, n_directions, hidden_dim).
        """
        refusal_directions = [
            F.normalize(bad_mean - good_means, p=2, dim=1) for bad_mean in bad_means
        ]

        if orthogonalize_direction:
            # Implements https://huggingface.co/blog/grimjim/projected-abliteration
            # Adjust the refusal directions so that only the component that is
            # orthogonal to the good direction is subtracted during abliteration.
            good_directions = F.normalize(
                good_means,
                p=2,
                dim=1,
            )
            projection_vectors = [
                torch.sum(refusal_direction * good_directions, dim=1)
                for refusal_direction in refusal_directions
            ]
            refusal_directions = [
                refusal_direction - projection_vector.unsqueeze(1) * good_directions
                for (
                    refusal_direction,
                    projection_vector,
                ) in zip(refusal_directions, projection_vectors)
            ]
            refusal_directions = [
                F.normalize(refusal_direction, p=2, dim=1)
                for refusal_direction in refusal_directions
            ]

        # (n_directions, n_layers, hidden_dim) -> (n_layers, n_directions, hidden_dim)
        return torch.stack(refusal_directions, dim=0).permute(1, 0, 2)

    # Trains a SOM on the bad residuals of each layer and returns the weights
    # of the most representative neurons as the per-layer "bad" centroids.
    def obtain_som_means(self, bad_residuals: Tensor) -> Tensor:
        """
        Obtains the per-layer "bad" centroids by training a SOM per layer
        on the per-prompt bad residuals.

        Args:
            bad_residuals (Tensor): Per-prompt bad residuals with shape
                (n_prompts, n_layers, hidden_dim).

        Returns:
            Tensor: The "bad" centroids with shape
            (n_directions, n_layers, hidden_dim), where n_directions is
            `som_k` (weights of the most representative, de-duplicated
            neurons; repeated if the SOM found fewer distinct neurons).
        """
        bad_means: list[Tensor] = []
        n_layers = bad_residuals.shape[1]

        print("  - Retrieving multi-directions through self-organizing maps...")

        for layer_index in range(n_layers):
            print(f"  - Processing Layer {layer_index + 1}/{n_layers}...")

            # Extract the residuals for the current layer.
            # Shape: (n_prompts, hidden_dim)
            layer_residuals = bad_residuals[:, layer_index, :].cpu().float().numpy()

            # Initialize and fit the SOM for this layer's residuals
            som_calc = SOMCalculator(
                som_x=self.settings.som_x,
                som_y=self.settings.som_y,
                iterations=self.settings.som_iterations,
                lr=self.settings.som_lr,
                sigma=self.settings.som_sigma,
            )
            som_calc.fit(layer_residuals)

            # Get the weights of the top-k neurons as our "bad means"
            top_k_weights = som_calc.get_top_k_neuron_weights(k=self.settings.som_k)

            # The SOM de-duplicates neurons if they are too close to each other,
            # so fewer than som_k neurons may be found. Repeat them back.
            t = torch.tensor(
                top_k_weights,
                dtype=bad_residuals.dtype,
                device=bad_residuals.device,
            )
            num_found_neurons = t.shape[0]

            if num_found_neurons < self.settings.som_k:
                t = t.repeat(self.settings.som_k // num_found_neurons, 1)
                if t.shape[0] < self.settings.som_k:
                    t = torch.cat([t, t[0 : self.settings.som_k - t.shape[0]]], dim=0)

            # t has shape (som_k, hidden_dim)
            bad_means.append(t)

        # (n_layers, n_directions, hidden_dim) -> (n_directions, n_layers, hidden_dim)
        return torch.stack(bad_means, dim=0).permute(1, 0, 2)

    def suggest_parameters(self, ctx: Context, trial: Trial) -> Parameters:
        model = ctx.get_model()

        n_directions = self.residual_directions.shape[1]

        direction_scope = trial.suggest_categorical(
            "direction_scope",
            [
                "global",
                "per layer",
            ],
        )

        last_layer_index = len(model.get_layers()) - 1

        # Discrimination between "harmful" and "harmless" inputs is usually strongest
        # in layers slightly past the midpoint of the layer stack. See the original
        # abliteration paper (https://arxiv.org/abs/2406.11717) for a deeper analysis.
        #
        # Note that we always sample this parameter even though we only need it for
        # the "global" direction scope. The reason is that multivariate TPE doesn't
        # work with conditional or variable-range parameters.
        direction_index = trial.suggest_float(
            "direction_index",
            0.4 * last_layer_index,
            0.9 * last_layer_index,
        )

        if direction_scope == "per layer":
            direction_index = None

        weight_distributions = {}

        for component in model.get_modifiable_components():
            # The parameter ranges are based on experiments with various models
            # and much wider ranges. They are not set in stone and might have to be
            # adjusted for future models.
            #
            # The MLP gets a negative lower bound that is then clamped to 0, so the
            # optimizer can fully disable its ablation. The clamp puts a positive
            # probability mass on exactly 0 (the continuous sampler would otherwise
            # reach 0 with probability zero). Ablating the MLP is often unnecessary for
            # removing refusals and tends to damage model intelligence more than
            # ablating the attention output, so on many models the optimum is to leave
            # it (mostly) untouched. See issue #202.
            max_weight_lower_bound = -0.25 if component == "mlp.down_proj" else 0.8
            max_weights = [
                max(
                    0.0,
                    trial.suggest_float(
                        f"{component}.max_weight.{i}",
                        max_weight_lower_bound,
                        1.5,
                    ),
                )
                for i in range(n_directions)
            ]
            max_weight_position = trial.suggest_float(
                f"{component}.max_weight_position",
                0.6 * last_layer_index,
                1.0 * last_layer_index,
            )
            # For sampling purposes, min_weights are expressed as fractions of
            # max_weights, again because multivariate TPE doesn't support
            # variable-range parameters. The values are transformed into the
            # actual min_weight values below.
            min_weights = [
                trial.suggest_float(
                    f"{component}.min_weight.{i}",
                    0.0,
                    1.0,
                )
                for i in range(n_directions)
            ]
            min_weight_distance = trial.suggest_float(
                f"{component}.min_weight_distance",
                1.0,
                max(0.6 * last_layer_index, 1.0),
            )

            weight_distributions[component] = DirectionWeightDistribution(
                max_weights=max_weights,
                max_weight_position=max_weight_position,
                min_weights=[
                    min_weight * max_weight
                    for (min_weight, max_weight) in zip(min_weights, max_weights)
                ],
                min_weight_distance=min_weight_distance,
            )

        return Parameters(
            direction_index=direction_index,
            weight_distributions=weight_distributions,
        )

    def modify_model(self, ctx: Context, parameters: Parameters) -> None:
        model = ctx.get_model()

        # residual_directions has shape (n_layers, n_directions, hidden_dim).
        refusal_directions = self.residual_directions
        n_directions = refusal_directions.shape[1]

        if parameters.direction_index is not None:
            # A specific (possibly fractional) layer position was requested:
            # interpolate between the two adjacent layers, per direction, and
            # use the same blended set of directions for all layers.
            weight, index = math.modf(parameters.direction_index + 1)
            # Clamp index to be within the valid range [0, n_layers - 1]
            idx1 = int(index)
            idx2 = min(idx1 + 1, refusal_directions.shape[0] - 1)
            # Shape: (n_directions, hidden_dim)
            layer_refusal_directions = torch.stack(
                [
                    F.normalize(
                        direction[idx1].lerp(
                            direction[idx2],
                            weight,
                        ),
                        p=2,
                        dim=0,
                    )
                    for direction in refusal_directions.permute(1, 0, 2)
                ],
                dim=0,
            )
        else:
            # Per layer: each layer uses its own set of directions.
            # Shape: (n_layers, n_directions, hidden_dim)
            layer_refusal_directions = refusal_directions

        for layer_index in range(len(model.get_layers())):
            for component, modules in model.get_layer_modules(layer_index).items():
                params = parameters.weight_distributions[component]

                # Ensure the number of weights matches the number of directions
                if (
                    len(params.max_weights) != n_directions
                    or len(params.min_weights) != n_directions
                ):
                    raise ValueError(
                        f"Mismatch in number of directions and weights for component '{component}'. "
                        f"Found {n_directions} directions but {len(params.max_weights)} max_weights."
                    )

                # Get the layer-specific position scaling factor.
                # This factor is the same for all directions in this layer.
                distance = abs(layer_index - params.max_weight_position)
                if distance > params.min_weight_distance:
                    # If this layer is too far from the optimal position, skip it entirely.
                    continue

                # Interpolate linearly between max_weight and min_weight
                # over min_weight_distance (anchored at direction 0).
                layer_base_weight = params.max_weights[0] + (
                    distance / params.min_weight_distance
                ) * (params.min_weights[0] - params.max_weights[0])

                # A weight of 0 disables this component's ablation. reset_model() has
                # already left the adapter at identity, so abort before the otherwise
                # wasteful decomposition (which would also be operating on a zero matrix).
                if layer_base_weight == 0:
                    continue

                if parameters.direction_index is None:
                    # The index must be shifted by 1 because the first element
                    # of residual_directions is the direction for the embeddings.
                    current_layer_directions = layer_refusal_directions[layer_index + 1]
                else:
                    current_layer_directions = layer_refusal_directions

                for module in modules:
                    self._modify_module(
                        module,
                        current_layer_directions,
                        layer_base_weight,
                        params,
                        model.peft_config.r,
                    )

    # Applies the summed multi-direction delta to a single module's LoRA adapter.
    def _modify_module(
        self,
        module: Any,
        directions: Tensor,
        layer_base_weight: float,
        params: DirectionWeightDistribution,
        rank: int,
    ) -> None:
        # Type inference fails here for some reason.
        module = cast(Linear, module)

        # Get W (dequantize if necessary).
        #
        # FIXME: This cast is valid only under the assumption that the original
        #        module wrapped by the LoRA adapter has a weight attribute.
        base_weight = cast(Tensor, module.base_layer.weight)
        quant_state = getattr(base_weight, "quant_state", None)

        if quant_state is None:
            W = base_weight.to(torch.float32)
        else:
            # 4-bit quantization.
            W = BNB_F.dequantize_4bit(
                base_weight.data,
                quant_state,
            ).to(torch.float32)

        # Flatten weight matrix to (out_features, in_features).
        W = W.view(W.shape[0], -1)

        if self.settings.row_normalization == RowNormalization.FULL:
            # Keep a reference to the original weight matrix so we can subtract it later.
            W_org = W

        if self.settings.row_normalization != RowNormalization.NONE:
            # Get the row norms.
            W_row_norms = LA.vector_norm(W, dim=1, keepdim=True)
            # Normalize the weight matrix along the rows.
            W = F.normalize(W, p=2, dim=1)

        # The core multi-directional step: calculate a delta for each direction
        # and sum them up.
        #
        # LoRA abliteration per direction: delta W = -lambda * v * (v^T W)
        # lora_B = -lambda * v
        # lora_A = v^T W
        total_delta_W = torch.zeros_like(W)

        for i in range(len(params.max_weights)):
            # Get the specific refusal direction for this component.
            v = directions[i].to(module.weight.device)

            # Get the optimized weight for this specific direction (i):
            # the layer's base weight scaled by the direction's relative weight.
            direction_specific_weight = layer_base_weight * (
                params.max_weights[i] / params.max_weights[0]
                if params.max_weights[0] != 0
                else 1.0
            )

            # Calculate lora_A = v^T W -> (in_features,)
            lora_A = (v @ W).view(1, -1)
            # Calculate lora_B = -direction_specific_weight * v -> (out_features, 1)
            lora_B = (-direction_specific_weight * v).view(-1, 1)

            # The delta for this direction is lora_B @ lora_A
            total_delta_W += lora_B @ lora_A

        if self.settings.row_normalization == RowNormalization.PRE:
            # Decompose the total delta into LoRA matrices.
            lora_B, lora_A = decompose_delta(
                total_delta_W,
                rank,
                seed=self.heretic_settings.seed,
            )
            # Make the LoRA adapter apply to the original weight matrix.
            lora_B = W_row_norms * lora_B
        elif self.settings.row_normalization == RowNormalization.FULL:
            # Approximates https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
            W = W + total_delta_W
            # Normalize the adjusted weight matrix along the rows.
            W = F.normalize(W, p=2, dim=1)
            # Restore the original row norms of the weight matrix.
            W = W * W_row_norms
            # Subtract the original matrix to turn W into a delta.
            W = W - W_org
            # Decompose the full delta into LoRA matrices.
            lora_B, lora_A = decompose_delta(
                W,
                rank,
                seed=self.heretic_settings.seed,
            )
        else:
            # Decompose the total delta into LoRA matrices.
            lora_B, lora_A = decompose_delta(
                total_delta_W,
                rank,
                seed=self.heretic_settings.seed,
            )

        # Assign to adapters. The adapter name is "default", because that's
        # what PEFT uses when no name is explicitly specified, as above.
        # These casts are therefore valid.
        weight_A = cast(Tensor, module.lora_A["default"].weight)
        weight_B = cast(Tensor, module.lora_B["default"].weight)
        weight_A.data = lora_A.to(weight_A.dtype)
        weight_B.data = lora_B.to(weight_B.dtype)

    def reset_model(self, ctx: Context) -> None:
        model = ctx.get_model()
        fast_path = model.reset_model()
        if not fast_path:
            model.apply_lora(self.lora_rank)
