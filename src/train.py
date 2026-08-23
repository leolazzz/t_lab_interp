"""Small sampler-driven training loop for activation denoisers.

This module initially provides Gaussian corruption, but optimization accepts a
generic corruption sampler. Future SAE samplers must receive *training split*
direction IDs only. Validation/test directions must never affect corruption,
training-time direction scoring, optimization, or hyperparameter fitting.
"""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm.auto import tqdm

from src.directions import (
    direction_split_hash,
    validate_direction_ids_for_usage,
    validate_direction_split,
)
from src.utils import (
    CHECKPOINT_SCHEMA_VERSION,
    PIPELINE_VERSION,
    denormalize_activations,
    normalize_activations,
    resolve_device,
    resolve_dtype,
    seed_everything,
)


CorruptionSampler = Callable[[Tensor, torch.Generator], tuple[Tensor, Tensor]]
ActivationDataset = Dataset[Tensor] | IterableDataset[Tensor]

CORRUPTION_IDENTITY = 0
CORRUPTION_GAUSSIAN = 1
CORRUPTION_SAE_RAWSCALE = 2
CORRUPTION_SAE_CALIBRATED = 3
CORRUPTION_SAE_HARMFUL = 4
CORRUPTION_TYPE_NAMES = {
    CORRUPTION_IDENTITY: "identity",
    CORRUPTION_GAUSSIAN: "gaussian",
    CORRUPTION_SAE_RAWSCALE: "sae_rawscale_legacy",
    CORRUPTION_SAE_CALIBRATED: "sae_calibrated",
    CORRUPTION_SAE_HARMFUL: "sae_harmful",
}


def split_activation_shards(*args: Any, **kwargs: Any) -> tuple[Dataset, Dataset]:
    """Compatibility wrapper for the cache split helper used in notebooks."""
    from src.experiment import split_activation_shards as _split_activation_shards

    return _split_activation_shards(*args, **kwargs)


def _sample_corruption(
    sampler: CorruptionSampler,
    clean_raw: Tensor,
    clean_standardized: Tensor,
    mean: Tensor,
    std: Tensor,
    generator: torch.Generator,
) -> dict[str, Any]:
    """Sample corruption once and retain metadata used by runtime assertions."""
    clean_raw_before = clean_raw.detach().clone()
    clean_standardized_before = clean_standardized.detach().clone()
    sample_batch = getattr(sampler, "sample_batch", None)
    if sample_batch is not None:
        result = sample_batch(
            clean_raw if getattr(sampler, "input_space", "standardized") == "raw" else clean_standardized,
            generator,
        )
        assert isinstance(result, dict)
        corrupted_value = result["corrupted"]
        noise_level = result["magnitude"]
    else:
        corrupted_value, noise_level = sampler(
            clean_raw if getattr(sampler, "input_space", "standardized") == "raw" else clean_standardized,
            generator,
        )
        result = {
            "corrupted": corrupted_value,
            "magnitude": noise_level,
            "direction_id": torch.full_like(noise_level, -1, dtype=torch.long),
            "corruption_type": torch.full_like(
                noise_level, CORRUPTION_GAUSSIAN, dtype=torch.long
            ),
        }
    if getattr(sampler, "input_space", "standardized") == "raw":
        corrupted = normalize_activations(corrupted_value, mean, std)
    else:
        corrupted = corrupted_value
    assert corrupted.shape == clean_standardized.shape
    assert noise_level.shape == clean_standardized.shape[:-1]
    torch.testing.assert_close(clean_raw, clean_raw_before, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        clean_standardized, clean_standardized_before, rtol=0.0, atol=0.0
    )
    result["corrupted"] = corrupted
    result["magnitude"] = noise_level
    assert isinstance(result.get("direction_id"), Tensor)
    assert isinstance(result.get("corruption_type"), Tensor)
    return result


def prepare_training_corruption_batch(
    clean_raw: Tensor,
    corruption_sampler: CorruptionSampler,
    mean: Tensor,
    std: Tensor,
    generator: torch.Generator,
    identity_probability: float,
) -> tuple[Tensor, dict[str, Any]]:
    """Create the exact standardized corruption consumed by optimization."""
    assert clean_raw.ndim == 2
    assert 0.0 <= identity_probability <= 1.0
    clean_before = clean_raw.detach().clone()
    clean_standardized = normalize_activations(clean_raw, mean, std)
    corruption = _sample_corruption(
        corruption_sampler, clean_raw, clean_standardized, mean, std, generator
    )
    identity_mask = torch.rand(
        corruption["magnitude"].shape, device=clean_raw.device, generator=generator
    ) < identity_probability
    corruption["corrupted"] = torch.where(
        identity_mask.unsqueeze(-1), clean_standardized, corruption["corrupted"]
    )
    corruption["magnitude"] = torch.where(
        identity_mask, torch.zeros_like(corruption["magnitude"]),
        corruption["magnitude"],
    )
    corruption["direction_id"] = torch.where(
        identity_mask, torch.full_like(corruption["direction_id"], -1),
        corruption["direction_id"],
    )
    corruption["corruption_type"] = torch.where(
        identity_mask,
        torch.full_like(corruption["corruption_type"], CORRUPTION_IDENTITY),
        corruption["corruption_type"],
    )
    corruption["identity_mask"] = identity_mask
    torch.testing.assert_close(clean_raw, clean_before, rtol=0.0, atol=0.0)
    return clean_standardized, corruption


def direction_sampling_probabilities(
    scores: Tensor,
    mode: str = "uniform",
    gamma: float = 0.5,
    eps: float = 1e-8,
    clip_quantile: float | None = None,
) -> Tensor:
    """Return stable direction probabilities without concentrating excessively."""
    assert scores.ndim == 1 and scores.numel() > 0
    assert scores.is_floating_point() and gamma >= 0.0 and eps > 0.0
    if not torch.isfinite(scores).all().item():
        raise ValueError("Direction scores must all be finite.")

    # Calculate weights in float32 even if model activations use float16, so
    # small eps values and low harmfulness scores do not underflow to zero.
    nonnegative = scores.float().clamp_min(0)
    if clip_quantile is not None:
        assert 0.0 < clip_quantile <= 1.0
        upper = torch.quantile(nonnegative, clip_quantile)
        nonnegative = nonnegative.clamp_max(upper)

    if mode == "uniform":
        weights = torch.ones_like(nonnegative)
    elif mode == "score_proportional":
        weights = (nonnegative + eps).pow(gamma)
    elif mode == "rank_based":
        ranks = torch.empty_like(nonnegative)
        order = torch.argsort(nonnegative)
        ranks[order] = torch.arange(
            1,
            nonnegative.numel() + 1,
            device=scores.device,
            dtype=scores.dtype,
        )
        weights = ranks.pow(gamma)
    else:
        raise ValueError(
            "sampling mode must be 'uniform', 'score_proportional', or 'rank_based'."
        )

    total = weights.sum()
    assert total.item() > 0.0
    probabilities = weights / total
    assert torch.isfinite(probabilities).all().item()
    torch.testing.assert_close(
        probabilities.sum(), probabilities.new_ones(()), rtol=1e-5, atol=1e-7
    )
    return probabilities


def directions_to_standardized_space(
    directions: Tensor,
    normalization_stats: Mapping[str, Any] | str | Path,
    eps: float = 1e-6,
) -> Tensor:
    """Map raw SAE directions through ``z = (h - mean) / std`` and renormalize."""
    if isinstance(normalization_stats, (str, Path)):
        stats = load_activation_stats(normalization_stats)
    else:
        stats = normalization_stats
    std_value = stats.get("std")
    assert isinstance(std_value, Tensor) and std_value.shape == (directions.shape[1],)
    std = std_value.to(device=directions.device, dtype=directions.dtype).clamp_min(eps)
    standardized = directions / std
    norms = torch.linalg.vector_norm(standardized, dim=-1, keepdim=True)
    assert torch.all(norms > torch.finfo(directions.dtype).tiny).item()
    return standardized / norms


def load_train_direction_scores(
    path: str | Path,
    expected_train_ids: list[int],
    direction_split: Mapping[str, Any],
    score_key: str = "mean_kl",
    expected_hook_name: str | None = None,
) -> Tensor:
    """Load scores and align them to the persisted training split order."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    assert payload.get("split") == "train", "Score file is not marked train-only."
    assert payload.get("pipeline_version") == PIPELINE_VERSION, (
        "Direction scores were produced by an incompatible pipeline; recompute them."
    )
    assert payload.get("direction_split_hash") == direction_split_hash(direction_split), (
        "Direction scores use a different train/validation/test split."
    )
    validate_direction_ids_for_usage(
        expected_train_ids,
        direction_split,
        usage="damage_scoring",
        require_complete_split=True,
    )
    if expected_hook_name is not None:
        assert payload.get("hook_name") == expected_hook_name, (
            "Direction-score hook does not match the configured intervention hook."
        )
    direction_ids_value = payload.get("direction_ids")
    scores_value = payload.get(score_key)
    assert isinstance(direction_ids_value, (Tensor, list))
    assert isinstance(scores_value, Tensor) and scores_value.ndim == 1
    direction_ids = (
        direction_ids_value.tolist()
        if isinstance(direction_ids_value, Tensor)
        else direction_ids_value
    )
    assert len(direction_ids) == len(set(direction_ids)) == scores_value.numel()
    assert set(direction_ids) == set(expected_train_ids), (
        "Score file IDs do not exactly match the persisted training split."
    )
    score_by_id = {
        int(direction_id): scores_value[index]
        for index, direction_id in enumerate(direction_ids)
    }
    return torch.stack([score_by_id[direction_id] for direction_id in expected_train_ids])


class SAEDirectionCorruptionSampler:
    """Sample relative-norm corruptions from verified training directions only."""

    def __init__(
        self,
        train_directions: Tensor,
        train_direction_ids: list[int],
        direction_split: Mapping[str, Any],
        relative_magnitude_min: float,
        relative_magnitude_max: float,
        sampling_mode: str = "uniform",
        scores: Tensor | None = None,
        gamma: float = 0.5,
        score_eps: float = 1e-8,
        clip_quantile: float | None = None,
    ) -> None:
        assert train_directions.ndim == 2 and train_directions.is_floating_point()
        assert len(train_direction_ids) == train_directions.shape[0]
        assert len(train_direction_ids) == len(set(train_direction_ids))
        assert 0.0 <= relative_magnitude_min <= relative_magnitude_max

        persisted_train_ids = list(direction_split["train"])
        validation_ids = list(direction_split["val"])
        test_ids = list(direction_split["test"])
        validate_direction_split(
            persisted_train_ids,
            validation_ids,
            test_ids,
            int(direction_split["num_features"]),
        )
        validate_direction_ids_for_usage(
            train_direction_ids,
            direction_split,
            usage="training",
            require_complete_split=True,
        )
        assert set(train_direction_ids) == set(persisted_train_ids), (
            "Sampler directions must exactly match the persisted training split."
        )
        forbidden_ids = set(validation_ids) | set(test_ids)
        assert set(train_direction_ids).isdisjoint(forbidden_ids), (
            "Validation/test directions are forbidden in corruption samplers."
        )

        norms = torch.linalg.vector_norm(train_directions, dim=-1, keepdim=True)
        stability_eps = torch.finfo(train_directions.dtype).tiny
        assert torch.all(norms > stability_eps).item()
        self.directions = (train_directions / norms).detach().clone()
        self.train_direction_ids = torch.tensor(
            train_direction_ids,
            dtype=torch.long,
            device=train_directions.device,
        )
        self.relative_magnitude_min = relative_magnitude_min
        self.relative_magnitude_max = relative_magnitude_max

        if scores is None:
            scores = torch.zeros(
                len(train_direction_ids),
                device=train_directions.device,
                dtype=train_directions.dtype,
            )
        else:
            assert scores.shape == (len(train_direction_ids),)
            scores = scores.to(
                device=train_directions.device,
                dtype=train_directions.dtype,
            )
        self.probabilities = direction_sampling_probabilities(
            scores,
            mode=sampling_mode,
            gamma=gamma,
            eps=score_eps,
            clip_quantile=clip_quantile,
        )

    def sample_direction_ids(
        self,
        num_samples: int,
        generator: torch.Generator,
    ) -> Tensor:
        """Draw global SAE feature IDs according to the configured distribution."""
        assert num_samples > 0
        local_indices = torch.multinomial(
            self.probabilities,
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )
        return self.train_direction_ids.index_select(0, local_indices)

    def __call__(
        self,
        clean: Tensor,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        assert clean.ndim == 2
        assert clean.shape[-1] == self.directions.shape[-1]
        assert clean.device == self.directions.device
        assert clean.dtype == self.directions.dtype

        local_indices = torch.multinomial(
            self.probabilities,
            num_samples=clean.shape[0],
            replacement=True,
            generator=generator,
        )
        sampled_directions = self.directions.index_select(0, local_indices)
        relative = torch.rand(
            clean.shape[0],
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        relative = self.relative_magnitude_min + (
            self.relative_magnitude_max - self.relative_magnitude_min
        ) * relative
        activation_norm = torch.linalg.vector_norm(clean, dim=-1)
        magnitude = relative * activation_norm
        return clean + magnitude.unsqueeze(-1) * sampled_directions, magnitude

    def sample_batch(self, clean: Tensor, generator: torch.Generator) -> dict[str, Any]:
        """Compatibility-preserving metadata path used by real training."""
        local_indices = torch.multinomial(
            self.probabilities,
            num_samples=clean.shape[0],
            replacement=True,
            generator=generator,
        )
        sampled_directions = self.directions.index_select(0, local_indices)
        relative = torch.rand(
            clean.shape[0], device=clean.device, dtype=clean.dtype, generator=generator
        )
        relative = self.relative_magnitude_min + (
            self.relative_magnitude_max - self.relative_magnitude_min
        ) * relative
        magnitude = relative * torch.linalg.vector_norm(clean, dim=-1)
        return {
            "corrupted": clean + magnitude.unsqueeze(-1) * sampled_directions,
            "magnitude": magnitude,
            "direction_id": self.train_direction_ids.index_select(0, local_indices),
            "corruption_type": torch.full(
                (clean.shape[0],), CORRUPTION_SAE_RAWSCALE,
                device=clean.device, dtype=torch.long,
            ),
        }


class SAEStandardizedUnitCorruptionSampler:
    """Calibrated SAE corruption with unit norm in standardized coordinates.

    Unlike :class:`SAEDirectionCorruptionSampler`, this mode does not use
    ``r * ||h||``. It maps each raw decoder direction through ``1 / std`` and
    renormalizes it, so a sampled magnitude has the same norm for every
    direction in denoiser training space.
    """

    input_space = "standardized"

    def __init__(
        self,
        train_directions: Tensor,
        train_direction_ids: list[int],
        direction_split: Mapping[str, Any],
        std: Tensor,
        magnitude_min: float,
        magnitude_max: float,
        eps: float = 1e-6,
        sampling_mode: str = "uniform",
        scores: Tensor | None = None,
        gamma: float = 0.5,
        score_eps: float = 1e-8,
        clip_quantile: float | None = None,
        corruption_type: int = CORRUPTION_SAE_CALIBRATED,
    ) -> None:
        assert train_directions.ndim == 2 and std.ndim == 1
        assert train_directions.shape[1] == std.shape[0]
        validate_direction_ids_for_usage(
            train_direction_ids,
            direction_split,
            usage="training",
            require_complete_split=True,
        )
        assert 0.0 <= magnitude_min <= magnitude_max
        std_safe = std.to(train_directions).clamp_min(eps)
        mapped = train_directions / std_safe
        mapped = mapped / torch.linalg.vector_norm(mapped, dim=-1, keepdim=True).clamp_min(eps)
        self.directions = mapped.detach().clone()
        self.train_direction_ids = list(train_direction_ids)
        self.magnitude_min = float(magnitude_min)
        self.magnitude_max = float(magnitude_max)
        score_values = (
            torch.zeros(len(train_direction_ids), device=train_directions.device)
            if scores is None else scores.to(train_directions.device, torch.float32)
        )
        self.probabilities = direction_sampling_probabilities(
            score_values,
            mode=sampling_mode,
            gamma=gamma,
            eps=score_eps,
            clip_quantile=clip_quantile,
        )
        self.corruption_type = int(corruption_type)

    def __call__(self, clean: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        assert clean.ndim == 2 and clean.shape[-1] == self.directions.shape[1]
        indices = torch.multinomial(
            self.probabilities.to(clean.device), clean.shape[0], replacement=True,
            generator=generator,
        )
        direction = self.directions.to(clean.device, clean.dtype).index_select(0, indices)
        if self.magnitude_min == self.magnitude_max:
            magnitude = clean.new_full((clean.shape[0],), self.magnitude_min)
        else:
            magnitude = self.magnitude_min + (self.magnitude_max - self.magnitude_min) * torch.rand(
                clean.shape[0], device=clean.device, dtype=clean.dtype, generator=generator
            )
        return clean + magnitude.unsqueeze(-1) * direction, magnitude

    def sample_batch(self, clean: Tensor, generator: torch.Generator) -> dict[str, Any]:
        assert clean.ndim == 2 and clean.shape[-1] == self.directions.shape[1]
        clean_before = clean.detach().clone()
        indices = torch.multinomial(
            self.probabilities.to(clean.device), clean.shape[0], replacement=True,
            generator=generator,
        )
        direction = self.directions.to(clean.device, clean.dtype).index_select(0, indices)
        if self.magnitude_min == self.magnitude_max:
            magnitude = clean.new_full((clean.shape[0],), self.magnitude_min)
        else:
            magnitude = self.magnitude_min + (
                self.magnitude_max - self.magnitude_min
            ) * torch.rand(
                clean.shape[0], device=clean.device, dtype=clean.dtype, generator=generator
            )
        corrupted = clean + magnitude.unsqueeze(-1) * direction
        torch.testing.assert_close(clean, clean_before, rtol=0.0, atol=0.0)
        ids = torch.tensor(
            self.train_direction_ids, device=clean.device, dtype=torch.long
        ).index_select(0, indices)
        return {
            "corrupted": corrupted,
            "magnitude": magnitude,
            "direction_id": ids,
            "corruption_type": torch.full(
                (clean.shape[0],), self.corruption_type,
                device=clean.device, dtype=torch.long,
            ),
        }


@torch.inference_mode()
def calibrated_sampler_math_gate(
    sampler: SAEStandardizedUnitCorruptionSampler,
    d_model: int,
    magnitudes: Sequence[float] = (3.7878, 9.4695, 18.9390),
    num_examples: int = 8,
    atol: float = 2e-4,
    rtol: float = 2e-3,
) -> Any:
    """Verify unit standardized corruption geometry before model training."""
    import pandas as pd

    assert d_model > 0 and num_examples > 0
    clean = torch.zeros((num_examples, d_model), device=sampler.directions.device, dtype=sampler.directions.dtype)
    original_min, original_max = sampler.magnitude_min, sampler.magnitude_max
    rows: list[dict[str, float]] = []
    for magnitude in magnitudes:
        sampler.magnitude_min = float(magnitude)
        sampler.magnitude_max = float(magnitude)
        corrupted, _ = sampler(clean, torch.Generator(device=clean.device).manual_seed(123))
        delta = corrupted - clean
        actual_norm = torch.linalg.vector_norm(delta, dim=-1).mean()
        actual_mse = delta.square().mean()
        expected_mse = float(magnitude) ** 2 / d_model
        torch.testing.assert_close(actual_norm, clean.new_tensor(magnitude), atol=atol, rtol=rtol)
        torch.testing.assert_close(actual_mse, clean.new_tensor(expected_mse), atol=atol, rtol=rtol)
        rows.append({
            "magnitude": float(magnitude),
            "actual_delta_norm": float(actual_norm.detach().cpu()),
            "expected_delta_norm": float(magnitude),
            "actual_mse": float(actual_mse.detach().cpu()),
            "expected_mse": float(expected_mse),
            "relative_error": float((actual_mse - expected_mse).abs().detach().cpu() / max(expected_mse, 1e-12)),
        })
    sampler.magnitude_min, sampler.magnitude_max = original_min, original_max
    return pd.DataFrame(rows)


@torch.inference_mode()
def assert_calibrated_training_batch_math(
    clean_z: Tensor,
    corruption: Mapping[str, Any],
    *,
    atol: float = 3e-4,
    rtol: float = 3e-3,
) -> dict[str, float]:
    """Validate calibrated SAE geometry on the batch used by optimization."""
    corrupted = corruption["corrupted"]
    magnitude = corruption["magnitude"]
    direction_ids = corruption["direction_id"]
    corruption_type = corruption["corruption_type"]
    assert isinstance(corrupted, Tensor) and corrupted.shape == clean_z.shape
    assert isinstance(magnitude, Tensor) and magnitude.shape == clean_z.shape[:-1]
    assert isinstance(direction_ids, Tensor) and direction_ids.shape == magnitude.shape
    assert isinstance(corruption_type, Tensor) and corruption_type.shape == magnitude.shape
    mask = (corruption_type == CORRUPTION_SAE_CALIBRATED) | (
        corruption_type == CORRUPTION_SAE_HARMFUL
    )
    if not mask.any().item():
        return {"num_sae": 0.0}
    delta = corrupted[mask].float() - clean_z[mask].float()
    actual_norm = torch.linalg.vector_norm(delta, dim=-1)
    expected_norm = magnitude[mask].float()
    actual_mse = delta.square().mean(dim=-1)
    expected_mse = expected_norm.square() / clean_z.shape[-1]
    try:
        torch.testing.assert_close(actual_norm, expected_norm, atol=atol, rtol=rtol)
        torch.testing.assert_close(actual_mse, expected_mse, atol=atol, rtol=rtol)
    except AssertionError as error:
        bad = ((actual_norm - expected_norm).abs() > atol + rtol * expected_norm.abs())
        bad_indices = torch.nonzero(mask, as_tuple=False).flatten()[bad][:8]
        details = {
            "example_indices": bad_indices.detach().cpu().tolist(),
            "direction_ids": direction_ids[bad_indices].detach().cpu().tolist(),
            "magnitudes": magnitude[bad_indices].detach().cpu().tolist(),
            "actual_norm": actual_norm[bad][:8].detach().cpu().tolist(),
            "expected_norm": expected_norm[bad][:8].detach().cpu().tolist(),
            "actual_mse": actual_mse[bad][:8].detach().cpu().tolist(),
            "expected_mse": expected_mse[bad][:8].detach().cpu().tolist(),
        }
        raise AssertionError(f"Calibrated REAL training batch math failed: {details}") from error
    return {
        "num_sae": float(mask.sum().detach().cpu()),
        "mean_actual_norm": actual_norm.mean().detach().cpu().item(),
        "mean_expected_norm": expected_norm.mean().detach().cpu().item(),
        "mean_actual_mse": actual_mse.mean().detach().cpu().item(),
        "mean_expected_mse": expected_mse.mean().detach().cpu().item(),
    }


@torch.inference_mode()
def correction_diagnostics(
    clean_z: Tensor,
    corrupted_z: Tensor,
    predicted_z: Tensor,
    sampled_magnitude: Tensor | None = None,
    calibrated_mask: Tensor | None = None,
    eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Compute correction geometry and enforce ``MSE * D == ||delta||^2``."""
    assert clean_z.shape == corrupted_z.shape == predicted_z.shape
    assert clean_z.ndim == 2 and clean_z.shape[-1] > 0
    assert clean_z.device == corrupted_z.device == predicted_z.device
    clean_before = clean_z.detach().clone()
    corruption = corrupted_z.float() - clean_z.float()
    target = clean_z.float() - corrupted_z.float()
    predicted = predicted_z.float() - corrupted_z.float()
    corruption_norm = torch.linalg.vector_norm(corruption, dim=-1)
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    predicted_norm = torch.linalg.vector_norm(predicted, dim=-1)
    corrupted_mse = corruption.square().mean(dim=-1)
    torch.testing.assert_close(corruption_norm, target_norm, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        corrupted_mse * clean_z.shape[-1], corruption_norm.square(),
        rtol=2e-5, atol=2e-5,
    )
    if sampled_magnitude is not None and calibrated_mask is not None:
        assert sampled_magnitude.shape == calibrated_mask.shape == corruption_norm.shape
        if calibrated_mask.any().item():
            torch.testing.assert_close(
                corruption_norm[calibrated_mask], sampled_magnitude[calibrated_mask].float(),
                rtol=3e-3, atol=3e-4,
            )
    cosine = F.cosine_similarity(predicted, target, dim=-1, eps=eps)
    torch.testing.assert_close(clean_z, clean_before, rtol=0.0, atol=0.0)
    return {
        "corrupted_mse": corrupted_mse,
        "corruption_norm": corruption_norm,
        "target_correction_norm": target_norm,
        "predicted_correction_norm": predicted_norm,
        "correction_norm_ratio": predicted_norm / target_norm.clamp_min(eps),
        "correction_cosine": cosine,
    }


class MixtureCorruptionSampler:
    """Apply per-example mixtures of corruption samplers."""

    def __init__(
        self,
        samplers: list[CorruptionSampler],
        probabilities: list[float],
        corruption_mode: str = "mixture",
    ) -> None:
        assert samplers and len(samplers) == len(probabilities)
        assert all(probability >= 0.0 for probability in probabilities)
        probability_tensor = torch.tensor(probabilities, dtype=torch.float64)
        torch.testing.assert_close(
            probability_tensor.sum(),
            torch.tensor(1.0, dtype=torch.float64),
            rtol=0.0,
            atol=1e-8,
        )
        self.samplers = samplers
        self.probabilities = probability_tensor
        self.corruption_mode = corruption_mode
        self.component_names = [
            CORRUPTION_TYPE_NAMES.get(
                int(getattr(sampler, "corruption_type", CORRUPTION_GAUSSIAN)),
                f"component_{index}",
            )
            for index, sampler in enumerate(samplers)
        ]

    def sample_assignments(
        self, num_samples: int, device: torch.device, generator: torch.Generator
    ) -> Tensor:
        assert num_samples > 0
        return torch.multinomial(
            self.probabilities.to(device=device, dtype=torch.float32),
            num_samples=num_samples,
            replacement=True,
            generator=generator,
        )

    def __call__(
        self,
        clean: Tensor,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        assert clean.ndim == 2
        result = self.sample_batch(clean, generator)
        return result["corrupted"], result["magnitude"]

    def sample_batch(self, clean: Tensor, generator: torch.Generator) -> dict[str, Any]:
        assert clean.ndim == 2
        assignments = self.sample_assignments(clean.shape[0], clean.device, generator)
        corrupted = torch.empty_like(clean)
        noise_level = clean.new_empty(clean.shape[0])
        direction_id = torch.full(
            (clean.shape[0],), -1, device=clean.device, dtype=torch.long
        )
        corruption_type = torch.full_like(direction_id, -1)
        for sampler_index, sampler in enumerate(self.samplers):
            mask = assignments == sampler_index
            if not mask.any().item():
                continue
            sample_batch = getattr(sampler, "sample_batch", None)
            if sample_batch is None:
                component_corrupted, component_level = sampler(clean[mask], generator)
                component = {
                    "corrupted": component_corrupted,
                    "magnitude": component_level,
                    "direction_id": torch.full_like(component_level, -1, dtype=torch.long),
                    "corruption_type": torch.full_like(
                        component_level, CORRUPTION_GAUSSIAN, dtype=torch.long
                    ),
                }
            else:
                component = sample_batch(clean[mask], generator)
                component_corrupted = component["corrupted"]
                component_level = component["magnitude"]
            assert component_corrupted.shape == clean[mask].shape
            assert component_level.shape == clean[mask].shape[:-1]
            corrupted[mask] = component_corrupted
            noise_level[mask] = component_level
            direction_id[mask] = component["direction_id"]
            corruption_type[mask] = component["corruption_type"]
        assert (corruption_type >= 0).all().item()
        return {
            "corrupted": corrupted,
            "magnitude": noise_level,
            "direction_id": direction_id,
            "corruption_type": corruption_type,
            "mixture_assignment": assignments,
        }


@torch.inference_mode()
def validate_mixture_sampling(
    sampler: MixtureCorruptionSampler,
    identity_probability: float,
    *,
    num_samples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Check requested versus empirical component frequencies, including identity."""
    assert 0.0 <= identity_probability <= 1.0 and num_samples >= 10_000
    generator = torch.Generator(device="cpu").manual_seed(seed)
    assignments = sampler.sample_assignments(num_samples, torch.device("cpu"), generator)
    identity = torch.rand(num_samples, generator=generator) < identity_probability
    empirical: dict[str, float] = {"identity": identity.float().mean().item()}
    requested: dict[str, float] = {"identity": identity_probability}
    for index, probability in enumerate(sampler.probabilities.tolist()):
        name = sampler.component_names[index]
        empirical[name] = ((assignments == index) & ~identity).float().mean().item()
        requested[name] = (1.0 - identity_probability) * float(probability)
    for name, expected in requested.items():
        observed = empirical[name]
        sigma = (expected * (1.0 - expected) / num_samples) ** 0.5
        tolerance = max(0.01, 6.0 * sigma)
        assert abs(observed - expected) <= tolerance, (
            f"Implausible corruption mixture frequency for {name}: "
            f"requested={expected:.4f}, observed={observed:.4f}, tolerance={tolerance:.4f}"
        )
    return {"requested": requested, "empirical": empirical, "num_samples": num_samples}


def gaussian_corruption(
    clean: Tensor,
    generator: torch.Generator,
    sigma_min: float = 0.0,
    sigma_max: float = 2.0,
    distribution: str = "uniform",
) -> tuple[Tensor, Tensor]:
    """Add isotropic Gaussian noise with independently sampled magnitudes."""
    assert clean.ndim == 2, f"Expected [batch, d_model], got {clean.shape}."
    assert 0.0 <= sigma_min <= sigma_max
    sample_shape = clean.shape[:-1]

    if sigma_min == sigma_max:
        sigma = clean.new_full(sample_shape, sigma_min)
    else:
        uniform = torch.rand(
            sample_shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        if distribution == "uniform":
            sigma = sigma_min + (sigma_max - sigma_min) * uniform
        elif distribution == "log_uniform":
            assert sigma_min > 0.0, "log_uniform requires sigma_min > 0."
            log_min = torch.log(clean.new_tensor(sigma_min))
            log_max = torch.log(clean.new_tensor(sigma_max))
            sigma = torch.exp(log_min + (log_max - log_min) * uniform)
        else:
            raise ValueError("distribution must be 'uniform' or 'log_uniform'.")

    epsilon = torch.randn(
        clean.shape,
        device=clean.device,
        dtype=clean.dtype,
        generator=generator,
    )
    corrupted = clean + sigma.unsqueeze(-1) * epsilon
    return corrupted, sigma


class GaussianCorruptionSampler:
    """Metadata-aware wrapper around :func:`gaussian_corruption`."""

    input_space = "standardized"
    corruption_mode = "gaussian"

    def __init__(self, sigma_min: float, sigma_max: float, distribution: str) -> None:
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.distribution = distribution

    def __call__(self, clean: Tensor, generator: torch.Generator) -> tuple[Tensor, Tensor]:
        return gaussian_corruption(
            clean, generator, self.sigma_min, self.sigma_max, self.distribution
        )

    def sample_batch(self, clean: Tensor, generator: torch.Generator) -> dict[str, Any]:
        corrupted, magnitude = self(clean, generator)
        return {
            "corrupted": corrupted,
            "magnitude": magnitude,
            "direction_id": torch.full_like(magnitude, -1, dtype=torch.long),
            "corruption_type": torch.full_like(
                magnitude, CORRUPTION_GAUSSIAN, dtype=torch.long
            ),
        }


def make_gaussian_corruption_sampler(config: Mapping[str, Any]) -> CorruptionSampler:
    """Build the Gaussian baseline sampler from the plain configuration dict."""
    noise_config = config.get("noise", config)
    return GaussianCorruptionSampler(
        sigma_min=float(noise_config.get("gaussian_sigma_min", 0.0)),
        sigma_max=float(noise_config.get("gaussian_sigma_max", 2.0)),
        distribution=str(noise_config.get("gaussian_sigma_distribution", "uniform")),
    )


def _training_direction_subset(
    all_directions: Tensor,
    direction_split: Mapping[str, Any],
) -> tuple[Tensor, list[int]]:
    assert all_directions.ndim == 2
    num_features = int(direction_split["num_features"])
    assert all_directions.shape[0] == num_features
    train_ids = list(direction_split["train"])
    validate_direction_split(
        train_ids,
        list(direction_split["val"]),
        list(direction_split["test"]),
        num_features,
    )
    validate_direction_ids_for_usage(
        train_ids,
        direction_split,
        usage="training",
        require_complete_split=True,
    )
    index = torch.tensor(train_ids, dtype=torch.long, device=all_directions.device)
    return all_directions.index_select(0, index), train_ids


def make_structured_sae_corruption_sampler(
    all_directions: Tensor,
    direction_split: Mapping[str, Any],
    config: Mapping[str, Any],
    normalization_stats: Mapping[str, Any] | str | Path | None = None,
) -> MixtureCorruptionSampler:
    """Build the uniform train-SAE/Gaussian baseline mixture."""
    noise_config = config.get("noise", config)
    # Structured SAE corruption is defined in raw activation space so its
    # magnitude has the same semantic meaning as inference steering:
    # ``h + r * ||h|| * u``.  The training loop standardizes the resulting
    # corrupted activation afterwards, using the exact cache statistics.
    sampling_directions = all_directions
    train_directions, train_ids = _training_direction_subset(
        sampling_directions, direction_split
    )
    sae_sampler = SAEDirectionCorruptionSampler(
        train_directions=train_directions,
        train_direction_ids=train_ids,
        direction_split=direction_split,
        relative_magnitude_min=float(
            noise_config.get("sae_relative_magnitude_min", 0.1)
        ),
        relative_magnitude_max=float(
            noise_config.get("sae_relative_magnitude_max", 1.0)
        ),
        sampling_mode="uniform",
    )
    gaussian_sampler = make_gaussian_corruption_sampler(config)
    mixture = MixtureCorruptionSampler(
        samplers=[sae_sampler, gaussian_sampler],
        probabilities=[
            float(noise_config.get("structured_sae_probability", 0.85)),
            float(noise_config.get("structured_gaussian_probability", 0.15)),
        ],
        corruption_mode="sae_rawscale_legacy",
    )
    mixture.input_space = "raw"
    return mixture


def make_calibrated_sae_corruption_sampler(
    all_directions: Tensor,
    direction_split: Mapping[str, Any],
    normalization_stats: Mapping[str, Any] | str | Path,
    magnitude_min: float,
    magnitude_max: float,
    gaussian_probability: float = 0.15,
    gaussian_sigma_min: float = 0.0,
    gaussian_sigma_max: float = 2.0,
    eps: float = 1e-6,
) -> MixtureCorruptionSampler:
    """Build the new calibrated standardized-unit SAE/Gaussian mixture."""
    stats = load_activation_stats(normalization_stats) if isinstance(normalization_stats, (str, Path)) else normalization_stats
    std = stats["std"]
    train_directions, train_ids = _training_direction_subset(all_directions, direction_split)
    sae_sampler = SAEStandardizedUnitCorruptionSampler(
        train_directions, train_ids, direction_split, std,
        magnitude_min, magnitude_max, eps=eps,
    )
    gaussian_sampler = make_gaussian_corruption_sampler(
        {"noise": {
            "gaussian_sigma_min": gaussian_sigma_min,
            "gaussian_sigma_max": gaussian_sigma_max,
        }}
    )
    mixture = MixtureCorruptionSampler(
        [sae_sampler, gaussian_sampler],
        [1.0 - gaussian_probability, gaussian_probability],
        corruption_mode="sae_calibrated",
    )
    mixture.input_space = "standardized"
    return mixture


def make_fluency_sensitive_corruption_sampler(
    all_directions: Tensor,
    direction_split: Mapping[str, Any],
    config: Mapping[str, Any],
    scores_path: str | Path | None = None,
    normalization_stats: Mapping[str, Any] | str | Path | None = None,
) -> MixtureCorruptionSampler:
    """Build weighted-SAE, uniform-SAE, and Gaussian main-method mixture."""
    noise_config = config.get("noise", config)
    score_config = config.get("damage_score", {})
    assert normalization_stats is not None, (
        "Fluency-sensitive calibrated corruption requires activation statistics."
    )
    stats = (
        load_activation_stats(normalization_stats)
        if isinstance(normalization_stats, (str, Path)) else normalization_stats
    )
    train_directions, train_ids = _training_direction_subset(
        all_directions, direction_split
    )
    resolved_scores_path = Path(
        scores_path
        or score_config.get(
            "output_path", "outputs/direction_scores/train_scores.pt"
        )
    )
    scores = load_train_direction_scores(
        resolved_scores_path,
        train_ids,
        direction_split,
        expected_hook_name=config.get("model", {}).get("hook_name"),
    )
    clip_value = noise_config.get("score_clip_quantile", 0.99)
    clip_quantile = None if clip_value is None else float(clip_value)
    common_kwargs = {
        "train_directions": train_directions,
        "train_direction_ids": train_ids,
        "direction_split": direction_split,
        "std": stats["std"],
        "magnitude_min": float(noise_config.get("calibrated_magnitude_min", 3.8)),
        "magnitude_max": float(noise_config.get("calibrated_magnitude_max", 19.1)),
    }
    weighted_sampler = SAEStandardizedUnitCorruptionSampler(
        **common_kwargs,
        sampling_mode=str(
            noise_config.get("fluency_direction_sampling_mode", "score_proportional")
        ),
        scores=scores,
        gamma=float(noise_config.get("score_gamma", 0.5)),
        score_eps=float(noise_config.get("score_eps", 1e-8)),
        clip_quantile=clip_quantile,
        corruption_type=CORRUPTION_SAE_HARMFUL,
    )
    uniform_sampler = SAEStandardizedUnitCorruptionSampler(
        **common_kwargs,
        sampling_mode="uniform",
        corruption_type=CORRUPTION_SAE_CALIBRATED,
    )
    gaussian_sampler = make_gaussian_corruption_sampler(config)
    mixture = MixtureCorruptionSampler(
        samplers=[weighted_sampler, uniform_sampler, gaussian_sampler],
        probabilities=[
            float(noise_config.get("fluency_weighted_sae_probability", 0.60)),
            float(noise_config.get("fluency_uniform_sae_probability", 0.25)),
            float(noise_config.get("fluency_gaussian_probability", 0.15)),
        ],
        corruption_mode="fluency_sensitive",
    )
    mixture.input_space = "standardized"
    return mixture


def load_activation_stats(path: str | Path) -> dict[str, Any]:
    """Load activation normalization statistics saved by the cache pipeline."""
    stats = torch.load(Path(path), map_location="cpu", weights_only=True)
    assert isinstance(stats, dict)
    assert isinstance(stats.get("mean"), Tensor)
    assert isinstance(stats.get("std"), Tensor)
    return stats


def _resolve_normalization(
    normalization_stats: Mapping[str, Any] | str | Path | None,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    if normalization_stats is None:
        mean_cpu = torch.zeros(d_model, dtype=torch.float32)
        std_cpu = torch.ones(d_model, dtype=torch.float32)
        reference: str | None = None
        enabled = False
    else:
        if isinstance(normalization_stats, (str, Path)):
            stats_path = Path(normalization_stats)
            stats = load_activation_stats(stats_path)
            reference = str(stats_path)
        else:
            stats = dict(normalization_stats)
            reference = None
        mean_cpu = stats["mean"].detach().to(device="cpu", dtype=torch.float32)
        std_cpu = stats["std"].detach().to(device="cpu", dtype=torch.float32)
        enabled = True

    assert mean_cpu.shape == (d_model,)
    assert std_cpu.shape == (d_model,)
    mean = mean_cpu.to(device=device, dtype=dtype)
    std = std_cpu.to(device=device, dtype=dtype).clamp_min(eps)
    checkpoint_data = {
        "enabled": enabled,
        "mean": mean_cpu,
        "std": std_cpu,
        "eps": eps,
        "stats_path": reference,
    }
    return mean, std, checkpoint_data


def standardize_activations(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Backward-compatible alias for the central normalization helper."""
    return normalize_activations(x, mean, std)


def destandardize_activations(z: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Backward-compatible alias for the central denormalization helper."""
    return denormalize_activations(z, mean, std)


@torch.inference_mode()
def denoise_activations(
    denoiser: nn.Module,
    x: Tensor,
    normalization: Mapping[str, Any],
    noise_level: Tensor | float | None = None,
) -> Tensor:
    """Run inference with the same standardization stored in a checkpoint."""
    parameter = next(denoiser.parameters())
    assert x.device == parameter.device and x.dtype == parameter.dtype
    eps = float(normalization.get("eps", 1e-6))
    mean, std, _ = _resolve_normalization(
        normalization,
        d_model=x.shape[-1],
        device=x.device,
        dtype=x.dtype,
        eps=eps,
    )
    standardized = standardize_activations(x, mean, std)
    if bool(getattr(denoiser, "condition_on_noise", False)):
        predicted = denoiser(standardized, noise_level=noise_level)
    else:
        predicted = denoiser(standardized)
    return destandardize_activations(predicted, mean, std)


def _prepare_batch(batch: Any, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(batch, (tuple, list)):
        assert len(batch) == 1
        batch = batch[0]
    assert isinstance(batch, Tensor) and batch.ndim == 2
    return batch.to(device=device, dtype=dtype, non_blocking=True)


@torch.inference_mode()
def _validate(
    denoiser: nn.Module,
    validation_loader: DataLoader[Tensor],
    corruption_sampler: CorruptionSampler,
    mean: Tensor,
    std: Tensor,
    generator: torch.Generator,
    max_batches: int,
) -> dict[str, float]:
    denoiser.eval()
    reconstruction_sum = 0.0
    identity_sum = 0.0
    num_examples = 0
    parameter = next(denoiser.parameters())

    for batch_index, batch in enumerate(validation_loader):
        if batch_index >= max_batches:
            break
        clean = _prepare_batch(batch, parameter.device, parameter.dtype)
        clean_standardized = standardize_activations(clean, mean, std)
        corruption = _sample_corruption(
            corruption_sampler, clean, clean_standardized, mean, std, generator
        )
        corrupted = corruption["corrupted"]
        noise_level = corruption["magnitude"]
        assert corrupted.shape == clean_standardized.shape
        assert noise_level.shape == clean_standardized.shape[:-1]
        predicted = denoiser(corrupted, noise_level=noise_level)
        zero_noise = torch.zeros_like(noise_level)
        identity_prediction = denoiser(clean_standardized, noise_level=zero_noise)

        batch_size = clean.shape[0]
        reconstruction_sum += F.mse_loss(predicted, clean_standardized).item() * batch_size
        identity_sum += (
            F.mse_loss(identity_prediction, clean_standardized).item() * batch_size
        )
        num_examples += batch_size

    assert num_examples > 0, "Validation dataset produced no examples."
    return {
        "val_loss": reconstruction_sum / num_examples,
        "val_clean_identity_loss": identity_sum / num_examples,
    }


def _save_checkpoint(
    path: Path,
    denoiser: nn.Module,
    config: Mapping[str, Any],
    step: int,
    normalization: Mapping[str, Any],
    history: list[dict[str, float | int]],
    corruption_sampler: CorruptionSampler,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config_section = config.get("model", {})
    split_path = Path(config.get("directions", {}).get(
        "split_path", "outputs/direction_split.json"
    ))
    if split_path.exists():
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        split_hash = direction_split_hash(split_payload)
    else:
        split_hash = "not_applicable"
    corruption_mode = str(
        getattr(corruption_sampler, "corruption_mode", "custom")
    )
    checkpoint = {
        "checkpoint_version": CHECKPOINT_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "model_config": {
            "d_model": int(denoiser.d_model),
            "hidden_dim": int(denoiser.hidden_dim),
            "condition_on_noise": bool(denoiser.condition_on_noise),
            "conditioning_hidden_dim": int(denoiser.conditioning_hidden_dim),
        },
        "model_class": type(denoiser).__name__,
        "parameter_dtype": str(next(denoiser.parameters()).dtype).removeprefix(
            "torch."
        ),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in denoiser.state_dict().items()
        },
        "config": deepcopy(dict(config)),
        "step": step,
        "normalization": dict(normalization),
        "history": history,
        "training_seed": int(config.get("seed", 42)),
        "d_model": int(denoiser.d_model),
        "hidden_dim": int(denoiser.hidden_dim),
        "hook_name": model_config_section.get("hook_name"),
        "model_name": model_config_section.get("name"),
        "corruption_mode": corruption_mode,
        "corruption_config": deepcopy(dict(config.get("noise", {}))),
        "direction_split_hash": split_hash,
        "real_training_math_verified": bool(
            getattr(corruption_sampler, "real_training_math_verified", False)
        ),
        "real_training_math_summary": getattr(
            corruption_sampler, "real_training_math_summary", None
        ),
    }
    torch.save(checkpoint, path)


def load_denoiser_checkpoint(
    path: str | Path,
    device: str | torch.device,
    dtype: str | torch.dtype | None = None,
    *,
    expected_hook_name: str | None = None,
    expected_model_name: str | None = None,
    expected_corruption_mode: str | None = None,
    expected_direction_split_hash: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Reconstruct a denoiser and return it with its saved inference metadata."""
    from src.denoiser import ResidualDenoiser

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    assert isinstance(checkpoint, dict)
    required = {
        "checkpoint_version", "pipeline_version", "model_state_dict",
        "model_config", "normalization", "config", "step", "training_seed",
        "d_model", "hidden_dim", "hook_name", "model_name", "corruption_mode",
        "corruption_config", "direction_split_hash",
    }
    missing = required - set(checkpoint)
    assert not missing, f"Checkpoint is missing required keys: {sorted(missing)}"
    assert checkpoint["checkpoint_version"] == CHECKPOINT_SCHEMA_VERSION, (
        "Incompatible checkpoint schema; retraining is required."
    )
    assert checkpoint["pipeline_version"] == PIPELINE_VERSION, (
        "Checkpoint was created by an incompatible experiment pipeline."
    )
    model_config = checkpoint["model_config"]
    assert isinstance(model_config, dict)
    denoiser = ResidualDenoiser(**model_config)
    denoiser.load_state_dict(checkpoint["model_state_dict"], strict=True)
    resolved_device = resolve_device(device)
    saved_dtype = checkpoint.get("parameter_dtype", "float32")
    resolved_dtype = resolve_dtype(saved_dtype if dtype is None else dtype)
    denoiser.to(device=resolved_device, dtype=resolved_dtype).eval()
    normalization = checkpoint["normalization"]
    assert isinstance(normalization, dict)
    assert normalization["mean"].shape == (denoiser.d_model,)
    assert normalization["std"].shape == (denoiser.d_model,)
    assert torch.isfinite(normalization["mean"]).all().item()
    assert torch.isfinite(normalization["std"]).all().item()
    assert (normalization["std"] > 0).all().item()
    assert int(checkpoint["d_model"]) == denoiser.d_model
    assert int(checkpoint["hidden_dim"]) == denoiser.hidden_dim
    if expected_hook_name is not None:
        assert checkpoint["hook_name"] == expected_hook_name
    if expected_model_name is not None:
        assert checkpoint["model_name"] == expected_model_name
    if expected_corruption_mode is not None:
        assert checkpoint["corruption_mode"] == expected_corruption_mode
    if expected_direction_split_hash is not None:
        assert checkpoint["direction_split_hash"] == expected_direction_split_hash, (
            "Checkpoint direction split fingerprint does not match the current split."
        )
    if checkpoint["corruption_mode"] in {"sae_calibrated", "fluency_sensitive"}:
        assert checkpoint.get("real_training_math_verified") is True, (
            "Calibrated checkpoint lacks proof that real training-batch math passed."
        )
    return denoiser, checkpoint


def train_denoiser(
    denoiser: nn.Module,
    activation_dataset: ActivationDataset,
    config: Mapping[str, Any],
    corruption_sampler: CorruptionSampler,
    validation_dataset: ActivationDataset,
    normalization_stats: Mapping[str, Any] | str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, float | int]]:
    """Train a denoiser with generic corruption and held-out validation data.

    Use distinct ``checkpoint_path`` values for Gaussian, structured-SAE, and
    fluency-sensitive runs while keeping the same model and optimizer config.
    """
    training_config = config.get("training", config)
    seed = int(config.get("seed", 42))
    deterministic = bool(
        config.get("reproducibility", {}).get("deterministic_algorithms", False)
    )
    seed_everything(seed, deterministic=deterministic)

    if normalization_stats is None:
        configured_stats_path = training_config.get("normalization_stats_path")
        if configured_stats_path is not None and Path(configured_stats_path).exists():
            normalization_stats = Path(configured_stats_path)

    parameter = next(denoiser.parameters())
    device, dtype = parameter.device, parameter.dtype
    d_model = getattr(denoiser, "d_model", None)
    assert isinstance(d_model, int), "denoiser must expose an integer d_model."

    batch_size = int(training_config.get("batch_size", 64))
    num_workers = int(training_config.get("num_workers", 0))
    assert num_workers == 0, "Kaggle-safe denoiser training requires num_workers=0."
    assert not bool(training_config.get("persistent_workers", False)), (
        "Kaggle-safe denoiser training requires persistent_workers=False."
    )
    loader_generator = torch.Generator(device="cpu").manual_seed(seed)
    train_is_iterable = isinstance(activation_dataset, IterableDataset)
    train_loader = DataLoader(
        activation_dataset,
        batch_size=batch_size,
        shuffle=not train_is_iterable,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
        persistent_workers=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator(device="cpu").manual_seed(seed + 1),
        persistent_workers=False,
    )

    normalization_eps = float(training_config.get("normalization_eps", 1e-6))
    mean, std, normalization_data = _resolve_normalization(
        normalization_stats,
        d_model=d_model,
        device=device,
        dtype=dtype,
        eps=normalization_eps,
    )
    optimizer = torch.optim.AdamW(
        denoiser.parameters(),
        lr=float(training_config.get("learning_rate", 1e-4)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    gradient_clip_norm = float(training_config.get("gradient_clip_norm", 1.0))
    identity_probability = float(
        training_config.get("clean_identity_probability", 0.1)
    )
    assert 0.0 <= identity_probability <= 1.0
    if isinstance(corruption_sampler, MixtureCorruptionSampler):
        mixture_summary = validate_mixture_sampling(
            corruption_sampler, identity_probability, seed=seed
        )
        print("Corruption mixture gate:", mixture_summary)
    num_epochs = int(training_config.get("num_epochs", 1))
    max_steps_value = training_config.get("max_steps")
    max_steps = None if max_steps_value is None else int(max_steps_value)
    log_every = int(training_config.get("log_every", 50))
    max_validation_batches = int(
        training_config.get("max_validation_batches", 20)
    )
    checkpoint_every = int(training_config.get("checkpoint_every", 1))
    assert log_every > 0 and max_validation_batches > 0 and checkpoint_every > 0
    resolved_checkpoint_path = Path(
        checkpoint_path
        or training_config.get("checkpoint_path", "outputs/checkpoints/gaussian.pt")
    )

    train_generator = torch.Generator(device=device).manual_seed(seed)
    validation_generator = torch.Generator(device=device).manual_seed(seed + 1)
    history: list[dict[str, float | int]] = []
    step = 0
    stop_training = False

    for epoch in range(num_epochs):
        denoiser.train()
        progress = tqdm(train_loader, desc=f"Denoiser epoch {epoch + 1}/{num_epochs}")
        for batch in progress:
            clean = _prepare_batch(batch, device, dtype)
            assert clean.shape[-1] == d_model
            clean_standardized, corruption = prepare_training_corruption_batch(
                clean, corruption_sampler, mean, std, train_generator,
                identity_probability,
            )
            corrupted = corruption["corrupted"]
            noise_level = corruption["magnitude"]
            assert corrupted.shape == clean_standardized.shape
            assert corrupted.device == device and corrupted.dtype == dtype
            assert noise_level.shape == clean_standardized.shape[:-1]
            assert noise_level.device == device and noise_level.dtype == dtype

            identity_mask = corruption["identity_mask"]

            math_assert_every = int(training_config.get("math_assert_every", 250))
            if step < 5 or (math_assert_every > 0 and (step + 1) % math_assert_every == 0):
                math_summary = assert_calibrated_training_batch_math(
                    clean_standardized, corruption
                )
                if math_summary.get("num_sae", 0.0) > 0 and step == 0:
                    print("REAL training-batch calibrated math:", math_summary)
                if math_summary.get("num_sae", 0.0) > 0:
                    setattr(corruption_sampler, "real_training_math_verified", True)
                    setattr(corruption_sampler, "real_training_math_summary", math_summary)

            optimizer.zero_grad(set_to_none=True)
            predicted_clean = denoiser(corrupted, noise_level=noise_level)
            loss = F.mse_loss(predicted_clean, clean_standardized)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), gradient_clip_norm)
            optimizer.step()
            step += 1

            identity_loss: Tensor | None = None
            if identity_mask.any().item():
                identity_loss = F.mse_loss(
                    predicted_clean[identity_mask], clean_standardized[identity_mask]
                )
            progress.set_postfix(loss=f"{loss.item():.5f}", step=step)
            if step == 1 or step % log_every == 0:
                identity_text = (
                    f"{identity_loss.item():.6f}" if identity_loss is not None else "n/a"
                )
                print(
                    f"step={step} train_loss={loss.item():.6f} "
                    f"clean_identity_loss={identity_text}"
                )

            if max_steps is not None and step >= max_steps:
                stop_training = True
                break

        validation_metrics = _validate(
            denoiser,
            validation_loader,
            corruption_sampler,
            mean,
            std,
            validation_generator,
            max_batches=max_validation_batches,
        )
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch + 1,
            "step": step,
            **validation_metrics,
        }
        history.append(epoch_metrics)
        print(
            f"epoch={epoch + 1} step={step} "
            f"val_loss={validation_metrics['val_loss']:.6f} "
            "val_clean_identity_loss="
            f"{validation_metrics['val_clean_identity_loss']:.6f}"
        )

        if (epoch + 1) % checkpoint_every == 0 or stop_training:
            _save_checkpoint(
                resolved_checkpoint_path,
                denoiser,
                config,
                step,
                normalization_data,
                history,
                corruption_sampler,
            )
        if stop_training:
            break

    _save_checkpoint(
        resolved_checkpoint_path,
        denoiser,
        config,
        step,
        normalization_data,
        history,
        corruption_sampler,
    )
    return history


@torch.inference_mode()
def denoising_sanity_metrics(
    denoiser: nn.Module,
    clean: Tensor,
    corruption_sampler: CorruptionSampler,
    normalization_stats: Mapping[str, Any] | str | Path,
    seed: int = 42,
) -> dict[str, float | bool]:
    """Compare corrupted-vs-clean and denoised-vs-clean MSE on held-out rows."""
    parameter = next(denoiser.parameters())
    clean = clean.to(device=parameter.device, dtype=parameter.dtype)
    assert clean.ndim == 2 and clean.shape[-1] == getattr(denoiser, "d_model", -1)
    mean, std, _ = _resolve_normalization(
        normalization_stats,
        clean.shape[-1],
        clean.device,
        clean.dtype,
        1e-6,
    )
    standardized = standardize_activations(clean, mean, std)
    generator = torch.Generator(device=clean.device).manual_seed(seed)
    corruption = _sample_corruption(
        corruption_sampler, clean, standardized, mean, std, generator
    )
    corrupted = corruption["corrupted"]
    noise_level = corruption["magnitude"]
    denoised = denoiser(corrupted, noise_level=noise_level)
    corrupted_mse = F.mse_loss(corrupted, standardized).item()
    denoised_mse = F.mse_loss(denoised, standardized).item()
    geometry = correction_diagnostics(
        standardized,
        corrupted,
        denoised,
        sampled_magnitude=noise_level,
        calibrated_mask=(
            (corruption["corruption_type"] == CORRUPTION_SAE_CALIBRATED)
            | (corruption["corruption_type"] == CORRUPTION_SAE_HARMFUL)
        ),
    )
    return {
        "corrupted_mse": float(corrupted_mse),
        "denoised_mse": float(denoised_mse),
        "improves": bool(denoised_mse < corrupted_mse),
        "mean_corruption_norm": geometry["corruption_norm"].mean().item(),
        "mean_correction_cosine": geometry["correction_cosine"].mean().item(),
    }


@torch.inference_mode()
def corruption_scale_diagnostics(
    clean: Tensor,
    directions: Tensor,
    std: Tensor,
    strengths: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0),
) -> tuple[Any, Any]:
    """Return per-strength corruption summaries and direction norm diagnostics."""
    import pandas as pd

    assert clean.ndim == 2 and directions.ndim == 2 and std.ndim == 1
    assert clean.shape[1] == directions.shape[1] == std.shape[0]
    h_norm = torch.linalg.vector_norm(clean.float(), dim=-1)
    rows: list[dict[str, float]] = []
    for strength in strengths:
        delta_raw = strength * h_norm[:, None, None] * directions[None].float()
        delta_raw_norm = torch.linalg.vector_norm(delta_raw, dim=-1)
        delta_z = delta_raw / std.float()[None, None, :].clamp_min(1e-6)
        delta_z_norm = torch.linalg.vector_norm(delta_z, dim=-1)
        z_mse = delta_z.square().mean(dim=-1)
        values = delta_z_norm.flatten()
        rows.append({
            "strength": float(strength),
            "raw_norm_mean": float(delta_raw_norm.mean()),
            "raw_norm_std": float(delta_raw_norm.std(unbiased=False)),
            "raw_norm_median": float(delta_raw_norm.median()),
            "standardized_norm_mean": float(delta_z_norm.mean()),
            "standardized_norm_std": float(delta_z_norm.std(unbiased=False)),
            "standardized_norm_median": float(values.median()),
            "standardized_norm_p90": float(torch.quantile(values, 0.90)),
            "standardized_norm_p95": float(torch.quantile(values, 0.95)),
            "standardized_norm_p99": float(torch.quantile(values, 0.99)),
            "standardized_norm_max": float(values.max()),
            "standardized_mse_mean": float(z_mse.mean()),
        })
    mapped = directions.float() / std.float().clamp_min(1e-6)
    direction_rows = pd.DataFrame({
        "direction_id": list(range(directions.shape[0])),
        "raw_norm": torch.linalg.vector_norm(directions.float(), dim=-1).cpu().numpy(),
        "standardized_norm": torch.linalg.vector_norm(mapped, dim=-1).cpu().numpy(),
    })
    return pd.DataFrame(rows), direction_rows


def _configured_normalization_stats(
    config: Mapping[str, Any],
    normalization_stats: Mapping[str, Any] | str | Path | None,
) -> Mapping[str, Any] | str | Path | None:
    if normalization_stats is not None:
        return normalization_stats
    configured_path = config.get("training", config).get("normalization_stats_path")
    if configured_path is not None and Path(configured_path).exists():
        return Path(configured_path)
    return None


def train_gaussian_denoiser(
    denoiser: nn.Module,
    activation_dataset: ActivationDataset,
    validation_dataset: ActivationDataset,
    config: Mapping[str, Any],
    normalization_stats: Mapping[str, Any] | str | Path | None = None,
) -> list[dict[str, float | int]]:
    """Train the Gaussian baseline using the common optimizer/checkpoint path."""
    resolved_stats = _configured_normalization_stats(config, normalization_stats)
    checkpoint_path = config.get("training", config).get(
        "checkpoint_paths", {}
    ).get("gaussian", "outputs/checkpoints/gaussian.pt")
    return train_denoiser(
        denoiser, activation_dataset, config, make_gaussian_corruption_sampler(config),
        validation_dataset, normalization_stats=resolved_stats,
        checkpoint_path=checkpoint_path,
    )


def train_structured_sae_denoiser(
    denoiser: nn.Module,
    activation_dataset: ActivationDataset,
    validation_dataset: ActivationDataset,
    all_directions: Tensor,
    direction_split: Mapping[str, Any],
    config: Mapping[str, Any],
    normalization_stats: Mapping[str, Any] | str | Path | None = None,
) -> list[dict[str, float | int]]:
    """Train the historical raw-scale SAE baseline to ``sae.pt``.

    This is retained only for legacy comparison. The canonical structured
    baseline is :func:`train_calibrated_sae_denoiser`.
    """
    resolved_stats = _configured_normalization_stats(config, normalization_stats)
    sampler = make_structured_sae_corruption_sampler(
        all_directions,
        direction_split,
        config,
        normalization_stats=resolved_stats,
    )
    checkpoint_paths = config.get("training", config).get("checkpoint_paths", {})
    checkpoint_path = checkpoint_paths.get("sae", "outputs/checkpoints/sae.pt")
    return train_denoiser(
        denoiser,
        activation_dataset,
        config,
        sampler,
        validation_dataset,
        normalization_stats=resolved_stats,
        checkpoint_path=checkpoint_path,
    )


def train_calibrated_sae_denoiser(
    denoiser: nn.Module,
    activation_dataset: ActivationDataset,
    validation_dataset: ActivationDataset,
    all_directions: Tensor,
    direction_split: Mapping[str, Any],
    config: Mapping[str, Any],
    normalization_stats: Mapping[str, Any] | str | Path,
    magnitude_min: float,
    magnitude_max: float,
) -> list[dict[str, float | int]]:
    """Train calibrated standardized-unit SAE baseline to ``sae_calibrated.pt``."""
    sampler = make_calibrated_sae_corruption_sampler(
        all_directions,
        direction_split,
        normalization_stats,
        magnitude_min,
        magnitude_max,
        gaussian_probability=float(
            config.get("noise", {}).get("calibrated_gaussian_probability", 0.15)
        ),
        gaussian_sigma_min=float(
            config.get("noise", {}).get("gaussian_sigma_min", 0.0)
        ),
        gaussian_sigma_max=float(
            config.get("noise", {}).get("gaussian_sigma_max", 2.0)
        ),
    )
    training_config = config.get("training", config)
    checkpoint_path = training_config.get(
        "checkpoint_paths", {}
    ).get("sae_calibrated", "outputs/checkpoints/sae_calibrated.pt")
    return train_denoiser(
        denoiser,
        activation_dataset,
        config,
        sampler,
        validation_dataset,
        normalization_stats=normalization_stats,
        checkpoint_path=checkpoint_path,
    )


def train_fluency_sensitive_denoiser(
    denoiser: nn.Module,
    activation_dataset: ActivationDataset,
    validation_dataset: ActivationDataset,
    all_directions: Tensor,
    direction_split: Mapping[str, Any],
    config: Mapping[str, Any],
    scores_path: str | Path | None = None,
    normalization_stats: Mapping[str, Any] | str | Path | None = None,
) -> list[dict[str, float | int]]:
    """Train the score-weighted main method to ``fluency.pt``."""
    resolved_stats = _configured_normalization_stats(config, normalization_stats)
    sampler = make_fluency_sensitive_corruption_sampler(
        all_directions,
        direction_split,
        config,
        scores_path=scores_path,
        normalization_stats=resolved_stats,
    )
    checkpoint_paths = config.get("training", config).get("checkpoint_paths", {})
    checkpoint_path = checkpoint_paths.get(
        "fluency", "outputs/checkpoints/fluency.pt"
    )
    return train_denoiser(
        denoiser,
        activation_dataset,
        config,
        sampler,
        validation_dataset,
        normalization_stats=resolved_stats,
        checkpoint_path=checkpoint_path,
    )
