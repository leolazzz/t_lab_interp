"""Train-only loss calibration and causal ablations for final V3."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.denoiser_v3 import GatedConditionedDenoiser, apply_gated_conditioned_denoiser
from src.directions import direction_split_hash, validate_direction_ids_for_usage
from src.metrics import next_token_nll
from src.train_v2 import (
    TokenBatch,
    _relative_steer_batch,
    _run_modified_logits,
    _sample_training_conditions,
    _target_feature_means,
    _unpack_batch,
    freeze_teacher_models,
    gradient_diagnostics,
)
from src.utils import seed_everything
from src.v2_objectives import correction_geometry, downstream_kl, relative_correction_loss


V3_PIPELINE_VERSION = "final_v3"
V3_CHECKPOINT_VERSION = 1

ABLATION_SPECS: dict[str, dict[str, Any]] = {
    "conditioned_reconstruction": {
        "uses_conditioning": True, "uses_downstream_kl": False,
        "uses_retention_constraint": False, "uses_reconstruction": True,
        "uses_structural_identity": True,
        "weights": {"kl": 0.0, "retention": 0.0, "reconstruction": 1.0, "correction": 0.01},
    },
    "conditioned_kl": {
        "uses_conditioning": True, "uses_downstream_kl": True,
        "uses_retention_constraint": False, "uses_reconstruction": False,
        "uses_structural_identity": True,
        "weights": {"kl": 1.0, "retention": 0.0, "reconstruction": 0.0, "correction": 0.01},
    },
    "conditioned_kl_retention": {
        "uses_conditioning": True, "uses_downstream_kl": True,
        "uses_retention_constraint": True, "uses_reconstruction": False,
        "uses_structural_identity": True,
        "weights": {"kl": 1.0, "retention": 1.0, "reconstruction": 0.0, "correction": 0.01},
    },
    "conditioned_full": {
        "uses_conditioning": True, "uses_downstream_kl": True,
        "uses_retention_constraint": True, "uses_reconstruction": True,
        "uses_structural_identity": True,
        "weights": {"kl": 1.0, "retention": 1.0, "reconstruction": 0.05, "correction": 0.01},
    },
}


def resolved_ablation_spec(ablation: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve pre-registered ablation metadata with config-owned weights."""
    if ablation not in ABLATION_SPECS:
        raise ValueError(f"Unknown V3 ablation: {ablation}")
    spec = deepcopy(ABLATION_SPECS[ablation])
    configured = config.get("ablation_weights", {}).get(ablation)
    if configured is not None:
        assert set(configured) == {"kl", "retention", "reconstruction", "correction"}
        spec["weights"] = {key: float(value) for key, value in configured.items()}
    return spec


def _prediction_mask(attention_mask: Tensor) -> Tensor:
    mask = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    assert mask.any()
    return mask


def _retention_terms(
    raw_feature: Tensor,
    denoised_feature: Tensor,
    strengths: Tensor,
    target: float,
    threshold: float,
    eps: float,
) -> dict[str, Tensor]:
    """Retention hinge excluding zero strength and unstable denominators."""
    assert raw_feature.shape == denoised_feature.shape == strengths.shape
    valid = (strengths > 0) & (raw_feature > threshold)
    ratio = torch.full_like(raw_feature, torch.nan)
    ratio[valid] = denoised_feature[valid] / raw_feature[valid].clamp_min(eps)
    zero = denoised_feature.sum() * 0.0
    violation = F.relu(target - ratio[valid]).square() if valid.any() else None
    return {
        "loss": violation.mean() if violation is not None else zero,
        "ratio": ratio,
        "valid_mask": valid,
        "mean_ratio": ratio[valid].mean() if valid.any() else zero,
        "median_ratio": ratio[valid].median() if valid.any() else zero,
        "fraction_violating": (ratio[valid] < target).float().mean() if valid.any() else zero,
    }


def compute_v3_batch_losses(
    model: nn.Module,
    sae: nn.Module,
    denoiser: GatedConditionedDenoiser,
    tokens: Tensor,
    attention_mask: Tensor,
    directions: Tensor,
    direction_ids: Tensor,
    strengths: Tensor,
    normalization: Mapping[str, Any],
    loss_scales: Mapping[str, float],
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    hook_name: str,
) -> dict[str, Tensor]:
    """Compute raw, normalized, and weighted V3 objective components."""
    assert all(float(loss_scales[key]) > 0 for key in ("reconstruction", "correction"))
    with torch.no_grad():
        clean_logits, cache = model.run_with_cache(tokens, names_filter=hook_name)
        clean_h = cache[hook_name].detach(); clean_logits = clean_logits.detach(); del cache
    raw_h = _relative_steer_batch(clean_h, directions, strengths)
    denoised_h = apply_gated_conditioned_denoiser(
        denoiser, raw_h, directions, strengths, normalization
    )
    modified_logits = _run_modified_logits(model, tokens, hook_name, denoised_h)
    kl = downstream_kl(clean_logits[:, :-1], modified_logits[:, :-1], _prediction_mask(attention_mask))
    mask3 = attention_mask.bool().unsqueeze(-1).expand_as(clean_h)
    reconstruction = F.mse_loss(denoised_h[mask3], clean_h[mask3])
    correction = relative_correction_loss(denoised_h, raw_h, attention_mask.bool())
    raw_feature = _target_feature_means(sae, raw_h.detach(), direction_ids, attention_mask.bool())
    denoised_feature = _target_feature_means(sae, denoised_h, direction_ids, attention_mask.bool())
    retention = _retention_terms(
        raw_feature, denoised_feature, strengths,
        float(config["retention_target"]), float(config["retention_mask_threshold"]),
        float(config["retention_eps"]),
    )
    weights = spec["weights"]
    normalized_reconstruction = reconstruction / float(loss_scales["reconstruction"])
    normalized_correction = correction / float(loss_scales["correction"])
    weighted = {
        "weighted_kl": float(weights["kl"]) * kl,
        "weighted_retention": float(weights["retention"]) * retention["loss"],
        "weighted_reconstruction": float(weights["reconstruction"]) * normalized_reconstruction,
        "weighted_correction": float(weights["correction"]) * normalized_correction,
    }
    total = sum(weighted.values())
    geometry = correction_geometry(denoised_h, raw_h, directions)
    clean_nll = next_token_nll(clean_logits, tokens, attention_mask)
    modified_nll = next_token_nll(modified_logits, tokens, attention_mask)
    assert isinstance(clean_nll, Tensor) and isinstance(modified_nll, Tensor)
    outputs = {
        "loss": total, "fluency_kl": kl, "retention_loss": retention["loss"],
        "retention_ratio": retention["mean_ratio"],
        "retention_median": retention["median_ratio"],
        "fraction_below_retention_threshold": retention["fraction_violating"],
        "reconstruction_loss": reconstruction, "correction_loss": correction,
        "normalized_reconstruction": normalized_reconstruction,
        "normalized_correction": normalized_correction,
        "delta_nll": modified_nll - clean_nll,
        "correction_norm": geometry["correction_norm"][attention_mask.bool()].mean(),
        **weighted,
    }
    assert all(torch.isfinite(value).all() for value in outputs.values())
    return outputs


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    assert array.size and np.isfinite(array).all()
    return {
        "median": float(np.median(array)), "mean": float(array.mean()),
        "std": float(array.std()), "p90": float(np.quantile(array, 0.9)),
    }


@torch.inference_mode()
def calibrate_v3_loss_scales(
    model: nn.Module,
    sae: nn.Module,
    token_batches: Sequence[TokenBatch],
    all_directions: Tensor,
    split: Mapping[str, Any],
    strengths: Sequence[float],
    hook_name: str,
    seed: int,
    num_batches: int,
    output_path: str | Path,
    retention_target: float = 0.8,
    retention_threshold: float = 1e-3,
    retention_eps: float = 1e-6,
) -> dict[str, Any]:
    """Freeze natural auxiliary scales using TRAIN directions and contexts only."""
    train_ids = list(split["train"])
    validate_direction_ids_for_usage(train_ids, split, "training", require_complete_split=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows: dict[str, list[float]] = {
        "kl": [], "reconstruction": [], "correction": [], "retention_when_active": []
    }
    for batch in token_batches[:num_batches]:
        tokens, mask = _unpack_batch(batch)
        directions, direction_ids, sampled_strengths, _ = _sample_training_conditions(
            tokens.shape[0], train_ids, all_directions, strengths, generator
        )
        clean_logits, cache = model.run_with_cache(tokens, names_filter=hook_name)
        clean_h = cache[hook_name]; del cache
        raw_h = _relative_steer_batch(clean_h, directions, sampled_strengths)
        raw_logits = _run_modified_logits(model, tokens, hook_name, raw_h)
        rows["kl"].append(float(downstream_kl(
            clean_logits[:, :-1], raw_logits[:, :-1], _prediction_mask(mask)
        )))
        valid_coordinates = mask.bool().unsqueeze(-1).expand_as(clean_h)
        rows["reconstruction"].append(float(F.mse_loss(
            raw_h[valid_coordinates], clean_h[valid_coordinates]
        )))
        rows["correction"].append(float(relative_correction_loss(clean_h, raw_h, mask.bool())))
        raw_feature = _target_feature_means(sae, raw_h, direction_ids, mask.bool())
        clean_feature = _target_feature_means(sae, clean_h, direction_ids, mask.bool())
        retention = _retention_terms(
            raw_feature, clean_feature, sampled_strengths,
            target=retention_target, threshold=retention_threshold, eps=retention_eps,
        )
        rows["retention_when_active"].append(float(retention["loss"]))
    diagnostics = {key: _summary(value) for key, value in rows.items()}
    frozen_scales = {
        "reconstruction": max(diagnostics["reconstruction"]["median"], 1e-8),
        "correction": max(diagnostics["correction"]["median"], 1e-8),
        "retention": 1.0,
    }
    payload = {
        "pipeline_version": V3_PIPELINE_VERSION,
        "source_split": "train", "num_batches": min(num_batches, len(token_batches)),
        "statistics": diagnostics, "frozen_reference_scales": frozen_scales,
        "structural_identity_reference": 0.0,
        "validation_or_holdout_used": False,
    }
    for loss_name, values in diagnostics.items():
        for statistic, value in values.items():
            payload[f"{statistic}_{loss_name}"] = value
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _checkpoint_payload(
    denoiser: GatedConditionedDenoiser,
    optimizer: torch.optim.Optimizer,
    step: int,
    ablation: str,
    spec: Mapping[str, Any],
    config: Mapping[str, Any],
    normalization: Mapping[str, Any],
    loss_scales: Mapping[str, float],
    split: Mapping[str, Any],
    hook_name: str,
) -> dict[str, Any]:
    return {
        "checkpoint_version": V3_CHECKPOINT_VERSION,
        "pipeline_version": V3_PIPELINE_VERSION,
        "ablation": ablation, "ablation_metadata": deepcopy(dict(spec)),
        "model_state_dict": denoiser.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "step": int(step),
        "architecture": {
            "d_model": denoiser.d_model, "hidden_dim": denoiser.hidden_dim,
            "conditioning_dim": denoiser.conditioning_dim,
            "gate_scale": denoiser.gate_scale, "type": "GatedConditionedDenoiser",
        },
        "normalization": {k: v.detach().cpu() if isinstance(v, Tensor) else v for k, v in normalization.items()},
        "loss_scales": dict(loss_scales), "config": deepcopy(dict(config)),
        "direction_split_hash": direction_split_hash(split), "hook_name": hook_name,
        "torch_version": str(torch.__version__),
    }


def train_v3_ablation(
    model: nn.Module,
    sae: nn.Module,
    denoiser: GatedConditionedDenoiser,
    token_batches: Sequence[TokenBatch],
    all_directions: Tensor,
    split: Mapping[str, Any],
    normalization: Mapping[str, Any],
    loss_scales: Mapping[str, float],
    config: Mapping[str, Any],
    hook_name: str,
    ablation: str,
    checkpoint_path: str | Path,
) -> list[dict[str, float]]:
    """Train one pre-registered ablation using TRAIN directions only."""
    spec = resolved_ablation_spec(ablation, config)
    seed = int(config["seed"]); seed_everything(seed); freeze_teacher_models(model, sae)
    train_ids = list(split["train"])
    validate_direction_ids_for_usage(train_ids, split, "training", require_complete_split=True)
    optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, float]] = []
    max_steps = int(config["max_steps"])
    for step in range(1, max_steps + 1):
        tokens, mask = _unpack_batch(token_batches[(step - 1) % len(token_batches)])
        directions, ids, strengths, _ = _sample_training_conditions(
            tokens.shape[0], train_ids, all_directions, config["training_strengths"], generator
        )
        optimizer.zero_grad(set_to_none=True)
        losses = compute_v3_batch_losses(
            model, sae, denoiser, tokens, mask, directions, ids, strengths,
            normalization, loss_scales, spec, config, hook_name,
        )
        losses["loss"].backward()
        gradients = gradient_diagnostics(denoiser)
        assert gradients["denoiser_gradient_norm"] > 0
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), float(config["gradient_clip_norm"]))
        optimizer.step()
        row = {"step": float(step), **{k: float(v.detach()) for k, v in losses.items()}, **gradients}
        history.append(row)
        if step == 1 or step % int(config["log_every"]) == 0 or step == max_steps:
            concise = {key: row[key] for key in (
                "step", "loss", "weighted_kl", "weighted_retention",
                "weighted_reconstruction", "weighted_correction", "retention_ratio",
            )}
            print(ablation, json.dumps(concise))
    path = Path(checkpoint_path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(
        denoiser, optimizer, max_steps, ablation, spec, config, normalization,
        loss_scales, split, hook_name,
    ), path)
    return history


def load_v3_checkpoint(
    path: str | Path,
    device: str | torch.device,
    expected_split_hash: str | None = None,
) -> tuple[GatedConditionedDenoiser, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    assert checkpoint["checkpoint_version"] == V3_CHECKPOINT_VERSION
    assert checkpoint["pipeline_version"] == V3_PIPELINE_VERSION
    if expected_split_hash is not None:
        assert checkpoint["direction_split_hash"] == expected_split_hash
    architecture = checkpoint["architecture"]
    model = GatedConditionedDenoiser(
        architecture["d_model"], architecture["hidden_dim"],
        architecture["conditioning_dim"], architecture["gate_scale"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
