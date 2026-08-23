"""Leakage-safe validation and freezing for conditioned V2."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.denoiser_v2 import ConditionedSteeringDenoiser
from src.directions import (
    direction_split_hash,
    validate_direction_ids_for_usage,
    validate_direction_split,
)
from src.metrics import activation_norm_ratio, next_token_nll, token_level_kl
from src.model import get_logits_with_intervention
from src.steering import apply_steering, literal_raw_steering, relative_raw_steering
from src.train_v2 import V2_PIPELINE_VERSION, checkpoint_sha256
from src.v2_objectives import correction_geometry


def _feature_mean(sae: nn.Module, activations: Tensor, feature_id: int, mask: Tensor) -> Tensor:
    features = sae.encode(activations)[..., feature_id]
    assert features.shape == mask.shape
    return features[mask.bool()].mean()


@torch.inference_mode()
def evaluate_v2_methods(
    model: nn.Module,
    sae: nn.Module,
    token_batches: Sequence[Tensor | tuple[Tensor, Tensor | None]],
    all_directions: Tensor,
    split: Mapping[str, Any],
    direction_ids: Sequence[int],
    relative_strengths: Sequence[float],
    literal_calibration: Mapping[str, Any] | None,
    hook_name: str,
    conditioned_denoiser: ConditionedSteeringDenoiser,
    conditioned_normalization: Mapping[str, Any],
    v1_denoisers: Mapping[str, nn.Module] | None = None,
    v1_normalizations: Mapping[str, Mapping[str, Any]] | None = None,
    methods: Sequence[str] = (
        "literal_raw", "relative_raw", "gaussian_denoiser", "sae_calibrated",
        "fluency_denoiser", "conditioned_kl_denoiser",
    ),
    evaluation_split: str = "val",
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Evaluate V2 and unchanged V1 baselines on matched held-out sequences."""
    assert evaluation_split in {"val", "test"}
    usage = "preliminary_evaluation" if evaluation_split == "val" else "final_evaluation"
    ids = [int(value) for value in direction_ids]
    validate_direction_ids_for_usage(ids, split, usage)
    assert set(ids).isdisjoint(set(split["train"]))
    validate_direction_split(
        list(split["train"]), list(split["val"]), list(split["test"]),
        int(split["num_features"]),
    )
    v1_denoisers = {} if v1_denoisers is None else v1_denoisers
    v1_normalizations = {} if v1_normalizations is None else v1_normalizations
    strengths = [float(value) for value in relative_strengths]
    alpha_by_strength = {}
    if "literal_raw" in methods:
        assert literal_calibration is not None
        calibration_strengths = [float(value) for value in literal_calibration["relative_calibration_strengths"]]
        alphas = [float(value) for value in literal_calibration["literal_alphas"]]
        alpha_by_strength = dict(zip(calibration_strengths, alphas))
        assert set(strengths).issubset(alpha_by_strength)
    rows: list[dict[str, Any]] = []
    selected_batches = token_batches if max_batches is None else token_batches[:max_batches]
    total = len(selected_batches) * len(ids) * len(strengths) * len(methods)
    progress = tqdm(total=total, desc=f"V2 {evaluation_split} evaluation")
    for batch_index, batch in enumerate(selected_batches):
        if isinstance(batch, Tensor):
            tokens, attention_mask = batch, torch.ones_like(batch, dtype=torch.bool)
        else:
            tokens, supplied_mask = batch
            attention_mask = (
                torch.ones_like(tokens, dtype=torch.bool)
                if supplied_mask is None else supplied_mask.bool()
            )
        clean_logits, cache = model.run_with_cache(tokens, names_filter=hook_name)
        clean_h = cache[hook_name]
        clean_nll = next_token_nll(clean_logits, tokens, attention_mask)
        assert isinstance(clean_nll, Tensor)
        for direction_id in ids:
            direction = all_directions[direction_id]
            for strength in strengths:
                relative_x = relative_raw_steering(clean_h, direction, strength)
                relative_raw_feature = _feature_mean(
                    sae, relative_x, direction_id, attention_mask
                )
                for method in methods:
                    literal_alpha = alpha_by_strength.get(strength)
                    if method == "literal_raw":
                        assert literal_alpha is not None
                        steered_x = literal_raw_steering(clean_h, direction, literal_alpha)
                        modified_h = steered_x
                        parameter_name, parameter_value = "literal_alpha", literal_alpha
                    elif method == "relative_raw":
                        steered_x = relative_x
                        modified_h = steered_x
                        parameter_name, parameter_value = "relative_strength", strength
                    elif method == "conditioned_kl_denoiser":
                        steered_x = relative_x
                        modified_h = apply_steering(
                            clean_h, direction, strength, method="conditioned_kl_denoise",
                            denoiser=conditioned_denoiser,
                            normalization_stats=conditioned_normalization,
                        )
                        parameter_name, parameter_value = "relative_strength", strength
                    else:
                        key_by_method = {
                            "gaussian_denoiser": "gaussian",
                            "sae_calibrated": "sae_calibrated",
                            "fluency_denoiser": "fluency",
                            "projected_fluency_denoiser": "fluency",
                        }
                        assert method in key_by_method
                        key = key_by_method[method]
                        steering_method = (
                            "relative_projected_denoise"
                            if method == "projected_fluency_denoiser" else "relative_denoise"
                        )
                        steered_x = relative_x
                        modified_h = apply_steering(
                            clean_h, direction, strength, method=steering_method,
                            denoiser=v1_denoisers[key],
                            normalization_stats=v1_normalizations[key],
                        )
                        parameter_name, parameter_value = "relative_strength", strength

                    def replace(_: Tensor) -> Tensor:
                        return modified_h

                    modified_logits = get_logits_with_intervention(
                        model, tokens, hook_name, replace
                    )
                    modified_nll = next_token_nll(modified_logits, tokens, attention_mask)
                    assert isinstance(modified_nll, Tensor)
                    target_activation = _feature_mean(
                        sae, modified_h, direction_id, attention_mask
                    )
                    raw_reference = (
                        target_activation.detach() if method == "literal_raw"
                        else relative_raw_feature
                    )
                    retention_ratio = float(
                        (target_activation + 1e-6) / (raw_reference + 1e-6)
                    )
                    geometry = correction_geometry(modified_h, steered_x, direction)
                    valid = attention_mask.bool()
                    correction_norm = geometry["correction_norm"][valid].mean()
                    steered_norm = steered_x.norm(dim=-1)[valid].mean()
                    row = {
                        "pipeline_version": V2_PIPELINE_VERSION,
                        "evaluation_split": evaluation_split,
                        "batch_id": batch_index,
                        "method": method,
                        "direction_id": direction_id,
                        "relative_strength": strength,
                        "literal_alpha": literal_alpha if method == "literal_raw" else None,
                        "nominal_relative_calibration": (
                            literal_alpha / float(literal_calibration["median_activation_norm"])
                            if method == "literal_raw" else strength
                        ),
                        "parameter_name": parameter_name,
                        "parameter_value": parameter_value,
                        "kl": float(token_level_kl(clean_logits, modified_logits, attention_mask)),
                        "clean_nll": float(clean_nll),
                        "modified_nll": float(modified_nll),
                        "delta_nll": float(modified_nll - clean_nll),
                        "target_sae_activation": float(target_activation),
                        "raw_target_sae_activation": float(raw_reference),
                        "activation_norm_ratio": float(activation_norm_ratio(clean_h, modified_h)),
                        "activation_mse": float((modified_h - clean_h).float().square().mean()),
                        "correction_norm": float(correction_norm),
                        "relative_correction_norm": float(correction_norm / steered_norm.clamp_min(1e-8)),
                        "correction_parallel_norm": float(geometry["correction_parallel_norm"][valid].mean()),
                        "correction_orthogonal_norm": float(geometry["correction_orthogonal_norm"][valid].mean()),
                        "correction_cosine_with_v": float(geometry["correction_cosine_with_v"][valid].mean()),
                        "parallel_fraction": float(geometry["parallel_fraction"][valid].mean()),
                        "concept_retention_ratio": retention_ratio,
                        "concept_retention_violation": max(0.0, 0.8 - retention_ratio) ** 2,
                        "retention_below_80": retention_ratio < 0.8,
                    }
                    assert all(np.isfinite(value) for value in row.values() if isinstance(value, float))
                    rows.append(row)
                    progress.update(1)
    progress.close()
    return pd.DataFrame(rows)


def aggregate_v2_results(results: pd.DataFrame) -> pd.DataFrame:
    assert not results.empty
    aggregate = results.groupby(
        ["evaluation_split", "method", "relative_strength", "literal_alpha"],
        dropna=False, as_index=False,
    ).agg(
        mean_kl=("kl", "mean"),
        mean_delta_nll=("delta_nll", "mean"),
        mean_target_sae_activation=("target_sae_activation", "mean"),
        mean_concept_retention=("concept_retention_ratio", "mean"),
        mean_norm_ratio=("activation_norm_ratio", "mean"),
        mean_correction_norm=("correction_norm", "mean"),
        mean_parallel_fraction=("parallel_fraction", "mean"),
        fraction_violating_80=("retention_below_80", "mean"),
        count=("kl", "size"),
    )
    return aggregate


def validate_conditioned_checkpoint(
    model: nn.Module,
    sae: nn.Module,
    denoiser: ConditionedSteeringDenoiser,
    token_batches: Sequence[Tensor | tuple[Tensor, Tensor | None]],
    all_directions: Tensor,
    split: Mapping[str, Any],
    val_ids: Sequence[int],
    strengths: Sequence[float],
    normalization: Mapping[str, Any],
    config: Mapping[str, Any],
    hook_name: str,
) -> dict[str, float]:
    results = evaluate_v2_methods(
        model, sae, token_batches, all_directions, split, val_ids, strengths,
        literal_calibration=None, hook_name=hook_name,
        conditioned_denoiser=denoiser,
        conditioned_normalization=normalization,
        methods=("relative_raw", "conditioned_kl_denoiser"),
        evaluation_split="val",
        max_batches=int(config["max_validation_batches"]),
    )
    positive = results[
        (results.method == "conditioned_kl_denoiser")
        & (results.relative_strength > 0)
    ]
    return {
        "mean_delta_nll": float(positive.delta_nll.mean()),
        "mean_kl": float(positive.kl.mean()),
        "mean_concept_retention": float(positive.concept_retention_ratio.mean()),
        "fraction_violating_80": float(positive.retention_below_80.mean()),
    }


def save_v2_validation(
    results: pd.DataFrame,
    results_path: str | Path,
    aggregate_path: str | Path,
    figure_dir: str | Path = "outputs/final_figures",
) -> pd.DataFrame:
    results_file = Path(results_path); results_file.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_file, index=False)
    aggregate = aggregate_v2_results(results)
    aggregate_file = Path(aggregate_path); aggregate_file.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(aggregate_file, index=False)
    comparison = compare_conditioned_to_calibrated(aggregate)
    if not comparison.empty:
        comparison.to_csv(
            aggregate_file.parent / "comparison_vs_sae_calibrated.csv", index=False
        )
    _plot_v2(aggregate, figure_dir)
    return aggregate


def compare_conditioned_to_calibrated(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Matched-strength answer to the primary V2-vs-calibrated question."""
    conditioned = aggregate[
        (aggregate.method == "conditioned_kl_denoiser")
        & (aggregate.relative_strength > 0)
    ].copy()
    calibrated = aggregate[
        (aggregate.method == "sae_calibrated")
        & (aggregate.relative_strength > 0)
    ].copy()
    if conditioned.empty or calibrated.empty:
        return pd.DataFrame()
    columns = ["evaluation_split", "relative_strength"]
    merged = conditioned.merge(
        calibrated, on=columns, suffixes=("_conditioned", "_sae_calibrated"),
        validate="one_to_one",
    )
    merged["delta_nll_improvement_vs_sae_calibrated"] = (
        merged["mean_delta_nll_sae_calibrated"]
        - merged["mean_delta_nll_conditioned"]
    )
    merged["conditioned_retains_at_least_80"] = (
        merged["mean_concept_retention_conditioned"] >= 0.8
    )
    merged["conditioned_has_lower_nll"] = (
        merged["mean_delta_nll_conditioned"] < merged["mean_delta_nll_sae_calibrated"]
    )
    merged["conditioned_pareto_dominates_calibrated"] = (
        merged["conditioned_has_lower_nll"]
        & (merged["mean_target_sae_activation_conditioned"]
           >= merged["mean_target_sae_activation_sae_calibrated"])
    )
    return merged


def _plot_v2(aggregate: pd.DataFrame, figure_dir: str | Path) -> None:
    import matplotlib.pyplot as plt
    root = Path(figure_dir); root.mkdir(parents=True, exist_ok=True)
    specifications = [
        ("v2_pareto.png", "mean_delta_nll", "mean_target_sae_activation",
         "Delta NLL (lower is better)", "Target SAE activation (higher is better)"),
        ("v2_concept_retention.png", "relative_strength", "mean_concept_retention",
         "Relative strength", "Concept retention ratio (target >= 0.8)"),
        ("v2_kl_vs_strength.png", "relative_strength", "mean_kl",
         "Relative strength", "KL(clean || modified), lower is better"),
        ("v2_correction_parallel_fraction.png", "relative_strength", "mean_parallel_fraction",
         "Relative strength", "Parallel fraction of denoiser correction"),
    ]
    for filename, x_column, y_column, xlabel, ylabel in specifications:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, group in aggregate.groupby("method"):
            group = group.sort_values("relative_strength")
            ax.plot(group[x_column], group[y_column], marker="o", label=method)
            if filename == "v2_pareto.png":
                for x, y, strength in zip(group[x_column], group[y_column], group.relative_strength):
                    ax.annotate(f"{strength:g}", (x, y), fontsize=7)
        if filename == "v2_concept_retention.png":
            ax.axhline(0.8, linestyle="--", color="black", linewidth=1)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(root / filename, dpi=160)
        plt.close(fig)


def freeze_v2_test_config(
    path: str | Path,
    split: Mapping[str, Any],
    model_name: str,
    hook_name: str,
    relative_strengths: Sequence[float],
    literal_calibration: Mapping[str, Any],
    checkpoint_path: str | Path,
    config: Mapping[str, Any],
    evaluation_methods: Sequence[str],
    prompt_count: int,
    generation_settings: Mapping[str, Any],
) -> Path:
    """Freeze V2 after validation; never overwrite an existing frozen file."""
    resolved = Path(path)
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite frozen V2 config: {resolved}")
    checkpoint = Path(checkpoint_path)
    assert checkpoint.exists(), "Cannot freeze V2 without the selected validation checkpoint."
    payload = {
        "pipeline_version": V2_PIPELINE_VERSION,
        "status": "frozen_after_validation",
        "direction_split_hash": direction_split_hash(split),
        "model": model_name,
        "hook": hook_name,
        "relative_strengths": list(map(float, relative_strengths)),
        "literal_alpha_calibration": dict(literal_calibration),
        "conditioned_architecture": {
            "d_model": 768,
            "hidden_dim": int(config["hidden_dim"]),
            "conditioning_dim": int(config["conditioning_dim"]),
            "film_blocks": 2,
        },
        "loss_formula": "KL + lambda_retain*soft_hinge + lambda_identity*identity + lambda_correction*relative_correction + lambda_reconstruction*MSE",
        "lambdas": {key: float(config[key]) for key in (
            "lambda_fluency", "lambda_retain", "lambda_identity",
            "lambda_correction", "lambda_reconstruction",
        )},
        "retention_target": float(config["retention_target"]),
        "training_steps": int(config["max_steps"]),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "evaluation_methods": list(evaluation_methods),
        "prompt_count": int(prompt_count),
        "generation_settings": dict(generation_settings),
    }
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return resolved


def validate_v2_test_gate(
    run_test_v2: bool,
    frozen_path: str | Path,
    split: Mapping[str, Any],
) -> dict[str, Any]:
    """TEST cannot run accidentally or before validation settings are frozen."""
    if not run_test_v2:
        raise PermissionError("RUN_TEST_V2 is False; refusing V2 TEST access.")
    path = Path(frozen_path)
    if not path.exists():
        raise FileNotFoundError("frozen_test_config_v2.json does not exist.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["pipeline_version"] == V2_PIPELINE_VERSION
    assert payload["status"] == "frozen_after_validation"
    assert payload["direction_split_hash"] == direction_split_hash(split)
    return payload
