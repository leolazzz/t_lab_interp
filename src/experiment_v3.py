"""Leakage-safe evaluation, statistics, freezing, and reporting for final V3."""

from __future__ import annotations

import json
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from src.denoiser_v3 import GatedConditionedDenoiser
from src.directions import direction_split_hash, validate_direction_split
from src.metrics import token_level_kl
from src.model import get_logits_with_intervention
from src.steering import apply_steering, literal_raw_steering, relative_raw_steering
from src.train_v3 import (
    ABLATION_SPECS, V3_PIPELINE_VERSION, file_sha256, resolved_ablation_spec,
)
from src.v2_objectives import correction_geometry


def create_v3_direction_split(
    canonical_split: Mapping[str, Any],
    num_features: int,
    num_validation: int,
    num_holdout: int,
    seed: int,
    excluded_external_ids: Sequence[int] = (),
    path: str | Path = "outputs/final_v3/configs/direction_split_v3.json",
) -> dict[str, Any]:
    """Keep canonical TRAIN and draw new val/holdout IDs from never-used features."""
    resolved = Path(path)
    if resolved.exists():
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        validate_direction_split(payload["train"], payload["val"], payload["test"], num_features)
        assert payload["train"] == list(map(int, canonical_split["train"]))
        assert payload["seed"] == seed
        assert payload["num_val"] == num_validation and payload["num_test"] == num_holdout
        assert payload["canonical_observed_split_hash"] == direction_split_hash(canonical_split)
        assert set(payload["val"]) | set(payload["test"]) <= (
            set(range(num_features))
            - set(canonical_split["train"])
            - set(canonical_split["val"])
            - set(canonical_split["test"])
            - set(map(int, excluded_external_ids))
        )
        return payload
    observed = set(canonical_split["train"]) | set(canonical_split["val"]) | set(canonical_split["test"])
    observed |= {int(value) for value in excluded_external_ids}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    candidates = [value for value in torch.randperm(num_features, generator=generator).tolist() if value not in observed]
    assert len(candidates) >= num_validation + num_holdout
    val = candidates[:num_validation]
    holdout = candidates[num_validation:num_validation + num_holdout]
    train = list(map(int, canonical_split["train"]))
    validate_direction_split(train, val, holdout, num_features)
    payload = {
        "num_features": num_features, "num_train": len(train),
        "num_val": len(val), "num_test": len(holdout), "seed": seed,
        "train": train, "val": val, "test": holdout,
        "holdout_label": "new_unseen_v3_holdout",
        "canonical_observed_split_hash": direction_split_hash(canonical_split),
        "canonical_val_and_test_excluded": True,
        "excluded_external_feature_ids": sorted(map(int, excluded_external_ids)),
    }
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def retention_fields(
    raw_activation: float,
    denoised_activation: float,
    relative_strength: float,
    threshold: float,
    target: float = 0.8,
) -> dict[str, Any]:
    """Return stable retention reporting fields; zero strength is always invalid."""
    valid = relative_strength > 0 and raw_activation > threshold
    ratio = denoised_activation / max(raw_activation, 1e-12) if valid else float("nan")
    return {
        "raw_target_sae_activation": raw_activation,
        "denoised_target_sae_activation": denoised_activation,
        "concept_retention": ratio,
        "valid_retention_ratio": bool(valid),
        "violates_80_percent": bool(ratio < target) if valid else None,
    }


def _per_example_mean(values: Tensor, mask: Tensor) -> Tensor:
    assert values.shape == mask.shape
    weights = mask.to(values.dtype)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1)


def _per_example_feature(sae: nn.Module, h: Tensor, feature_id: int, mask: Tensor) -> Tensor:
    values = sae.encode(h)[..., feature_id]
    return _per_example_mean(values, mask.bool())


def _tokenize_fixed_texts(
    model: nn.Module, texts: Sequence[str], max_length: int,
) -> tuple[Tensor, Tensor]:
    tokenizer = model.tokenizer
    assert tokenizer is not None and texts and max_length >= 2
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_side = tokenizer.padding_side; tokenizer.padding_side = "right"
    try:
        encoded = tokenizer(
            list(texts), padding=True, truncation=True, max_length=max_length - 1,
            add_special_tokens=False, return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = old_side
    bos = torch.full(
        (len(texts), 1), int(tokenizer.bos_token_id), dtype=torch.long,
    )
    tokens = torch.cat((bos, encoded["input_ids"]), dim=1)
    mask = torch.cat((torch.ones_like(bos), encoded["attention_mask"]), dim=1).bool()
    device = next(model.parameters()).device
    return tokens.to(device), mask.to(device)


@torch.inference_mode()
def build_sentiment_contrast_direction(
    model: nn.Module,
    positive_examples: Sequence[str],
    negative_examples: Sequence[str],
    hook_name: str,
    max_length: int = 96,
) -> tuple[Tensor, dict[str, Any]]:
    """Build one fixed non-SAE sentiment direction after model selection.

    The direction is the unit-normalized difference between mean final-token
    residual activations for paired positive and negative sentences. It is
    never passed to training, loss calibration, or validation selection.
    """
    assert positive_examples and len(positive_examples) == len(negative_examples)

    def final_activations(texts: Sequence[str]) -> Tensor:
        tokens, mask = _tokenize_fixed_texts(model, texts, max_length)
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        h = cache[hook_name]; del cache
        indices = mask.sum(dim=1).long() - 1
        return h[torch.arange(h.shape[0], device=h.device), indices]

    positive = final_activations(positive_examples)
    negative = final_activations(negative_examples)
    direction = positive.mean(0) - negative.mean(0)
    norm = direction.norm(); assert torch.isfinite(norm) and norm > 1e-8
    direction = direction / norm
    metadata = {
        "family": "sentiment_contrastive_residual",
        "construction": "unit(mean_positive_final_residual - mean_negative_final_residual)",
        "hook_name": hook_name, "num_pairs": len(positive_examples),
        "used_for_training": False, "used_for_model_selection": False,
        "examples_sha256": hashlib.sha256(
            json.dumps([list(positive_examples), list(negative_examples)], ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    return direction, metadata


@torch.inference_mode()
def evaluate_cross_concept_direction(
    model: nn.Module,
    prompts: Sequence[str],
    direction: Tensor,
    direction_id: str,
    strengths: Sequence[float],
    hook_name: str,
    v3_models: Mapping[str, GatedConditionedDenoiser],
    v3_normalizations: Mapping[str, Mapping[str, Any]],
    v1_models: Mapping[str, nn.Module],
    v1_normalizations: Mapping[str, Mapping[str, Any]],
    methods: Sequence[str],
    retention_threshold: float,
    max_length: int = 96,
    positive_logit_tokens: Sequence[str] = (" good", " great", " happy", " excellent"),
    negative_logit_tokens: Sequence[str] = (" bad", " terrible", " sad", " awful"),
) -> pd.DataFrame:
    """Post-freeze transfer test on a non-SAE contrastive sentiment direction."""
    tokens, mask = _tokenize_fixed_texts(model, prompts, max_length)
    clean_logits, cache = model.run_with_cache(tokens, names_filter=hook_name)
    clean_h = cache[hook_name]; del cache
    direction = direction.to(device=clean_h.device, dtype=clean_h.dtype)
    assert direction.shape == (clean_h.shape[-1],)
    target_mask = mask[:, 1:] & mask[:, :-1]
    clean_losses = torch.nn.functional.cross_entropy(
        clean_logits[:, :-1].reshape(-1, clean_logits.shape[-1]),
        tokens[:, 1:].reshape(-1), reduction="none",
    ).reshape(tokens.shape[0], -1)
    clean_nll = _per_example_mean(clean_losses, target_mask)
    tokenizer = model.tokenizer; assert tokenizer is not None
    def single_token_ids(words: Sequence[str]) -> Tensor:
        ids = [tokenizer.encode(word, add_special_tokens=False) for word in words]
        assert all(len(value) == 1 for value in ids), f"Sentiment proxy words must be single GPT-2 tokens: {ids}"
        return torch.tensor([value[0] for value in ids], device=tokens.device, dtype=torch.long)
    positive_ids = single_token_ids(positive_logit_tokens)
    negative_ids = single_token_ids(negative_logit_tokens)
    last_positions = mask.sum(1).long() - 1
    batch_indices = torch.arange(tokens.shape[0], device=tokens.device)

    def sentiment_logit_score(logits: Tensor) -> Tensor:
        final_logits = logits[batch_indices, last_positions]
        return (
            torch.logsumexp(final_logits[:, positive_ids], dim=-1)
            - torch.logsumexp(final_logits[:, negative_ids], dim=-1)
        )

    clean_sentiment = sentiment_logit_score(clean_logits)
    rows: list[dict[str, Any]] = []
    for strength in map(float, strengths):
        steered = relative_raw_steering(clean_h, direction, strength)
        raw_signal = _per_example_mean(((steered - clean_h) * direction).sum(-1), mask)
        for method in methods:
            if method == "relative_raw":
                modified = steered
            elif method == "hard_projected_conditioned_kl_retention":
                modified = apply_steering(
                    clean_h, direction, strength,
                    method="hard_projected_gated_conditioned_denoise",
                    denoiser=v3_models["conditioned_kl_retention"],
                    normalization_stats=v3_normalizations["conditioned_kl_retention"],
                )
            elif method in v3_models:
                modified = apply_steering(
                    clean_h, direction, strength, method="gated_conditioned_denoise",
                    denoiser=v3_models[method], normalization_stats=v3_normalizations[method],
                )
            else:
                key = {"sae_calibrated": "sae_calibrated", "gaussian_denoiser": "gaussian",
                       "fluency_denoiser": "fluency"}[method]
                modified = apply_steering(
                    clean_h, direction, strength, method="relative_denoise",
                    denoiser=v1_models[key], normalization_stats=v1_normalizations[key],
                )
            modified_logits = get_logits_with_intervention(
                model, tokens, hook_name, lambda _: modified
            )
            kl_tokens = token_level_kl(
                clean_logits[:, :-1], modified_logits[:, :-1], target_mask, reduction="none"
            )
            kl = _per_example_mean(kl_tokens, target_mask)
            modified_losses = torch.nn.functional.cross_entropy(
                modified_logits[:, :-1].reshape(-1, modified_logits.shape[-1]),
                tokens[:, 1:].reshape(-1), reduction="none",
            ).reshape(tokens.shape[0], -1)
            modified_nll = _per_example_mean(modified_losses, target_mask)
            modified_sentiment = sentiment_logit_score(modified_logits)
            output_signal = _per_example_mean(((modified - clean_h) * direction).sum(-1), mask)
            geometry = correction_geometry(modified, steered, direction)
            for prompt_id in range(tokens.shape[0]):
                valid = mask[prompt_id]
                retention = retention_fields(
                    float(raw_signal[prompt_id]), float(output_signal[prompt_id]),
                    strength, retention_threshold,
                )
                rows.append({
                    "pipeline_version": V3_PIPELINE_VERSION,
                    "evaluation_split": "cross_concept_confirmation",
                    "concept_family": "sentiment_contrastive_residual",
                    "method": method, "direction_id": direction_id,
                    "prompt_id": prompt_id, "steering_mode": "relative",
                    "relative_strength": strength, "literal_alpha": float("nan"),
                    "kl": float(kl[prompt_id]), "clean_nll": float(clean_nll[prompt_id]),
                    "modified_nll": float(modified_nll[prompt_id]),
                    "delta_nll": float(modified_nll[prompt_id] - clean_nll[prompt_id]),
                    "semantic_proxy_score": float(modified_sentiment[prompt_id]),
                    "semantic_proxy_delta_vs_clean": float(
                        modified_sentiment[prompt_id] - clean_sentiment[prompt_id]
                    ),
                    "semantic_proxy_definition": "fixed positive-vs-negative clean-GPT next-token logit contrast; not human ground truth",
                    "activation_norm_ratio": float(
                        (modified[prompt_id][valid].norm(dim=-1)
                         / clean_h[prompt_id][valid].norm(dim=-1).clamp_min(1e-8)).mean()
                    ),
                    "activation_mse": float((modified[prompt_id][valid] - clean_h[prompt_id][valid]).float().square().mean()),
                    "correction_norm": float(geometry["correction_norm"][prompt_id][valid].mean()),
                    "correction_parallel_norm": float(geometry["correction_parallel_norm"][prompt_id][valid].mean()),
                    "correction_orthogonal_norm": float(geometry["correction_orthogonal_norm"][prompt_id][valid].mean()),
                    "parallel_fraction": float(geometry["parallel_fraction"][prompt_id][valid].mean()),
                    "correction_cosine_with_v": float(geometry["correction_cosine_with_v"][prompt_id][valid].mean()),
                    "raw_target_signal": retention.pop("raw_target_sae_activation"),
                    "denoised_target_signal": retention.pop("denoised_target_sae_activation"),
                    "target_signal_definition": "mean intervention-induced residual projection onto sentiment direction",
                    "token_positions": "all_tokens", **retention,
                })
    frame = pd.DataFrame(rows)
    raw_semantic = frame[frame.method == "relative_raw"][
        ["prompt_id", "relative_strength", "semantic_proxy_delta_vs_clean"]
    ].rename(columns={"semantic_proxy_delta_vs_clean": "raw_semantic_proxy_delta"})
    frame = frame.merge(
        raw_semantic, on=["prompt_id", "relative_strength"], how="left", validate="many_to_one"
    )
    frame["valid_semantic_retention_ratio"] = (
        (frame.relative_strength > 0)
        & (frame.raw_semantic_proxy_delta.abs() > retention_threshold)
    )
    frame["semantic_proxy_retention"] = np.where(
        frame.valid_semantic_retention_ratio,
        frame.semantic_proxy_delta_vs_clean / frame.raw_semantic_proxy_delta,
        np.nan,
    )
    assert np.isfinite(frame[["kl", "clean_nll", "modified_nll", "delta_nll"]]).all().all()
    return frame


@torch.inference_mode()
def evaluate_v3_methods(
    model: nn.Module,
    sae: nn.Module,
    token_batches: Sequence[Tensor | tuple[Tensor, Tensor | None]],
    all_directions: Tensor,
    split: Mapping[str, Any],
    direction_ids: Sequence[int],
    strengths: Sequence[float],
    literal_calibration: Mapping[str, Any],
    hook_name: str,
    v3_models: Mapping[str, GatedConditionedDenoiser],
    v3_normalizations: Mapping[str, Mapping[str, Any]],
    v1_models: Mapping[str, nn.Module],
    v1_normalizations: Mapping[str, Mapping[str, Any]],
    methods: Sequence[str],
    evaluation_split: str,
    retention_threshold: float,
    max_batches: int | None = None,
    token_positions: str = "all_tokens",
    neighbor_index: Any | None = None,
) -> pd.DataFrame:
    """Deterministic matched token evaluation with one row per prompt/config."""
    holdout_label = str(split.get("holdout_label", "replication_test_observed"))
    assert evaluation_split in {"validation", holdout_label}
    allowed = set(split["val"] if evaluation_split == "validation" else split["test"])
    ids = list(map(int, direction_ids)); assert ids and set(ids).issubset(allowed)
    assert set(ids).isdisjoint(set(split["train"]))
    strengths = list(map(float, strengths))
    alpha_map = dict(zip(
        map(float, literal_calibration["relative_calibration_strengths"]),
        map(float, literal_calibration["literal_alphas"]),
    ))
    rows: list[dict[str, Any]] = []
    batches = token_batches if max_batches is None else token_batches[:max_batches]
    prompt_offset = 0
    for batch_id, batch in enumerate(batches):
        tokens, supplied = (batch, None) if isinstance(batch, Tensor) else batch
        mask = torch.ones_like(tokens, dtype=torch.bool) if supplied is None else supplied.bool()
        clean_logits, cache = model.run_with_cache(tokens, names_filter=hook_name)
        clean_h = cache[hook_name]; del cache
        clean_nll_tokens = torch.nn.functional.cross_entropy(
            clean_logits[:, :-1].reshape(-1, clean_logits.shape[-1]),
            tokens[:, 1:].reshape(-1), reduction="none",
        ).reshape(tokens.shape[0], -1)
        target_mask = mask[:, 1:].bool() & mask[:, :-1].bool()
        clean_nll = _per_example_mean(clean_nll_tokens, target_mask)
        for direction_id in ids:
            direction = all_directions[direction_id]
            for strength in strengths:
                relative_x = relative_raw_steering(clean_h, direction, strength)
                raw_feature = _per_example_feature(sae, relative_x, direction_id, mask)
                literal_alpha = alpha_map[strength]
                for method in methods:
                    steering_mode = "literal" if method == "literal_raw" else "relative"
                    if method == "literal_raw":
                        steered = literal_raw_steering(clean_h, direction, literal_alpha); modified = steered
                    elif method == "relative_raw":
                        steered = relative_x; modified = steered
                    elif method == "norm_preserving":
                        steered = relative_x
                        modified = apply_steering(clean_h, direction, strength, method="relative_norm_preserving")
                    elif method == "hard_projected_conditioned_kl_retention":
                        steered = relative_x
                        modified = apply_steering(
                            clean_h, direction, strength,
                            method="hard_projected_gated_conditioned_denoise",
                            denoiser=v3_models["conditioned_kl_retention"],
                            normalization_stats=v3_normalizations["conditioned_kl_retention"],
                        )
                    elif method in v3_models:
                        steered = relative_x
                        modified = apply_steering(
                            clean_h, direction, strength, method="gated_conditioned_denoise",
                            denoiser=v3_models[method], normalization_stats=v3_normalizations[method],
                        )
                    else:
                        key = {"gaussian_denoiser": "gaussian", "sae_calibrated": "sae_calibrated", "fluency_denoiser": "fluency"}[method]
                        steered = relative_x
                        modified = apply_steering(
                            clean_h, direction, strength, method="relative_denoise",
                            denoiser=v1_models[key], normalization_stats=v1_normalizations[key],
                        )
                    modified_logits = get_logits_with_intervention(model, tokens, hook_name, lambda _: modified)
                    # Match the train objective exactly: only logits with an
                    # observed next-token target contribute downstream KL.
                    kl_tokens = token_level_kl(
                        clean_logits[:, :-1], modified_logits[:, :-1],
                        target_mask, reduction="none",
                    )
                    kl = _per_example_mean(kl_tokens, target_mask)
                    modified_nll_tokens = torch.nn.functional.cross_entropy(
                        modified_logits[:, :-1].reshape(-1, modified_logits.shape[-1]),
                        tokens[:, 1:].reshape(-1), reduction="none",
                    ).reshape(tokens.shape[0], -1)
                    modified_nll = _per_example_mean(modified_nll_tokens, target_mask)
                    feature = _per_example_feature(sae, modified, direction_id, mask)
                    # Retention must be measured against the matching raw
                    # steering scale. Literal and relative steering are not
                    # interchangeable baselines.
                    raw_reference = feature if method == "literal_raw" else raw_feature
                    geometry = correction_geometry(modified, steered, direction)
                    knn_values = nearest_cosine = None
                    if neighbor_index is not None:
                        from src.metrics import denoiser_nearest_clean_cosine
                        knn_values = neighbor_index.knn_distance(
                            modified, reduction="none"
                        ).reshape(tokens.shape).astype(np.float64)
                        nearest_cosine = denoiser_nearest_clean_cosine(
                            steered, modified, neighbor_index
                        )
                    clean_norm = clean_h.norm(dim=-1); modified_norm = modified.norm(dim=-1)
                    for index in range(tokens.shape[0]):
                        valid = mask[index]
                        retention = retention_fields(
                            float(raw_reference[index]), float(feature[index]), strength,
                            retention_threshold,
                        )
                        row = {
                            "pipeline_version": V3_PIPELINE_VERSION,
                            "evaluation_split": evaluation_split, "method": method,
                            "direction_id": direction_id, "prompt_id": prompt_offset + index,
                            "batch_id": batch_id, "steering_mode": steering_mode,
                            "relative_strength": strength,
                            "literal_alpha": literal_alpha if method == "literal_raw" else float("nan"),
                            "kl": float(kl[index]), "clean_nll": float(clean_nll[index]),
                            "modified_nll": float(modified_nll[index]),
                            "delta_nll": float(modified_nll[index] - clean_nll[index]),
                            "activation_norm_ratio": float((modified_norm[index][valid] / clean_norm[index][valid].clamp_min(1e-8)).mean()),
                            "activation_mse": float((modified[index][valid] - clean_h[index][valid]).float().square().mean()),
                            "correction_norm": float(geometry["correction_norm"][index][valid].mean()),
                            "correction_parallel_norm": float(geometry["correction_parallel_norm"][index][valid].mean()),
                            "correction_orthogonal_norm": float(geometry["correction_orthogonal_norm"][index][valid].mean()),
                            "parallel_fraction": float(geometry["parallel_fraction"][index][valid].mean()),
                            "correction_cosine_with_v": float(geometry["correction_cosine_with_v"][index][valid].mean()),
                            "token_positions": token_positions, **retention,
                        }
                        if knn_values is not None and nearest_cosine is not None:
                            row["knn_distance"] = float(knn_values[index][valid.detach().cpu().numpy()].mean())
                            row["nearest_clean_correction_cosine"] = float(
                                nearest_cosine[index][valid].mean()
                            )
                        finite_columns = ["kl", "clean_nll", "modified_nll", "delta_nll", "activation_norm_ratio", "activation_mse", "correction_norm"]
                        assert all(math.isfinite(float(row[key])) for key in finite_columns)
                        rows.append(row)
        prompt_offset += tokens.shape[0]
    return pd.DataFrame(rows)


def aggregate_v3_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate with retention computed only from valid positive-strength rows."""
    keys = ["evaluation_split", "method", "steering_mode", "relative_strength"]
    basic = results.groupby(keys, as_index=False).agg(
        mean_kl=("kl", "mean"), median_kl=("kl", "median"),
        mean_delta_nll=("delta_nll", "mean"), median_delta_nll=("delta_nll", "median"),
        mean_activation_norm_ratio=("activation_norm_ratio", "mean"),
        mean_parallel_fraction=("parallel_fraction", "mean"), count=("kl", "size"),
    )
    valid = results[results.valid_retention_ratio & (results.relative_strength > 0)]
    retention = valid.groupby(keys, as_index=False).agg(
        mean_concept_retention=("concept_retention", "mean"),
        median_concept_retention=("concept_retention", "median"),
        fraction_violating_80=("violates_80_percent", "mean"),
        valid_retention_count=("concept_retention", "size"),
    )
    return basic.merge(retention, on=keys, how="left", validate="one_to_one")


def causal_ablation_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Return the narrow five-row table isolating conditioning, KL, retention."""
    definitions = [
        ("relative_raw", "raw", False, False, False, False, "reference"),
        ("sae_calibrated", "unconditional reconstruction D(x)", False, False, False, True, "unconditional repair"),
        ("conditioned_reconstruction", "conditioned reconstruction D(x,v,s)", True, False, False, True, "conditioning contribution"),
        ("conditioned_kl", "conditioned KL D(x,v,s)", True, True, False, False, "downstream KL contribution"),
        ("conditioned_kl_retention", "conditioned KL + retention", True, True, True, False, "retention contribution"),
    ]
    positive = aggregate[aggregate.relative_strength > 0]
    rows = []
    for method, label, conditioning, kl, retention, reconstruction, increment in definitions:
        selected = positive[positive.method == method]
        assert not selected.empty, f"Missing causal ablation method: {method}"
        rows.append({
            "method": method, "ablation_label": label,
            "uses_conditioning": conditioning, "uses_downstream_kl": kl,
            "uses_retention_constraint": retention, "uses_reconstruction": reconstruction,
            "isolated_increment": increment,
            "mean_delta_nll": float(selected.mean_delta_nll.mean()),
            "mean_kl": float(selected.mean_kl.mean()),
            "mean_concept_retention": float(selected.mean_concept_retention.mean()),
            "fraction_violating_80": float(selected.fraction_violating_80.mean()),
        })
    return pd.DataFrame(rows)


def _normalized_pareto_hypervolume(
    rows: pd.DataFrame,
    delta_column: str,
    retention_column: str,
    bounds: tuple[float, float, float, float],
) -> float:
    """2D dominated HV after one fixed within-experiment normalization."""
    x_min, x_max, y_min, y_max = bounds
    assert x_max > x_min and y_max > y_min
    points = rows.groupby("relative_strength", as_index=False).agg(
        delta=(delta_column, "mean"), retention=(retention_column, "mean")
    )
    quality_x = np.clip((x_max - points.delta.to_numpy()) / (x_max - x_min), 0, 1)
    quality_y = np.clip((points.retention.to_numpy() - y_min) / (y_max - y_min), 0, 1)
    order = np.argsort(quality_x); xs, ys = quality_x[order], quality_y[order]
    area = 0.0; previous = 0.0
    for x in np.unique(xs):
        area += (float(x) - previous) * float(ys[xs >= x].max())
        previous = float(x)
    return float(area)


def paired_hierarchical_bootstrap(
    results: pd.DataFrame,
    comparisons: Sequence[tuple[str, str]],
    seed: int = 42,
    samples: int = 2000,
) -> pd.DataFrame:
    """Paired direction/concept→prompt bootstrap; token rows are never iid.

    Pareto-HV uses one fixed set of bounds per supplied experiment frame and
    reports only paired method-minus-reference differences. Absolute HV values
    are intentionally not exposed for cross-study comparison.
    """
    rng = np.random.default_rng(seed); output: list[dict[str, Any]] = []
    positive = results[results.relative_strength > 0]
    valid_hv = positive[positive.valid_retention_ratio & positive.concept_retention.notna()]
    x_min, x_max = float(valid_hv.delta_nll.min()), float(valid_hv.delta_nll.max())
    y_min, y_max = float(valid_hv.concept_retention.min()), float(valid_hv.concept_retention.max())
    if x_max <= x_min: x_max = x_min + 1.0
    if y_max <= y_min: y_max = y_min + 1.0
    hv_bounds = (x_min, x_max, y_min, y_max)
    keys = ["direction_id", "prompt_id", "relative_strength"]
    for method, reference in comparisons:
        left = positive[positive.method == method]
        right = positive[positive.method == reference]
        merged = left.merge(right, on=keys, suffixes=("_method", "_reference"), validate="one_to_one")
        assert not merged.empty
        for metric in ("delta_nll", "kl", "concept_retention"):
            metric_rows = merged.copy()
            if metric == "concept_retention":
                metric_rows = metric_rows[
                    metric_rows["valid_retention_ratio_method"]
                    & metric_rows["valid_retention_ratio_reference"]
                ].copy()
                metric_rows["difference"] = (
                    metric_rows[f"{metric}_method"]
                    - metric_rows[f"{metric}_reference"]
                )
                effect_definition = "method_minus_reference_higher_is_better"
            else:
                metric_rows["difference"] = (
                    metric_rows[f"{metric}_reference"]
                    - metric_rows[f"{metric}_method"]
                )
                effect_definition = "reference_minus_method_positive_is_recovery"
            if metric_rows.empty:
                continue
            direction_ids = metric_rows.direction_id.unique(); estimates = []
            for _ in range(samples):
                selected_directions = rng.choice(direction_ids, len(direction_ids), replace=True)
                values = []
                for direction_id in selected_directions:
                    group = metric_rows[metric_rows.direction_id == direction_id]
                    prompt_ids = group.prompt_id.unique()
                    selected_prompts = rng.choice(prompt_ids, len(prompt_ids), replace=True)
                    values.extend(group.set_index("prompt_id").loc[selected_prompts, "difference"].tolist())
                estimates.append(float(np.mean(values)))
            output.append({
                "method": method, "reference": reference, "metric": metric,
                "effect_definition": effect_definition,
                "mean_recovery": float(metric_rows.difference.mean()),
                "median_recovery": float(metric_rows.difference.median()),
                "ci95_low": float(np.quantile(estimates, 0.025)),
                "ci95_high": float(np.quantile(estimates, 0.975)),
                "matched_examples": len(metric_rows),
                "num_directions": int(metric_rows.direction_id.nunique()),
                "fraction_directions_improved": float(metric_rows.groupby("direction_id").difference.mean().gt(0).mean()),
            })

        hv_rows = merged[
            merged.valid_retention_ratio_method
            & merged.valid_retention_ratio_reference
            & merged.concept_retention_method.notna()
            & merged.concept_retention_reference.notna()
        ].copy()
        if not hv_rows.empty:
            observed = _normalized_pareto_hypervolume(
                hv_rows, "delta_nll_method", "concept_retention_method", hv_bounds
            ) - _normalized_pareto_hypervolume(
                hv_rows, "delta_nll_reference", "concept_retention_reference", hv_bounds
            )
            estimates: list[float] = []
            direction_ids = hv_rows.direction_id.unique()
            for _ in range(samples):
                sampled_groups = []
                for direction_id in rng.choice(direction_ids, len(direction_ids), replace=True):
                    group = hv_rows[hv_rows.direction_id == direction_id]
                    prompts = group.prompt_id.unique()
                    for prompt_id in rng.choice(prompts, len(prompts), replace=True):
                        sampled_groups.append(group[group.prompt_id == prompt_id])
                sampled = pd.concat(sampled_groups, ignore_index=True)
                estimates.append(
                    _normalized_pareto_hypervolume(
                        sampled, "delta_nll_method", "concept_retention_method", hv_bounds
                    ) - _normalized_pareto_hypervolume(
                        sampled, "delta_nll_reference", "concept_retention_reference", hv_bounds
                    )
                )
            per_direction = []
            for direction_id, group in hv_rows.groupby("direction_id"):
                del direction_id
                per_direction.append(
                    _normalized_pareto_hypervolume(
                        group, "delta_nll_method", "concept_retention_method", hv_bounds
                    ) - _normalized_pareto_hypervolume(
                        group, "delta_nll_reference", "concept_retention_reference", hv_bounds
                    )
                )
            output.append({
                "method": method, "reference": reference, "metric": "pareto_hv",
                "effect_definition": "paired_method_minus_reference_same_fixed_bounds",
                "mean_recovery": observed, "median_recovery": float(np.median(estimates)),
                "ci95_low": float(np.quantile(estimates, 0.025)),
                "ci95_high": float(np.quantile(estimates, 0.975)),
                "matched_examples": len(hv_rows), "num_directions": len(direction_ids),
                "fraction_directions_improved": float(np.mean(np.asarray(per_direction) > 0)),
                "hv_delta_nll_min": x_min, "hv_delta_nll_max": x_max,
                "hv_retention_min": y_min, "hv_retention_max": y_max,
                "hv_cross_study_comparable": False,
            })
    return pd.DataFrame(output)


def audit_v3_leakage(
    split: Mapping[str, Any],
    harmfulness_ids: Sequence[int],
    calibration_ids: Sequence[int],
    validation_ids: Sequence[int],
    frozen_protocol_exists: bool,
    holdout_requested: bool,
) -> dict[str, Any]:
    train, val, holdout = map(set, (split["train"], split["val"], split["test"]))
    checks = {
        "split_disjoint": train.isdisjoint(val) and train.isdisjoint(holdout) and val.isdisjoint(holdout),
        "harmfulness_train_only": set(harmfulness_ids).issubset(train),
        "calibration_train_only": set(calibration_ids).issubset(train),
        "validation_excludes_holdout": set(validation_ids).issubset(val) and set(validation_ids).isdisjoint(holdout),
        "holdout_after_freeze": (not holdout_requested) or frozen_protocol_exists,
    }
    if not all(checks.values()):
        raise AssertionError(f"V3 leakage audit failed: {checks}")
    return {"pipeline_version": V3_PIPELINE_VERSION, "checks": checks, "passed": True}


def freeze_v3_protocol(
    path: str | Path,
    config: Mapping[str, Any],
    split: Mapping[str, Any],
    selected_method: str,
    checkpoint_paths: Mapping[str, str],
    code_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Write-once protocol after validation and before new holdout access."""
    resolved = Path(path)
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite frozen V3 protocol: {resolved}")
    checkpoints = {name: {"path": value, "sha256": file_sha256(value)} for name, value in checkpoint_paths.items()}
    code_hashes = {str(path): file_sha256(path) for path in code_paths if Path(path).exists()}
    payload = {
        "pipeline_version": V3_PIPELINE_VERSION, "status": "frozen_after_validation",
        "seed": int(config["seed"]), "model": config["model"], "sae": config["sae"],
        "hook": config["model"]["hook_name"], "direction_split_hash": direction_split_hash(split),
        "holdout_label": split.get("holdout_label", "replication_test_observed"),
        "prompt_split_metadata": config["final_v3"]["prompt_split"],
        "strength_grid": config["final_v3"]["strengths"],
        "retention_target": config["final_v3"]["retention_target"],
        "architecture": config["final_v3"]["architecture"],
        "training": config["final_v3"]["training"],
        "ablation_specs": {
            name: resolved_ablation_spec(name, config["final_v3"])
            for name in config["final_v3"]["ablations"]
        },
        "selected_method": selected_method,
        "selection_rule": config["final_v3"]["selection_rule"],
        "generation": config["final_v3"]["generation"],
        "checkpoints": checkpoints, "code_sha256": code_hashes,
        "holdout_used_for_selection": False,
    }
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def save_v3_figures(
    aggregate: pd.DataFrame,
    harmfulness: pd.DataFrame,
    output_dir: str | Path,
    neighbor_results: pd.DataFrame | None = None,
    generation_results: pd.DataFrame | None = None,
) -> list[str]:
    """Save the pre-registered V3 figures from observed results only."""
    import matplotlib.pyplot as plt
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True); created = []
    primary_methods = [
        "relative_raw", "sae_calibrated", "conditioned_reconstruction",
        "conditioned_kl", "conditioned_kl_retention",
    ]

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            target = root / f"{name}.{suffix}"
            fig.savefig(target, dpi=180)
            created.append(str(target))
        plt.close(fig)

    specs = [
        ("strength_delta_nll", "relative_strength", "mean_delta_nll", "Relative strength", "Mean delta NLL (lower is better)"),
        ("strength_retention", "relative_strength", "mean_concept_retention", "Relative strength", "Mean concept retention (higher is better)"),
        ("strength_parallel_fraction", "relative_strength", "mean_parallel_fraction", "Relative strength", "Parallel correction fraction"),
        ("strength_activation_norm", "relative_strength", "mean_activation_norm_ratio", "Relative strength", "Activation norm ratio"),
    ]
    for name, x, y, xlabel, ylabel in specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        frame = aggregate[aggregate.method.isin(primary_methods)]
        frame = frame[frame.relative_strength > 0] if "retention" in name else frame
        for method, group in frame.groupby("method"):
            group = group.sort_values(x); ax.plot(group[x], group[y], marker="o", label=method)
        if "retention" in name: ax.axhline(0.8, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel=xlabel, ylabel=ylabel); ax.legend(fontsize=7)
        save(fig, name)

    positive = aggregate[
        (aggregate.relative_strength > 0) & aggregate.mean_concept_retention.notna()
        & aggregate.method.isin(primary_methods)
    ]
    for name, x, xlabel in (
        ("pareto_delta_nll_retention", "mean_delta_nll", "Mean delta NLL (left is better)"),
        ("kl_vs_retention", "mean_kl", "Mean KL(clean || modified) (left is better)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, group in positive.groupby("method"):
            group = group.sort_values("relative_strength")
            ax.plot(group[x], group.mean_concept_retention, marker="o", label=method)
        ax.axhline(0.8, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel=xlabel, ylabel="Mean concept retention (up is better)")
        ax.legend(fontsize=7)
        save(fig, name)

    ablation_methods = primary_methods
    ablation = positive[positive.method.isin(ablation_methods)]
    if not ablation.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, group in ablation.groupby("method"):
            ax.scatter(group.mean_delta_nll, group.mean_concept_retention, s=36, label=method)
        ax.axhline(0.8, color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Mean delta NLL (left is better)",
               ylabel="Mean concept retention (up is better)")
        ax.legend(fontsize=7)
        save(fig, "causal_ablation_summary")

    hard_soft = aggregate[
        aggregate.method.isin([
            "conditioned_kl_retention", "hard_projected_conditioned_kl_retention"
        ]) & (aggregate.relative_strength > 0)
    ]
    if not hard_soft.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for method, group in hard_soft.groupby("method"):
            group = group.sort_values("relative_strength")
            axes[0].plot(group.relative_strength, group.mean_delta_nll, marker="o", label=method)
            axes[1].plot(group.relative_strength, group.mean_concept_retention, marker="o", label=method)
        axes[0].set(xlabel="Relative strength", ylabel="Mean delta NLL (lower is better)")
        axes[1].set(xlabel="Relative strength", ylabel="Mean retention (higher is better)")
        axes[1].axhline(0.8, color="black", linestyle="--", linewidth=1)
        axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
        save(fig, "hard_projection_vs_soft_retention")

    if not harmfulness.empty:
        fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(harmfulness["mean_kl"], bins=25)
        ax.set(xlabel="Train-direction mean KL", ylabel="Directions", title="Harmfulness heterogeneity (descriptive)")
        save(fig, "harmfulness_distribution")
    if neighbor_results is not None and not neighbor_results.empty:
        valid = neighbor_results.dropna(subset=["knn_distance", "delta_nll"])
        rho = valid.knn_distance.corr(valid.delta_nll, method="spearman")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(valid.knn_distance, valid.delta_nll, alpha=0.5, s=16)
        ax.set(xlabel="kNN distance in clean PCA space (lower is more natural)",
               ylabel="delta NLL (lower is better)", title=f"Natural-neighbor diagnostic; Spearman rho={rho:.3f}")
        save(fig, "knn_distance_vs_delta_nll")
    if generation_results is not None and not generation_results.empty:
        semantic_column = "semantic_proxy_score" if "semantic_proxy_score" in generation_results else "concept_score"
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, group in generation_results.groupby("method"):
            ax.scatter(group.clean_model_nll, group[semantic_column], s=18, alpha=0.6, label=method)
        ax.set(xlabel="Clean-model continuation NLL (lower is better)",
               ylabel=f"{semantic_column} (higher is stronger)")
        ax.legend(fontsize=7)
        save(fig, "generation_fluency_vs_semantic_proxy")
    cross_path = root.parent / "results" / "cross_concept_aggregate.csv"
    if cross_path.exists():
        cross = pd.read_csv(cross_path)
        cross = cross[(cross.relative_strength > 0) & cross.method.isin(primary_methods)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for method, group in cross.groupby("method"):
            group = group.sort_values("relative_strength")
            axes[0].plot(group.relative_strength, group.mean_delta_nll, marker="o", label=method)
            axes[1].plot(group.relative_strength, group.mean_concept_retention, marker="o", label=method)
        axes[0].set(xlabel="Relative strength", ylabel="Sentiment-family delta NLL")
        axes[1].set(xlabel="Relative strength", ylabel="Sentiment-direction retention")
        axes[1].axhline(0.8, color="black", linestyle="--", linewidth=1)
        axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
        save(fig, "cross_concept_sentiment_confirmation")
    return created


def write_v3_report(
    output_dir: str | Path,
    validation: pd.DataFrame,
    holdout: pd.DataFrame | None,
    statistics: pd.DataFrame,
    audit: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Write factual report artifacts from observed rows only."""
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    artifact_root = root
    aggregate = aggregate_v3_results(validation)
    holdout_aggregate = (
        aggregate_v3_results(holdout) if holdout is not None and not holdout.empty
        else pd.DataFrame()
    )
    identity = validation[validation.relative_strength == 0].groupby("method", as_index=False).agg(
        mean_kl=("kl", "mean"), mean_delta_nll=("delta_nll", "mean"),
        mean_activation_mse=("activation_mse", "mean"),
        mean_correction_norm=("correction_norm", "mean"),
    )
    positive = aggregate[aggregate.relative_strength > 0]
    harmfulness_path = artifact_root / "results" / "harmfulness.csv"
    neighbor_path = artifact_root / "results" / "natural_neighbor_diagnostics.csv"
    generation_path = artifact_root / "results" / "generation_aggregate.csv"
    semantic_path = artifact_root / "semantic" / "semantic_aggregate.csv"
    causal_path = artifact_root / "results" / "causal_ablation_table.csv"
    cross_path = artifact_root / "results" / "cross_concept_confirmation.csv"
    cross_statistics_path = artifact_root / "results" / "cross_concept_statistics.csv"
    harmfulness = pd.read_csv(harmfulness_path) if harmfulness_path.exists() else pd.DataFrame()
    neighbors = pd.read_csv(neighbor_path) if neighbor_path.exists() else pd.DataFrame()
    generation = pd.read_csv(generation_path) if generation_path.exists() else pd.DataFrame()
    semantic = pd.read_csv(semantic_path) if semantic_path.exists() else pd.DataFrame()
    causal = pd.read_csv(causal_path) if causal_path.exists() else causal_ablation_table(aggregate)
    cross = pd.read_csv(cross_path) if cross_path.exists() else pd.DataFrame()
    cross_statistics = pd.read_csv(cross_statistics_path) if cross_statistics_path.exists() else pd.DataFrame()
    cross_aggregate = aggregate_v3_results(cross) if not cross.empty else pd.DataFrame()

    def relation(method: str, reference: str, metric: str, frame: pd.DataFrame = validation) -> float:
        left = frame[(frame.method == method) & (frame.relative_strength > 0)]
        right = frame[(frame.method == reference) & (frame.relative_strength > 0)]
        merged = left.merge(
            right, on=["direction_id", "prompt_id", "relative_strength"],
            suffixes=("_method", "_reference"), validate="one_to_one",
        )
        return float((merged[f"{metric}_reference"] - merged[f"{metric}_method"]).mean())

    raw = validation[(validation.method == "relative_raw") & (validation.relative_strength > 0)]
    raw_strength = raw.groupby("relative_strength").delta_nll.mean()
    h1 = len(raw_strength) > 1 and raw_strength.corr(pd.Series(raw_strength.index, index=raw_strength.index), method="spearman") > 0.5
    kl_nll_rho = float(validation[["kl", "delta_nll"]].corr(method="spearman").iloc[0, 1])
    h3 = (not harmfulness.empty and harmfulness.mean_kl.quantile(.9) > 2 * max(harmfulness.mean_kl.quantile(.1), 1e-8))
    norm_preserved = validation[
        (validation.method == "norm_preserving") & (validation.relative_strength > 0)
    ]
    norm_preserved_damage = float(norm_preserved.delta_nll.mean())
    gaussian_recovery = relation("gaussian_denoiser", "relative_raw", "delta_nll")
    sae_vs_gaussian = relation("sae_calibrated", "gaussian_denoiser", "delta_nll")
    conditioning_gain = relation("conditioned_reconstruction", "sae_calibrated", "delta_nll")
    kl_gain = relation("conditioned_kl", "conditioned_reconstruction", "delta_nll")
    retention_means = positive.set_index("method").groupby(level=0).mean(numeric_only=True)
    def violation_fraction(method: str) -> float:
        values = validation[
            (validation.method == method)
            & (validation.relative_strength > 0)
            & validation.valid_retention_ratio
        ].violates_80_percent.astype(float)
        return float(values.mean())

    retention_gain = violation_fraction("conditioned_kl") - violation_fraction("conditioned_kl_retention")
    soft_method = str(protocol["selected_method"])
    generalization_frame = holdout if holdout is not None and not holdout.empty else validation
    generalization_scope = (
        protocol["holdout_label"] if holdout is not None and not holdout.empty else "validation_only"
    )
    generalization_aggregate = aggregate_v3_results(generalization_frame)
    generalization_positive = generalization_aggregate[generalization_aggregate.relative_strength > 0]
    generalization_means = generalization_positive.set_index("method").groupby(level=0).mean(numeric_only=True)
    full_recovery = relation(soft_method, "relative_raw", "delta_nll", generalization_frame)
    full_retention = float(generalization_means.loc[soft_method, "mean_concept_retention"])
    full_parallel = float(generalization_means.loc[soft_method, "mean_parallel_fraction"])
    soft_vs_hard = relation(
        "conditioned_kl_retention", "hard_projected_conditioned_kl_retention", "delta_nll"
    )
    neighbor_rho = (
        float(neighbors.knn_distance.corr(neighbors.delta_nll, method="spearman"))
        if not neighbors.empty else float("nan")
    )
    full_rows = generalization_frame[(generalization_frame.method == soft_method) & (generalization_frame.relative_strength > 0)]
    raw_rows = generalization_frame[(generalization_frame.method == "relative_raw") & (generalization_frame.relative_strength > 0)]
    joint = full_rows.merge(
        raw_rows, on=["direction_id", "prompt_id", "relative_strength"],
        suffixes=("_full", "_raw"), validate="one_to_one",
    )
    joint_fraction = float((
        (joint.delta_nll_full < joint.delta_nll_raw)
        & joint.valid_retention_ratio_full
        & (joint.concept_retention_full >= 0.8)
    ).mean())
    cross_available = not cross.empty
    cross_recovery = relation(soft_method, "relative_raw", "delta_nll", cross) if cross_available else float("nan")
    cross_retention = (
        float(cross[(cross.method == soft_method) & (cross.relative_strength > 0)
                    & cross.valid_retention_ratio].concept_retention.mean())
        if cross_available else float("nan")
    )
    raw_semantic_delta = (
        float(cross[(cross.method == "relative_raw") & (cross.relative_strength > 0)]
              .semantic_proxy_delta_vs_clean.mean())
        if cross_available else float("nan")
    )
    cross_semantic_retention = (
        float(cross[(cross.method == soft_method) & cross.valid_semantic_retention_ratio]
              .semantic_proxy_retention.mean())
        if cross_available else float("nan")
    )
    cross_supported = (
        cross_available and cross_recovery > 0 and cross_retention >= 0.8
        and raw_semantic_delta > 0 and cross_semantic_retention >= 0.8
    )
    within_supported = full_recovery > 0 and full_retention >= 0.8
    if within_supported and not cross_supported and cross_available:
        transfer_boundary = (
            "Conditioned denoising generalizes across unseen SAE directions but "
            "not across the tested sentiment steering family."
        )
    elif within_supported and cross_supported:
        transfer_boundary = (
            "Conditioned denoising generalizes across unseen SAE directions and "
            "the tested sentiment steering family; broader family transfer remains untested."
        )
    elif cross_available:
        transfer_boundary = "The pre-registered within-SAE criterion was not supported; cross-family evidence is reported separately."
    else:
        transfer_boundary = "Cross-concept confirmation was not run, so no cross-family claim is made."

    def label(condition: bool, available: bool = True) -> str:
        return "supported" if available and bool(condition) else ("not supported" if available else "inconclusive")

    hypotheses = {
        "H1": {"status": label(h1), "evidence": "Spearman strength vs relative_raw delta NLL"},
        "H2": {"status": label(kl_nll_rho >= 0.7), "value": kl_nll_rho, "evidence": "Spearman KL vs delta NLL"},
        "H3": {"status": label(h3, not harmfulness.empty), "evidence": "train-direction harmfulness p90/p10 rule"},
        "H4": {"status": label(norm_preserved_damage > 0), "value": norm_preserved_damage, "evidence": "positive damage remains under norm-preserving steering"},
        "H5": {"status": label(gaussian_recovery <= 0), "value": gaussian_recovery, "evidence": "Gaussian recovery vs relative_raw"},
        "H6": {"status": label(sae_vs_gaussian > 0), "value": sae_vs_gaussian, "evidence": "calibrated SAE recovery vs Gaussian"},
        "H7": {"status": label(conditioning_gain > 0), "value": conditioning_gain, "evidence": "conditioned reconstruction vs unconditional SAE reconstruction"},
        "H8": {"status": label(kl_gain > 0), "value": kl_gain, "evidence": "conditioned KL vs conditioned reconstruction"},
        "H9": {"status": label(retention_gain > 0), "value": retention_gain, "evidence": "reduction in <80% retention violations for KL+retention vs KL-only"},
        "H10": {"status": label(within_supported), "evidence": f"validation-selected {soft_method} recovery with mean valid retention >= 0.8"},
        "H11": {"status": label(full_parallel < 0.5), "value": full_parallel, "evidence": "mean correction parallel fraction < 0.5; correlational"},
        "H12": {"status": label(soft_vs_hard > 0), "value": soft_vs_hard, "evidence": "soft conditioned KL+retention recovery vs hard c_parallel=0 control; projection/incremental remain supporting controls"},
        "H13": {"status": label(neighbor_rho > 0.3, math.isfinite(neighbor_rho)), "value": neighbor_rho, "evidence": "Spearman kNN distance vs delta NLL; correlational"},
        "H14": {"status": label(joint_fraction > 0.5), "value": joint_fraction, "evidence": "matched fraction with lower delta NLL and >=80% target-SAE retention"},
    }

    def json_safe(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, (np.floating, float)) and not math.isfinite(float(value)):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    summary = {
        "pipeline_version": V3_PIPELINE_VERSION, "audit": dict(audit),
        "holdout_status": protocol["holdout_label"],
        "validation_rows": len(validation), "holdout_rows": 0 if holdout is None else len(holdout),
        "identity": identity.to_dict(orient="records"),
        "validation_positive_strength": positive.to_dict(orient="records"),
        "holdout_positive_strength": (
            holdout_aggregate[holdout_aggregate.relative_strength > 0].to_dict(orient="records")
            if not holdout_aggregate.empty else []
        ),
        "paired_statistics": statistics.to_dict(orient="records"),
        "causal_ablation_table": causal.to_dict(orient="records"),
        "within_sae_generalization": {"scope": generalization_scope, "supported": within_supported, "mean_recovery": full_recovery, "mean_retention": full_retention},
        "cross_concept_generalization": {
            "available": cross_available, "supported": cross_supported,
            "mean_recovery": cross_recovery, "mean_direction_projection_retention": cross_retention,
            "raw_sentiment_logit_delta": raw_semantic_delta,
            "mean_sentiment_proxy_retention": cross_semantic_retention,
            "semantic_proxy_is_human_ground_truth": False,
        },
        "transfer_boundary": transfer_boundary,
        "cross_concept_statistics": cross_statistics.to_dict(orient="records"),
        "hypotheses": hypotheses,
        "generation_rows": 0 if generation.empty else int(generation.shape[0]),
        "semantic_proxy_rows": 0 if semantic.empty else int(semantic.shape[0]),
        "results_are_empirical": True,
    }
    json_path = root / "final_summary.json"
    json_path.write_text(json.dumps(json_safe(summary), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
        selected = frame if columns is None else frame[[column for column in columns if column in frame]]
        return "```text\n" + selected.to_string(index=False) + "\n```" if not selected.empty else "No rows were produced."

    hypothesis_rows = pd.DataFrame([
        {"hypothesis": key, **value} for key, value in hypotheses.items()
    ])
    report = [
        "# Final V3 research report", "", "## A. Experimental setup and narrow research question", "",
        f"Pipeline `{V3_PIPELINE_VERSION}`; GPT-2 Small at `blocks.6.hook_resid_pre`; holdout label `{protocol['holdout_label']}`. The central comparison is unconditional reconstruction `D(x)` versus intervention-aware `D(x,v,s)`. Gaussian, harmfulness weighting, projection, and incremental steering are supporting or negative controls, not co-equal central proposals.", "",
        "## B. Audit results", "", f"All automated checks passed: `{audit.get('passed')}`.", "",
        "## C. Main token-level result", "", "### Validation", "", table(positive), "",
        "### Frozen V3 holdout", "", table(
            holdout_aggregate[holdout_aggregate.relative_strength > 0]
            if not holdout_aggregate.empty else holdout_aggregate
        ), "",
        "## D. Causal ablations", "", table(causal), "", "### Paired hierarchical uncertainty", "", "ΔNLL, KL, and Pareto-HV intervals resample direction/concept then prompts. Pareto-HV is reported only as a paired difference under identical fixed bounds within each experiment; absolute HV is not compared across studies.", "", table(statistics), "",
        "## E. Concept retention", "", "Zero-strength rows and unstable raw denominators are excluded from retention aggregates; numerator and denominator remain in per-example CSV rows.", "",
        "## F. Identity", "", table(identity), "",
        "## G. Harmfulness heterogeneity", "", "This TRAIN-direction analysis is descriptive, not causal.", "", table(harmfulness.head(20)), "",
        "## H. Correction geometry and density diagnostic", "", "Parallel and orthogonal components and kNN distances are correlational diagnostics. Repair must preserve the intended causal/semantic steering signal, not merely move activations toward high-density clean regions. Low kNN distance is not evidence of semantic correctness.", "",
        "## I. Generation evaluation", "", "Generation is secondary to deterministic token-level evaluation.", "", table(generation), "",
        "## J. Semantic evaluation", "", "Independent external semantic proxy, not human ground truth.", "", table(semantic), "",
        "## K. Generalization scopes", "", "### Within-SAE-family generalization", "", f"Scope: `{generalization_scope}`. Supported under the pre-registered joint criterion: `{within_supported}`.", "", "### Cross-concept generalization", "", f"Supported on the fixed sentiment contrast direction: `{cross_supported if cross_available else 'not run'}`.", "", table(cross_aggregate), "", "### Cross-concept paired uncertainty", "", table(cross_statistics), "", f"Boundary statement: {transfer_boundary}", "",
        "## L. Limitations", "", f"Holdout status: `{protocol['holdout_label']}`. Results cover the fixed sampled directions and prompts. The cross-concept experiment contains one fixed sentiment family and is confirmation, not a new selection stage. Prompt partitions are exactly disjoint by text and begin after a configured streaming offset; the upstream OpenWebText stream does not expose a global document registry, so corpus-level near-duplicate exclusion is not claimed.", "",
        "## M. Hypothesis-by-hypothesis conclusions", "", "Labels use pre-declared numerical rules in the report generator; none is described as proved.", "", table(hypothesis_rows), "",
    ]
    md_path = root / "final_report.md"; md_path.write_text("\n".join(report), encoding="utf-8")
    mirror = root / "reports"; mirror.mkdir(parents=True, exist_ok=True)
    (mirror / "final_report.md").write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    (mirror / "final_summary.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    return md_path, json_path
