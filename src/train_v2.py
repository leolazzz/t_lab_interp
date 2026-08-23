"""Training utilities for the separate conditioned downstream-aware V2 experiment."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.denoiser_v2 import ConditionedSteeringDenoiser, apply_conditioned_denoiser
from src.directions import direction_split_hash, validate_direction_ids_for_usage
from src.metrics import next_token_nll
from src.v2_objectives import (
    correction_geometry,
    downstream_kl,
    relative_correction_loss,
    retention_hinge_loss,
)
from src.utils import seed_everything


V2_PIPELINE_VERSION = "conditioned_v2_v1"
V2_CHECKPOINT_VERSION = 1
TokenBatch = Tensor | tuple[Tensor, Tensor | None]


def resolved_v2_config(config: Mapping[str, Any], quick: bool = False) -> dict[str, Any]:
    """Return an isolated V2 config with optional small, identical-path overrides."""
    result = deepcopy(dict(config["conditioned_v2"]))
    if quick:
        result.update(dict(result.get("quick_overrides", {})))
    result["quick"] = bool(quick)
    assert result["pipeline_version"] == V2_PIPELINE_VERSION
    return result


def calibrate_literal_alphas(
    activation_stats: Mapping[str, Any],
    relative_strengths: Sequence[float],
) -> dict[str, Any]:
    """Pre-register literal alphas as ``s * median_bank_activation_norm``."""
    if "median_activation_norm" not in activation_stats:
        raise KeyError("Activation stats lack median_activation_norm required by V2 calibration.")
    median_norm = float(activation_stats["median_activation_norm"])
    assert median_norm > 0 and torch.isfinite(torch.tensor(median_norm))
    strengths = [float(value) for value in relative_strengths]
    return {
        "median_activation_norm": median_norm,
        "relative_calibration_strengths": strengths,
        "literal_alphas": [strength * median_norm for strength in strengths],
        "formula": "alpha_k = s_k * median_h_norm_from_activation_bank",
    }


def _unpack_batch(batch: TokenBatch) -> tuple[Tensor, Tensor]:
    if isinstance(batch, Tensor):
        tokens, attention_mask = batch, torch.ones_like(batch, dtype=torch.bool)
    else:
        tokens, supplied_mask = batch
        attention_mask = (
            torch.ones_like(tokens, dtype=torch.bool)
            if supplied_mask is None else supplied_mask.bool()
        )
    assert tokens.ndim == 2 and tokens.dtype == torch.long
    assert attention_mask.shape == tokens.shape and tokens.shape[1] >= 2
    return tokens, attention_mask


def _relative_steer_batch(clean_h: Tensor, directions: Tensor, strengths: Tensor) -> Tensor:
    assert clean_h.ndim == 3 and directions.shape == (clean_h.shape[0], clean_h.shape[-1])
    assert strengths.shape == (clean_h.shape[0],)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    update = (
        strengths[:, None, None]
        * clean_h.norm(dim=-1, keepdim=True)
        * directions[:, None, :]
    )
    return clean_h + update


def _target_feature_means(
    sae: nn.Module,
    activations: Tensor,
    direction_ids: Tensor,
    token_mask: Tensor,
) -> Tensor:
    assert direction_ids.shape == (activations.shape[0],)
    features = sae.encode(activations)
    selected = features.gather(
        -1,
        direction_ids[:, None, None].expand(-1, activations.shape[1], 1),
    ).squeeze(-1)
    weights = token_mask.to(selected.dtype)
    denominator = weights.sum(dim=1).clamp_min(1)
    return (selected * weights).sum(dim=1) / denominator


def _run_modified_logits(
    model: nn.Module,
    tokens: Tensor,
    hook_name: str,
    replacement: Tensor,
) -> Tensor:
    """Run full GPT with a differentiable hook replacement at layer 6."""
    hook_called = False

    def replace(activation: Tensor, hook: Any = None) -> Tensor:
        del hook
        nonlocal hook_called
        hook_called = True
        assert activation.shape == replacement.shape
        return replacement

    logits = model.run_with_hooks(
        tokens, return_type="logits", fwd_hooks=[(hook_name, replace)]
    )
    assert hook_called and logits.shape[:2] == tokens.shape
    return logits


def freeze_teacher_models(model: nn.Module, sae: nn.Module) -> None:
    """Freeze weights while retaining gradients with respect to their inputs."""
    model.eval().requires_grad_(False)
    sae.eval().requires_grad_(False)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert not any(parameter.requires_grad for parameter in sae.parameters())


def compute_v2_batch_losses(
    model: nn.Module,
    sae: nn.Module,
    denoiser: ConditionedSteeringDenoiser,
    tokens: Tensor,
    attention_mask: Tensor,
    directions: Tensor,
    direction_ids: Tensor,
    strengths: Tensor,
    identity_directions: Tensor,
    normalization: Mapping[str, Any],
    config: Mapping[str, Any],
    hook_name: str,
) -> dict[str, Tensor]:
    """Compute the full differentiable V2 objective on intact sequences.

    Implemented objective::

      L = lambda_fluency KL(p_clean || p_D)
        + lambda_retain ReLU(rho - (a_D+eps)/(a_raw+eps))^2
        + lambda_identity [identity_KL + eta identity_MSE]
        + lambda_correction ||D(x)-x||^2/(||x||^2+eps)
        + lambda_reconstruction MSE(D(x), h)

    The retention term is masked for ``a_raw <= retention_mask_threshold``.
    """
    assert tokens.shape == attention_mask.shape
    assert not any(parameter.requires_grad for parameter in model.parameters()), (
        "GPT-2 must be frozen before differentiable V2 downstream training."
    )
    assert not any(parameter.requires_grad for parameter in sae.parameters()), (
        "SAE weights must be frozen; gradients may flow only through its input."
    )
    assert directions.shape == identity_directions.shape == (tokens.shape[0], model.cfg.d_model)
    assert direction_ids.shape == strengths.shape == (tokens.shape[0],)
    assert torch.isfinite(strengths).all() and (strengths > 0).all()
    prediction_mask = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    assert prediction_mask.any()
    with torch.no_grad():
        clean_logits, cache = model.run_with_cache(tokens, names_filter=hook_name)
        clean_h = cache[hook_name].detach()
        clean_logits = clean_logits.detach()
        del cache
    raw_h = _relative_steer_batch(clean_h, directions, strengths)
    denoised_h = apply_conditioned_denoiser(
        denoiser, raw_h, directions, strengths, normalization
    )
    modified_logits = _run_modified_logits(model, tokens, hook_name, denoised_h)
    fluency = downstream_kl(
        clean_logits[:, :-1], modified_logits[:, :-1], prediction_mask
    )

    raw_feature = _target_feature_means(
        sae, raw_h.detach(), direction_ids, attention_mask.bool()
    )
    denoised_feature = _target_feature_means(
        sae, denoised_h, direction_ids, attention_mask.bool()
    )
    retention = retention_hinge_loss(
        raw_feature,
        denoised_feature,
        target=float(config["retention_target"]),
        threshold=float(config["retention_mask_threshold"]),
        eps=float(config["retention_eps"]),
    )
    correction = relative_correction_loss(
        denoised_h, raw_h, attention_mask.bool()
    )
    mask3 = attention_mask.bool().unsqueeze(-1).expand_as(clean_h)
    reconstruction = F.mse_loss(denoised_h[mask3], clean_h[mask3])

    zero_strength = strengths.new_zeros(strengths.shape)
    identity_h = apply_conditioned_denoiser(
        denoiser, clean_h, identity_directions, zero_strength, normalization
    )
    identity_logits = _run_modified_logits(model, tokens, hook_name, identity_h)
    identity_kl = downstream_kl(
        clean_logits[:, :-1], identity_logits[:, :-1], prediction_mask
    )
    identity_mse = F.mse_loss(identity_h[mask3], clean_h[mask3])
    identity_loss = identity_kl + float(config["identity_mse_eta"]) * identity_mse
    identity_correction_norm = (identity_h - clean_h).norm(dim=-1)[attention_mask.bool()].mean()
    clean_nll = next_token_nll(clean_logits, tokens, attention_mask)
    identity_nll = next_token_nll(identity_logits, tokens, attention_mask)
    assert isinstance(clean_nll, Tensor) and isinstance(identity_nll, Tensor)

    objective_mode = str(config.get("objective_mode", "full"))
    if objective_mode == "conditioned_reconstruction":
        total = reconstruction + float(config["identity_mse_eta"]) * identity_mse
    else:
        retain_weight = 0.0 if objective_mode == "downstream_no_retain" else float(config["lambda_retain"])
        assert objective_mode in {"full", "downstream_no_retain"}
        total = (
            float(config["lambda_fluency"]) * fluency
            + retain_weight * retention["loss"]
            + float(config["lambda_identity"]) * identity_loss
            + float(config["lambda_correction"]) * correction
            + float(config["lambda_reconstruction"]) * reconstruction
        )
    geometry = correction_geometry(denoised_h, raw_h, directions)
    token_mask = attention_mask.bool()
    outputs = {
        "loss": total,
        "fluency_kl": fluency,
        "retention_loss": retention["loss"],
        "retention_ratio": retention["mean_retention_ratio"],
        "retention_violation": retention["retention_violation"],
        "fraction_below_retention_threshold": retention["fraction_below_retention_threshold"],
        "identity_loss": identity_loss,
        "identity_kl": identity_kl,
        "identity_delta_nll": identity_nll - clean_nll,
        "identity_activation_mse": identity_mse,
        "identity_correction_norm": identity_correction_norm,
        "correction_loss": correction,
        "reconstruction_loss": reconstruction,
        "correction_norm": geometry["correction_norm"][token_mask].mean(),
        "correction_parallel_norm": geometry["correction_parallel_norm"][token_mask].mean(),
        "correction_orthogonal_norm": geometry["correction_orthogonal_norm"][token_mask].mean(),
        "correction_cosine_with_v": geometry["correction_cosine_with_v"][token_mask].mean(),
        "parallel_fraction": geometry["parallel_fraction"][token_mask].mean(),
        "raw_feature_activation": raw_feature.mean(),
        "denoised_feature_activation": denoised_feature.mean(),
    }
    assert all(torch.isfinite(value).all() for value in outputs.values())
    return outputs


def gradient_diagnostics(denoiser: nn.Module) -> dict[str, float]:
    groups = {
        "denoiser_gradient_norm": [],
        "film_gradient_norm": [],
        "direction_embedding_gradient_norm": [],
        "correction_head_gradient_norm": [],
    }
    for name, parameter in denoiser.named_parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().float().norm())
        groups["denoiser_gradient_norm"].append(norm)
        if "film_head" in name:
            groups["film_gradient_norm"].append(norm)
        if "direction_mlp" in name:
            groups["direction_embedding_gradient_norm"].append(norm)
        if "output_projection" in name:
            groups["correction_head_gradient_norm"].append(norm)
    return {
        key: float(sum(value * value for value in values) ** 0.5)
        for key, values in groups.items()
    }


def _sample_training_conditions(
    batch_size: int,
    train_ids: Sequence[int],
    all_directions: Tensor,
    strengths: Sequence[float],
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    assert train_ids and strengths
    train_id_tensor = torch.tensor(train_ids, dtype=torch.long)
    selected = torch.randint(len(train_ids), (batch_size,), generator=generator)
    identity_selected = torch.randint(len(train_ids), (batch_size,), generator=generator)
    strength_index = torch.randint(len(strengths), (batch_size,), generator=generator)
    ids = train_id_tensor[selected].to(all_directions.device)
    identity_ids = train_id_tensor[identity_selected].to(all_directions.device)
    strength_tensor = torch.tensor(strengths, dtype=all_directions.dtype)[strength_index]
    strength_tensor = strength_tensor.to(all_directions.device)
    return all_directions[ids], ids, strength_tensor, all_directions[identity_ids]


def run_loss_scale_diagnostic(
    model: nn.Module,
    sae: nn.Module,
    denoiser: ConditionedSteeringDenoiser,
    token_batches: Sequence[TokenBatch],
    all_directions: Tensor,
    split: Mapping[str, Any],
    normalization: Mapping[str, Any],
    config: Mapping[str, Any],
    hook_name: str,
    output_path: str | Path = "outputs/conditioned_v2/loss_scale_diagnostic.json",
) -> dict[str, Any]:
    """Measure unweighted loss scales without selecting lambdas by validation."""
    freeze_teacher_models(model, sae)
    train_ids = list(split["train"])
    validate_direction_ids_for_usage(train_ids, split, "training", require_complete_split=True)
    generator = torch.Generator(device="cpu").manual_seed(int(config.get("seed", 42)))
    rows = []
    diagnostic_optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    denoiser.train()
    count = min(int(config["loss_scale_diagnostic_batches"]), len(token_batches))
    assert count > 0
    for batch in token_batches[:count]:
        tokens, mask = _unpack_batch(batch)
        directions, ids, strengths, identity_directions = _sample_training_conditions(
            tokens.shape[0], train_ids, all_directions,
            config["training_strengths"], generator,
        )
        losses = compute_v2_batch_losses(
            model, sae, denoiser, tokens, mask, directions, ids, strengths,
            identity_directions, normalization, config, hook_name,
        )
        rows.append({key: float(losses[key].detach()) for key in (
            "fluency_kl", "retention_loss", "identity_kl",
            "identity_activation_mse", "correction_loss", "reconstruction_loss",
        )})
        # This is a disposable scratch warm-up: it makes post-initialization
        # auxiliary scales observable. The diagnostic model is never saved or
        # reused as the training initialization.
        diagnostic_optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), float(config["gradient_clip_norm"]))
        diagnostic_optimizer.step()
    denoiser.eval()
    means = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
    payload = {
        "pipeline_version": V2_PIPELINE_VERSION,
        "num_batches": len(rows),
        "unweighted_mean_losses": means,
        "configured_lambdas": {
            key: float(config[key]) for key in (
                "lambda_fluency", "lambda_retain", "lambda_identity",
                "lambda_correction", "lambda_reconstruction",
            )
        },
        "automatic_lambda_adjustment": False,
        "scratch_warmup_updates": len(rows),
        "first_batch_unweighted_losses": rows[0],
        "last_batch_unweighted_losses": rows[-1],
        "note": "Engineering scale diagnostic only; no validation or TEST performance was used.",
    }
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _checkpoint_payload(
    denoiser: ConditionedSteeringDenoiser,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: Mapping[str, Any],
    normalization: Mapping[str, Any],
    split: Mapping[str, Any],
    hook_name: str,
    model_name: str,
    literal_calibration: Mapping[str, Any],
    validation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    return {
        "checkpoint_version": V2_CHECKPOINT_VERSION,
        "pipeline_version": V2_PIPELINE_VERSION,
        "model_state_dict": denoiser.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "architecture": {
            "d_model": denoiser.d_model,
            "hidden_dim": denoiser.hidden_dim,
            "conditioning_dim": denoiser.conditioning_dim,
            "type": "ConditionedSteeringDenoiser",
            "film_blocks": 2,
            "strength_features": ["s", "log1p(s)", "s^2"],
        },
        "step": int(step),
        "config": deepcopy(dict(config)),
        "normalization": {
            key: value.detach().cpu() if isinstance(value, Tensor) else value
            for key, value in normalization.items()
        },
        "direction_split_hash": direction_split_hash(split),
        "hook_name": hook_name,
        "model_name": model_name,
        "literal_alpha_calibration": dict(literal_calibration),
        "validation": None if validation is None else dict(validation),
        # ``torch.__version__`` is a ``TorchVersion`` object in recent PyTorch
        # releases.  Store plain metadata so the checkpoint remains compatible
        # with the restricted ``weights_only=True`` loader used on Kaggle.
        "torch_version": str(torch.__version__),
        "transformers_version": (
            None if transformers_version is None else str(transformers_version)
        ),
        "python_version": platform.python_version(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "reproducibility": {
            "random_seed": int(config.get("seed", 42)),
            "optimizer": "AdamW",
            "learning_rate": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "batch_size": int(config["batch_size"]),
            "sae_release": config.get("sae_release"),
            "activation_bank_metadata": config.get("activation_bank_metadata"),
            "git_commit": config.get("git_commit"),
        },
    }


def train_conditioned_v2(
    model: nn.Module,
    sae: nn.Module,
    denoiser: ConditionedSteeringDenoiser,
    train_token_batches: Sequence[TokenBatch],
    validation_token_batches: Sequence[TokenBatch],
    all_directions: Tensor,
    split: Mapping[str, Any],
    normalization: Mapping[str, Any],
    config: Mapping[str, Any],
    hook_name: str,
    model_name: str,
    literal_calibration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Train V2 using TRAIN directions and validation-only checkpoint selection."""
    from src.experiment_v2 import validate_conditioned_checkpoint

    seed = int(config.get("seed", 42))
    seed_everything(seed)
    freeze_teacher_models(model, sae)
    train_ids = list(split["train"])
    validate_direction_ids_for_usage(train_ids, split, "training", require_complete_split=True)
    val_ids = list(split["val"])[: int(config["num_validation_directions"])]
    validate_direction_ids_for_usage(val_ids, split, "preliminary_evaluation")
    optimizer = torch.optim.AdamW(
        denoiser.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    max_steps = int(config["max_steps"])
    best_key: tuple[int, float] | None = None
    zero_grad_streak = 0
    history: list[dict[str, Any]] = []
    progress = tqdm(range(1, max_steps + 1), desc="Conditioned V2 training")
    for step in progress:
        tokens, mask = _unpack_batch(train_token_batches[(step - 1) % len(train_token_batches)])
        directions, ids, strengths, identity_directions = _sample_training_conditions(
            tokens.shape[0], train_ids, all_directions,
            config["training_strengths"], generator,
        )
        assert set(ids.detach().cpu().tolist()).issubset(set(train_ids))
        optimizer.zero_grad(set_to_none=True)
        losses = compute_v2_batch_losses(
            model, sae, denoiser, tokens, mask, directions, ids, strengths,
            identity_directions, normalization, config, hook_name,
        )
        losses["loss"].backward()
        gradients = gradient_diagnostics(denoiser)
        assert all(parameter.grad is None for parameter in model.parameters())
        if gradients["denoiser_gradient_norm"] == 0:
            zero_grad_streak += 1
        else:
            zero_grad_streak = 0
        if zero_grad_streak >= 3:
            raise RuntimeError("Denoiser gradient was zero for three consecutive V2 steps.")
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), float(config["gradient_clip_norm"]))
        optimizer.step()
        row = {"step": step, **{key: float(value.detach()) for key, value in losses.items()}, **gradients}
        history.append(row)
        if step == 1 or step % int(config["log_every"]) == 0:
            progress.write(json.dumps(row))

        validation_summary = None
        if step % int(config["validation_every"]) == 0 or step == max_steps:
            validation_summary = validate_conditioned_checkpoint(
                model, sae, denoiser, validation_token_batches,
                all_directions, split, val_ids,
                config["validation_strengths"], normalization, config, hook_name,
            )
            retention = float(validation_summary["mean_concept_retention"])
            delta_nll = float(validation_summary["mean_delta_nll"])
            feasible = int(retention >= float(config["retention_target"]))
            # Feasible checkpoints always outrank infeasible checkpoints. Within
            # each class, minimize downstream NLL plus a fixed constraint penalty.
            constrained_score = delta_nll + 1000.0 * max(
                0.0, float(config["retention_target"]) - retention
            )
            candidate_key = (-feasible, constrained_score)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                payload = _checkpoint_payload(
                    denoiser, optimizer, step, config, normalization, split,
                    hook_name, model_name, literal_calibration, validation_summary,
                )
                best_path = Path(config["best_checkpoint_path"])
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, best_path)
        last_payload = _checkpoint_payload(
            denoiser, optimizer, step, config, normalization, split,
            hook_name, model_name, literal_calibration, validation_summary,
        )
        last_path = Path(config["last_checkpoint_path"])
        last_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(last_payload, last_path)
    constraint_satisfied = best_key is not None and best_key[0] == -1
    selection_report = {
        "pipeline_version": V2_PIPELINE_VERSION,
        "selection_rule": (
            "require mean validation retention >= target, then minimize validation delta NLL; "
            "if infeasible minimize fixed constrained objective"
        ),
        "retention_target": float(config["retention_target"]),
        "retention_constraint_satisfied": constraint_satisfied,
        "test_used_for_selection": False,
    }
    selection_path = Path(config["best_checkpoint_path"]).with_suffix(".selection.json")
    selection_path.write_text(json.dumps(selection_report, indent=2) + "\n", encoding="utf-8")
    if not constraint_satisfied:
        print("WARNING: no V2 checkpoint met mean validation concept retention >= 0.8.")
    return history


def load_conditioned_v2_checkpoint(
    path: str | Path,
    device: str | torch.device,
    expected_split_hash: str | None = None,
) -> tuple[ConditionedSteeringDenoiser, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V2 checkpoint does not exist: {checkpoint_path}")

    # Version 31 checkpoints stored ``torch.__version__`` before it was
    # converted to ``str``.  PyTorch represents that value as TorchVersion,
    # which the restricted loader correctly rejects unless that exact harmless
    # metadata class is allow-listed.  The context keeps weights-only loading
    # enabled and does not permit arbitrary pickle execution.
    from torch.torch_version import TorchVersion

    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    if not isinstance(checkpoint, dict):
        raise TypeError("V2 checkpoint payload must be a dictionary.")
    assert checkpoint["checkpoint_version"] == V2_CHECKPOINT_VERSION
    assert checkpoint["pipeline_version"] == V2_PIPELINE_VERSION
    if expected_split_hash is not None:
        assert checkpoint["direction_split_hash"] == expected_split_hash
    architecture = checkpoint["architecture"]
    denoiser = ConditionedSteeringDenoiser(
        architecture["d_model"], architecture["hidden_dim"], architecture["conditioning_dim"]
    )
    denoiser.load_state_dict(checkpoint["model_state_dict"], strict=True)
    denoiser.to(device).eval()
    return denoiser, checkpoint


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
