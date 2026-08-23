"""Differentiable losses and diagnostics for conditioned downstream-aware V2."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def downstream_kl(
    clean_logits: Tensor,
    modified_logits: Tensor,
    prediction_mask: Tensor | None = None,
) -> Tensor:
    """Stable ``KL(p_clean || p_modified)`` with a detached clean teacher."""
    assert clean_logits.shape == modified_logits.shape and clean_logits.ndim == 3
    clean_log_probs = F.log_softmax(clean_logits.float(), dim=-1).detach()
    clean_probs = clean_log_probs.exp().detach()
    modified_log_probs = F.log_softmax(modified_logits.float(), dim=-1)
    per_position = F.kl_div(
        modified_log_probs, clean_probs, reduction="none"
    ).sum(dim=-1)
    if prediction_mask is None:
        return per_position.mean()
    assert prediction_mask.shape == per_position.shape
    mask = prediction_mask.to(device=per_position.device, dtype=per_position.dtype)
    assert mask.sum() > 0
    return (per_position * mask).sum() / mask.sum()


def retention_hinge_loss(
    raw_activation: Tensor,
    denoised_activation: Tensor,
    target: float = 0.8,
    threshold: float = 1e-3,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Normalized soft retention hinge, masked when raw activation is tiny."""
    assert raw_activation.shape == denoised_activation.shape
    assert 0 < target <= 1 and threshold >= 0 and eps > 0
    valid = raw_activation > threshold
    ratio = (denoised_activation + eps) / (raw_activation + eps)
    violation = F.relu(target - ratio).square()
    zero = denoised_activation.sum() * 0.0
    loss = violation[valid].mean() if valid.any() else zero
    mean_ratio = ratio[valid].mean() if valid.any() else zero
    fraction_violating = (ratio[valid] < target).float().mean() if valid.any() else zero
    return {
        "loss": loss,
        "ratio": ratio,
        "valid_mask": valid,
        "mean_retention_ratio": mean_ratio,
        "fraction_below_retention_threshold": fraction_violating,
        "retention_violation": loss,
    }


def relative_correction_loss(
    denoised: Tensor,
    steered: Tensor,
    token_mask: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Mean ``||D(x)-x||^2 / (||x||^2 + eps)`` over valid positions."""
    assert denoised.shape == steered.shape and denoised.ndim == 3
    values = (denoised - steered).square().sum(-1) / (steered.square().sum(-1) + eps)
    if token_mask is None:
        return values.mean()
    assert token_mask.shape == values.shape and token_mask.any()
    return values[token_mask.bool()].mean()


def correction_geometry(
    denoised: Tensor,
    steered: Tensor,
    direction: Tensor,
    eps: float = 1e-8,
) -> dict[str, Tensor]:
    """Decompose a correction into steering-parallel and orthogonal parts."""
    assert denoised.shape == steered.shape and denoised.ndim == 3
    if direction.ndim == 1:
        direction = direction.reshape(1, 1, -1)
    elif direction.ndim == 2:
        direction = direction[:, None, :]
    assert direction.shape[-1] == denoised.shape[-1]
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    correction = denoised - steered
    parallel = (correction * direction).sum(-1, keepdim=True) * direction
    orthogonal = correction - parallel
    # This is a diagnostic invariant, not an exact bitwise identity: the
    # subtraction above followed by addition can differ by a few ULPs when
    # correction and parallel have different scales.  Use a dtype-aware
    # tolerance so normal float32 round-off cannot abort a long holdout run,
    # while non-finite values or a material decomposition error still fail.
    finfo = torch.finfo(correction.dtype)
    torch.testing.assert_close(
        parallel + orthogonal,
        correction,
        rtol=max(1e-4, 16.0 * finfo.eps),
        atol=max(5e-6, 16.0 * finfo.eps),
        equal_nan=False,
        msg="Correction parallel/orthogonal decomposition is numerically inconsistent.",
    )
    correction_norm = correction.norm(dim=-1)
    parallel_norm = parallel.norm(dim=-1)
    orthogonal_norm = orthogonal.norm(dim=-1)
    cosine = (correction * direction).sum(-1) / correction_norm.clamp_min(eps)
    return {
        "correction": correction,
        "parallel": parallel,
        "orthogonal": orthogonal,
        "correction_norm": correction_norm,
        "correction_parallel_norm": parallel_norm,
        "correction_orthogonal_norm": orthogonal_norm,
        "correction_cosine_with_v": cosine,
        "parallel_fraction": parallel_norm / correction_norm.clamp_min(eps),
    }
