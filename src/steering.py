"""Vectorized residual-stream steering operations."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn


TokenPositions = int | slice | Sequence[int] | Tensor


def _validate_inputs(h: Tensor, v: Tensor) -> None:
    assert h.ndim == 3, f"Expected h with shape [batch, seq, d_model], got {h.shape}."
    assert v.ndim == 1, f"Expected v with shape [d_model], got {v.shape}."
    assert h.shape[-1] == v.shape[0], (
        f"Direction size {v.shape[0]} does not match d_model {h.shape[-1]}."
    )
    assert h.is_floating_point() and v.is_floating_point()
    assert h.dtype == v.dtype, f"h has dtype {h.dtype}, but v has dtype {v.dtype}."
    assert h.device == v.device, f"h is on {h.device}, but v is on {v.device}."


def _validate_token_positions(h: Tensor, token_positions: TokenPositions) -> None:
    if isinstance(token_positions, Tensor):
        assert token_positions.ndim == 1, "token_positions must be a 1D tensor."
        assert token_positions.dtype in (torch.long, torch.bool), (
            "Tensor token_positions must have dtype torch.long or torch.bool."
        )
        assert token_positions.device == h.device, (
            f"token_positions is on {token_positions.device}, but h is on {h.device}."
        )
        if token_positions.dtype == torch.bool:
            assert token_positions.shape[0] == h.shape[1], (
                "Boolean token_positions must have one entry per sequence position."
            )


def _apply_to_positions(
    h: Tensor,
    token_positions: TokenPositions | None,
    transform: Callable[[Tensor], Tensor],
) -> Tensor:
    if token_positions is None:
        return transform(h)

    _validate_token_positions(h, token_positions)
    result = h.clone()
    selected = h[:, token_positions, :]
    modified = transform(selected)
    assert modified.shape == selected.shape
    result[:, token_positions, :] = modified
    return result


def normalize_direction(v: Tensor, eps: float = 1e-8) -> Tensor:
    """Return a unit-L2 direction without changing dtype or device."""
    assert v.ndim == 1, f"Expected v with shape [d_model], got {v.shape}."
    assert v.is_floating_point(), "Direction must be floating point."
    norm = torch.linalg.vector_norm(v)
    stability_eps = max(eps, torch.finfo(v.dtype).tiny)
    assert norm.item() > stability_eps, "Cannot normalize a near-zero direction."
    return v / norm


def raw_steering(
    h: Tensor,
    v: Tensor,
    alpha: float,
    token_positions: TokenPositions | None = None,
) -> Tensor:
    """Apply ``h_new = h + alpha * v`` at the selected token positions."""
    _validate_inputs(h, v)
    return _apply_to_positions(h, token_positions, lambda values: values + alpha * v)


def literal_raw_steering(
    h: Tensor,
    v: Tensor,
    alpha: float,
    token_positions: TokenPositions | None = None,
) -> Tensor:
    """Literal V2 baseline ``h' = h + alpha * unit(v)``.

    ``alpha`` has activation-norm units and is deliberately distinct from the
    dimensionless relative strength ``s`` used by :func:`relative_raw_steering`.
    """
    _validate_inputs(h, v)
    unit_v = normalize_direction(v)
    return _apply_to_positions(h, token_positions, lambda values: values + alpha * unit_v)


def relative_raw_steering(
    h: Tensor,
    v: Tensor,
    strength: float,
    token_positions: TokenPositions | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Apply the project's authoritative relative steering update.

    For every selected activation vector independently, the update is
    ``strength * ||h||_2 * (v / ||v||_2)``. Consequently the update norm is
    ``abs(strength) * ||h||_2`` for shapes ``[B, S, D]``. No tensor is moved,
    cast, or modified in place.
    """
    _validate_inputs(h, v)
    unit_v = normalize_direction(v, eps=eps)

    def transform(values: Tensor) -> Tensor:
        activation_norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
        return values + strength * activation_norm * unit_v

    return _apply_to_positions(h, token_positions, transform)


def relative_norm_preserving_steering(
    h: Tensor,
    v: Tensor,
    strength: float,
    token_positions: TokenPositions | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Apply relative steering and restore each selected activation norm."""
    _validate_inputs(h, v)
    stability_eps = max(eps, torch.finfo(h.dtype).tiny)
    unit_v = normalize_direction(v, eps=eps)

    def transform(values: Tensor) -> Tensor:
        original_norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
        steered = values + strength * original_norm * unit_v
        steered_norm = torch.linalg.vector_norm(steered, dim=-1, keepdim=True)
        return steered * (original_norm / steered_norm.clamp_min(stability_eps))

    return _apply_to_positions(h, token_positions, transform)


def norm_preserving_steering(
    h: Tensor,
    v: Tensor,
    alpha: float,
    token_positions: TokenPositions | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Apply raw steering, then restore each modified vector's original norm."""
    _validate_inputs(h, v)
    stability_eps = max(eps, torch.finfo(h.dtype).tiny)

    def transform(values: Tensor) -> Tensor:
        steered = values + alpha * v
        original_norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
        steered_norm = torch.linalg.vector_norm(steered, dim=-1, keepdim=True)
        scale = original_norm / steered_norm.clamp_min(stability_eps)
        return steered * scale

    return _apply_to_positions(h, token_positions, transform)


def tangent_steering(
    h: Tensor,
    v: Tensor,
    alpha: float,
    token_positions: TokenPositions | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Add the component of ``v`` tangent to the activation-norm sphere.

    The projected direction is not renormalized: the update is exactly
    ``alpha * v_perp`` at each position. Thus ``alpha`` uses the same units as
    raw steering, while nearly radial directions produce smaller updates.
    """
    _validate_inputs(h, v)
    stability_eps = max(eps, torch.finfo(h.dtype).tiny)

    def transform(values: Tensor) -> Tensor:
        squared_norm = values.square().sum(dim=-1, keepdim=True)
        projection_scale = (values * v).sum(dim=-1, keepdim=True)
        projection_scale = projection_scale / (squared_norm + stability_eps)
        v_perp = v - projection_scale * values
        return values + alpha * v_perp

    return _apply_to_positions(h, token_positions, transform)


def _run_denoiser(
    denoiser: nn.Module,
    x: Tensor,
    normalization_stats: Mapping[str, Any] | None,
    noise_level: Tensor | None = None,
) -> Tensor:
    if normalization_stats is None:
        if bool(getattr(denoiser, "condition_on_noise", False)):
            denoised = denoiser(x, noise_level=noise_level)
        else:
            denoised = denoiser(x)
    else:
        # Reuse the exact standardize -> denoise -> destandardize path used by
        # checkpoint inference rather than duplicating normalization logic.
        from src.train import denoise_activations

        denoised = denoise_activations(
            denoiser,
            x,
            normalization=normalization_stats,
            noise_level=noise_level,
        )
    assert isinstance(denoised, Tensor)
    assert denoised.shape == x.shape
    assert denoised.dtype == x.dtype
    assert denoised.device == x.device
    return denoised


def _inference_noise_level(
    clean: Tensor,
    corrupted: Tensor,
    normalization_stats: Mapping[str, Any] | None,
) -> Tensor:
    """Measure update L2 norm in the denoiser's training coordinates."""
    if normalization_stats is None:
        delta = corrupted - clean
    else:
        from src.utils import normalize_activations

        mean_value = normalization_stats["mean"]
        std_value = normalization_stats["std"]
        assert isinstance(mean_value, Tensor) and isinstance(std_value, Tensor)
        mean = mean_value.to(device=clean.device, dtype=clean.dtype)
        std = std_value.to(device=clean.device, dtype=clean.dtype)
        eps = float(normalization_stats.get("eps", 1e-6))
        clean_z = normalize_activations(clean, mean, std, eps)
        corrupted_z = normalize_activations(corrupted, mean, std, eps)
        delta = corrupted_z - clean_z
    return torch.linalg.vector_norm(delta, dim=-1)


def denoised_steering(
    h: Tensor,
    v: Tensor,
    alpha: float,
    denoiser: nn.Module,
    normalization_stats: Mapping[str, Any] | None = None,
) -> Tensor:
    """Apply raw steering and then denoise in the training coordinate system."""
    _validate_inputs(h, v)
    steered = h + alpha * v
    noise_level = _inference_noise_level(h, steered, normalization_stats)
    return _run_denoiser(denoiser, steered, normalization_stats, noise_level)


def relative_denoised_steering(
    h: Tensor,
    v: Tensor,
    strength: float,
    denoiser: nn.Module,
    normalization_stats: Mapping[str, Any] | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Apply relative raw steering and denoise using training normalization."""
    steered = relative_raw_steering(h, v, strength, eps=eps)
    noise_level = _inference_noise_level(h, steered, normalization_stats)
    return _run_denoiser(denoiser, steered, normalization_stats, noise_level)


def projected_denoised_steering(
    h: Tensor,
    v: Tensor,
    alpha: float,
    denoiser: nn.Module,
    normalization_stats: Mapping[str, Any] | None = None,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Denoise while preserving the component along the steering direction.

    If ``x = h + alpha * v`` and ``correction = D(x) - x``, this returns
    ``x + beta * correction_perp`` where ``correction_perp`` has zero component
    parallel to ``v`` up to numerical precision.
    """
    _validate_inputs(h, v)
    stability_eps = max(eps, torch.finfo(h.dtype).tiny)
    steered = h + alpha * v
    noise_level = _inference_noise_level(h, steered, normalization_stats)
    denoised = _run_denoiser(denoiser, steered, normalization_stats, noise_level)
    correction = denoised - steered
    direction_norm_squared = v.square().sum()
    parallel_scale = (correction * v).sum(dim=-1, keepdim=True)
    parallel_scale = parallel_scale / (direction_norm_squared + stability_eps)
    correction_perp = correction - parallel_scale * v
    return steered + beta * correction_perp


def relative_projected_denoised_steering(
    h: Tensor,
    v: Tensor,
    strength: float,
    denoiser: nn.Module,
    normalization_stats: Mapping[str, Any] | None = None,
    beta: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Relative steering followed by correction orthogonal to ``v``."""
    _validate_inputs(h, v)
    stability_eps = max(eps, torch.finfo(h.dtype).tiny)
    unit_v = normalize_direction(v, eps=eps)
    steered = relative_raw_steering(h, unit_v, strength, eps=eps)
    noise_level = _inference_noise_level(h, steered, normalization_stats)
    denoised = _run_denoiser(denoiser, steered, normalization_stats, noise_level)
    correction = denoised - steered
    parallel = (correction * unit_v).sum(dim=-1, keepdim=True) * unit_v
    correction_perp = correction - parallel
    assert torch.isfinite(correction_perp).all()
    return steered + beta * correction_perp


def incremental_relative_denoised_steering(
    h: Tensor,
    v: Tensor,
    strength: float,
    denoiser: nn.Module,
    n_steps: int,
    projected: bool = False,
    beta: float = 1.0,
    normalization_stats: Mapping[str, Any] | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Alternate relative steering increments and denoising corrections."""
    _validate_inputs(h, v)
    assert n_steps in {1, 2, 4, 8}, "n_steps must be one of 1, 2, 4, or 8."
    increment = strength / n_steps
    current = h
    for _ in range(n_steps):
        if projected:
            current = relative_projected_denoised_steering(
                current, v, increment, denoiser,
                normalization_stats=normalization_stats, beta=beta, eps=eps,
            )
        else:
            current = relative_denoised_steering(
                current, v, increment, denoiser,
                normalization_stats=normalization_stats, eps=eps,
            )
    return current


def incremental_denoised_steering(
    h: Tensor,
    v: Tensor,
    alpha: float,
    denoiser: nn.Module,
    n_steps: int,
    projected: bool = False,
    beta: float = 1.0,
    normalization_stats: Mapping[str, Any] | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Alternate small steering updates with denoising corrections.

    Supported step counts are 1, 2, 4, and 8. With one step this calls the
    corresponding one-shot denoised or projected-denoised implementation
    exactly, including its normalization path.
    """
    _validate_inputs(h, v)
    assert n_steps in {1, 2, 4, 8}, "n_steps must be one of 1, 2, 4, or 8."
    delta = alpha / n_steps
    current = h
    for _ in range(n_steps):
        if projected:
            current = projected_denoised_steering(
                current,
                v,
                delta,
                denoiser,
                normalization_stats=normalization_stats,
                beta=beta,
                eps=eps,
            )
        else:
            current = denoised_steering(
                current,
                v,
                delta,
                denoiser,
                normalization_stats=normalization_stats,
            )
    return current


def incremental_projected_denoised_steering(
    h: Tensor,
    v: Tensor,
    alpha: float,
    denoiser: nn.Module,
    n_steps: int,
    beta: float = 1.0,
    normalization_stats: Mapping[str, Any] | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Convenience wrapper for projected correction after every increment."""
    return incremental_denoised_steering(
        h,
        v,
        alpha,
        denoiser,
        n_steps,
        projected=True,
        beta=beta,
        normalization_stats=normalization_stats,
        eps=eps,
    )


def conditioned_kl_denoised_steering(
    h: Tensor,
    v: Tensor,
    strength: float,
    denoiser: nn.Module,
    normalization_stats: Mapping[str, Any],
    eps: float = 1e-8,
) -> Tensor:
    """Relative steering followed by the separate conditioned V2 denoiser."""
    _validate_inputs(h, v)
    from src.denoiser_v2 import apply_conditioned_denoiser

    unit_v = normalize_direction(v, eps=eps)
    steered = relative_raw_steering(h, unit_v, strength, eps=eps)
    output = apply_conditioned_denoiser(
        denoiser, steered, unit_v, strength, normalization_stats
    )
    assert output.shape == h.shape and output.dtype == h.dtype and output.device == h.device
    return output


def gated_conditioned_denoised_steering(
    h: Tensor,
    v: Tensor,
    strength: float,
    denoiser: nn.Module,
    normalization_stats: Mapping[str, Any],
    eps: float = 1e-8,
) -> Tensor:
    """Relative steering followed by the structurally gated V3 denoiser."""
    _validate_inputs(h, v)
    from src.denoiser_v3 import apply_gated_conditioned_denoiser

    unit_v = normalize_direction(v, eps=eps)
    steered = relative_raw_steering(h, unit_v, strength, eps=eps)
    output = apply_gated_conditioned_denoiser(
        denoiser, steered, unit_v, strength, normalization_stats
    )
    assert output.shape == h.shape and output.dtype == h.dtype and output.device == h.device
    return output


def hard_projected_gated_conditioned_steering(
    h: Tensor,
    v: Tensor,
    strength: float,
    denoiser: nn.Module,
    normalization_stats: Mapping[str, Any],
    eps: float = 1e-8,
) -> Tensor:
    """Apply V3 denoising but hard-remove correction parallel to ``v``.

    This is a control for the learned soft retention constraint. The raw
    relative steering update is unchanged; only the denoiser correction is
    projected so ``c_parallel = 0`` up to floating-point error.
    """
    _validate_inputs(h, v)
    unit_v = normalize_direction(v, eps=eps)
    steered = relative_raw_steering(h, unit_v, strength, eps=eps)
    from src.denoiser_v3 import apply_gated_conditioned_denoiser

    denoised = apply_gated_conditioned_denoiser(
        denoiser, steered, unit_v, strength, normalization_stats
    )
    correction = denoised - steered
    parallel = (correction * unit_v).sum(dim=-1, keepdim=True) * unit_v
    output = steered + correction - parallel
    assert output.shape == h.shape and torch.isfinite(output).all()
    return output


def apply_steering(
    h: Tensor,
    v: Tensor,
    alpha: float,
    method: str = "raw",
    **kwargs: object,
) -> Tensor:
    """Dispatch to one of the supported activation-steering methods."""
    methods = {
        "raw": raw_steering,
        "literal_raw": literal_raw_steering,
        "relative_raw": relative_raw_steering,
        "norm_preserving": norm_preserving_steering,
        "relative_norm_preserving": relative_norm_preserving_steering,
        "tangent": tangent_steering,
        "denoise": denoised_steering,
        "relative_denoise": relative_denoised_steering,
        "projected_denoise": projected_denoised_steering,
        "relative_projected_denoise": relative_projected_denoised_steering,
        "incremental_denoise": incremental_denoised_steering,
        "incremental_projected_denoise": incremental_projected_denoised_steering,
        "incremental_relative_denoise": incremental_relative_denoised_steering,
        "conditioned_kl_denoise": conditioned_kl_denoised_steering,
        "gated_conditioned_denoise": gated_conditioned_denoised_steering,
        "hard_projected_gated_conditioned_denoise": hard_projected_gated_conditioned_steering,
    }
    if method not in methods:
        supported = ", ".join(methods)
        raise ValueError(f"Unknown steering method {method!r}. Expected one of: {supported}.")
    return methods[method](h, v, alpha, **kwargs)
