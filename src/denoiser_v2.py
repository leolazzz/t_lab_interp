"""Direction- and strength-conditioned residual denoiser for V2.

Inputs and outputs are standardized GPT-2 residual activations. The direction
condition must be the unit vector in standardized geometry, not raw ``v/std``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from src.utils import denormalize_activations, normalize_activations


def standardized_condition_direction(
    direction: Tensor,
    std: Tensor,
    eps: float = 1e-6,
    tolerance: float = 2e-4,
) -> Tensor:
    """Convert raw unit directions to unit directions in standardized geometry."""
    assert direction.ndim >= 1 and std.ndim == 1
    assert direction.shape[-1] == std.numel()
    assert direction.device == std.device and direction.dtype == std.dtype
    assert direction.is_floating_point() and eps > 0 and tolerance > 0
    scaled = direction / std.clamp_min(eps)
    assert torch.isfinite(scaled).all(), "Non-finite v/std conditioning direction."
    norms = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    assert (norms > eps).all(), "Cannot normalize a near-zero conditioned direction."
    normalized = scaled / norms
    normalized_norms = torch.linalg.vector_norm(normalized, dim=-1)
    assert torch.isfinite(normalized).all()
    assert abs(float(normalized_norms.mean().detach()) - 1.0) < tolerance
    return normalized


class ConditionedSteeringDenoiser(nn.Module):
    """Small residual MLP with two FiLM blocks conditioned on ``(v_z, s)``.

    ``x`` has shape ``[..., d_model]``. ``steering_direction`` can be
    ``[d_model]`` or broadcast-compatible ``[..., d_model]``. ``strength`` is
    broadcast over the leading dimensions. Both FiLM heads and the final
    correction projection are zero-initialized, so the initialized model is
    exactly the identity.
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 768,
        conditioning_dim: int = 128,
    ) -> None:
        super().__init__()
        assert d_model > 0 and hidden_dim > 0 and conditioning_dim > 0
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.conditioning_dim = conditioning_dim
        self.input_norm = nn.LayerNorm(d_model)
        self.input_projection = nn.Linear(d_model, hidden_dim)
        self.hidden_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, d_model)
        self.direction_mlp = nn.Sequential(
            nn.Linear(d_model, conditioning_dim), nn.SiLU(),
            nn.Linear(conditioning_dim, hidden_dim),
        )
        self.strength_mlp = nn.Sequential(
            nn.Linear(3, conditioning_dim), nn.SiLU(),
            nn.Linear(conditioning_dim, hidden_dim),
        )
        self.film_head_1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.film_head_2 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.activation = nn.SiLU()
        for layer in (self.film_head_1, self.film_head_2, self.output_projection):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _broadcast_direction(self, x: Tensor, direction: Tensor) -> Tensor:
        assert direction.shape[-1] == self.d_model
        assert direction.device == x.device and direction.dtype == x.dtype
        if direction.ndim == 1:
            direction = direction.reshape(*([1] * (x.ndim - 1)), self.d_model)
        elif direction.ndim == 2 and x.ndim >= 3 and direction.shape[0] == x.shape[0]:
            direction = direction.reshape(
                direction.shape[0], *([1] * (x.ndim - 2)), self.d_model
            )
        try:
            return torch.broadcast_to(direction, x.shape)
        except RuntimeError as error:
            raise AssertionError(
                f"Direction shape {tuple(direction.shape)} cannot broadcast to {tuple(x.shape)}."
            ) from error

    def _broadcast_strength(self, x: Tensor, strength: Tensor | float) -> Tensor:
        leading = x.shape[:-1]
        if isinstance(strength, (float, int)):
            return x.new_full((*leading, 1), float(strength))
        assert strength.device == x.device and strength.dtype == x.dtype
        if strength.ndim > 0 and strength.shape[-1] == 1:
            strength = strength.squeeze(-1)
        if strength.ndim == 1 and x.ndim >= 3 and strength.shape[0] == x.shape[0]:
            strength = strength.reshape(strength.shape[0], *([1] * (x.ndim - 2)))
        try:
            return torch.broadcast_to(strength, leading).unsqueeze(-1)
        except RuntimeError as error:
            raise AssertionError(
                f"Strength shape {tuple(strength.shape)} cannot broadcast to {leading}."
            ) from error

    @staticmethod
    def _film(hidden: Tensor, parameters: Tensor) -> Tensor:
        gamma, beta = parameters.chunk(2, dim=-1)
        return (1.0 + gamma) * hidden + beta

    def forward(
        self,
        x: Tensor,
        steering_direction: Tensor,
        strength: Tensor | float,
    ) -> Tensor:
        assert x.ndim >= 2 and x.shape[-1] == self.d_model and x.is_floating_point()
        direction = self._broadcast_direction(x, steering_direction)
        direction_norm = torch.linalg.vector_norm(direction, dim=-1)
        assert torch.isfinite(direction).all()
        assert torch.allclose(
            direction_norm.mean(), direction_norm.new_tensor(1.0), rtol=2e-4, atol=2e-4
        ), "Conditioned direction must have unit norm in standardized geometry."
        strength_value = self._broadcast_strength(x, strength)
        assert torch.isfinite(strength_value).all() and (strength_value >= 0).all()
        strength_features = torch.cat(
            (strength_value, torch.log1p(strength_value), strength_value.square()), dim=-1
        )
        condition = self.direction_mlp(direction) + self.strength_mlp(strength_features)
        hidden = self.activation(self.input_projection(self.input_norm(x)))
        hidden = self._film(hidden, self.film_head_1(condition))
        hidden = self.activation(self.hidden_projection(hidden))
        hidden = self._film(hidden, self.film_head_2(condition))
        output = x + self.output_projection(hidden)
        assert output.shape == x.shape and torch.isfinite(output).all()
        return output

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def apply_conditioned_denoiser(
    denoiser: ConditionedSteeringDenoiser,
    steered_raw: Tensor,
    raw_direction: Tensor,
    strength: Tensor | float,
    normalization: Mapping[str, Any],
) -> Tensor:
    """Apply the exact train/inference normalization and conditioning path."""
    mean_value, std_value = normalization["mean"], normalization["std"]
    assert isinstance(mean_value, Tensor) and isinstance(std_value, Tensor)
    mean = mean_value.to(device=steered_raw.device, dtype=steered_raw.dtype)
    std = std_value.to(device=steered_raw.device, dtype=steered_raw.dtype)
    eps = float(normalization.get("eps", 1e-6))
    standardized = normalize_activations(steered_raw, mean, std, eps)
    direction = raw_direction
    if direction.ndim == 1:
        direction = direction / direction.norm().clamp_min(eps)
    else:
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(eps)
    direction_z = standardized_condition_direction(direction, std, eps)
    output_z = denoiser(standardized, direction_z, strength)
    return denormalize_activations(output_z, mean, std, eps)


@torch.inference_mode()
def sanity_check_conditioned_identity() -> None:
    model = ConditionedSteeringDenoiser(d_model=8, hidden_dim=12, conditioning_dim=4)
    x = torch.randn(2, 3, 8)
    direction = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    output = model(x, direction, torch.zeros(2))
    torch.testing.assert_close(output, x, rtol=0.0, atol=0.0)
