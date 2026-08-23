"""Structurally identity-preserving conditioned denoiser for final V3."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from src.denoiser_v2 import standardized_condition_direction
from src.utils import denormalize_activations, normalize_activations


class GatedConditionedDenoiser(nn.Module):
    """FiLM-conditioned correction network with exact ``D(x, v, 0) = x``.

    The correction gate is ``g(s) = s / (s + gate_scale)`` for non-negative
    relative steering strength.  The explicit zero branch makes the returned
    tensor exactly the input at ``s=0`` rather than relying on a learned
    identity penalty.
    """

    def __init__(
        self,
        d_model: int = 768,
        hidden_dim: int = 768,
        conditioning_dim: int = 128,
        gate_scale: float = 0.25,
    ) -> None:
        super().__init__()
        assert d_model > 0 and hidden_dim > 0 and conditioning_dim > 0
        assert gate_scale > 0
        self.d_model = int(d_model)
        self.hidden_dim = int(hidden_dim)
        self.conditioning_dim = int(conditioning_dim)
        self.gate_scale = float(gate_scale)
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

    def _direction(self, x: Tensor, direction: Tensor) -> Tensor:
        assert direction.shape[-1] == self.d_model
        assert direction.device == x.device and direction.dtype == x.dtype
        if direction.ndim == 1:
            direction = direction.reshape(*([1] * (x.ndim - 1)), self.d_model)
        elif direction.ndim == 2 and x.ndim >= 3 and direction.shape[0] == x.shape[0]:
            direction = direction.reshape(
                direction.shape[0], *([1] * (x.ndim - 2)), self.d_model
            )
        try:
            result = torch.broadcast_to(direction, x.shape)
        except RuntimeError as error:
            raise AssertionError("Condition direction is not broadcast-compatible with x.") from error
        norms = result.norm(dim=-1)
        assert torch.allclose(norms.mean(), norms.new_tensor(1.0), atol=2e-4, rtol=2e-4)
        return result

    def _strength(self, x: Tensor, strength: Tensor | float) -> Tensor:
        leading = x.shape[:-1]
        if isinstance(strength, (float, int)):
            result = x.new_full((*leading, 1), float(strength))
        else:
            assert strength.device == x.device and strength.dtype == x.dtype
            value = strength.squeeze(-1) if strength.ndim and strength.shape[-1] == 1 else strength
            if value.ndim == 1 and x.ndim >= 3 and value.shape[0] == x.shape[0]:
                value = value.reshape(value.shape[0], *([1] * (x.ndim - 2)))
            try:
                result = torch.broadcast_to(value, leading).unsqueeze(-1)
            except RuntimeError as error:
                raise AssertionError("Strength is not broadcast-compatible with x.") from error
        assert torch.isfinite(result).all() and (result >= 0).all()
        return result

    @staticmethod
    def _film(hidden: Tensor, parameters: Tensor) -> Tensor:
        gamma, beta = parameters.chunk(2, dim=-1)
        return (1.0 + gamma) * hidden + beta

    def forward(self, x: Tensor, steering_direction: Tensor, strength: Tensor | float) -> Tensor:
        assert x.ndim >= 2 and x.shape[-1] == self.d_model and x.is_floating_point()
        direction = self._direction(x, steering_direction)
        value = self._strength(x, strength)
        features = torch.cat((value, torch.log1p(value), value.square()), dim=-1)
        condition = self.direction_mlp(direction) + self.strength_mlp(features)
        hidden = self.activation(self.input_projection(self.input_norm(x)))
        hidden = self._film(hidden, self.film_head_1(condition))
        hidden = self.activation(self.hidden_projection(hidden))
        hidden = self._film(hidden, self.film_head_2(condition))
        correction = self.output_projection(hidden)
        gate = torch.where(
            value == 0,
            torch.zeros_like(value),
            value / (value + self.gate_scale),
        )
        output = x + gate * correction
        assert output.shape == x.shape and torch.isfinite(output).all()
        return output

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def apply_gated_conditioned_denoiser(
    denoiser: GatedConditionedDenoiser,
    steered_raw: Tensor,
    raw_direction: Tensor,
    strength: Tensor | float,
    normalization: Mapping[str, Any],
) -> Tensor:
    """Apply the shared train/inference normalization path for V3."""
    if isinstance(strength, (float, int)) and float(strength) == 0.0:
        # Preserve exact raw-domain identity as well as standardized-domain
        # identity; a normalize/denormalize round trip can otherwise round.
        return steered_raw
    mean_value, std_value = normalization["mean"], normalization["std"]
    assert isinstance(mean_value, Tensor) and isinstance(std_value, Tensor)
    mean = mean_value.to(device=steered_raw.device, dtype=steered_raw.dtype)
    std = std_value.to(device=steered_raw.device, dtype=steered_raw.dtype)
    eps = float(normalization.get("eps", 1e-6))
    standardized = normalize_activations(steered_raw, mean, std, eps)
    direction = raw_direction / raw_direction.norm(dim=-1, keepdim=True).clamp_min(eps) \
        if raw_direction.ndim > 1 else raw_direction / raw_direction.norm().clamp_min(eps)
    direction_z = standardized_condition_direction(direction, std, eps)
    output_z = denoiser(standardized, direction_z, strength)
    output = denormalize_activations(output_z, mean, std, eps)
    if isinstance(strength, Tensor):
        zero_mask = strength == 0
        if zero_mask.ndim and zero_mask.shape[-1] == 1:
            zero_mask = zero_mask.squeeze(-1)
        while zero_mask.ndim < output.ndim:
            zero_mask = zero_mask.unsqueeze(-1)
        output = torch.where(zero_mask, steered_raw, output)
    assert output.shape == steered_raw.shape
    return output


@torch.inference_mode()
def structural_identity_gate() -> float:
    """Return and assert the maximum exact-zero-strength identity error."""
    model = GatedConditionedDenoiser(8, 12, 4)
    x = torch.randn(2, 3, 8)
    direction = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    error = float((model(x, direction, torch.zeros(2)) - x).abs().max())
    assert error == 0.0
    return error
