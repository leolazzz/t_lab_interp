import unittest

import torch
from torch import nn

from src.steering import (
    apply_steering,
    denoised_steering,
    incremental_denoised_steering,
    normalize_direction,
    norm_preserving_steering,
    projected_denoised_steering,
    raw_steering,
    relative_raw_steering,
    tangent_steering,
)


class AddCorrection(nn.Module):
    def __init__(self, correction: torch.Tensor) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=correction.dtype))
        self.register_buffer("correction", correction)

    def forward(self, x, noise_level=None):
        return x + self.correction


class RecordNoise(nn.Module):
    condition_on_noise = True

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=dtype))
        self.seen = None

    def forward(self, x, noise_level=None):
        self.seen = noise_level.detach().clone()
        return x


class SteeringTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.h = torch.randn(2, 4, 6, dtype=torch.float64)
        self.v = normalize_direction(torch.randn(6, dtype=torch.float64))

    def test_alpha_zero_returns_h(self) -> None:
        for method in ("raw", "norm_preserving", "tangent"):
            actual = apply_steering(self.h, self.v, alpha=0.0, method=method)
            torch.testing.assert_close(actual, self.h)

    def test_raw_steering_matches_formula_and_selected_positions(self) -> None:
        alpha = 1.5
        actual = raw_steering(self.h, self.v, alpha, token_positions=[1, 3])
        expected = self.h.clone()
        expected[:, [1, 3], :] += alpha * self.v
        torch.testing.assert_close(actual, expected)

    def test_relative_steering_update_has_requested_per_token_norm(self) -> None:
        strength = 0.4
        original = self.h.clone()
        actual = relative_raw_steering(
            self.h, 7.0 * self.v, strength, token_positions=[1, 3]
        )
        torch.testing.assert_close(self.h, original)
        update_norm = (actual[:, [1, 3]] - self.h[:, [1, 3]]).norm(dim=-1)
        expected = strength * self.h[:, [1, 3]].norm(dim=-1)
        torch.testing.assert_close(update_norm, expected)
        torch.testing.assert_close(actual[:, [0, 2]], self.h[:, [0, 2]])

    def test_norm_preserving_steering_preserves_norms(self) -> None:
        actual = norm_preserving_steering(self.h, self.v, alpha=3.0)
        torch.testing.assert_close(
            torch.linalg.vector_norm(actual, dim=-1),
            torch.linalg.vector_norm(self.h, dim=-1),
        )

    def test_tangent_update_is_orthogonal_to_h(self) -> None:
        alpha = 2.0
        actual = tangent_steering(self.h, self.v, alpha=alpha)
        tangent_component = (actual - self.h) / alpha
        inner_products = (self.h * tangent_component).sum(dim=-1)
        torch.testing.assert_close(
            inner_products,
            torch.zeros_like(inner_products),
            atol=1e-7,
            rtol=0.0,
        )

    def test_projected_beta_zero_equals_raw_steering(self) -> None:
        denoiser = AddCorrection(self.v + torch.ones_like(self.v))
        actual = projected_denoised_steering(
            self.h, self.v, alpha=1.2, denoiser=denoiser, beta=0.0
        )
        expected = raw_steering(self.h, self.v, alpha=1.2)
        torch.testing.assert_close(actual, expected)

    def test_projected_correction_is_orthogonal_to_direction(self) -> None:
        denoiser = AddCorrection(2.0 * self.v + torch.arange(6, dtype=self.h.dtype))
        raw = raw_steering(self.h, self.v, alpha=0.5)
        projected = projected_denoised_steering(
            self.h, self.v, alpha=0.5, denoiser=denoiser
        )
        projection = ((projected - raw) * self.v).sum(dim=-1)
        torch.testing.assert_close(
            projection, torch.zeros_like(projection), atol=1e-7, rtol=0.0
        )

    def test_denoised_alpha_zero_and_shapes(self) -> None:
        identity = nn.Identity()
        actual = denoised_steering(self.h, self.v, alpha=0.0, denoiser=identity)
        self.assertEqual(actual.shape, self.h.shape)
        torch.testing.assert_close(actual, self.h)

    def test_denoised_steering_uses_checkpoint_normalization(self) -> None:
        denoiser = AddCorrection(torch.ones(6, dtype=self.h.dtype))
        stats = {
            "mean": torch.arange(6, dtype=torch.float32),
            "std": torch.full((6,), 2.0, dtype=torch.float32),
            "eps": 1e-6,
        }
        actual = apply_steering(
            self.h,
            self.v,
            alpha=0.0,
            method="denoise",
            denoiser=denoiser,
            normalization_stats=stats,
        )
        torch.testing.assert_close(actual, self.h + 2.0)

    def test_conditioned_inference_receives_standardized_update_norm(self) -> None:
        denoiser = RecordNoise(self.h.dtype)
        stats = {
            "mean": torch.zeros(6, dtype=torch.float32),
            "std": torch.arange(1, 7, dtype=torch.float32),
            "eps": 1e-6,
        }
        actual = apply_steering(
            self.h, self.v, 0.5, method="relative_denoise",
            denoiser=denoiser, normalization_stats=stats,
        )
        raw = apply_steering(self.h, self.v, 0.5, method="relative_raw")
        expected_level = ((raw - self.h) / stats["std"].to(self.h)).norm(dim=-1)
        torch.testing.assert_close(denoiser.seen, expected_level)
        torch.testing.assert_close(actual, raw)

    def test_incremental_one_step_matches_one_shot_methods(self) -> None:
        denoiser = AddCorrection(torch.arange(6, dtype=self.h.dtype))
        one_shot = denoised_steering(self.h, self.v, 1.3, denoiser)
        incremental = incremental_denoised_steering(
            self.h, self.v, 1.3, denoiser, n_steps=1
        )
        torch.testing.assert_close(incremental, one_shot)

        projected_one_shot = projected_denoised_steering(
            self.h, self.v, 1.3, denoiser, beta=0.7
        )
        projected_incremental = apply_steering(
            self.h,
            self.v,
            1.3,
            method="incremental_projected_denoise",
            denoiser=denoiser,
            n_steps=1,
            beta=0.7,
        )
        torch.testing.assert_close(projected_incremental, projected_one_shot)

    def test_incremental_supported_step_counts_preserve_shape(self) -> None:
        identity = nn.Identity()
        expected = raw_steering(self.h, self.v, alpha=2.0)
        for n_steps in (1, 2, 4, 8):
            actual = incremental_denoised_steering(
                self.h, self.v, 2.0, identity, n_steps=n_steps
            )
            self.assertEqual(actual.shape, self.h.shape)
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
