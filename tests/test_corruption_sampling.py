import tempfile
import unittest
from pathlib import Path

import torch

from src.directions import direction_split_hash
from src.utils import PIPELINE_VERSION

from src.train import (
    SAEDirectionCorruptionSampler,
    directions_to_standardized_space,
    direction_sampling_probabilities,
    load_train_direction_scores,
    make_fluency_sensitive_corruption_sampler,
)


class CorruptionSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.num_features = 100
        self.train_ids = list(range(80))
        self.split = {
            "num_features": self.num_features,
            "train": self.train_ids,
            "val": list(range(80, 90)),
            "test": list(range(90, 100)),
        }
        self.directions = torch.randn(80, 6)

    def test_score_proportional_sampling_favors_harmful_directions(self) -> None:
        scores = torch.linspace(0.1, 1.0, 80)
        sampler = SAEDirectionCorruptionSampler(
            self.directions,
            self.train_ids,
            self.split,
            relative_magnitude_min=0.2,
            relative_magnitude_max=0.8,
            sampling_mode="score_proportional",
            scores=scores,
            gamma=0.5,
            clip_quantile=None,
        )
        generator = torch.Generator().manual_seed(4)
        sampled_ids = sampler.sample_direction_ids(10_000, generator)
        counts = torch.bincount(sampled_ids, minlength=100).float()
        self.assertGreater(counts[60:80].mean(), counts[:20].mean())
        self.assertGreater((counts[:80] > 0).sum().item(), 70)
        self.assertTrue(set(sampled_ids.tolist()).issubset(set(self.train_ids)))

    def test_relative_corruption_has_expected_magnitude(self) -> None:
        sampler = SAEDirectionCorruptionSampler(
            self.directions,
            self.train_ids,
            self.split,
            relative_magnitude_min=0.5,
            relative_magnitude_max=0.5,
        )
        clean = torch.randn(16, 6)
        corrupted, magnitude = sampler(clean, torch.Generator().manual_seed(1))
        expected = 0.5 * clean.norm(dim=-1)
        torch.testing.assert_close(magnitude, expected)
        torch.testing.assert_close((corrupted - clean).norm(dim=-1), expected)

    def test_probability_modes_are_finite_and_normalized(self) -> None:
        scores = torch.tensor([0.0, 1.0, 1000.0])
        for mode in ("uniform", "score_proportional", "rank_based"):
            probabilities = direction_sampling_probabilities(
                scores, mode=mode, gamma=0.5, clip_quantile=0.9
            )
            self.assertTrue(torch.isfinite(probabilities).all())
            torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))

    def test_score_loader_rejects_non_train_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "scores.pt"
            torch.save(
                {
                    "pipeline_version": PIPELINE_VERSION,
                    "direction_split_hash": direction_split_hash(self.split),
                    "split": "train",
                    "direction_ids": torch.tensor(self.train_ids),
                    "mean_kl": torch.linspace(0.0, 1.0, len(self.train_ids)),
                },
                path,
            )
            scores = load_train_direction_scores(path, self.train_ids, self.split)
            self.assertEqual(scores.shape, (80,))
            with self.assertRaises(AssertionError):
                load_train_direction_scores(
                    path, self.train_ids[:-1] + [80], self.split
                )

    def test_standardization_maps_and_renormalizes_directions(self) -> None:
        directions = torch.tensor([[1.0, 1.0]])
        transformed = directions_to_standardized_space(
            directions,
            {"mean": torch.zeros(2), "std": torch.tensor([1.0, 2.0])},
        )
        expected = torch.tensor([[1.0, 0.5]])
        expected = expected / expected.norm(dim=-1, keepdim=True)
        torch.testing.assert_close(transformed, expected)

    def test_main_mixture_uses_configured_60_25_15_weights(self) -> None:
        all_directions = torch.randn(100, 6)
        with tempfile.TemporaryDirectory() as temporary_directory:
            score_path = Path(temporary_directory) / "scores.pt"
            torch.save(
                {
                    "pipeline_version": PIPELINE_VERSION,
                    "direction_split_hash": direction_split_hash(self.split),
                    "split": "train",
                    "direction_ids": torch.tensor(self.train_ids),
                    "mean_kl": torch.linspace(0.1, 1.0, len(self.train_ids)),
                },
                score_path,
            )
            config = {
                "noise": {
                    "sae_relative_magnitude_min": 0.1,
                    "sae_relative_magnitude_max": 1.0,
                    "fluency_direction_sampling_mode": "score_proportional",
                    "score_gamma": 0.5,
                    "score_eps": 1e-8,
                    "score_clip_quantile": 0.99,
                    "fluency_weighted_sae_probability": 0.60,
                    "fluency_uniform_sae_probability": 0.25,
                    "fluency_gaussian_probability": 0.15,
                    "calibrated_magnitude_min": 0.1,
                    "calibrated_magnitude_max": 1.0,
                }
            }
            mixture = make_fluency_sensitive_corruption_sampler(
                all_directions,
                self.split,
                config,
                scores_path=score_path,
                normalization_stats={"mean": torch.zeros(6), "std": torch.ones(6)},
            )
            torch.testing.assert_close(
                mixture.probabilities,
                torch.tensor([0.60, 0.25, 0.15], dtype=torch.float64),
            )


if __name__ == "__main__":
    unittest.main()
