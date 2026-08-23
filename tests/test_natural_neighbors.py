import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from src.metrics import (
    NaturalNeighborIndex,
    build_neighbor_diagnostic_frame,
    denoiser_nearest_clean_cosine,
    spearman_neighbor_correlations,
)
from src.utils import generate_standard_figures


class NaturalNeighborTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(4)
        self.clean = torch.randn(80, 6, generator=generator)
        self.index = NaturalNeighborIndex(
            n_components=4,
            n_neighbors=3,
            max_fit_samples=80,
            seed=7,
        ).fit(self.clean)

    def test_distance_mahalanobis_and_serialization(self) -> None:
        near = self.clean[:5] + 0.01
        far = self.clean[:5] + 10.0
        self.assertLess(
            self.index.knn_distance(near),
            self.index.knn_distance(far),
        )
        self.assertGreater(self.index.mahalanobis_distance(far), 0.0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "neighbors.pkl"
            self.index.save(path)
            loaded = NaturalNeighborIndex.load(path)
            self.assertAlmostEqual(
                loaded.knn_distance(near), self.index.knn_distance(near)
            )

    def test_correction_alignment_and_correlations(self) -> None:
        corrupted = self.clean[:4] + 0.2
        nearest = torch.from_numpy(
            self.index.nearest_clean_activations(corrupted)
        )
        cosine = denoiser_nearest_clean_cosine(corrupted, nearest, self.index)
        torch.testing.assert_close(cosine, torch.ones_like(cosine), atol=1e-5, rtol=0)

        metadata = pd.DataFrame(
            {
                "method": ["raw", "raw"],
                "alpha": [1.0, 2.0],
                "direction_id": [3, 3],
                "delta_nll": [0.1, 0.9],
                "kl": [0.2, 1.1],
            }
        )
        frame = build_neighbor_diagnostic_frame(
            metadata,
            [self.clean[:3], self.clean[:3] + 5.0],
            self.index,
            k=3,
        )
        correlations = spearman_neighbor_correlations(frame)
        self.assertAlmostEqual(
            correlations["knn_distance_vs_delta_nll"], 1.0
        )
        self.assertAlmostEqual(correlations["knn_distance_vs_kl"], 1.0)

    def test_standard_figures_are_saved(self) -> None:
        methods = [
            "raw",
            "gaussian_denoiser",
            "sae_denoiser",
            "fluency_denoiser",
        ]
        rows = []
        for method_index, method in enumerate(methods):
            for alpha in (1.0, 2.0):
                rows.append(
                    {
                        "method": method,
                        "alpha": alpha,
                        "delta_nll": 0.1 * alpha + method_index * 0.01,
                        "concept_score": alpha - method_index * 0.05,
                        "activation_norm_ratio": 1.0 + 0.01 * alpha,
                        "n_steps": 2 if method == "fluency_denoiser" else None,
                    }
                )
        results = pd.DataFrame(rows)
        diagnostics = pd.DataFrame(
            {
                "method": ["raw", "fluency_denoiser"],
                "knn_distance": [0.2, 0.1],
                "delta_nll": [0.4, 0.2],
                "kl": [0.3, 0.1],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            created = generate_standard_figures(
                results,
                temporary_directory,
                damage_scores=torch.tensor([0.1, 0.2, 0.5]),
                neighbor_diagnostics=diagnostics,
            )
            self.assertEqual(len(created), 7)
            self.assertTrue(all(path.exists() for path in created.values()))


if __name__ == "__main__":
    unittest.main()
