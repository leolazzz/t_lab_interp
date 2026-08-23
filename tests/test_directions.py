import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.directions import (
    extract_decoder_directions,
    get_or_create_direction_split,
    load_direction_split,
    model_sae_compatibility_report,
    select_directions,
    split_direction_ids,
    validate_direction_ids_for_usage,
)


class DirectionTests(unittest.TestCase):
    def test_decoder_directions_are_unit_normalized(self) -> None:
        sae = SimpleNamespace(W_dec=torch.randn(12, 6))
        directions = extract_decoder_directions(sae, model_d_model=6)
        self.assertEqual(directions.shape, (12, 6))
        torch.testing.assert_close(
            directions.norm(dim=-1), torch.ones(12), atol=1e-6, rtol=1e-6
        )

    def test_split_is_deterministic_disjoint_and_persisted(self) -> None:
        first = split_direction_ids(100, 20, 10, 10, seed=7)
        second = split_direction_ids(100, 20, 10, 10, seed=7)
        self.assertEqual(first, second)
        self.assertTrue(set(first[0]).isdisjoint(first[1]))
        self.assertTrue(set(first[0]).isdisjoint(first[2]))
        self.assertTrue(set(first[1]).isdisjoint(first[2]))

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "direction_split.json"
            created = get_or_create_direction_split(100, 20, 10, 10, 7, path)
            reused = get_or_create_direction_split(100, 20, 10, 10, 7, path)
            self.assertEqual(created, reused)
            payload = load_direction_split(path)
            self.assertEqual(payload["train"], created[0])

    def test_select_directions_uses_requested_rows(self) -> None:
        directions = torch.arange(30, dtype=torch.float32).reshape(5, 6)
        selected = select_directions(directions, [3, 1])
        torch.testing.assert_close(selected, directions[[3, 1]])

    def test_usage_policy_rejects_split_leakage(self) -> None:
        split = {
            "num_features": 6,
            "train": [0, 1],
            "val": [2, 3],
            "test": [4, 5],
        }
        validate_direction_ids_for_usage([0, 1], split, "training", True)
        validate_direction_ids_for_usage([2], split, "hyperparameter_selection")
        validate_direction_ids_for_usage([5], split, "final_evaluation")
        with self.assertRaises(AssertionError):
            validate_direction_ids_for_usage([4], split, "preliminary_evaluation")

    def test_sae_preprocessing_is_checked_against_load_provenance(self) -> None:
        model = SimpleNamespace(
            cfg=SimpleNamespace(d_model=6),
            _steering_denoiser_load_kwargs={"center_writing_weights": True},
        )
        sae = SimpleNamespace(
            W_dec=torch.randn(12, 6),
            cfg=SimpleNamespace(metadata={
                "hook_name": "blocks.6.hook_resid_pre",
                "model_from_pretrained_kwargs": {
                    "center_writing_weights": True,
                },
            }),
        )
        report = model_sae_compatibility_report(
            model, sae, "blocks.6.hook_resid_pre"
        )
        self.assertTrue(report["compatible"])
        self.assertEqual(
            report["compared_model_cfg"]["center_writing_weights"]["source"],
            "load_kwargs",
        )

    def test_unknown_required_sae_preprocessing_fails(self) -> None:
        model = SimpleNamespace(cfg=SimpleNamespace(d_model=6))
        sae = SimpleNamespace(
            W_dec=torch.randn(12, 6),
            cfg=SimpleNamespace(metadata={
                "hook_name": "blocks.6.hook_resid_pre",
                "model_from_pretrained_kwargs": {
                    "center_writing_weights": True,
                },
            }),
        )
        with self.assertRaisesRegex(AssertionError, "Cannot verify required"):
            model_sae_compatibility_report(
                model, sae, "blocks.6.hook_resid_pre"
            )


if __name__ == "__main__":
    unittest.main()
