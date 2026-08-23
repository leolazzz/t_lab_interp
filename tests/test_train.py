import tempfile
import unittest
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from src.denoiser import ResidualDenoiser
from src.train import (
    CORRUPTION_SAE_CALIBRATED,
    SAEStandardizedUnitCorruptionSampler,
    assert_calibrated_training_batch_math,
    calibrated_sampler_math_gate,
    correction_diagnostics,
    gaussian_corruption,
    load_denoiser_checkpoint,
    prepare_training_corruption_batch,
    train_denoiser,
)


def test_calibrated_sampler_math_gate() -> None:
    split = {"num_features": 4, "train": [0, 1, 2], "val": [3], "test": []}
    sampler = SAEStandardizedUnitCorruptionSampler(
        torch.randn(3, 8), [0, 1, 2], split, torch.ones(8), 1.0, 2.0
    )
    table = calibrated_sampler_math_gate(sampler, 8, magnitudes=[1.0, 2.0])
    assert (table["relative_error"] < 1e-4).all()


def test_real_batch_math_and_correction_invariants() -> None:
    d_model = 768
    batch = 6
    clean = torch.randn(batch, d_model)
    split = {"num_features": 5, "train": [0, 1, 2], "val": [3], "test": [4]}
    std = torch.rand(d_model) + 0.2
    sampler = SAEStandardizedUnitCorruptionSampler(
        torch.randn(3, d_model), split["train"], split, std,
        3.0, 3.0,
    )
    clean_z, sample = prepare_training_corruption_batch(
        clean, sampler, torch.zeros(d_model), std,
        torch.Generator().manual_seed(4), identity_probability=0.0,
    )
    summary = assert_calibrated_training_batch_math(clean_z, sample)
    assert summary["num_sae"] == batch
    assert abs(summary["mean_actual_norm"] - 3.0) < 1e-5
    assert abs(summary["mean_actual_mse"] - 9.0 / d_model) < 1e-6

    diagnostics = correction_diagnostics(
        clean_z,
        sample["corrupted"],
        clean_z,
        sampled_magnitude=sample["magnitude"],
        calibrated_mask=sample["corruption_type"] == CORRUPTION_SAE_CALIBRATED,
    )
    torch.testing.assert_close(
        diagnostics["corruption_norm"].square(),
        diagnostics["corrupted_mse"] * d_model,
    )
    torch.testing.assert_close(
        diagnostics["target_correction_norm"],
        sample["magnitude"],
    )


def test_synthetic_correction_diagnostic_exact_values() -> None:
    d_model, magnitude = 768, 4.0
    clean = torch.zeros(3, d_model)
    direction = torch.zeros(d_model)
    direction[0] = 1.0
    corrupted = clean + magnitude * direction
    diagnostics = correction_diagnostics(clean, corrupted, clean)
    torch.testing.assert_close(
        diagnostics["corruption_norm"], torch.full((3,), magnitude)
    )
    torch.testing.assert_close(
        diagnostics["target_correction_norm"], torch.full((3,), magnitude)
    )
    torch.testing.assert_close(
        diagnostics["corrupted_mse"], torch.full((3,), magnitude**2 / d_model)
    )


class TrainTests(unittest.TestCase):
    def test_checkpoint_loader_rejects_legacy_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy.pt"
            torch.save({"checkpoint_version": 1, "model_state_dict": {}}, path)
            with self.assertRaises(AssertionError):
                load_denoiser_checkpoint(path, device="cpu")

    def test_short_gaussian_training_writes_complete_checkpoint(self) -> None:
        torch.manual_seed(0)
        train_dataset = TensorDataset(torch.randn(16, 4))
        validation_dataset = TensorDataset(torch.randn(8, 4))
        denoiser = ResidualDenoiser(
            d_model=4,
            hidden_dim=8,
            condition_on_noise=True,
            conditioning_hidden_dim=4,
        )
        sampler = partial(
            gaussian_corruption,
            sigma_min=0.1,
            sigma_max=0.2,
            distribution="uniform",
        )
        normalization = {"mean": torch.zeros(4), "std": torch.ones(4)}

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "gaussian.pt"
            config = {
                "seed": 5,
                "training": {
                    "batch_size": 4,
                    "num_workers": 0,
                    "num_epochs": 1,
                    "max_steps": 2,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "gradient_clip_norm": 1.0,
                    "clean_identity_probability": 0.25,
                    "max_validation_batches": 2,
                    "log_every": 1,
                    "checkpoint_every": 1,
                    "checkpoint_path": str(checkpoint_path),
                },
            }
            history = train_denoiser(
                denoiser,
                train_dataset,
                config,
                sampler,
                validation_dataset,
                normalization_stats=normalization,
            )
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            self.assertEqual(checkpoint["step"], 2)
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("normalization", checkpoint)
            self.assertIn("model_config", checkpoint)
            self.assertEqual(len(history), 1)
            loaded, loaded_checkpoint = load_denoiser_checkpoint(
                checkpoint_path, device="cpu", dtype="float32"
            )
            self.assertEqual(loaded.d_model, 4)
            self.assertEqual(loaded_checkpoint["step"], 2)


if __name__ == "__main__":
    unittest.main()
