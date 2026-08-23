import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


if "transformer_lens" not in sys.modules and importlib.util.find_spec("transformer_lens") is None:
    fake_transformer_lens = types.ModuleType("transformer_lens")
    fake_transformer_lens.HookedTransformer = object
    sys.modules["transformer_lens"] = fake_transformer_lens

from src.denoiser_v2 import (  # noqa: E402
    ConditionedSteeringDenoiser,
    standardized_condition_direction,
)
from src.experiment_v2 import validate_v2_test_gate  # noqa: E402
from src.steering import literal_raw_steering, relative_raw_steering  # noqa: E402
from src.train_v2 import (  # noqa: E402
    V2_CHECKPOINT_VERSION,
    V2_PIPELINE_VERSION,
    _checkpoint_payload,
    compute_v2_batch_losses,
    freeze_teacher_models,
    load_conditioned_v2_checkpoint,
)
from src.v2_objectives import correction_geometry, downstream_kl, retention_hinge_loss  # noqa: E402


HOOK = "blocks.6.hook_resid_pre"


class TinyGPT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 4)
        self.downstream = torch.nn.Linear(4, 7, bias=False)
        self.cfg = types.SimpleNamespace(d_model=4)

    def run_with_cache(self, tokens, names_filter):
        assert names_filter == HOOK
        h = self.embedding(tokens)
        return self.downstream(h), {HOOK: h}

    def run_with_hooks(self, tokens, return_type, fwd_hooks):
        assert return_type == "logits" and len(fwd_hooks) == 1
        h = self.embedding(tokens)
        replacement = fwd_hooks[0][1](h, object())
        return self.downstream(replacement)


class TinySAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W_dec = torch.nn.Parameter(torch.eye(4))

    def encode(self, activations):
        return F.relu(activations @ self.W_dec.T)


class V2Tests(unittest.TestCase):
    def test_v2_checkpoint_uses_weights_only_safe_version_metadata(self):
        denoiser = ConditionedSteeringDenoiser(4, 8, 4)
        optimizer = torch.optim.AdamW(denoiser.parameters())
        config = {
            "seed": 42,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 2,
        }
        split = {"num_features": 4, "train": [0, 1], "val": [2], "test": [3]}
        payload = _checkpoint_payload(
            denoiser,
            optimizer,
            1,
            config,
            {"mean": torch.zeros(4), "std": torch.ones(4), "eps": 1e-6},
            split,
            HOOK,
            "tiny",
            {"literal_alphas": [0.0, 1.0]},
            None,
        )
        self.assertIsInstance(payload["torch_version"], str)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditioned.pt"
            torch.save(payload, path)
            loaded_model, loaded = load_conditioned_v2_checkpoint(
                path, "cpu", expected_split_hash=payload["direction_split_hash"]
            )
        self.assertIsInstance(loaded_model, ConditionedSteeringDenoiser)
        self.assertEqual(loaded["checkpoint_version"], V2_CHECKPOINT_VERSION)

    def test_v2_loader_accepts_legacy_torch_version_metadata_safely(self):
        denoiser = ConditionedSteeringDenoiser(4, 8, 4)
        payload = {
            "checkpoint_version": V2_CHECKPOINT_VERSION,
            "pipeline_version": V2_PIPELINE_VERSION,
            "direction_split_hash": "legacy-split",
            "architecture": {
                "d_model": 4,
                "hidden_dim": 8,
                "conditioning_dim": 4,
            },
            "model_state_dict": denoiser.state_dict(),
            "torch_version": torch.__version__,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(payload, path)
            loaded_model, loaded = load_conditioned_v2_checkpoint(
                path, "cpu", expected_split_hash="legacy-split"
            )
        self.assertIsInstance(loaded_model, ConditionedSteeringDenoiser)
        self.assertEqual(str(loaded["torch_version"]), str(torch.__version__))

    def test_literal_and_relative_formulas_are_distinct_and_exact(self):
        h = torch.tensor([[[3.0, 4.0]]])
        v = torch.tensor([2.0, 0.0])
        literal = literal_raw_steering(h, v, 2.5)
        relative = relative_raw_steering(h, v, 0.5)
        torch.testing.assert_close(literal, h + torch.tensor([[[2.5, 0.0]]]))
        torch.testing.assert_close(relative, h + torch.tensor([[[2.5, 0.0]]]))

    def test_standardized_direction_has_unit_norm(self):
        directions = F.normalize(torch.randn(5, 8), dim=-1)
        std = torch.linspace(0.5, 2.0, 8)
        standardized = standardized_condition_direction(directions, std)
        torch.testing.assert_close(standardized.norm(dim=-1), torch.ones(5), atol=1e-6, rtol=1e-6)

    def test_zero_strength_initial_identity_is_finite(self):
        denoiser = ConditionedSteeringDenoiser(8, 12, 4)
        x = torch.randn(2, 3, 8)
        directions = F.normalize(torch.randn(2, 8), dim=-1)
        output = denoiser(x, directions, torch.zeros(2))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, x, rtol=0, atol=0)

    def test_downstream_kl_direction_matches_manual_distribution(self):
        clean_probability = torch.tensor([0.8, 0.2])
        modified_probability = torch.tensor([0.5, 0.5])
        clean_logits = clean_probability.log().reshape(1, 1, 2)
        modified_logits = modified_probability.log().reshape(1, 1, 2)
        expected = (clean_probability * (clean_probability.log() - modified_probability.log())).sum()
        actual = downstream_kl(clean_logits, modified_logits)
        torch.testing.assert_close(actual, expected)

    def test_retention_hinge_and_tiny_mask(self):
        at_target = retention_hinge_loss(torch.tensor([2.0]), torch.tensor([1.6]))
        below = retention_hinge_loss(torch.tensor([2.0]), torch.tensor([1.0]))
        tiny = retention_hinge_loss(torch.tensor([1e-5]), torch.tensor([0.0]), threshold=1e-3)
        self.assertAlmostEqual(float(at_target["loss"]), 0.0, places=6)
        self.assertGreater(float(below["loss"]), 0.0)
        self.assertAlmostEqual(float(tiny["loss"]), 0.0)
        self.assertFalse(bool(tiny["valid_mask"].item()))

    def test_correction_decomposition_reconstructs_correction(self):
        steered = torch.randn(2, 3, 4)
        denoised = steered + torch.randn_like(steered)
        direction = F.normalize(torch.randn(2, 4), dim=-1)
        geometry = correction_geometry(denoised, steered, direction)
        torch.testing.assert_close(
            geometry["parallel"] + geometry["orthogonal"], geometry["correction"]
        )

    def test_kl_gradient_reaches_denoiser_but_not_frozen_gpt(self):
        torch.manual_seed(4)
        model, sae = TinyGPT(), TinySAE()
        freeze_teacher_models(model, sae)
        denoiser = ConditionedSteeringDenoiser(4, 8, 4)
        tokens = torch.tensor([[1, 2, 3], [3, 2, 1]])
        mask = torch.ones_like(tokens, dtype=torch.bool)
        directions = F.normalize(torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]]), dim=-1)
        config = {
            "retention_target": 0.8, "retention_mask_threshold": 1e-3,
            "retention_eps": 1e-6, "identity_mse_eta": 0.01,
            "lambda_fluency": 1.0, "lambda_retain": 1.0,
            "lambda_identity": 0.5, "lambda_correction": 0.01,
            "lambda_reconstruction": 0.05, "objective_mode": "full",
        }
        losses = compute_v2_batch_losses(
            model, sae, denoiser, tokens, mask, directions,
            torch.tensor([0, 1]), torch.tensor([0.5, 1.0]), directions.flip(0),
            {"mean": torch.zeros(4), "std": torch.ones(4), "eps": 1e-6},
            config, HOOK,
        )
        losses["loss"].backward()
        gradients = [parameter.grad for parameter in denoiser.parameters()]
        self.assertTrue(any(gradient is not None and gradient.abs().sum() > 0 for gradient in gradients))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_v2_test_gate_rejects_default_and_missing_freeze(self):
        split = {"num_features": 6, "train": [0, 1], "val": [2], "test": [3]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            with self.assertRaises(PermissionError):
                validate_v2_test_gate(False, path, split)
            with self.assertRaises(FileNotFoundError):
                validate_v2_test_gate(True, path, split)

    def test_v2_test_gate_rejects_wrong_split_hash(self):
        split = {"num_features": 6, "train": [0, 1], "val": [2], "test": [3]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            path.write_text(json.dumps({
                "pipeline_version": "conditioned_v2_v1",
                "status": "frozen_after_validation",
                "direction_split_hash": "wrong",
            }))
            with self.assertRaises(AssertionError):
                validate_v2_test_gate(True, path, split)


if __name__ == "__main__":
    unittest.main()
