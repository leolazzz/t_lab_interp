import unittest
import json

import torch

from src.utils import (
    denormalize_activations,
    load_config,
    normalize_activations,
    resolve_device,
    resolve_dtype,
)


class UtilsTests(unittest.TestCase):
    def test_numpy_boolean_audit_values_are_json_serializable(self) -> None:
        # Regression guard for the final Kaggle audit artifact. numpy.bool_
        # prints as True/False but json.dumps rejects it unless converted.
        audit = {"check": bool(torch.tensor(True).numpy())}
        self.assertEqual(json.loads(json.dumps(audit)), {"check": True})

    def test_debug_overrides_are_applied(self) -> None:
        config = load_config("config.yaml", debug=True)
        self.assertEqual(config["training"]["max_steps"], 100)
        self.assertEqual(config["directions"]["num_train"], 8)
        self.assertEqual(config["data"]["target_num_activations"], 5000)

    def test_full_evaluation_optimization_defaults(self) -> None:
        config = load_config("config.yaml", debug=False)
        evaluation = config["evaluation"]
        self.assertEqual(evaluation["token_eval_prompt_batch_size"], 16)
        self.assertEqual(evaluation["intervention_batch_size"], 8)
        self.assertTrue(evaluation["use_inference_autocast"])
        self.assertEqual(config["damage_score"]["max_contexts"], 32)
        self.assertEqual(config["damage_score"]["relative_strengths"], [0.25, 0.5])

    def test_normalization_roundtrip_broadcast_and_no_mutation(self) -> None:
        for shape in ((5, 7), (2, 3, 7)):
            values = torch.randn(*shape, dtype=torch.float64)
            mean = torch.randn(7, dtype=torch.float64)
            std = torch.rand(7, dtype=torch.float64) + 0.1
            before = values.clone()
            standardized = normalize_activations(values, mean, std)
            reconstructed = denormalize_activations(standardized, mean, std)
            torch.testing.assert_close(reconstructed, values)
            torch.testing.assert_close(values, before, rtol=0, atol=0)
            self.assertEqual(standardized.dtype, values.dtype)
            self.assertEqual(standardized.device, values.device)

    def test_device_and_dtype_resolution(self) -> None:
        self.assertEqual(resolve_device("cpu").type, "cpu")
        self.assertEqual(resolve_dtype("float32").is_floating_point, True)
        with self.assertRaises(ValueError):
            resolve_dtype("int8")


if __name__ == "__main__":
    unittest.main()
