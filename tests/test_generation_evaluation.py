import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch


if "transformer_lens" not in sys.modules and importlib.util.find_spec("transformer_lens") is None:
    fake_transformer_lens = types.ModuleType("transformer_lens")
    fake_transformer_lens.HookedTransformer = object
    sys.modules["transformer_lens"] = fake_transformer_lens

from src.experiment import (  # noqa: E402
    _generate_fixed_seed,
    _generation_intervention,
    evaluate_generation_methods,
    evaluate_token_level_methods,
    evaluate_token_level_methods_fast,
    validate_generation_methods,
)
from src.semantic_eval import generate_semantic_cache  # noqa: E402


HOOK_NAME = "blocks.6.hook_resid_pre"


class FakeTokenizer:
    eos_token_id = None

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = types.SimpleNamespace(d_model=3, n_ctx=8)
        self.hook_dict = {HOOK_NAME: object()}
        self.tokenizer = FakeTokenizer()
        self.clean_forwards = 0
        self.modified_forwards = 0
        self.logit_projection = torch.tensor(
            [
                [0.5, -0.2, 0.1, 0.3, -0.4, 0.2, 0.0, 0.1],
                [0.1, 0.4, -0.3, 0.2, 0.5, -0.1, 0.3, -0.2],
                [-0.2, 0.1, 0.5, -0.4, 0.2, 0.3, -0.1, 0.4],
            ]
        )

    def to_tokens(self, prompt, prepend_bos=True):
        values = [0] if prepend_bos else []
        values.extend(1 + (ord(character) % 6) for character in prompt[:2])
        return torch.tensor([values], dtype=torch.long, device=self.anchor.device)

    def _activations(self, tokens):
        offsets = torch.arange(3, dtype=self.anchor.dtype, device=tokens.device)
        return tokens.to(self.anchor.dtype).unsqueeze(-1) / 4.0 + offsets

    def _logits(self, activations):
        projection = self.logit_projection.to(
            device=activations.device, dtype=activations.dtype
        )
        return activations @ projection

    def forward(self, tokens, return_type="logits"):
        assert return_type == "logits"
        return self._logits(self._activations(tokens))

    def run_with_hooks(self, tokens, return_type, fwd_hooks):
        self.modified_forwards += 1
        assert return_type == "logits" and len(fwd_hooks) == 1
        hook_name, hook_fn = fwd_hooks[0]
        assert hook_name == HOOK_NAME
        activations = hook_fn(self._activations(tokens), object())
        return self._logits(activations)

    def run_with_cache(self, tokens, names_filter, return_type="logits"):
        self.clean_forwards += 1
        assert names_filter == HOOK_NAME
        activations = self._activations(tokens)
        logits = None if return_type is None else self._logits(activations)
        return logits, {HOOK_NAME: activations}


class FakeSAE:
    def __init__(self) -> None:
        self.W_dec = torch.eye(3)
        metadata = types.SimpleNamespace(hook_name=HOOK_NAME)
        self.cfg = types.SimpleNamespace(metadata=metadata)

    def encode(self, activations):
        return torch.relu(activations @ self.W_dec.T)


class BatchSensitiveFakeModel(FakeModel):
    """Simulate harmless batch-size-dependent CUDA roundoff before the hook."""

    def _activations(self, tokens):
        activations = super()._activations(tokens)
        if tokens.shape[0] > 1:
            activations = activations + 1e-4
        return activations


class IdentityDenoiser(torch.nn.Module):
    condition_on_noise = False

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, values):
        return values


class IdentityConditionedDenoiser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, values, steering_direction, strength):
        del steering_direction, strength
        return values


class GenerationEvaluationTests(unittest.TestCase):
    def test_generation_hook_is_identity_at_zero_and_only_modifies_final_token(self) -> None:
        activation = torch.randn(2, 4, 3)
        direction = torch.tensor([2.0, 0.0, 0.0])
        zero_hook = _generation_intervention("raw", direction, 0.0, {}, {}, {})
        torch.testing.assert_close(zero_hook(activation), activation)
        hook = _generation_intervention("raw", direction, 0.5, {}, {}, {})
        modified = hook(activation)
        torch.testing.assert_close(modified[:, :-1], activation[:, :-1])
        update_norm = (modified[:, -1] - activation[:, -1]).norm(dim=-1)
        torch.testing.assert_close(update_norm, 0.5 * activation[:, -1].norm(dim=-1))

    def test_v3_relative_raw_alias_matches_raw(self) -> None:
        """The production V3 method name must use the established raw path."""
        direction = torch.randn(8)
        activation = torch.randn(2, 4, 8)
        legacy_hook = _generation_intervention("raw", direction, 0.5, {}, {}, {})
        v3_hook = _generation_intervention("relative_raw", direction, 0.5, {}, {}, {})
        torch.testing.assert_close(v3_hook(activation), legacy_hook(activation))

    def test_production_v3_generation_method_list_passes_preflight(self) -> None:
        config = __import__("yaml").safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        validate_generation_methods(config["final_v3"]["generation"]["methods"])

    def test_production_v3_generation_methods_execute_end_to_end(self) -> None:
        config = __import__("yaml").safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        methods = config["final_v3"]["generation"]["methods"]
        model, sae = FakeModel().eval(), FakeSAE()
        unconditional = IdentityDenoiser()
        conditioned = IdentityConditionedDenoiser()
        denoisers = {
            "sae_calibrated": unconditional,
            "conditioned_reconstruction": conditioned,
            "conditioned_kl": conditioned,
            "conditioned_kl_retention": conditioned,
        }
        stats = {"mean": torch.zeros(3), "std": torch.ones(3), "eps": 1e-6}
        normalizations = {key: stats for key in denoisers}
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results = evaluate_generation_methods(
                model=model, sae=sae, prompts=["hi"], all_directions=sae.W_dec.clone(),
                direction_split={"train": [0], "val": [1], "test": [2], "num_features": 3},
                direction_ids=[1], alphas=[0.5], seeds=[0], hook_name=HOOK_NAME,
                denoisers=denoisers, normalization_stats=normalizations,
                generation_config={"max_new_tokens": 1, "do_sample": False, "top_k": 4,
                                   "generation_positions": "final_token_only"},
                evaluation_split="val", methods=methods,
                jsonl_path=root / "generations.jsonl", results_path=root / "results.csv",
                aggregate_path=root / "aggregate.csv",
            )
        self.assertEqual(set(results.method), set(methods))
        self.assertEqual(len(results), len(methods))

    def test_unknown_generation_method_fails_preflight(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown_method"):
            validate_generation_methods(["relative_raw", "unknown_method"])

    def test_v3_semantic_generation_cache_executes_all_conditioned_methods(self) -> None:
        model, sae = FakeModel().eval(), FakeSAE()
        methods = [
            "raw", "sae_calibrated", "conditioned_reconstruction",
            "conditioned_kl", "conditioned_kl_retention", "conditioned_full",
        ]
        unconditional = IdentityDenoiser()
        conditioned = IdentityConditionedDenoiser()
        denoisers = {
            "sae_calibrated": unconditional,
            **{method: conditioned for method in methods if method.startswith("conditioned_")},
        }
        stats = {"mean": torch.zeros(3), "std": torch.ones(3), "eps": 1e-6}
        with tempfile.TemporaryDirectory() as temporary_directory:
            frame = generate_semantic_cache(
                model=model, sae=sae, all_directions=sae.W_dec.clone(),
                split={"train": [0], "val": [1], "test": [], "num_features": 3},
                selection={"feature_id": 1, "negative_control_feature_id": 2},
                denoisers=denoisers,
                normalization_stats={key: stats for key in denoisers},
                config={"methods": methods, "strengths": [0.5], "seed": 0,
                        "max_new_tokens": 1, "temperature": 1.0, "top_k": 4,
                        "do_sample": False, "hook_name": HOOK_NAME},
                prompts=["hi"],
                output_path=Path(temporary_directory) / "semantic.jsonl",
            )
        self.assertEqual(len(frame), len(methods) + 1)
        self.assertEqual(set(frame[frame.experiment_role == "primary"].method), set(methods))

    def test_all_token_hook_changes_prediction_positions(self) -> None:
        activation = torch.randn(2, 4, 3)
        hook = _generation_intervention(
            "raw", torch.tensor([1.0, 0.0, 0.0]), 0.5, {}, {}, {},
            token_positions="all_tokens",
        )
        modified = hook(activation)
        self.assertTrue(torch.all((modified - activation).norm(dim=-1) > 0))

    def test_token_level_nll_is_not_forced_to_zero(self) -> None:
        model = FakeModel().eval()
        sae = FakeSAE()
        tokens = model.to_tokens("hi")
        results = evaluate_token_level_methods(
            model=model,
            token_batches=[tokens],
            all_directions=sae.W_dec.clone(),
            direction_split={
                "train": [0], "val": [1], "test": [2], "num_features": 3,
            },
            direction_ids=[1],
            strengths=[0.5],
            hook_name=HOOK_NAME,
            methods=["raw"],
            evaluation_config={"token_evaluation_positions": "all_tokens"},
            evaluation_split="val",
            sae=sae,
        )
        self.assertNotAlmostEqual(float(results.loc[0, "delta_nll"]), 0.0)

    def test_batched_token_evaluation_matches_reference(self) -> None:
        model = FakeModel().eval()
        sae = FakeSAE()
        split = {
            "train": [0], "val": [1, 2], "test": [], "num_features": 3,
        }
        tokens = model.to_tokens("hi")
        kwargs = dict(
            model=model,
            token_batches=[tokens],
            all_directions=sae.W_dec.clone(),
            direction_split=split,
            direction_ids=[1, 2],
            strengths=[0.0, 0.5],
            hook_name=HOOK_NAME,
            methods=["raw", "norm_preserving", "sae_calibrated"],
            denoisers={"sae_calibrated": IdentityDenoiser().eval()},
            normalization_stats={
                "sae_calibrated": {
                    "mean": torch.zeros(3), "std": torch.ones(3), "eps": 1e-6,
                }
            },
            evaluation_config={"token_evaluation_positions": "all_tokens"},
            evaluation_split="val",
            sae=sae,
        )
        reference = evaluate_token_level_methods(**kwargs)
        optimized = evaluate_token_level_methods_fast(
            **kwargs, intervention_batch_size=4
        )
        keys = ["prompt_id", "method", "direction_id", "strength"]
        merged = reference.merge(
            optimized, on=keys, suffixes=("_reference", "_optimized")
        )
        self.assertEqual(len(merged), len(reference))
        for metric in (
            "kl", "delta_nll", "target_sae_activation",
            "activation_norm_ratio", "activation_mse",
        ):
            torch.testing.assert_close(
                torch.tensor(merged[f"{metric}_optimized"].to_numpy()),
                torch.tensor(merged[f"{metric}_reference"].to_numpy()),
                rtol=2e-5,
                atol=2e-6,
            )

    def test_batched_token_evaluation_resumes_without_forwards(self) -> None:
        model = FakeModel().eval()
        sae = FakeSAE()
        kwargs = dict(
            model=model,
            token_batches=[model.to_tokens("hi")],
            all_directions=sae.W_dec.clone(),
            direction_split={
                "train": [0], "val": [1, 2], "test": [], "num_features": 3,
            },
            direction_ids=[1, 2],
            strengths=[0.0, 0.5],
            hook_name=HOOK_NAME,
            methods=["raw"],
            evaluation_config={"token_evaluation_positions": "all_tokens"},
            evaluation_split="val",
            sae=sae,
            intervention_batch_size=4,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            partial = Path(temporary_directory) / "partial.jsonl"
            first = evaluate_token_level_methods_fast(
                **kwargs, partial_jsonl_path=partial
            )
            first_counts = (model.clean_forwards, model.modified_forwards)
            second = evaluate_token_level_methods_fast(
                **kwargs, partial_jsonl_path=partial
            )
            self.assertEqual(first_counts, (1, 1))
            self.assertEqual(
                (model.clean_forwards, model.modified_forwards), first_counts
            )
            self.assertEqual(len(first), len(second))

    def test_zero_raw_uses_exact_clean_identity_despite_batch_roundoff(self) -> None:
        model = BatchSensitiveFakeModel().eval()
        sae = FakeSAE()
        results = evaluate_token_level_methods_fast(
            model=model,
            token_batches=[model.to_tokens("hi")],
            all_directions=sae.W_dec.clone(),
            direction_split={
                "train": [0], "val": [1], "test": [2], "num_features": 3,
            },
            direction_ids=[1],
            strengths=[0.0, 0.5],
            hook_name=HOOK_NAME,
            methods=["raw"],
            evaluation_config={"token_evaluation_positions": "all_tokens"},
            evaluation_split="val",
            sae=sae,
            intervention_batch_size=2,
        )
        identity = results[results.strength == 0.0].iloc[0]
        self.assertEqual(float(identity.kl), 0.0)
        self.assertEqual(float(identity.delta_nll), 0.0)
        self.assertEqual(float(identity.activation_mse), 0.0)
        self.assertEqual(float(identity.activation_norm_ratio), 1.0)

    def test_zero_strength_generation_matches_clean_identity_hook(self) -> None:
        model = FakeModel().eval()
        prompt = model.to_tokens("hi")
        zero_hook = _generation_intervention(
            "raw", torch.tensor([1.0, 0.0, 0.0]), 0.0, {}, {}, {}
        )
        kwargs = dict(
            model=model, prompt_tokens=prompt, hook_name=HOOK_NAME, seed=9,
            max_new_tokens=3, temperature=1.0, top_k=4, do_sample=True,
            eos_token_id=None,
        )
        steered = _generate_fixed_seed(intervention_fn=zero_hook, **kwargs)
        clean = _generate_fixed_seed(intervention_fn=lambda activation: activation, **kwargs)
        torch.testing.assert_close(steered, clean)

    def test_held_out_evaluation_and_resume(self) -> None:
        model = FakeModel().eval()
        sae = FakeSAE()
        split = {
            "train": [0],
            "val": [1],
            "test": [2],
            "num_features": 3,
        }
        config = {
            "max_new_tokens": 2,
            "temperature": 1.0,
            "top_k": 4,
            "do_sample": False,
            "concept_threshold": 0.0,
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            kwargs = dict(
                model=model,
                sae=sae,
                prompts=["hi"],
                all_directions=sae.W_dec.clone(),
                direction_split=split,
                direction_ids=[1],
                alphas=[0.5],
                seeds=[7],
                hook_name=HOOK_NAME,
                denoisers={},
                normalization_stats={},
                generation_config=config,
                evaluation_split="val",
                methods=["raw", "norm_preserving"],
                jsonl_path=root / "generations.jsonl",
                results_path=root / "results.csv",
                aggregate_path=root / "aggregate.csv",
            )
            first = evaluate_generation_methods(**kwargs)
            second = evaluate_generation_methods(**kwargs)

            self.assertEqual(len(first), 2)
            self.assertEqual(len(second), 2)
            self.assertEqual(set(first["direction_id"]), {1})
            self.assertIn("concept_score", first.columns)
            self.assertTrue((first["num_generated_tokens"] == 2).all())
            self.assertTrue((first["prompt_was_truncated"] == False).all())  # noqa: E712
            with (root / "generations.jsonl").open(encoding="utf-8") as file:
                records = [json.loads(line) for line in file if line.strip()]
            self.assertEqual(len(records), 2)

    def test_long_prompt_is_truncated_to_leave_generation_room(self) -> None:
        model = FakeModel().eval()
        model.cfg.n_ctx = 4
        sae = FakeSAE()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            results = evaluate_generation_methods(
                model=model,
                sae=sae,
                prompts=["hi"],
                all_directions=sae.W_dec.clone(),
                direction_split={
                    "train": [0], "val": [1], "test": [2], "num_features": 3,
                },
                direction_ids=[1],
                alphas=[0.5],
                seeds=[7],
                hook_name=HOOK_NAME,
                denoisers={},
                normalization_stats={},
                generation_config={
                    "max_new_tokens": 2, "do_sample": False, "top_k": 4,
                },
                evaluation_split="val",
                methods=["raw"],
                jsonl_path=root / "generations.jsonl",
                results_path=root / "results.csv",
                aggregate_path=None,
            )
            self.assertTrue(bool(results.loc[0, "prompt_was_truncated"]))
            self.assertEqual(int(results.loc[0, "prompt_tokens_used"]), 2)
            self.assertEqual(int(results.loc[0, "num_generated_tokens"]), 2)

    def test_train_direction_is_rejected(self) -> None:
        model = FakeModel().eval()
        sae = FakeSAE()
        with self.assertRaises(AssertionError):
            evaluate_generation_methods(
                model=model,
                sae=sae,
                prompts=["x"],
                all_directions=sae.W_dec.clone(),
                direction_split={
                    "train": [0],
                    "val": [1],
                    "test": [2],
                    "num_features": 3,
                },
                direction_ids=[0],
                alphas=[1.0],
                seeds=[0],
                hook_name=HOOK_NAME,
                denoisers={},
                normalization_stats={},
                generation_config={"max_new_tokens": 1},
                evaluation_split="val",
                methods=["raw"],
            )


if __name__ == "__main__":
    unittest.main()
