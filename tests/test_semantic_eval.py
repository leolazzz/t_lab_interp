import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


if "transformer_lens" not in sys.modules and importlib.util.find_spec("transformer_lens") is None:
    fake_transformer_lens = types.ModuleType("transformer_lens")
    fake_transformer_lens.HookedTransformer = object
    sys.modules["transformer_lens"] = fake_transformer_lens

from src.semantic_eval import (  # noqa: E402
    FIXED_PROMPTS,
    aggregate_semantic_scores,
    classifier_label_ids,
    direction_split_membership,
    finalize_semantic_outputs,
    mask_greeting_keywords,
    prompt_set_hash,
    score_cached_generations,
    score_texts_nli,
    select_semantic_directions,
)


class FakeTokenizer:
    def __call__(self, premises, hypotheses, **kwargs):
        del kwargs
        assert all(isinstance(value, str) for value in premises)
        assert all(isinstance(value, str) for value in hypotheses)
        lengths = [len(text.split()) for text in premises]
        width = max(lengths)
        input_ids = torch.zeros(len(lengths), width, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, length in enumerate(lengths):
            input_ids[row, :length] = 1
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = types.SimpleNamespace(
            id2label={0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"},
            label2id={"CONTRADICTION": 0, "NEUTRAL": 1, "ENTAILMENT": 2},
        )

    def forward(self, input_ids, attention_mask):
        lengths = (input_ids * attention_mask).sum(dim=1).float()
        logits = torch.stack((-lengths, torch.zeros_like(lengths), lengths), dim=-1)
        return types.SimpleNamespace(logits=logits)


def split_with(*, train=(), val=(), test=()):
    return {
        "num_features": 24576,
        "train": list(train),
        "val": list(val),
        "test": list(test),
    }


class SemanticEvaluationTests(unittest.TestCase):
    def test_classifier_label_mapping(self):
        self.assertEqual(
            classifier_label_ids(FakeClassifier()),
            {"entailment": 2, "neutral": 1, "contradiction": 0},
        )

    def test_entailment_probability_and_batch_agreement(self):
        texts = ["one two", "one two three four", "one"]
        tokenizer, classifier = FakeTokenizer(), FakeClassifier()
        batched = score_texts_nli(texts, "fixed hypothesis", tokenizer, classifier, 3)
        singles = pd.concat([
            score_texts_nli([text], "fixed hypothesis", tokenizer, classifier, 1)
            for text in texts
        ], ignore_index=True)
        np.testing.assert_allclose(batched.to_numpy(), singles.to_numpy(), rtol=1e-7)
        self.assertTrue(batched.semantic_concept_score.between(0.0, 1.0).all())

    def test_prompt_hash_is_order_sensitive_and_deterministic(self):
        first = prompt_set_hash(FIXED_PROMPTS)
        self.assertEqual(first, prompt_set_hash(list(FIXED_PROMPTS)))
        self.assertNotEqual(first, prompt_set_hash(list(reversed(FIXED_PROMPTS))))

    def test_direction_split_gate_and_primary_selection(self):
        split = split_with(train=[1], val=[2], test=[3])
        self.assertEqual(direction_split_membership(6516, split), "outside")
        selected = select_semantic_directions(split, output_path=None)
        self.assertEqual(selected["feature_id"], 6516)
        self.assertTrue(selected["feature_6516_legally_usable"])

    def test_test_direction_is_rejected_and_fixed_fallback_used(self):
        split = split_with(train=[1], val=[9696], test=[6516])
        selected = select_semantic_directions(split, output_path=None)
        self.assertEqual(selected["feature_id"], 9696)
        self.assertFalse(selected["feature_6516_legally_usable"])
        self.assertEqual(selected["feature_6516_split_membership"], "test")

    def test_all_test_candidates_stop(self):
        split = split_with(test=[6516, 9696, 7672])
        with self.assertRaises(RuntimeError):
            select_semantic_directions(split, output_path=None)

    def test_text_only_scoring_and_masking(self):
        generation = pd.DataFrame({
            "experiment_role": ["primary"], "method": ["raw"],
            "direction_id": [6516], "prompt_id": [0], "strength": [0.5],
            "seed": [0], "generated_text": ["Hello and welcome"],
            "target_sae_activation": [2.0],
            "clean_model_continuation_nll": [3.0],
            "dist_1": [1.0], "dist_2": [1.0], "dist_3": [1.0],
            "repetition_rate": [0.0],
        })
        selection = {"hypothesis": "This text is a greeting or welcoming interaction."}
        scores, masked = score_cached_generations(
            generation, selection, FakeTokenizer(), FakeClassifier(), batch_size=1
        )
        self.assertEqual(len(scores), 1)
        self.assertIn("[MASKED]", masked.loc[0, "masked_text"])
        self.assertEqual(mask_greeting_keywords("Hi, goodbye!"), "[MASKED], [MASKED]!")

    def test_aggregation_row_counts_and_finite_metrics(self):
        rows = []
        for method in ("raw", "sae_calibrated"):
            for strength in (0.0, 0.5):
                for prompt_id in (0, 1):
                    rows.append({
                        "experiment_role": "primary", "method": method,
                        "strength": strength, "prompt_id": prompt_id, "seed": 0,
                        "semantic_concept_score": 0.1 + strength,
                        "target_sae_activation": strength,
                        "clean_model_continuation_nll": 2.0,
                        "dist_1": 1.0, "dist_2": 1.0, "dist_3": 1.0,
                        "repetition_rate": 0.0,
                    })
        aggregate = aggregate_semantic_scores(pd.DataFrame(rows))
        self.assertEqual(len(aggregate), 4)
        self.assertTrue(np.isfinite(aggregate.select_dtypes(include=[np.number])).all().all())

    def test_generation_cache_key_is_reproducible(self):
        from src.semantic_eval import _generation_key
        record = {
            "experiment_role": "negative_control", "method": "raw",
            "direction_id": 9696, "prompt_id": 3, "strength": 0.5,
            "seed": 0, "generation_signature": "fixed",
        }
        self.assertEqual(_generation_key(record), _generation_key(dict(record)))

    def test_semantic_finalization_writes_all_late_stage_outputs(self):
        rows = []
        for method in ("raw", "sae_calibrated"):
            for strength in (0.0, 0.5):
                for prompt_id in (0, 1):
                    score = 0.2 + 0.4 * strength + (0.02 if method == "sae_calibrated" else 0.0)
                    rows.append({
                        "experiment_role": "primary", "method": method,
                        "direction_id": 1, "prompt_id": prompt_id, "strength": strength,
                        "seed": 0, "generated_text": "hello example",
                        "semantic_concept_score": score,
                        "semantic_concept_score_masked": score - 0.01,
                        "target_sae_activation": 1.0 + strength,
                        "clean_model_continuation_nll": 2.0 + strength,
                        "dist_1": 1.0, "dist_2": 1.0, "dist_3": 1.0,
                        "repetition_rate": 0.0,
                    })
        for strength in (0.0, 0.5):
            rows.append({
                "experiment_role": "negative_control", "method": "raw",
                "direction_id": 2, "prompt_id": 0, "strength": strength,
                "seed": 0, "generated_text": "control example",
                "semantic_concept_score": 0.1,
                "semantic_concept_score_masked": 0.1,
                "target_sae_activation": 0.1,
                "clean_model_continuation_nll": 2.0,
                "dist_1": 1.0, "dist_2": 1.0, "dist_3": 1.0,
                "repetition_rate": 0.0,
            })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = finalize_semantic_outputs(
                pd.DataFrame(rows), {"feature_id": 1},
                {"mean_positive_score": 0.8, "mean_negative_score": 0.2, "roc_auc": 1.0},
                pd.DataFrame({"semantic_concept_score": [0.1, 0.2]}),
                root / "semantic", root / "figures",
            )
            self.assertEqual(summary["selected_direction"]["feature_id"], 1)
            for name in ("semantic_scores.csv", "semantic_aggregate.csv", "summary.json"):
                self.assertTrue((root / "semantic" / name).exists())
            self.assertTrue((root / "figures" / "semantic_pareto_classifier.png").exists())


if __name__ == "__main__":
    unittest.main()
