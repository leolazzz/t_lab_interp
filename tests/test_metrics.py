import unittest

import torch

from src.metrics import (
    activation_norm_ratio,
    distinct_n,
    external_clean_lm_nll,
    next_token_nll,
    nll_increase,
    repetition_rate,
    sae_feature_activation_metrics,
    token_level_kl,
)


class MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.logits = torch.randn(2, 4, 7, dtype=torch.float64)
        self.tokens = torch.randint(0, 7, (2, 4), dtype=torch.long)
        self.attention_mask = torch.tensor(
            [[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long
        )

    def test_identical_logits_have_zero_kl(self) -> None:
        kl = token_level_kl(
            self.logits,
            self.logits,
            attention_mask=self.attention_mask,
        )
        torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-12, rtol=0.0)

    def test_identical_logits_have_zero_nll_increase(self) -> None:
        increase = nll_increase(
            self.logits,
            self.logits,
            self.tokens,
            attention_mask=self.attention_mask,
        )
        torch.testing.assert_close(
            increase, torch.zeros_like(increase), atol=1e-12, rtol=0.0
        )

    def test_per_token_nll_shape_and_mask(self) -> None:
        mean_nll, per_token_nll = next_token_nll(
            self.logits,
            self.tokens,
            attention_mask=self.attention_mask,
            return_per_token=True,
        )
        self.assertEqual(mean_nll.ndim, 0)
        self.assertEqual(per_token_nll.shape, (2, 3))
        self.assertEqual(per_token_nll[1, -1].item(), 0.0)

    def test_unchanged_activations_have_unit_norm_ratio(self) -> None:
        activations = torch.randn(2, 4, 6, dtype=torch.float64)
        ratio = activation_norm_ratio(activations, activations)
        torch.testing.assert_close(
            ratio, torch.ones_like(ratio), atol=1e-12, rtol=0.0
        )

    def test_distinct_n_and_repetition_rate(self) -> None:
        sequence = [1, 2, 1, 2]
        self.assertEqual(distinct_n(sequence, 1), 0.5)
        self.assertAlmostEqual(distinct_n(sequence, 2), 2 / 3)
        self.assertAlmostEqual(repetition_rate(sequence, n=2), 1 / 3)
        self.assertEqual(distinct_n([], 3), 0.0)

    def test_external_clean_nll_can_exclude_prompt_targets(self) -> None:
        class CleanModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.hook_dict = {}

            def forward(self, tokens, return_type="logits"):
                logits = torch.zeros(*tokens.shape, 7)
                logits.scatter_(-1, tokens.unsqueeze(-1), 2.0)
                return logits

        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        model = CleanModel().eval()
        score = external_clean_lm_nll(model, tokens, prefix_lengths=2)
        all_token_scores = external_clean_lm_nll(
            model, tokens, reduction="none"
        )
        torch.testing.assert_close(score, all_token_scores[:, 1:].mean())

    def test_sae_feature_metrics_use_only_selected_tokens(self) -> None:
        class FakeSAE:
            W_dec = torch.eye(3)

            @staticmethod
            def encode(activations):
                return activations.clamp_min(0)

        activations = torch.tensor(
            [[[100.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]
        )
        continuation_mask = torch.tensor([[False, True, True]])
        metrics = sae_feature_activation_metrics(
            FakeSAE(),
            activations,
            feature_id=0,
            token_mask=continuation_mask,
            threshold=3.0,
        )
        self.assertEqual(metrics["concept_score"].item(), 3.0)
        self.assertEqual(metrics["max_feature_activation"].item(), 4.0)
        self.assertEqual(metrics["feature_active_fraction"].item(), 0.5)


if __name__ == "__main__":
    unittest.main()
