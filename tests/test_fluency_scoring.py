import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch


if "transformer_lens" not in sys.modules and importlib.util.find_spec("transformer_lens") is None:
    fake_transformer_lens = types.ModuleType("transformer_lens")
    fake_transformer_lens.HookedTransformer = object
    sys.modules.setdefault("transformer_lens", fake_transformer_lens)

from src.experiment import score_train_direction_fluency  # noqa: E402


HOOK_NAME = "blocks.6.hook_resid_pre"


class FakeConfig:
    d_model = 3


class ScoringFakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = FakeConfig()
        self.hook_dict = {HOOK_NAME: object()}
        self.clean_forwards = 0
        self.modified_forwards = 0

    def activation(self, tokens):
        offsets = torch.arange(3, dtype=torch.float32)
        return tokens.unsqueeze(-1).float() + offsets

    def logits_from_activation(self, activation):
        value = activation.mean(dim=-1)
        scales = torch.arange(8, dtype=activation.dtype)
        return value.unsqueeze(-1) * scales

    def forward(self, tokens, return_type="logits"):
        self.clean_forwards += 1
        return self.logits_from_activation(self.activation(tokens))

    def run_with_hooks(self, tokens, return_type="logits", fwd_hooks=None):
        self.modified_forwards += 1
        hook_name, hook_fn = fwd_hooks[0]
        activation = hook_fn(self.activation(tokens), self.hook_dict[hook_name])
        return self.logits_from_activation(activation)


class FluencyScoringTests(unittest.TestCase):
    def test_train_only_scoring_batches_directions(self) -> None:
        model = ScoringFakeModel().eval()
        tokens = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
        directions = torch.randn(6, 3)
        split = {
            "num_features": 6,
            "train": [0, 1, 2, 3],
            "val": [4],
            "test": [5],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            output_path = temporary_path / "train_scores.pt"
            csv_path = temporary_path / "train_scores.csv"
            figure_path = temporary_path / "histogram.png"
            scores = score_train_direction_fluency(
                model,
                [tokens],
                directions,
                split,
                relative_strengths=[0.5, 1.0],
                hook_name=HOOK_NAME,
                max_contexts=2,
                direction_batch_size=2,
                output_path=output_path,
                csv_path=csv_path,
                figure_path=figure_path,
            )
            self.assertEqual(model.clean_forwards, 1)
            self.assertEqual(model.modified_forwards, 4)
            self.assertEqual(set(scores["direction_ids"].tolist()), set(split["train"]))
            self.assertTrue(set(scores["direction_ids"].tolist()).isdisjoint({4, 5}))
            self.assertEqual(scores["kl_by_strength"].shape, (4, 2))
            self.assertTrue(output_path.exists() and csv_path.exists() and figure_path.exists())


if __name__ == "__main__":
    unittest.main()
