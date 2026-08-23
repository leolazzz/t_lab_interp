import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch


if importlib.util.find_spec("transformer_lens") is None:
    fake_transformer_lens = types.ModuleType("transformer_lens")
    fake_transformer_lens.HookedTransformer = object
    sys.modules["transformer_lens"] = fake_transformer_lens

from src.experiment import (  # noqa: E402
    ActivationShardDataset,
    cache_residual_activations,
    split_activation_shards,
)


HOOK_NAME = "blocks.6.hook_resid_pre"


class FakeTokenizer:
    pad_token_id = 0
    bos_token_id = 9
    padding_side = "left"

    def __call__(
        self,
        texts,
        padding,
        truncation,
        max_length,
        add_special_tokens,
        return_tensors,
    ):
        token_rows = [
            [1 + (ord(character) % 7) for character in text][:max_length]
            for text in texts
        ]
        width = max(len(row) for row in token_rows)
        tokens = torch.full((len(texts), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(texts), width), dtype=torch.long)
        for row_index, row in enumerate(token_rows):
            tokens[row_index, : len(row)] = torch.tensor(row)
            mask[row_index, : len(row)] = 1
        return {"input_ids": tokens, "attention_mask": mask}


class FakeConfig:
    d_model = 3


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.cfg = FakeConfig()
        self.hook_dict = {HOOK_NAME: object()}
        self.tokenizer = FakeTokenizer()

    def run_with_cache(self, tokens, names_filter, return_type=None):
        offsets = torch.arange(3, device=tokens.device, dtype=torch.float32)
        activations = tokens.unsqueeze(-1).float() + offsets
        return None, {names_filter: activations}


class ActivationCacheTests(unittest.TestCase):
    def test_cache_shards_stats_and_reader(self) -> None:
        model = FakeModel().eval()
        dataset = [{"text": "ab"}, {"text": "cde"}, {"text": "unused"}]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "activations"
            stats = cache_residual_activations(
                dataset,
                model,
                HOOK_NAME,
                target_num_tokens=7,
                output_dir=output_dir,
                batch_size=2,
                max_length=4,
                shard_size=3,
            )
            shards = sorted(output_dir.glob("shard_*.pt"))
            self.assertEqual([torch.load(path, weights_only=True).shape[0] for path in shards], [3, 3, 1])
            all_activations = torch.cat(
                [torch.load(path, weights_only=True).float() for path in shards]
            )
            self.assertEqual(stats["num_tokens"], 7)
            torch.testing.assert_close(stats["mean"], all_activations.mean(dim=0))
            torch.testing.assert_close(
                stats["std"], all_activations.std(dim=0, unbiased=False)
            )

            reader = ActivationShardDataset(output_dir, shuffle_shards=False, shuffle_within_shard=False)
            self.assertEqual(len(reader), 7)
            self.assertEqual(list(reader)[0].dtype, torch.float32)
            train_reader, validation_reader = split_activation_shards(output_dir, seed=0)
            self.assertEqual(len(train_reader) + len(validation_reader), 7)


if __name__ == "__main__":
    unittest.main()
