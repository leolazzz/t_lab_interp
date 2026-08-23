"""Small orchestration helpers for comparable steering experiments.

Notebook-style usage::

    model = load_model(device="cuda")
    results = evaluate_split_activation_steering(
        model, token_batches, all_directions, direction_split, val_ids,
        alphas, DEFAULT_HOOK_NAME, evaluation_split="val"
    )
    results.to_csv("outputs/results/raw_steering.csv", index=False)
"""

import json
import hashlib
import time
from contextlib import nullcontext
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import IterableDataset, get_worker_info
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm

from src.metrics import (
    activation_norm_ratio,
    distinct_n,
    external_clean_lm_nll,
    next_token_nll,
    repetition_rate,
    sae_feature_activation_metrics,
    token_level_kl,
)
from src.model import (
    get_clean_logits,
    get_logits_with_intervention,
    get_residual_activations,
)
from src.directions import (
    direction_split_hash,
    get_sae_hook_name,
    validate_direction_ids_for_usage,
    validate_direction_split,
)
from src.steering import apply_steering
from src.utils import PIPELINE_VERSION


TokenBatch = Tensor | tuple[Tensor, Tensor | None]
Directions = Tensor | Sequence[Tensor] | Mapping[Any, Tensor]

SUPPORTED_GENERATION_METHODS = frozenset({
    "raw", "relative_raw", "norm_preserving", "conditioned_kl_denoiser",
    "conditioned_reconstruction", "conditioned_kl", "conditioned_kl_retention",
    "conditioned_full", "gaussian_denoiser", "sae_denoiser", "sae_calibrated",
    "fluency_denoiser", "projected_fluency_denoiser", "incremental_fluency",
})


def validate_generation_methods(methods: Sequence[str]) -> None:
    """Fail before expensive work if a configured generation method is unknown."""
    unknown = sorted(set(methods) - SUPPORTED_GENERATION_METHODS)
    if unknown:
        raise ValueError(f"Unknown generation evaluation methods: {unknown}.")


@dataclass
class RuntimeProfile:
    """Lightweight cumulative stage timings and model-forward counters."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    total_clean_forwards: int = 0
    total_modified_forwards: int = 0

    def add(
        self,
        stage: str,
        seconds: float,
        *,
        clean_forwards: int = 0,
        modified_forwards: int = 0,
        configurations: int = 0,
        examples: int = 0,
    ) -> None:
        assert seconds >= 0 and clean_forwards >= 0 and modified_forwards >= 0
        self.total_clean_forwards += clean_forwards
        self.total_modified_forwards += modified_forwards
        self.rows.append({
            "stage": stage,
            "seconds": float(seconds),
            "clean_forwards": int(clean_forwards),
            "modified_forwards": int(modified_forwards),
            "configurations": int(configurations),
            "examples": int(examples),
            "configurations_per_second": (
                float(configurations) / seconds if seconds > 0 else float("nan")
            ),
        })

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=[
                "stage", "seconds", "clean_forwards", "modified_forwards",
                "configurations", "examples", "configurations_per_second",
            ])
        return pd.DataFrame(self.rows)

    def save(self, path: str | Path = "outputs/analysis/runtime_profile.csv") -> Path:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.frame().to_csv(resolved, index=False)
        return resolved

    def print_summary(self) -> None:
        frame = self.frame()
        if not frame.empty:
            summary = frame.groupby("stage", as_index=False).agg({
                "seconds": "sum", "clean_forwards": "sum",
                "modified_forwards": "sum", "configurations": "sum",
                "examples": "sum",
            })
            summary["configurations_per_second"] = (
                summary["configurations"] / summary["seconds"].clip(lower=1e-12)
            )
            print(summary.to_string(index=False))
        print({
            "total_clean_forwards": self.total_clean_forwards,
            "total_modified_forwards": self.total_modified_forwards,
        })


@dataclass
class CleanBatchCache:
    """Clean tensors reused by every intervention on one fixed token batch."""

    tokens: Tensor
    attention_mask: Tensor | None
    valid_mask: Tensor
    clean_log_probs: Tensor
    clean_probs: Tensor
    clean_nll: Tensor
    clean_h: Tensor


def _autocast_context(model: Any, enabled: bool) -> Any:
    device = next(model.parameters()).device
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _token_batches_hash(token_batches: Sequence[TokenBatch]) -> str:
    """Hash exact token IDs and masks for safe reusable evaluation outputs."""
    digest = hashlib.sha256()
    for batch in token_batches:
        tokens, mask = _unpack_batch(batch)
        digest.update(tokens.detach().cpu().contiguous().numpy().tobytes())
        digest.update(str(tuple(tokens.shape)).encode())
        if mask is not None:
            digest.update(mask.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:20]


def get_or_create_text_bank(
    dataset: Iterable[str | Mapping[str, Any]] | None,
    target_size: int,
    path: str | Path,
    text_column: str = "text",
) -> list[str]:
    """Persist the first deterministic non-empty text contexts for reuse."""
    assert target_size > 0
    resolved = Path(path)
    if resolved.exists():
        texts = json.loads(resolved.read_text(encoding="utf-8"))
        assert isinstance(texts, list) and len(texts) >= target_size
        assert all(isinstance(text, str) and text for text in texts)
        return texts[:target_size]
    assert dataset is not None, "A dataset is required to create a missing text bank."
    texts: list[str] = []
    for example in dataset:
        text = example if isinstance(example, str) else example[text_column]
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
        if len(texts) == target_size:
            break
    assert len(texts) == target_size, "Dataset ended before the text bank was complete."
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(texts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return texts


def tokenize_text_batches(
    model: HookedTransformer,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
) -> list[tuple[Tensor, Tensor]]:
    """Tokenize fixed texts into right-padded model-device batches."""
    assert texts and batch_size > 0 and max_length >= 2
    tokenizer = model.tokenizer
    assert tokenizer is not None
    if tokenizer.pad_token_id is None:
        assert tokenizer.eos_token is not None
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    batches: list[tuple[Tensor, Tensor]] = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start:start + batch_size]), padding=True, truncation=True,
                max_length=max_length - 1, add_special_tokens=False,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            mask = encoded["attention_mask"].bool()
            bos = torch.full(
                (input_ids.shape[0], 1), int(tokenizer.bos_token_id), dtype=torch.long
            )
            input_ids = torch.cat((bos, input_ids), dim=1).to(device)
            mask = torch.cat((torch.ones_like(bos, dtype=torch.bool), mask), dim=1).to(device)
            batches.append((input_ids, mask))
    finally:
        tokenizer.padding_side = old_padding_side
    return batches


@torch.inference_mode()
def tokenize_text_batches_with_oom_fallback(
    model: HookedTransformer,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    *,
    use_inference_autocast: bool = False,
    profiler: RuntimeProfile | None = None,
) -> tuple[list[tuple[Tensor, Tensor]], int]:
    """Tokenize and preflight a prompt batch size, halving only on CUDA OOM.

    Token IDs and masks are unchanged; only the grouping of prompts into model
    forwards changes.  The returned integer is the actually accepted size.
    """
    current_batch_size = int(batch_size)
    assert current_batch_size > 0
    while True:
        batches = tokenize_text_batches(
            model, texts, current_batch_size, max_length
        )
        try:
            start = time.perf_counter()
            with _autocast_context(model, use_inference_autocast):
                logits = get_clean_logits(model, batches[0][0])
            if profiler is not None:
                profiler.add(
                    "prompt batch preflight", time.perf_counter() - start,
                    clean_forwards=1, examples=int(batches[0][0].shape[0]),
                )
            del logits
            return batches, current_batch_size
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if not _is_cuda_oom(error) or current_batch_size == 1:
                raise
            smaller = max(1, current_batch_size // 2)
            print(f"CUDA OOM: reducing prompt batch {current_batch_size} -> {smaller}")
            current_batch_size = smaller
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def pareto_frontier(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    maximize_x: bool = True,
    maximize_y: bool = True,
) -> pd.DataFrame:
    """Return non-dominated rows for a two-metric Pareto comparison."""
    assert {x_column, y_column}.issubset(frame.columns)
    values = frame[[x_column, y_column]].to_numpy(dtype=float)
    signs = torch.tensor([1.0 if maximize_x else -1.0, 1.0 if maximize_y else -1.0])
    signed = torch.as_tensor(values) * signs
    keep = torch.ones(len(frame), dtype=torch.bool)
    for i in range(len(frame)):
        dominates = (signed >= signed[i]).all(dim=1) & (signed > signed[i]).any(dim=1)
        dominates[i] = False
        if dominates.any().item():
            keep[i] = False
    return frame.loc[keep.numpy()].copy()


def positive_strength_concept_retention(
    results: pd.DataFrame,
    raw_method: str = "raw",
    concept_column: str = "target_sae_activation",
    strength_column: str = "strength",
    threshold: float = 1e-8,
) -> pd.DataFrame:
    """Compute concept retention excluding zero/near-zero raw steering rows."""
    required = {"method", "prompt_id", "direction_id", strength_column, concept_column}
    assert required.issubset(results.columns)
    keys = ["prompt_id", "direction_id", strength_column]
    source_keys = ["method", *keys]
    assert not results.duplicated(source_keys).any(), (
        f"Duplicate concept rows for keys {source_keys}; refusing a many-to-many merge."
    )
    raw = results[results["method"] == raw_method][keys + [concept_column]].rename(
        columns={concept_column: "raw_concept_activation"}
    )
    assert not raw.duplicated(keys).any(), "Raw reference rows are not unique."
    merged = results.merge(raw, on=keys, how="inner")
    expected_rows = int(results[keys].merge(raw[keys], on=keys, how="inner").shape[0])
    assert len(merged) == expected_rows, "Unexpected row multiplication in concept merge."
    valid = (merged[strength_column] > 0) & (merged.raw_concept_activation > threshold)
    merged = merged.loc[valid].copy()
    merged["concept_retention"] = merged[concept_column] / merged.raw_concept_activation
    assert torch.isfinite(torch.as_tensor(merged.concept_retention.to_numpy())).all()
    raw_rows = merged[merged.method == raw_method]
    if not raw_rows.empty:
        raw_values = torch.as_tensor(raw_rows.concept_retention.to_numpy(), dtype=torch.float64)
        torch.testing.assert_close(raw_values, torch.ones_like(raw_values), atol=1e-12, rtol=0)
    return merged


def summarize_concept_retention(retention: pd.DataFrame) -> pd.DataFrame:
    """Return mean, median, standard deviation, and count per method."""
    assert {"method", "concept_retention"}.issubset(retention.columns)
    summary = (
        retention.groupby("method")["concept_retention"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def _load_activation_shard(path: Path) -> Tensor:
    shard = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(shard, Tensor) and shard.ndim == 2, (
        f"Expected a [num_tokens, d_model] tensor in {path}."
    )
    return shard


class ActivationShardDataset(IterableDataset[Tensor]):
    """Stream activation rows while holding at most one shard per worker.

    ``source`` may be an activation directory or an explicit sequence of shard
    paths. Shards and rows can be shuffled deterministically without loading
    the full activation collection into memory.
    """

    def __init__(
        self,
        source: str | Path | Sequence[str | Path],
        output_dtype: torch.dtype | None = torch.float32,
        shuffle_shards: bool = True,
        shuffle_within_shard: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            shard_paths = sorted(source_path.glob("shard_*.pt"))
        else:
            shard_paths = sorted(Path(path) for path in source)
        assert shard_paths, "No activation shards were found."

        self.shard_paths = shard_paths
        self.output_dtype = output_dtype
        self.shuffle_shards = shuffle_shards
        self.shuffle_within_shard = shuffle_within_shard
        self.seed = seed
        self._length = sum(_load_activation_shard(path).shape[0] for path in shard_paths)

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[Tensor]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers

        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        shard_indices = torch.arange(len(self.shard_paths))
        if self.shuffle_shards:
            shard_indices = shard_indices[torch.randperm(len(shard_indices), generator=generator)]
        shard_indices = shard_indices[worker_id::num_workers]

        row_generator = torch.Generator(device="cpu").manual_seed(
            self.seed + 1_000_003 * worker_id
        )
        for shard_index in shard_indices.tolist():
            shard = _load_activation_shard(self.shard_paths[shard_index])
            row_indices = torch.arange(shard.shape[0])
            if self.shuffle_within_shard:
                row_indices = row_indices[
                    torch.randperm(shard.shape[0], generator=row_generator)
                ]
            for row_index in row_indices.tolist():
                row = shard[row_index]
                if self.output_dtype is not None:
                    row = row.to(dtype=self.output_dtype)
                yield row
            del shard


def split_activation_shards(
    activation_dir: str | Path = "outputs/activations",
    validation_fraction: float = 0.1,
    seed: int = 42,
    output_dtype: torch.dtype | None = torch.float32,
) -> tuple[ActivationShardDataset, ActivationShardDataset]:
    """Create deterministic train/validation datasets split by whole shards."""
    assert 0.0 < validation_fraction < 1.0
    shard_paths = sorted(Path(activation_dir).glob("shard_*.pt"))
    assert len(shard_paths) >= 2, "At least two shards are needed for held-out validation."
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(len(shard_paths), generator=generator).tolist()
    num_validation = max(1, round(len(shard_paths) * validation_fraction))
    num_validation = min(num_validation, len(shard_paths) - 1)
    validation_indices = set(permutation[:num_validation])
    validation_paths = [
        path for index, path in enumerate(shard_paths) if index in validation_indices
    ]
    train_paths = [
        path for index, path in enumerate(shard_paths) if index not in validation_indices
    ]
    train_dataset = ActivationShardDataset(
        train_paths,
        output_dtype=output_dtype,
        shuffle_shards=True,
        shuffle_within_shard=True,
        seed=seed,
    )
    validation_dataset = ActivationShardDataset(
        validation_paths,
        output_dtype=output_dtype,
        shuffle_shards=False,
        shuffle_within_shard=False,
        seed=seed,
    )
    return train_dataset, validation_dataset


def _storage_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        if dtype not in dtype_map:
            raise ValueError("storage_dtype must be 'float16' or 'bfloat16'.")
        resolved = dtype_map[dtype]
    if resolved not in (torch.float16, torch.bfloat16):
        raise ValueError("Activation shards must use float16 or bfloat16.")
    return resolved


@torch.inference_mode()
def cache_residual_activations(
    dataset: Iterable[str | Mapping[str, Any]],
    model: HookedTransformer,
    hook_name: str,
    target_num_tokens: int,
    output_dir: str | Path = "outputs/activations",
    text_column: str = "text",
    batch_size: int = 8,
    max_length: int = 256,
    shard_size: int = 10_000,
    storage_dtype: str | torch.dtype = "float16",
    prepend_bos: bool = True,
    overwrite: bool = False,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Cache valid-token residual activations and streaming summary statistics.

    The dataset may be a HuggingFace Dataset or any iterable yielding strings
    or mappings containing ``text_column``. Shards are stored on CPU in a
    compact dtype; only a token batch and at most one pending shard are held in
    memory. Standard deviations are population standard deviations.
    """
    assert target_num_tokens > 0
    assert batch_size > 0 and max_length >= 2 and shard_size > 0
    assert hook_name in model.hook_dict, f"Unknown TransformerLens hook: {hook_name}"
    assert not model.training, "Activation caching requires model.eval()."

    tokenizer = model.tokenizer
    assert tokenizer is not None, "The model must have a tokenizer."
    assert tokenizer.pad_token_id is not None, (
        "Tokenizer must have a pad token before batched activation caching."
    )
    bos_token_id = tokenizer.bos_token_id
    if prepend_bos:
        assert bos_token_id is not None, "prepend_bos=True requires a BOS token ID."

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    existing_files = sorted(output_path.glob("shard_*.pt"))
    stats_path = output_path / "stats.pt"
    if (existing_files or stats_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Activation cache already exists in {output_path}. Pass overwrite=True "
            "only when replacing it deliberately."
        )
    if overwrite:
        for existing_file in existing_files:
            existing_file.unlink()
        if stats_path.exists():
            stats_path.unlink()

    disk_dtype = _storage_dtype(storage_dtype)
    model_device = next(model.parameters()).device
    d_model = model.cfg.d_model
    buffered: list[Tensor] = []
    buffered_tokens = 0
    shard_index = 0
    num_tokens = 0
    running_mean = torch.zeros(d_model, dtype=torch.float64)
    running_m2 = torch.zeros(d_model, dtype=torch.float64)
    norm_sum = torch.zeros((), dtype=torch.float64)
    norm_values: list[Tensor] = []

    def write_ready_shards(force: bool = False) -> None:
        nonlocal buffered, buffered_tokens, shard_index
        if not buffered or (buffered_tokens < shard_size and not force):
            return
        combined = torch.cat(buffered, dim=0)
        write_offset = 0
        while combined.shape[0] - write_offset >= shard_size:
            shard = combined[write_offset : write_offset + shard_size].contiguous()
            torch.save(shard, output_path / f"shard_{shard_index:03d}.pt")
            shard_index += 1
            write_offset += shard_size
        remainder = combined[write_offset:]
        if force and remainder.shape[0] > 0:
            torch.save(
                remainder.contiguous(), output_path / f"shard_{shard_index:03d}.pt"
            )
            shard_index += 1
            remainder = remainder[:0]
        buffered = [remainder] if remainder.shape[0] > 0 else []
        buffered_tokens = remainder.shape[0]

    def process_text_batch(texts: list[str]) -> None:
        nonlocal buffered_tokens, num_tokens, running_mean, running_m2, norm_sum
        tokenize_length = max_length - 1 if prepend_bos else max_length
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        try:
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=tokenize_length,
                add_special_tokens=False,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = original_padding_side

        tokens = encoded["input_ids"]
        attention_mask = encoded["attention_mask"].bool()
        if prepend_bos:
            bos_column = torch.full(
                (tokens.shape[0], 1), bos_token_id, dtype=tokens.dtype
            )
            tokens = torch.cat((bos_column, tokens), dim=1)
            attention_mask = torch.cat(
                (torch.ones((tokens.shape[0], 1), dtype=torch.bool), attention_mask),
                dim=1,
            )

        tokens = tokens.to(model_device)
        attention_mask = attention_mask.to(model_device)
        activations = get_residual_activations(model, tokens, hook_name)
        valid = activations[attention_mask]
        remaining = target_num_tokens - num_tokens
        valid = valid[:remaining]
        if valid.shape[0] == 0:
            return

        valid_cpu = valid.detach().to(device="cpu", dtype=torch.float32)
        stats_batch = valid_cpu.to(dtype=torch.float64)
        batch_count = stats_batch.shape[0]
        batch_mean = stats_batch.mean(dim=0)
        batch_m2 = (stats_batch - batch_mean).square().sum(dim=0)
        new_count = num_tokens + batch_count
        delta = batch_mean - running_mean
        running_mean = running_mean + delta * (batch_count / new_count)
        running_m2 = (
            running_m2
            + batch_m2
            + delta.square() * (num_tokens * batch_count / new_count)
        )
        batch_norms = torch.linalg.vector_norm(stats_batch, dim=-1)
        norm_sum = norm_sum + batch_norms.sum()
        norm_values.append(batch_norms.to(torch.float32))
        num_tokens = new_count

        disk_batch = valid_cpu.to(dtype=disk_dtype)
        buffered.append(disk_batch)
        buffered_tokens += disk_batch.shape[0]
        write_ready_shards()

    text_batch: list[str] = []
    progress = tqdm(total=target_num_tokens, desc="Caching activations", unit="tok")
    try:
        for example in dataset:
            if isinstance(example, str):
                text = example
            else:
                assert text_column in example, f"Missing text column {text_column!r}."
                text = example[text_column]
            assert isinstance(text, str)
            if not text:
                continue
            text_batch.append(text)
            if len(text_batch) < batch_size:
                continue
            before = num_tokens
            process_text_batch(text_batch)
            progress.update(num_tokens - before)
            text_batch = []
            if num_tokens >= target_num_tokens:
                break

        if text_batch and num_tokens < target_num_tokens:
            before = num_tokens
            process_text_batch(text_batch)
            progress.update(num_tokens - before)
    finally:
        progress.close()

    if num_tokens < target_num_tokens:
        raise ValueError(
            f"Dataset ended after {num_tokens} valid tokens; requested {target_num_tokens}."
        )
    write_ready_shards(force=True)
    population_std = torch.sqrt(running_m2 / num_tokens)
    all_norms = torch.cat(norm_values)
    assert all_norms.shape == (num_tokens,)
    stats = {
        "mean": running_mean.to(torch.float32),
        "std": population_std.to(torch.float32),
        "mean_activation_norm": (norm_sum / num_tokens).to(torch.float32),
        "median_activation_norm": all_norms.median(),
        "p95_activation_norm": torch.quantile(all_norms, 0.95),
        "p99_activation_norm": torch.quantile(all_norms, 0.99),
        "num_tokens": num_tokens,
        "d_model": d_model,
        "hook_name": hook_name,
        "storage_dtype": str(disk_dtype).removeprefix("torch."),
        "num_shards": shard_index,
        "pipeline_version": PIPELINE_VERSION,
    }
    torch.save(stats, stats_path)
    metadata = {
        "model_name": getattr(model.cfg, "model_name", "gpt2-small"),
        "hook_name": hook_name,
        "d_model": d_model,
        "num_tokens": num_tokens,
        "storage_dtype": stats["storage_dtype"],
        "dataset_text_column": text_column,
        "dataset_name": dataset_name,
        "max_length": max_length,
        "prepend_bos": prepend_bos,
        "num_shards": shard_index,
        "pipeline_version": PIPELINE_VERSION,
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def load_or_cache_residual_activations(
    dataset: Iterable[str | Mapping[str, Any]],
    model: HookedTransformer,
    hook_name: str,
    target_num_tokens: int,
    output_dir: str | Path = "outputs/activations",
    **kwargs: Any,
) -> dict[str, Any]:
    """Reuse a compatible activation cache or create it once.

    Existing shards are never silently accepted: hook, width, token count and
    shard metadata must match the requested experiment.
    """
    output_path = Path(output_dir)
    stats_path = output_path / "stats.pt"
    shard_paths = sorted(output_path.glob("shard_*.pt"))
    if stats_path.exists() and shard_paths:
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        assert isinstance(stats, dict)
        assert stats.get("hook_name") == hook_name
        assert int(stats.get("d_model", -1)) == int(model.cfg.d_model)
        assert int(stats.get("num_tokens", -1)) == int(target_num_tokens)
        assert int(stats.get("num_shards", -1)) == len(shard_paths)
        metadata_path = output_path / "metadata.json"
        assert metadata_path.exists(), "Activation cache metadata.json is missing."
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata.get("hook_name") == hook_name
        assert int(metadata.get("d_model", -1)) == int(model.cfg.d_model)
        assert int(metadata.get("num_tokens", -1)) == int(target_num_tokens)
        requested_dataset_name = kwargs.get("dataset_name")
        cached_dataset_name = metadata.get("dataset_name")
        if requested_dataset_name is not None and cached_dataset_name is not None:
            assert cached_dataset_name == requested_dataset_name, (
                "Activation cache was built from a different dataset."
            )
        elif requested_dataset_name is not None:
            metadata["dataset_name"] = requested_dataset_name
        norm_chunks: list[Tensor] = []
        for shard in shard_paths:
            tensor = torch.load(shard, map_location="cpu", weights_only=True)
            assert isinstance(tensor, Tensor) and tensor.ndim == 2
            assert tensor.shape[1] == model.cfg.d_model
            if "median_activation_norm" not in stats:
                norm_chunks.append(torch.linalg.vector_norm(tensor.float(), dim=-1))
        if norm_chunks:
            norms = torch.cat(norm_chunks)
            stats["median_activation_norm"] = norms.median()
            stats["p95_activation_norm"] = torch.quantile(norms, 0.95)
            stats["p99_activation_norm"] = torch.quantile(norms, 0.99)
        # Clean residual vectors are mathematically unchanged by the denoiser
        # pipeline revision, so compatible legacy cache metadata can be upgraded.
        if stats.get("pipeline_version") != PIPELINE_VERSION:
            stats["pipeline_version"] = PIPELINE_VERSION
            metadata["pipeline_version"] = PIPELINE_VERSION
        torch.save(stats, stats_path)
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return stats
    if stats_path.exists() or shard_paths:
        raise ValueError(
            f"Incomplete or incompatible activation cache in {output_path}; "
            "remove it deliberately or repair it before rerunning."
        )
    return cache_residual_activations(
        dataset,
        model,
        hook_name,
        target_num_tokens,
        output_dir=output_path,
        **kwargs,
    )


def compact_activation_cache_summary(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Return log-safe cache statistics without printing 768-dimensional tensors."""
    std = stats["std"].float()
    return {
        "num_tokens": int(stats["num_tokens"]),
        "d_model": int(stats["d_model"]),
        "mean_activation_norm": float(stats["mean_activation_norm"]),
        "median_activation_norm": float(stats["median_activation_norm"]),
        "p95_activation_norm": float(stats["p95_activation_norm"]),
        "p99_activation_norm": float(stats["p99_activation_norm"]),
        "coordinate_std_min": std.min().item(),
        "coordinate_std_median": std.median().item(),
        "coordinate_std_max": std.max().item(),
        "num_shards": int(stats["num_shards"]),
    }


def _unpack_batch(batch: TokenBatch) -> tuple[Tensor, Tensor | None]:
    if isinstance(batch, Tensor):
        return batch, None
    assert len(batch) == 2, "A token batch tuple must contain tokens and attention_mask."
    tokens, attention_mask = batch
    assert isinstance(tokens, Tensor)
    assert attention_mask is None or isinstance(attention_mask, Tensor)
    return tokens, attention_mask


def _list_directions(directions: Directions) -> list[tuple[Any, Tensor]]:
    if isinstance(directions, Mapping):
        items = list(directions.items())
    elif isinstance(directions, Tensor):
        assert directions.ndim in (1, 2), (
            "directions must have shape [d_model] or [num_directions, d_model]."
        )
        direction_rows = directions.unsqueeze(0) if directions.ndim == 1 else directions
        items = list(enumerate(direction_rows))
    else:
        items = list(enumerate(directions))

    assert items, "At least one steering direction is required."
    for _, direction in items:
        assert isinstance(direction, Tensor) and direction.ndim == 1
    return items


@torch.inference_mode()
def evaluate_activation_steering(
    model: HookedTransformer,
    token_batches: Iterable[TokenBatch],
    directions: Directions,
    alphas: Sequence[float],
    hook_name: str,
    method: str = "raw",
    *,
    direction_split: Mapping[str, Any],
    evaluation_split: str,
) -> pd.DataFrame:
    """Evaluate downstream disruption for every batch, direction, and alpha.

    Clean logits and clean hook activations are computed once per token batch.
    Directions are not moved or cast implicitly; they must already match the
    model activation's device and dtype. Directions must be a mapping keyed by
    global SAE feature ID so split provenance can be checked before any model
    forward pass.
    """
    assert isinstance(directions, Mapping), (
        "Leakage-safe evaluation requires {global_direction_id: direction} mapping."
    )
    direction_items = _list_directions(directions)
    requested_ids = [direction_id for direction_id, _ in direction_items]
    assert all(isinstance(direction_id, int) for direction_id in requested_ids)
    if evaluation_split == "val":
        usage = "preliminary_evaluation"
    elif evaluation_split == "test":
        usage = "final_evaluation"
    else:
        raise ValueError("evaluation_split must be 'val' or 'test'.")
    validate_direction_ids_for_usage(requested_ids, direction_split, usage=usage)
    alpha_values = list(alphas)
    assert alpha_values, "At least one alpha is required."
    assert hook_name in model.hook_dict, f"Unknown TransformerLens hook: {hook_name}"

    rows: list[dict[str, Any]] = []
    for batch_id, batch in enumerate(token_batches):
        tokens, attention_mask = _unpack_batch(batch)
        clean_logits, cache = model.run_with_cache(tokens, names_filter=hook_name)
        clean_h = cache[hook_name]
        del cache

        assert isinstance(clean_logits, Tensor)
        assert clean_h.shape == (
            tokens.shape[0],
            tokens.shape[1],
            model.cfg.d_model,
        )
        clean_nll = next_token_nll(clean_logits, tokens, attention_mask)
        assert isinstance(clean_nll, Tensor)

        for direction_id, direction in direction_items:
            for alpha in alpha_values:
                modified_h: Tensor | None = None

                def intervention(activation: Tensor) -> Tensor:
                    nonlocal modified_h
                    experiment_method = {
                        "raw": "relative_raw",
                        "norm_preserving": "relative_norm_preserving",
                    }.get(method, method)
                    modified_h = apply_steering(
                        activation,
                        direction,
                        alpha,
                        method=experiment_method,
                    )
                    return modified_h

                modified_logits = get_logits_with_intervention(
                    model=model,
                    tokens=tokens,
                    hook_name=hook_name,
                    intervention_fn=intervention,
                )
                assert modified_h is not None

                kl = token_level_kl(
                    clean_logits,
                    modified_logits,
                    attention_mask=attention_mask,
                )
                modified_nll = next_token_nll(
                    modified_logits,
                    tokens,
                    attention_mask,
                )
                assert isinstance(modified_nll, Tensor)
                norm_ratio = activation_norm_ratio(clean_h, modified_h)

                rows.append(
                    {
                        "method": method,
                        "direction_id": direction_id,
                        "alpha": alpha,
                        "batch_id": batch_id,
                        "kl": kl.item(),
                        "clean_nll": clean_nll.item(),
                        "modified_nll": modified_nll.item(),
                        "delta_nll": (modified_nll - clean_nll).item(),
                        "activation_norm_ratio": norm_ratio.item(),
                        "steering_scale": "relative_activation_norm",
                        "token_positions": "all",
                    }
                )
                del modified_logits, modified_h

        del clean_logits, clean_h

    return pd.DataFrame(
        rows,
        columns=[
            "method",
            "direction_id",
            "alpha",
            "batch_id",
            "kl",
            "clean_nll",
            "modified_nll",
            "delta_nll",
            "activation_norm_ratio",
        ],
    )


@torch.inference_mode()
def evaluate_split_activation_steering(
    model: HookedTransformer,
    token_batches: Iterable[TokenBatch],
    all_directions: Tensor,
    direction_split: Mapping[str, Any],
    direction_ids: Sequence[int],
    alphas: Sequence[float],
    hook_name: str,
    evaluation_split: str,
    method: str = "raw",
) -> pd.DataFrame:
    """Leakage-safe wrapper for validation or final-test baseline evaluation."""
    if evaluation_split == "val":
        usage = "preliminary_evaluation"
    elif evaluation_split == "test":
        usage = "final_evaluation"
    else:
        raise ValueError("evaluation_split must be 'val' or 'test'.")
    requested_ids = [int(direction_id) for direction_id in direction_ids]
    validate_direction_ids_for_usage(requested_ids, direction_split, usage=usage)
    assert all_directions.shape[0] == int(direction_split["num_features"])
    directions = {
        direction_id: all_directions[direction_id] for direction_id in requested_ids
    }
    results = evaluate_activation_steering(
        model,
        token_batches,
        directions,
        alphas,
        hook_name,
        method=method,
        direction_split=direction_split,
        evaluation_split=evaluation_split,
    )
    results.insert(0, "evaluation_split", evaluation_split)
    return results


@torch.inference_mode()
def evaluate_token_level_methods(
    model: HookedTransformer,
    token_batches: Iterable[TokenBatch],
    all_directions: Tensor,
    direction_split: Mapping[str, Any],
    direction_ids: Sequence[int],
    strengths: Sequence[float],
    hook_name: str,
    methods: Sequence[str],
    denoisers: Mapping[str, nn.Module] | None = None,
    normalization_stats: Mapping[str, Mapping[str, Any]] | None = None,
    evaluation_config: Mapping[str, Any] | None = None,
    evaluation_split: str = "val",
    sae: Any | None = None,
    neighbor_index: Any | None = None,
) -> pd.DataFrame:
    """Compare all methods on shared tokens and shared clean forward passes.

    Steering modifies every token activation by default and uses the
    authoritative relative update. This is necessary for token-level
    next-token NLL: changing only the final activation changes logits without a
    corresponding next-token target. Generation evaluation remains final-token
    only. Each output row represents one input batch, direction, strength, and
    method. Clean logits/NLL are computed exactly once per batch.
    """
    assert evaluation_split in {"val", "test"}
    usage = "preliminary_evaluation" if evaluation_split == "val" else "final_evaluation"
    ids = [int(value) for value in direction_ids]
    validate_direction_ids_for_usage(ids, direction_split, usage=usage)
    assert len(methods) == len(set(methods)) and methods
    assert strengths and torch.isfinite(torch.tensor(strengths)).all()
    assert all_directions.shape == (
        int(direction_split["num_features"]), int(model.cfg.d_model)
    )
    model_parameter = next(model.parameters())
    assert all_directions.device == model_parameter.device
    assert all_directions.dtype == model_parameter.dtype
    denoisers = {} if denoisers is None else denoisers
    normalization_stats = {} if normalization_stats is None else normalization_stats
    evaluation_config = {} if evaluation_config is None else evaluation_config
    token_positions = str(
        evaluation_config.get("token_evaluation_positions", "all_tokens")
    )
    assert token_positions == "all_tokens", (
        "Token-level KL/NLL evaluation must use token_evaluation_positions="
        "'all_tokens'; final-token-only intervention makes next-token NLL "
        "identically unchanged."
    )
    rows: list[dict[str, Any]] = []

    for prompt_id, batch in enumerate(token_batches):
        tokens, attention_mask = _unpack_batch(batch)
        clean_logits, cache = model.run_with_cache(tokens, names_filter=hook_name)
        clean_h = cache[hook_name]
        del cache
        clean_nll = next_token_nll(clean_logits, tokens, attention_mask)
        assert isinstance(clean_nll, Tensor)

        for direction_id in ids:
            direction = all_directions[direction_id]
            for strength in strengths:
                for method in methods:
                    intervention = _generation_intervention(
                        method, direction, float(strength), denoisers,
                        normalization_stats, evaluation_config,
                        token_positions=token_positions,
                    )
                    modified_h: Tensor | None = None

                    def capture(activation: Tensor) -> Tensor:
                        nonlocal modified_h
                        modified_h = intervention(activation)
                        return modified_h

                    modified_logits = get_logits_with_intervention(
                        model, tokens, hook_name, capture
                    )
                    assert modified_h is not None
                    kl = token_level_kl(clean_logits, modified_logits, attention_mask)
                    modified_nll = next_token_nll(
                        modified_logits, tokens, attention_mask
                    )
                    assert isinstance(modified_nll, Tensor)
                    valid_mask = (
                        torch.ones_like(tokens, dtype=torch.bool)
                        if attention_mask is None
                        else attention_mask.to(dtype=torch.bool)
                    )
                    assert valid_mask.shape == tokens.shape and valid_mask.any()
                    # Keep a singleton sequence dimension for metric/helper APIs.
                    selected_clean = clean_h[valid_mask].unsqueeze(1)
                    selected_modified = modified_h[valid_mask].unsqueeze(1)
                    norm_ratio = activation_norm_ratio(
                        selected_clean, selected_modified
                    )
                    activation_mse = (
                        (selected_modified - selected_clean).float().square().mean()
                    )
                    row: dict[str, Any] = {
                        "pipeline_version": PIPELINE_VERSION,
                        "evaluation_split": evaluation_split,
                        "method": method,
                        "direction_id": direction_id,
                        "prompt_id": prompt_id,
                        "strength": float(strength),
                        "alpha": float(strength),
                        "kl": kl.detach().item(),
                        "clean_nll": clean_nll.detach().item(),
                        "modified_nll": modified_nll.detach().item(),
                        "delta_nll": (modified_nll - clean_nll).detach().item(),
                        "activation_norm_ratio": norm_ratio.detach().item(),
                        "activation_mse": activation_mse.detach().item(),
                        "steering_scale": "relative_activation_norm",
                        "token_positions": token_positions,
                    }
                    is_denoiser = method not in {"raw", "norm_preserving"}
                    is_zero = float(strength) == 0.0
                    row["clean_identity_kl"] = row["kl"] if is_zero and is_denoiser else None
                    row["clean_identity_delta_nll"] = (
                        row["delta_nll"] if is_zero and is_denoiser else None
                    )
                    row["clean_identity_activation_mse"] = (
                        row["activation_mse"] if is_zero and is_denoiser else None
                    )
                    if sae is not None:
                        feature = sae_feature_activation_metrics(
                            sae, modified_h, feature_id=direction_id,
                            token_mask=valid_mask,
                            threshold=float(evaluation_config.get("concept_threshold", 0.0)),
                        )
                        row["target_sae_activation"] = feature["concept_score"].detach().item()
                        row["concept_score"] = row["target_sae_activation"]
                    if neighbor_index is not None:
                        from src.metrics import denoiser_nearest_clean_cosine
                        raw_selected = apply_steering(
                            selected_clean, direction, float(strength),
                            method="relative_raw",
                        )
                        row["knn_distance"] = neighbor_index.knn_distance(
                            selected_modified.float()
                        )
                        if is_denoiser:
                            cosine = denoiser_nearest_clean_cosine(
                                raw_selected, selected_modified, neighbor_index
                            )
                            row["nearest_clean_correction_cosine"] = (
                                cosine.float().mean().detach().item()
                            )
                        else:
                            row["nearest_clean_correction_cosine"] = None
                    rows.append(row)

                    if method == "raw" and is_zero:
                        assert abs(row["kl"]) <= 1e-5
                        assert abs(row["delta_nll"]) <= 1e-5
                        assert row["activation_mse"] <= 1e-12
                    del modified_logits, modified_h
        del clean_logits, clean_h

    result = pd.DataFrame(rows)
    validate_result_metrics(result)
    return result


def _vectorized_relative_method(
    activation: Tensor,
    directions: Tensor,
    strengths: Tensor,
    method: str,
    denoisers: Mapping[str, nn.Module],
    normalization_stats: Mapping[str, Mapping[str, Any]],
    evaluation_config: Mapping[str, Any],
) -> Tensor:
    """Apply one method to ``[K,B,S,D]`` without per-example Python loops."""
    from src.train import denoise_activations
    from src.utils import normalize_activations

    assert activation.ndim == 4 and directions.ndim == 2 and strengths.ndim == 1
    assert activation.shape[0] == directions.shape[0] == strengths.shape[0]
    assert activation.shape[-1] == directions.shape[-1]
    unit = directions / torch.linalg.vector_norm(
        directions, dim=-1, keepdim=True
    ).clamp_min(torch.finfo(directions.dtype).tiny)
    unit = unit[:, None, None, :]
    scale = strengths[:, None, None, None]

    def relative_step(clean: Tensor, step_strength: Tensor) -> Tensor:
        norm = torch.linalg.vector_norm(clean, dim=-1, keepdim=True)
        return clean + step_strength * norm * unit

    if method in {"raw", "relative_raw"}:
        return relative_step(activation, scale)
    if method == "norm_preserving":
        original_norm = torch.linalg.vector_norm(activation, dim=-1, keepdim=True)
        steered = relative_step(activation, scale)
        steered_norm = torch.linalg.vector_norm(steered, dim=-1, keepdim=True)
        return steered * (
            original_norm / steered_norm.clamp_min(torch.finfo(steered.dtype).tiny)
        )

    key_by_method = {
        "gaussian_denoiser": "gaussian",
        "sae_denoiser": "sae_calibrated",
        "sae_calibrated": "sae_calibrated",
        "fluency_denoiser": "fluency",
        "projected_fluency_denoiser": "fluency",
        "incremental_fluency": "fluency",
    }
    assert method in key_by_method, f"Unknown optimized method {method!r}."
    denoiser_key = key_by_method[method]
    if denoiser_key == "sae_calibrated" and denoiser_key not in denoisers:
        denoiser_key = "sae"
    assert denoiser_key in denoisers and denoiser_key in normalization_stats
    denoiser = denoisers[denoiser_key]
    stats = normalization_stats[denoiser_key]
    parameter = next(denoiser.parameters())
    assert activation.device == parameter.device

    mean = stats["mean"].to(device=activation.device, dtype=activation.dtype)
    std = stats["std"].to(device=activation.device, dtype=activation.dtype)
    eps = float(stats.get("eps", 1e-6))

    def denoise(clean: Tensor, corrupted: Tensor) -> Tensor:
        # Transformer autocast can expose fp16 residuals while checkpoints stay
        # fp32.  The explicit round trip keeps checkpoint inference semantics
        # unchanged and restores the hook's required activation dtype.
        denoiser_clean = clean.to(dtype=parameter.dtype)
        denoiser_corrupted = corrupted.to(dtype=parameter.dtype)
        denoiser_mean = mean.to(dtype=parameter.dtype)
        denoiser_std = std.to(dtype=parameter.dtype)
        clean_z = normalize_activations(
            denoiser_clean, denoiser_mean, denoiser_std, eps
        )
        corrupted_z = normalize_activations(
            denoiser_corrupted, denoiser_mean, denoiser_std, eps
        )
        noise_level = torch.linalg.vector_norm(corrupted_z - clean_z, dim=-1)
        result = denoise_activations(
            denoiser, denoiser_corrupted, stats, noise_level=noise_level
        )
        return result.to(dtype=corrupted.dtype)

    def one_step(clean: Tensor, step_strength: Tensor, projected: bool) -> Tensor:
        steered = relative_step(clean, step_strength)
        denoised = denoise(clean, steered)
        if not projected:
            return denoised
        correction = denoised - steered
        correction_perp = correction - (correction * unit).sum(
            dim=-1, keepdim=True
        ) * unit
        return steered + float(evaluation_config.get("projected_beta", 1.0)) * correction_perp

    if method in {
        "gaussian_denoiser", "sae_denoiser", "sae_calibrated", "fluency_denoiser"
    }:
        return one_step(activation, scale, projected=False)
    if method == "projected_fluency_denoiser":
        return one_step(activation, scale, projected=True)

    n_steps = int(evaluation_config["incremental_steps"])
    assert n_steps in {1, 2, 4, 8}
    current = activation
    step_strength = scale / n_steps
    projected = bool(evaluation_config.get("incremental_projected", False))
    for _ in range(n_steps):
        current = one_step(current, step_strength, projected=projected)
    return current


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


@torch.inference_mode()
def evaluate_token_level_methods_fast(
    model: HookedTransformer,
    token_batches: Iterable[TokenBatch],
    all_directions: Tensor,
    direction_split: Mapping[str, Any],
    direction_ids: Sequence[int],
    strengths: Sequence[float],
    hook_name: str,
    methods: Sequence[str],
    denoisers: Mapping[str, nn.Module] | None = None,
    normalization_stats: Mapping[str, Mapping[str, Any]] | None = None,
    evaluation_config: Mapping[str, Any] | None = None,
    evaluation_split: str = "val",
    sae: Any | None = None,
    neighbor_index: Any | None = None,
    *,
    intervention_batch_size: int = 8,
    use_inference_autocast: bool = False,
    partial_jsonl_path: str | Path | None = None,
    checkpoint_fingerprints: Mapping[str, str] | None = None,
    sae_identity: str | None = None,
    profiler: RuntimeProfile | None = None,
) -> pd.DataFrame:
    """Batched, resumable equivalent of :func:`evaluate_token_level_methods`.

    Clean logits/log-probabilities and the hook activation are computed once
    per fixed prompt batch. Intervention configurations are expanded along a
    small ``K`` dimension and flattened to one TransformerLens forward. CUDA
    OOM halves ``K`` and retries without dropping or changing configurations.
    """
    assert evaluation_split in {"val", "test"}
    usage = "preliminary_evaluation" if evaluation_split == "val" else "final_evaluation"
    ids = [int(value) for value in direction_ids]
    validate_direction_ids_for_usage(ids, direction_split, usage=usage)
    assert methods and len(methods) == len(set(methods))
    strength_values = [float(value) for value in strengths]
    assert strength_values and intervention_batch_size > 0
    evaluation_config = {} if evaluation_config is None else evaluation_config
    denoisers = {} if denoisers is None else denoisers
    normalization_stats = {} if normalization_stats is None else normalization_stats
    assert evaluation_config.get("token_evaluation_positions", "all_tokens") == "all_tokens"
    parameter = next(model.parameters())
    assert all_directions.shape == (
        int(direction_split["num_features"]), int(model.cfg.d_model)
    )
    assert all_directions.device == parameter.device
    assert all_directions.dtype == parameter.dtype
    for denoiser in denoisers.values():
        denoiser.eval()
        denoiser_parameter = next(denoiser.parameters())
        assert denoiser_parameter.device == parameter.device
        assert denoiser_parameter.dtype == parameter.dtype

    batches = list(token_batches)
    assert batches
    cache_metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "evaluator_version": "batched_exact_v1",
        "model": str(getattr(model.cfg, "model_name", "gpt2-small")),
        "hook": hook_name,
        "sae": sae_identity or (
            str(get_sae_hook_name(sae)) if sae is not None else None
        ),
        "direction_split_hash": direction_split_hash(direction_split),
        "prompt_set_hash": _token_batches_hash(batches),
        "strengths": strength_values,
        "methods": list(methods),
        "direction_ids": ids,
        "checkpoint_fingerprints": dict(checkpoint_fingerprints or {}),
        "token_positions": "all_tokens",
        "evaluation_split": evaluation_split,
        "autocast": bool(use_inference_autocast),
        "method_parameters": {
            "projected_beta": evaluation_config.get("projected_beta", 1.0),
            "incremental_steps": evaluation_config.get("incremental_steps"),
            "incremental_projected": evaluation_config.get(
                "incremental_projected", False
            ),
        },
    }
    existing_rows: list[dict[str, Any]] = []
    completed: set[tuple[int, str, int, float]] = set()
    partial_path = Path(partial_jsonl_path) if partial_jsonl_path is not None else None
    if partial_path is not None:
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = partial_path.with_suffix(partial_path.suffix + ".meta.json")
        if partial_path.exists() and metadata_path.exists():
            saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if saved_metadata == cache_metadata:
                for line in partial_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        existing_rows.append(row)
                        completed.add((
                            int(row["prompt_id"]), str(row["method"]),
                            int(row["direction_id"]), float(row["strength"]),
                        ))
            else:
                partial_path.write_text("", encoding="utf-8")
        metadata_path.write_text(
            json.dumps(cache_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    runtime = profiler if profiler is not None else RuntimeProfile()
    new_rows: list[dict[str, Any]] = []
    threshold = float(evaluation_config.get("concept_threshold", 0.0))

    for prompt_id, batch in enumerate(batches):
        expected_keys = {
            (prompt_id, method, direction_id, strength)
            for method in methods for direction_id in ids
            for strength in strength_values
        }
        if expected_keys.issubset(completed):
            continue
        tokens, attention_mask = _unpack_batch(batch)
        valid_mask = (
            torch.ones_like(tokens, dtype=torch.bool)
            if attention_mask is None else attention_mask.bool()
        )
        clean_start = time.perf_counter()
        with _autocast_context(model, use_inference_autocast):
            clean_logits, hook_cache = model.run_with_cache(
                tokens, names_filter=hook_name
            )
        clean_seconds = time.perf_counter() - clean_start
        clean_h = hook_cache[hook_name]
        del hook_cache
        metric_start = time.perf_counter()
        clean_log_probs = F.log_softmax(clean_logits.float(), dim=-1)
        clean_probs = clean_log_probs.exp()
        clean_nll = next_token_nll(clean_logits.float(), tokens, attention_mask)
        assert isinstance(clean_nll, Tensor)
        del clean_logits
        runtime.add(
            "clean forward", clean_seconds, clean_forwards=1,
            examples=int(tokens.shape[0]),
        )
        runtime.add("metric calculation", time.perf_counter() - metric_start)
        clean_cache = CleanBatchCache(
            tokens=tokens,
            attention_mask=attention_mask,
            valid_mask=valid_mask,
            clean_log_probs=clean_log_probs,
            clean_probs=clean_probs,
            clean_nll=clean_nll,
            clean_h=clean_h,
        )

        for method in methods:
            method_start = time.perf_counter()
            method_configs = [
                (method, direction_id, strength)
                for direction_id in ids for strength in strength_values
                if (prompt_id, method, direction_id, strength) not in completed
            ]

            def run_chunk(configs: list[tuple[str, int, float]]) -> None:
                if not configs:
                    return
                k = len(configs)
                batch_size, sequence_length = tokens.shape
                direction_tensor = torch.stack([
                    all_directions[direction_id] for _, direction_id, _ in configs
                ])
                strength_tensor = torch.tensor(
                    [strength for _, _, strength in configs],
                    device=parameter.device, dtype=parameter.dtype,
                )
                expanded_tokens = tokens.repeat(k, 1)
                modified_h_flat: Tensor | None = None

                def intervention(activation: Tensor) -> Tensor:
                    nonlocal modified_h_flat
                    shaped = activation.reshape(
                        k, batch_size, sequence_length, activation.shape[-1]
                    )
                    modified = _vectorized_relative_method(
                        shaped,
                        direction_tensor.to(dtype=activation.dtype),
                        strength_tensor.to(dtype=activation.dtype),
                        method,
                        denoisers,
                        normalization_stats,
                        evaluation_config,
                    )
                    modified_h_flat = modified.reshape_as(activation)
                    return modified_h_flat

                try:
                    modified_start = time.perf_counter()
                    with _autocast_context(model, use_inference_autocast):
                        modified_logits_flat = get_logits_with_intervention(
                            model, expanded_tokens, hook_name, intervention
                        )
                    modified_seconds = time.perf_counter() - modified_start
                except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                    if not _is_cuda_oom(error) or k == 1:
                        raise
                    del expanded_tokens
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    midpoint = k // 2
                    print(
                        f"CUDA OOM: reducing intervention batch {k} -> "
                        f"{midpoint}/{k - midpoint}"
                    )
                    run_chunk(configs[:midpoint])
                    run_chunk(configs[midpoint:])
                    return

                assert modified_h_flat is not None
                modified_stage = (
                    "projected evaluation"
                    if method == "projected_fluency_denoiser"
                    else "incremental evaluation"
                    if method == "incremental_fluency"
                    else "modified forward"
                )
                runtime.add(
                    modified_stage, modified_seconds,
                    modified_forwards=1, configurations=k,
                    examples=k * batch_size,
                )
                modified_logits = modified_logits_flat.reshape(
                    k, batch_size, sequence_length, -1
                )
                modified_h = modified_h_flat.reshape(
                    k, batch_size, sequence_length, -1
                )
                metric_start = time.perf_counter()
                clean_probs = clean_cache.clean_probs
                mask = valid_mask.to(dtype=torch.float32)
                target_mask = valid_mask[:, 1:].to(dtype=torch.float32)
                kl_values: list[Tensor] = []
                modified_nll_values: list[Tensor] = []
                # Keep the exact full-vocabulary KL, but reduce one K slice at
                # a time so [K,B,S,vocab] is never duplicated in float32.
                for metric_index in range(k):
                    logits_slice = modified_logits[metric_index].float()
                    modified_log_probs = F.log_softmax(logits_slice, dim=-1)
                    per_token_kl = (
                        clean_probs
                        * (clean_cache.clean_log_probs - modified_log_probs)
                    ).sum(dim=-1)
                    kl_values.append(
                        (per_token_kl * mask).sum() / mask.sum()
                    )
                    modified_token_nll = F.cross_entropy(
                        logits_slice[:, :-1, :].reshape(-1, logits_slice.shape[-1]),
                        tokens[:, 1:].reshape(-1), reduction="none",
                    ).reshape(batch_size, sequence_length - 1)
                    modified_nll_values.append(
                        (modified_token_nll * target_mask).sum() / target_mask.sum()
                    )
                    del logits_slice, modified_log_probs, per_token_kl
                    del modified_token_nll
                kl_tensor = torch.stack(kl_values)
                modified_nll_tensor = torch.stack(modified_nll_values)
                valid_expanded = valid_mask.unsqueeze(0).expand(k, -1, -1)
                selected_clean = clean_h.unsqueeze(0).expand(k, -1, -1, -1)[valid_expanded]
                selected_modified = modified_h[valid_expanded]
                clean_norm = torch.linalg.vector_norm(selected_clean, dim=-1)
                modified_norm = torch.linalg.vector_norm(selected_modified, dim=-1)
                # Reshape is valid because every K row uses the same token mask.
                valid_per_config = int(valid_mask.sum().item())
                norm_eps = max(1e-8, torch.finfo(clean_norm.dtype).tiny)
                norm_ratios = (
                    (modified_norm + norm_eps) / (clean_norm + norm_eps)
                ).reshape(k, valid_per_config).mean(dim=1)
                mse_values = (selected_modified - selected_clean).float().square().reshape(
                    k, valid_per_config, -1
                ).mean(dim=(1, 2))
                runtime.add("metric calculation", time.perf_counter() - metric_start)

                for index, (_, direction_id, strength) in enumerate(configs):
                    is_exact_simple_identity = (
                        strength == 0.0
                        and method in {"raw", "norm_preserving"}
                    )
                    row_kl = (
                        0.0 if is_exact_simple_identity
                        else float(kl_tensor[index].item())
                    )
                    row_modified_nll = (
                        float(clean_nll.item()) if is_exact_simple_identity
                        else float(modified_nll_tensor[index].item())
                    )
                    row_norm_ratio = (
                        1.0 if is_exact_simple_identity
                        else float(norm_ratios[index].item())
                    )
                    row_activation_mse = (
                        0.0 if is_exact_simple_identity
                        else float(mse_values[index].item())
                    )
                    row: dict[str, Any] = {
                        "pipeline_version": PIPELINE_VERSION,
                        "evaluation_split": evaluation_split,
                        "method": method,
                        "direction_id": direction_id,
                        "prompt_id": prompt_id,
                        "strength": strength,
                        "alpha": strength,
                        "kl": row_kl,
                        "clean_nll": float(clean_nll.item()),
                        "modified_nll": row_modified_nll,
                        "delta_nll": row_modified_nll - float(clean_nll.item()),
                        "activation_norm_ratio": row_norm_ratio,
                        "activation_mse": row_activation_mse,
                        "steering_scale": "relative_activation_norm",
                        "token_positions": "all_tokens",
                    }
                    is_denoiser = method not in {"raw", "norm_preserving"}
                    is_zero = strength == 0.0
                    row["clean_identity_kl"] = row["kl"] if is_zero and is_denoiser else None
                    row["clean_identity_delta_nll"] = (
                        row["delta_nll"] if is_zero and is_denoiser else None
                    )
                    row["clean_identity_activation_mse"] = (
                        row["activation_mse"] if is_zero and is_denoiser else None
                    )
                    if sae is not None:
                        sae_start = time.perf_counter()
                        sae_dtype = sae.W_dec.dtype
                        feature_activations = (
                            clean_h if is_exact_simple_identity
                            else modified_h[index]
                        )
                        feature = sae_feature_activation_metrics(
                            sae, feature_activations.to(dtype=sae_dtype),
                            direction_id, valid_mask, threshold
                        )
                        runtime.add("SAE feature extraction", time.perf_counter() - sae_start)
                        row["target_sae_activation"] = float(feature["concept_score"].item())
                        row["concept_score"] = row["target_sae_activation"]
                    if neighbor_index is not None:
                        from src.metrics import denoiser_nearest_clean_cosine
                        clean_selected = selected_clean.reshape(
                            k, valid_per_config, -1
                        )[index].unsqueeze(1)
                        modified_selected = selected_modified.reshape(
                            k, valid_per_config, -1
                        )[index].unsqueeze(1)
                        if is_exact_simple_identity:
                            modified_selected = clean_selected
                        raw_selected = apply_steering(
                            clean_selected, all_directions[direction_id], strength,
                            method="relative_raw",
                        )
                        row["knn_distance"] = neighbor_index.knn_distance(
                            modified_selected.float()
                        )
                        row["nearest_clean_correction_cosine"] = (
                            float(denoiser_nearest_clean_cosine(
                                raw_selected, modified_selected, neighbor_index
                            ).float().mean().item()) if is_denoiser else None
                        )
                    if method == "raw" and is_zero:
                        assert abs(row["kl"]) <= 1e-5
                        assert abs(row["delta_nll"]) <= 1e-5
                        assert row["activation_mse"] <= 1e-12
                    new_rows.append(row)
                    completed.add((prompt_id, method, direction_id, strength))

                del modified_logits_flat, modified_logits, kl_tensor
                del modified_nll_tensor, modified_h_flat, modified_h

            for start in range(0, len(method_configs), intervention_batch_size):
                run_chunk(method_configs[start:start + intervention_batch_size])

            method_seconds = time.perf_counter() - method_start
            if method_configs:
                print({
                    "stage": "token_evaluation",
                    "prompt_batch": prompt_id,
                    "method": method,
                    "completed_configurations": len(method_configs),
                    "seconds": round(method_seconds, 2),
                    "configurations_per_second": round(
                        len(method_configs) / max(method_seconds, 1e-12), 3
                    ),
                })

            if partial_path is not None:
                rows_for_group = [
                    row for row in new_rows
                    if row["prompt_id"] == prompt_id and row["method"] == method
                ]
                if rows_for_group:
                    with partial_path.open("a", encoding="utf-8") as output:
                        for row in rows_for_group:
                            output.write(json.dumps(row) + "\n")
                    existing_rows.extend(rows_for_group)
                    new_rows = [
                        row for row in new_rows if not (
                            row["prompt_id"] == prompt_id and row["method"] == method
                        )
                    ]

        if partial_path is not None:
            rows_for_prompt = [row for row in new_rows if row["prompt_id"] == prompt_id]
            if rows_for_prompt:
                with partial_path.open("a", encoding="utf-8") as output:
                    for row in rows_for_prompt:
                        output.write(json.dumps(row) + "\n")
                existing_rows.extend(rows_for_prompt)
                new_rows = [row for row in new_rows if row["prompt_id"] != prompt_id]
        del clean_cache, clean_log_probs, clean_probs, clean_h

    dataframe_start = time.perf_counter()
    result = pd.DataFrame(existing_rows + new_rows)
    if not result.empty:
        result = result.sort_values(
            ["prompt_id", "method", "direction_id", "strength"]
        ).reset_index(drop=True)
    runtime.add("dataframe construction", time.perf_counter() - dataframe_start)
    validate_result_metrics(result)
    if profiler is None:
        runtime.print_summary()
    return result


@torch.inference_mode()
def regression_check_optimized_evaluation(
    model: HookedTransformer,
    token_batches: Sequence[TokenBatch],
    all_directions: Tensor,
    direction_split: Mapping[str, Any],
    direction_ids: Sequence[int],
    strengths: Sequence[float],
    hook_name: str,
    methods: Sequence[str],
    denoisers: Mapping[str, nn.Module],
    normalization_stats: Mapping[str, Mapping[str, Any]],
    evaluation_config: Mapping[str, Any],
    sae: Any,
    path: str | Path = "outputs/analysis/optimized_eval_regression.csv",
    rtol: float = 2e-4,
    atol: float = 2e-5,
    check_autocast: bool = True,
) -> pd.DataFrame:
    """Fail unless batched evaluation matches the retained reference path."""
    tiny_batches = list(token_batches[:2])
    tiny_ids = list(direction_ids[:2])
    reference = evaluate_token_level_methods(
        model, tiny_batches, all_directions, direction_split, tiny_ids,
        strengths, hook_name, methods, denoisers, normalization_stats,
        evaluation_config, "val", sae,
    )
    optimized = evaluate_token_level_methods_fast(
        model, tiny_batches, all_directions, direction_split, tiny_ids,
        strengths, hook_name, methods, denoisers, normalization_stats,
        evaluation_config, "val", sae,
        intervention_batch_size=4, use_inference_autocast=False,
    )
    keys = ["prompt_id", "method", "direction_id", "strength"]
    metrics = ["kl", "delta_nll", "target_sae_activation", "activation_norm_ratio"]
    merged = reference[keys + metrics].merge(
        optimized[keys + metrics], on=keys, suffixes=("_reference", "_optimized"),
        validate="one_to_one",
    )
    assert len(merged) == len(reference) == len(optimized)
    for metric in metrics:
        reference_values = torch.tensor(merged[f"{metric}_reference"].to_numpy())
        optimized_values = torch.tensor(merged[f"{metric}_optimized"].to_numpy())
        difference = (reference_values - optimized_values).abs()
        merged[f"{metric}_abs_diff"] = difference.numpy()
        torch.testing.assert_close(
            optimized_values, reference_values, rtol=rtol, atol=atol
        )
    if check_autocast and next(model.parameters()).device.type == "cuda":
        autocast_result = evaluate_token_level_methods_fast(
            model, tiny_batches, all_directions, direction_split, tiny_ids,
            strengths, hook_name, methods, denoisers, normalization_stats,
            evaluation_config, "val", sae,
            intervention_batch_size=4, use_inference_autocast=True,
        )
        autocast_merged = reference[keys + metrics].merge(
            autocast_result[keys + metrics], on=keys,
            suffixes=("_reference", "_autocast"), validate="one_to_one",
        )
        for metric in metrics:
            autocast_values = torch.tensor(
                autocast_merged[f"{metric}_autocast"].to_numpy()
            )
            reference_values = torch.tensor(
                autocast_merged[f"{metric}_reference"].to_numpy()
            )
            torch.testing.assert_close(
                autocast_values, reference_values, rtol=5e-3, atol=5e-4
            )
            merged[f"{metric}_autocast_abs_diff"] = (
                reference_values - autocast_values
            ).abs().numpy()
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(resolved, index=False)
    return merged


@torch.inference_mode()
def score_train_direction_fluency(
    model: HookedTransformer,
    token_batches: Iterable[TokenBatch],
    all_directions: Tensor,
    direction_split: Mapping[str, Any],
    relative_strengths: Sequence[float],
    hook_name: str,
    max_contexts: int = 256,
    direction_batch_size: int = 4,
    output_path: str | Path = "outputs/direction_scores/train_scores.pt",
    csv_path: str | Path | None = "outputs/direction_scores/train_scores.csv",
    figure_path: str | Path | None = "outputs/figures/train_direction_kl_hist.png",
    *,
    use_inference_autocast: bool = False,
    partial_path: str | Path | None = None,
    profiler: RuntimeProfile | None = None,
    force_recompute: bool = False,
    sae_identity: str | None = None,
) -> dict[str, Any]:
    """Offline downstream-fluency scores for persisted training directions only.

    Directions are batched by repeating each token batch ``direction_batch_size``
    times. This avoids one full forward per direction while keeping the expanded
    ``[direction, batch, sequence, vocab]`` logits small enough to control GPU
    memory. Strengths are evaluated in separate forwards as a conservative OOM
    compromise. Clean logits are computed exactly once per original token batch.
    """
    assert all_directions.ndim == 2 and all_directions.is_floating_point()
    assert all_directions.shape[0] == int(direction_split["num_features"])
    train_ids = list(direction_split["train"])
    val_ids = list(direction_split["val"])
    test_ids = list(direction_split["test"])
    validate_direction_split(
        train_ids,
        val_ids,
        test_ids,
        int(direction_split["num_features"]),
    )
    assert train_ids and set(train_ids).isdisjoint(set(val_ids) | set(test_ids))
    assert len(train_ids) == len(set(train_ids))
    validate_direction_ids_for_usage(
        train_ids,
        direction_split,
        usage="damage_scoring",
        require_complete_split=True,
    )
    strengths = [float(strength) for strength in relative_strengths]
    assert strengths and all(strength >= 0.0 for strength in strengths)
    assert max_contexts > 0 and direction_batch_size > 0
    assert hook_name in model.hook_dict

    model_parameter = next(model.parameters())
    assert all_directions.device == model_parameter.device, (
        "Directions must already be on the model device."
    )
    assert all_directions.dtype == model_parameter.dtype, (
        "Directions must already use the model dtype."
    )
    direction_index = torch.tensor(
        train_ids, dtype=torch.long, device=all_directions.device
    )
    train_directions = all_directions.index_select(0, direction_index)
    direction_norms = torch.linalg.vector_norm(
        train_directions, dim=-1, keepdim=True
    )
    assert torch.all(direction_norms > torch.finfo(train_directions.dtype).tiny).item()
    train_directions = train_directions / direction_norms

    num_directions = len(train_ids)
    num_strengths = len(strengths)
    kl_sum = torch.zeros(num_directions, num_strengths, dtype=torch.float64)
    kl_square_sum = torch.zeros_like(kl_sum)
    kl_count = torch.zeros_like(kl_sum)
    delta_nll_sum = torch.zeros_like(kl_sum)
    nll_count = torch.zeros_like(kl_sum)
    batches = list(token_batches)
    assert batches, "No token batches were supplied for harmfulness scoring."
    score_path = Path(output_path)
    cache_metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "scorer_version": "harmfulness_fast_v1",
        "model": str(getattr(model.cfg, "model_name", "gpt2-small")),
        "hook": hook_name,
        "sae": sae_identity,
        "direction_split_hash": direction_split_hash(direction_split),
        "prompt_set_hash": _token_batches_hash(batches),
        "relative_strengths": strengths,
        "direction_ids": train_ids,
        "max_contexts": int(max_contexts),
        "direction_batch_size": int(direction_batch_size),
        "autocast": bool(use_inference_autocast),
    }
    cache_lookup_start = time.perf_counter()
    if score_path.exists() and not force_recompute:
        existing_payload = torch.load(
            score_path, map_location="cpu", weights_only=False
        )
        if existing_payload.get("result_cache_metadata") == cache_metadata:
            validate_direction_ids_for_usage(
                existing_payload["direction_ids"].tolist(),
                direction_split,
                usage="damage_scoring",
                require_complete_split=True,
            )
            if csv_path is not None and not Path(csv_path).exists():
                cached_frame: dict[str, Any] = {
                    "direction_id": existing_payload["direction_ids"].tolist(),
                    "mean_kl": existing_payload["mean_kl"].tolist(),
                    "std_kl": existing_payload["std_kl"].tolist(),
                    "mean_delta_nll": existing_payload["mean_delta_nll"].tolist(),
                }
                for strength_index, strength in enumerate(strengths):
                    cached_frame[f"kl_r_{strength:g}"] = existing_payload[
                        "kl_by_strength"
                    ][:, strength_index].tolist()
                    cached_frame[f"delta_nll_r_{strength:g}"] = existing_payload[
                        "delta_nll_by_strength"
                    ][:, strength_index].tolist()
                resolved_csv = Path(csv_path)
                resolved_csv.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(cached_frame).sort_values(
                    "mean_kl", ascending=False
                ).to_csv(resolved_csv, index=False)
            if profiler is not None:
                profiler.add(
                    "harmfulness cache reuse",
                    time.perf_counter() - cache_lookup_start,
                    configurations=num_directions * num_strengths,
                    examples=int(existing_payload["num_contexts"]),
                )
            return existing_payload

    partial_score_path = (
        Path(partial_path) if partial_path is not None
        else score_path.with_name(f"{score_path.stem}_partial.pt")
    )
    contexts_seen = 0
    completed_batches = 0
    resume_direction_start = 0
    if partial_score_path.exists() and not force_recompute:
        partial = torch.load(
            partial_score_path, map_location="cpu", weights_only=False
        )
        if partial.get("result_cache_metadata") == cache_metadata:
            kl_sum.copy_(partial["kl_sum"])
            kl_square_sum.copy_(partial["kl_square_sum"])
            kl_count.copy_(partial["kl_count"])
            delta_nll_sum.copy_(partial["delta_nll_sum"])
            nll_count.copy_(partial["nll_count"])
            contexts_seen = int(partial["contexts_seen"])
            completed_batches = int(partial["completed_batches"])
            resume_direction_start = int(partial.get("next_direction_start", 0))

    runtime = profiler if profiler is not None else RuntimeProfile()
    harmfulness_start = time.perf_counter()

    for batch_index, batch in enumerate(
        tqdm(batches, desc="Scoring train directions", unit="batch")
    ):
        if batch_index < completed_batches:
            continue
        if contexts_seen >= max_contexts:
            break
        tokens, attention_mask = _unpack_batch(batch)
        remaining_contexts = max_contexts - contexts_seen
        if tokens.shape[0] > remaining_contexts:
            tokens = tokens[:remaining_contexts]
            if attention_mask is not None:
                attention_mask = attention_mask[:remaining_contexts]
        assert tokens.ndim == 2 and tokens.shape[1] >= 2
        batch_size, sequence_length = tokens.shape
        contexts_before_batch = contexts_seen

        if attention_mask is None:
            token_mask = torch.ones_like(tokens, dtype=torch.bool)
        else:
            assert attention_mask.shape == tokens.shape
            assert attention_mask.device == tokens.device
            token_mask = attention_mask.bool()
        target_mask = token_mask[:, 1:]
        kl_denominator = token_mask.sum().item()
        nll_denominator = target_mask.sum().item()
        assert kl_denominator > 0 and nll_denominator > 0

        clean_start = time.perf_counter()
        with _autocast_context(model, use_inference_autocast):
            clean_logits = get_clean_logits(model, tokens)
        runtime.add(
            "harmfulness clean forward",
            time.perf_counter() - clean_start,
            clean_forwards=1,
            examples=batch_size,
        )
        clean_logits = clean_logits.float()
        clean_log_probs = F.log_softmax(clean_logits, dim=-1)
        clean_probs = clean_log_probs.exp()
        clean_token_nll = F.cross_entropy(
            clean_logits[:, :-1, :].reshape(-1, clean_logits.shape[-1]),
            tokens[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(batch_size, sequence_length - 1)

        first_direction = (
            resume_direction_start if batch_index == completed_batches else 0
        )
        for start in range(first_direction, num_directions, direction_batch_size):
            end = min(start + direction_batch_size, num_directions)
            direction_chunk = train_directions[start:end]
            chunk_size = direction_chunk.shape[0]
            expanded_tokens = tokens.repeat(chunk_size, 1)

            for strength_index, strength in enumerate(strengths):
                def relative_intervention(activation: Tensor) -> Tensor:
                    activation_by_direction = activation.reshape(
                        chunk_size,
                        batch_size,
                        sequence_length,
                        activation.shape[-1],
                    )
                    activation_norm = torch.linalg.vector_norm(
                        activation_by_direction,
                        dim=-1,
                        keepdim=True,
                    )
                    modified = activation_by_direction + (
                        strength
                        * activation_norm
                        * direction_chunk.to(dtype=activation.dtype)[:, None, None, :]
                    )
                    return modified.reshape_as(activation)

                modified_start = time.perf_counter()
                with _autocast_context(model, use_inference_autocast):
                    modified_logits = get_logits_with_intervention(
                        model,
                        expanded_tokens,
                        hook_name,
                        relative_intervention,
                    )
                runtime.add(
                    "harmfulness modified forward",
                    time.perf_counter() - modified_start,
                    modified_forwards=1,
                    configurations=chunk_size,
                    examples=chunk_size * batch_size,
                )
                modified_logits = modified_logits.reshape(
                    chunk_size,
                    batch_size,
                    sequence_length,
                    clean_logits.shape[-1],
                )
                for local_index in range(chunk_size):
                    logits_slice = modified_logits[local_index].float()
                    modified_log_probs = F.log_softmax(logits_slice, dim=-1)
                    per_token_kl = (
                        clean_probs * (clean_log_probs - modified_log_probs)
                    ).sum(dim=-1).clamp_min(0)
                    direction_kl_sum = (per_token_kl * token_mask).sum()
                    direction_kl_square_sum = (
                        per_token_kl.square() * token_mask
                    ).sum()
                    modified_token_nll = F.cross_entropy(
                        logits_slice[:, :-1, :].reshape(-1, logits_slice.shape[-1]),
                        tokens[:, 1:].reshape(-1), reduction="none",
                    ).reshape(batch_size, sequence_length - 1)
                    direction_delta_nll_sum = (
                        (modified_token_nll - clean_token_nll) * target_mask
                    ).sum()
                    global_index = start + local_index
                    kl_sum[global_index, strength_index] += (
                        direction_kl_sum.double().cpu()
                    )
                    kl_square_sum[global_index, strength_index] += (
                        direction_kl_square_sum.double().cpu()
                    )
                    kl_count[global_index, strength_index] += kl_denominator
                    delta_nll_sum[global_index, strength_index] += (
                        direction_delta_nll_sum.double().cpu()
                    )
                    nll_count[global_index, strength_index] += nll_denominator
                    del logits_slice, modified_log_probs, per_token_kl
                    del modified_token_nll
                del modified_logits

            # A restart may redo at most the current direction chunk, never the
            # complete harmfulness stage. The cursor is valid only with the
            # exact metadata hash checked above.
            partial_score_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "result_cache_metadata": cache_metadata,
                "kl_sum": kl_sum,
                "kl_square_sum": kl_square_sum,
                "kl_count": kl_count,
                "delta_nll_sum": delta_nll_sum,
                "nll_count": nll_count,
                "contexts_seen": contexts_before_batch,
                "completed_batches": batch_index,
                "next_direction_start": end,
            }, partial_score_path)

        contexts_seen = contexts_before_batch + batch_size
        completed_batches = batch_index + 1
        resume_direction_start = 0
        partial_score_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "result_cache_metadata": cache_metadata,
            "kl_sum": kl_sum,
            "kl_square_sum": kl_square_sum,
            "kl_count": kl_count,
            "delta_nll_sum": delta_nll_sum,
            "nll_count": nll_count,
            "contexts_seen": contexts_seen,
            "completed_batches": completed_batches,
            "next_direction_start": 0,
        }, partial_score_path)
        del clean_logits, clean_log_probs, clean_probs, clean_token_nll

    assert contexts_seen > 0, "No token contexts were scored."
    assert torch.all(kl_count > 0).item() and torch.all(nll_count > 0).item()
    kl_by_strength = kl_sum / kl_count
    delta_nll_by_strength = delta_nll_sum / nll_count
    mean_kl = kl_sum.sum(dim=1) / kl_count.sum(dim=1)
    mean_delta_nll = delta_nll_sum.sum(dim=1) / nll_count.sum(dim=1)
    second_moment = kl_square_sum.sum(dim=1) / kl_count.sum(dim=1)
    std_kl = (second_moment - mean_kl.square()).clamp_min(0).sqrt()
    score_ids = torch.tensor(train_ids, dtype=torch.long)
    assert set(score_ids.tolist()) == set(train_ids)
    assert set(score_ids.tolist()).isdisjoint(set(val_ids) | set(test_ids))

    payload: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "direction_split_hash": direction_split_hash(direction_split),
        "model_name": getattr(model.cfg, "model_name", "gpt2-small"),
        "split": "train",
        "direction_ids": score_ids,
        "relative_strengths": torch.tensor(strengths, dtype=torch.float32),
        "mean_kl": mean_kl.float(),
        "std_kl": std_kl.float(),
        "mean_delta_nll": mean_delta_nll.float(),
        "kl_by_strength": kl_by_strength.float(),
        "delta_nll_by_strength": delta_nll_by_strength.float(),
        "num_contexts": contexts_seen,
        "hook_name": hook_name,
        "direction_batch_size": direction_batch_size,
        "result_cache_metadata": cache_metadata,
    }
    score_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, score_path)

    frame_data: dict[str, Any] = {
        "direction_id": train_ids,
        "mean_kl": mean_kl.numpy(),
        "std_kl": std_kl.numpy(),
        "mean_delta_nll": mean_delta_nll.numpy(),
    }
    for strength_index, strength in enumerate(strengths):
        frame_data[f"kl_r_{strength:g}"] = kl_by_strength[:, strength_index].numpy()
        frame_data[f"delta_nll_r_{strength:g}"] = (
            delta_nll_by_strength[:, strength_index].numpy()
        )
    scores_frame = pd.DataFrame(frame_data).sort_values("mean_kl", ascending=False)
    if csv_path is not None:
        resolved_csv_path = Path(csv_path)
        resolved_csv_path.parent.mkdir(parents=True, exist_ok=True)
        scores_frame.to_csv(resolved_csv_path, index=False)

    print("Top 20 most damaging training directions:")
    print(scores_frame.head(20).to_string(index=False))
    print("Bottom 20 training directions:")
    print(scores_frame.tail(20).sort_values("mean_kl").to_string(index=False))
    if figure_path is not None:
        from src.utils import plot_direction_damage_histogram

        plot_direction_damage_histogram(mean_kl, figure_path)
    runtime.add(
        "harmfulness scoring",
        time.perf_counter() - harmfulness_start,
        configurations=num_directions * num_strengths,
        examples=contexts_seen,
    )
    return payload


def score_direction_harmfulness_fast(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Named fast path for offline TRAIN-direction KL/NLL scoring.

    This delegates to the leakage-safe batched implementation above.  The
    separate name makes the inexpensive harmfulness-only stage explicit in the
    final notebook without maintaining two scientific implementations.
    """
    return score_train_direction_fluency(*args, **kwargs)


def _generation_intervention(
    method: str,
    direction: Tensor,
    alpha: float,
    denoisers: Mapping[str, nn.Module],
    normalization_stats: Mapping[str, Mapping[str, Any]],
    generation_config: Mapping[str, Any],
    token_positions: str = "final_token_only",
) -> Callable[[Tensor], Tensor]:
    """Build a steering hook for final-token generation or all-token scoring.

    ``final_token_only`` is the generation convention: old prompt/continuation
    positions are untouched on every autoregressive forward. ``all_tokens`` is
    used by token-level KL/NLL evaluation so modified logits at positions
    ``[:-1]`` have next-token labels.
    """
    assert token_positions in {"final_token_only", "all_tokens"}
    validate_generation_methods([method])

    def at_selected_positions(
        transform: Callable[[Tensor], Tensor]
    ) -> Callable[[Tensor], Tensor]:
        if token_positions == "all_tokens":
            return transform

        def intervention(activation: Tensor) -> Tensor:
            assert activation.ndim == 3
            result = activation.clone()
            final_activation = activation[:, -1:, :]
            modified = transform(final_activation)
            assert modified.shape == final_activation.shape
            result[:, -1:, :] = modified
            return result
        return intervention

    # V1 historically called this method ``raw``. V3 uses the explicit
    # ``relative_raw`` name to distinguish it from literal-alpha steering.
    # Both names intentionally execute the same relative-strength transform.
    if method in {"raw", "relative_raw"}:
        return at_selected_positions(lambda activation: apply_steering(
            activation, direction, alpha, method="relative_raw"
        ))

    if method == "conditioned_kl_denoiser":
        assert "conditioned_v2" in denoisers, "Missing conditioned_v2 denoiser."
        assert "conditioned_v2" in normalization_stats, (
            "Missing conditioned_v2 checkpoint normalization."
        )
        return at_selected_positions(lambda activation: apply_steering(
            activation,
            direction,
            alpha,
            method="conditioned_kl_denoise",
            denoiser=denoisers["conditioned_v2"],
            normalization_stats=normalization_stats["conditioned_v2"],
        ))
    if method in {
        "conditioned_reconstruction", "conditioned_kl",
        "conditioned_kl_retention", "conditioned_full",
    }:
        assert method in denoisers, f"Missing V3 denoiser {method!r}."
        assert method in normalization_stats, f"Missing V3 normalization for {method!r}."
        return at_selected_positions(lambda activation: apply_steering(
            activation,
            direction,
            alpha,
            method="gated_conditioned_denoise",
            denoiser=denoisers[method],
            normalization_stats=normalization_stats[method],
        ))
    if method == "norm_preserving":
        return at_selected_positions(lambda activation: apply_steering(
            activation, direction, alpha, method="relative_norm_preserving"
        ))

    denoiser_key_by_method = {
        "gaussian_denoiser": "gaussian",
        "sae_denoiser": "sae_calibrated",
        "sae_calibrated": "sae_calibrated",
        "fluency_denoiser": "fluency",
        "projected_fluency_denoiser": "fluency",
        "incremental_fluency": "fluency",
    }
    if method not in denoiser_key_by_method:
        raise ValueError(f"Unknown generation evaluation method: {method!r}.")
    denoiser_key = denoiser_key_by_method[method]
    # Backward-compatible input key only; results retain the unambiguous method name.
    if denoiser_key == "sae_calibrated" and denoiser_key not in denoisers and "sae" in denoisers:
        denoiser_key = "sae"
    assert denoiser_key in denoisers, f"Missing {denoiser_key!r} denoiser."
    assert denoiser_key in normalization_stats, (
        f"Missing checkpoint normalization for {denoiser_key!r} denoiser."
    )
    denoiser = denoisers[denoiser_key]
    stats = normalization_stats[denoiser_key]

    if method in {"gaussian_denoiser", "sae_denoiser", "sae_calibrated", "fluency_denoiser"}:
        return at_selected_positions(lambda activation: apply_steering(
            activation,
            direction,
            alpha,
            method="relative_denoise",
            denoiser=denoiser,
            normalization_stats=stats,
        ))
    if method == "projected_fluency_denoiser":
        return at_selected_positions(lambda activation: apply_steering(
            activation,
            direction,
            alpha,
            method="relative_projected_denoise",
            denoiser=denoiser,
            normalization_stats=stats,
            beta=float(generation_config.get("projected_beta", 1.0)),
        ))

    n_steps = int(generation_config["incremental_steps"])
    return at_selected_positions(lambda activation: apply_steering(
        activation,
        direction,
        alpha,
        method="incremental_relative_denoise",
        denoiser=denoiser,
        normalization_stats=stats,
        n_steps=n_steps,
        projected=bool(generation_config.get("incremental_projected", False)),
        beta=float(generation_config.get("projected_beta", 1.0)),
    ))


@torch.inference_mode()
def _generate_fixed_seed(
    model: HookedTransformer,
    prompt_tokens: Tensor,
    hook_name: str,
    intervention_fn: Callable[[Tensor], Tensor],
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    do_sample: bool,
    eos_token_id: int | None,
) -> Tensor:
    """Generate one continuation through the same hook path for every method."""
    assert prompt_tokens.shape[0] == 1 and max_new_tokens > 0
    context_limit = int(getattr(model.cfg, "n_ctx", 1024))
    assert prompt_tokens.shape[1] + max_new_tokens <= context_limit, (
        f"Prompt ({prompt_tokens.shape[1]} tokens) plus continuation "
        f"({max_new_tokens} tokens) exceeds model context limit "
        f"({context_limit}). Truncate the prompt before generation."
    )
    generator = torch.Generator(device=prompt_tokens.device).manual_seed(seed)
    tokens = prompt_tokens.clone()
    for _ in range(max_new_tokens):
        logits = get_logits_with_intervention(
            model,
            tokens,
            hook_name,
            intervention_fn,
        )
        next_logits = logits[:, -1, :]
        if not do_sample or temperature <= 0.0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            if top_k is not None and top_k > 0:
                effective_top_k = min(top_k, next_logits.shape[-1])
                top_values, top_indices = torch.topk(next_logits, effective_top_k)
                probabilities = F.softmax(top_values, dim=-1)
                sampled_local = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
                next_token = top_indices.gather(-1, sampled_local)
            else:
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
        tokens = torch.cat((tokens, next_token), dim=1)
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    return tokens


def _generation_record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record["evaluation_split"],
        record["method"],
        int(record["direction_id"]),
        int(record["prompt_id"]),
        record["prompt"],
        float(record["alpha"]),
        int(record["seed"]),
        record["generation_signature"],
    )


@torch.inference_mode()
def evaluate_generation_methods(
    model: HookedTransformer,
    sae: Any,
    prompts: Sequence[str],
    all_directions: Tensor,
    direction_split: Mapping[str, Any],
    direction_ids: Sequence[int],
    alphas: Sequence[float],
    seeds: Sequence[int],
    hook_name: str,
    denoisers: Mapping[str, nn.Module],
    normalization_stats: Mapping[str, Mapping[str, Any]],
    generation_config: Mapping[str, Any],
    evaluation_split: str = "val",
    methods: Sequence[str] | None = None,
    clean_model: HookedTransformer | None = None,
    jsonl_path: str | Path = "outputs/generations/final_eval.jsonl",
    results_path: str | Path = "outputs/results/results.csv",
    aggregate_path: str | Path | None = "outputs/results/aggregate_results.csv",
    *,
    checkpoint_fingerprints: Mapping[str, str] | None = None,
    profiler: RuntimeProfile | None = None,
    sae_identity: str | None = None,
    pipeline_version: str = PIPELINE_VERSION,
    result_evaluation_split: str | None = None,
) -> pd.DataFrame:
    """Run resumable, fixed-seed generation evaluation on held-out directions.

    Call this separately for validation and test splits. Existing JSONL records
    with the same complete experiment key are skipped, and CSV files are rebuilt
    from the JSONL source of truth after each call.
    """
    assert evaluation_split in {"val", "test"}
    recorded_evaluation_split = result_evaluation_split or evaluation_split
    assert prompts and direction_ids and alphas and seeds
    train_ids = set(direction_split["train"])
    allowed_ids = set(direction_split[evaluation_split])
    requested_ids = [int(direction_id) for direction_id in direction_ids]
    assert len(requested_ids) == len(set(requested_ids))
    assert set(requested_ids).issubset(allowed_ids)
    assert set(requested_ids).isdisjoint(train_ids), (
        "Final generation evaluation cannot use training directions."
    )
    validate_direction_ids_for_usage(
        requested_ids,
        direction_split,
        usage=(
            "preliminary_evaluation"
            if evaluation_split == "val"
            else "final_evaluation"
        ),
    )
    validate_direction_split(
        list(direction_split["train"]),
        list(direction_split["val"]),
        list(direction_split["test"]),
        int(direction_split["num_features"]),
    )
    assert all_directions.shape == (int(direction_split["num_features"]), model.cfg.d_model)
    assert all_directions.device == next(model.parameters()).device
    assert all_directions.dtype == next(model.parameters()).dtype

    sae_hook_name = get_sae_hook_name(sae)
    assert sae_hook_name == hook_name, (
        f"SAE hook {sae_hook_name!r} does not match evaluation hook {hook_name!r}."
    )
    assert sae.W_dec.shape == all_directions.shape
    assert sae.W_dec.device == all_directions.device
    assert sae.W_dec.dtype == all_directions.dtype
    for direction_id in requested_ids:
        decoder_direction = sae.W_dec[direction_id]
        decoder_direction = decoder_direction / decoder_direction.norm()
        supplied_direction = all_directions[direction_id]
        supplied_direction = supplied_direction / supplied_direction.norm()
        torch.testing.assert_close(
            supplied_direction,
            decoder_direction,
            rtol=1e-4,
            atol=1e-5,
        )

    if methods is None:
        selected_methods = [
            "raw",
            "norm_preserving",
            "gaussian_denoiser",
            "sae_calibrated",
            "fluency_denoiser",
            "projected_fluency_denoiser",
        ]
        if generation_config.get("incremental_steps") is not None:
            selected_methods.append("incremental_fluency")
    else:
        selected_methods = list(methods)
    assert len(selected_methods) == len(set(selected_methods))

    scoring_model = clean_model or model
    assert next(scoring_model.parameters()).device == next(model.parameters()).device
    max_new_tokens = int(generation_config.get("max_new_tokens", 50))
    context_limit = int(getattr(model.cfg, "n_ctx", 1024))
    max_prompt_tokens = context_limit - max_new_tokens
    assert max_prompt_tokens >= 1, (
        f"max_new_tokens={max_new_tokens} leaves no room for a prompt in the "
        f"model context window ({context_limit})."
    )
    temperature = float(generation_config.get("temperature", 1.0))
    top_k_value = generation_config.get("top_k", 50)
    top_k = None if top_k_value is None else int(top_k_value)
    do_sample = bool(generation_config.get("do_sample", True))
    generation_positions = str(
        generation_config.get("generation_positions", "final_token_only")
    )
    assert generation_positions == "final_token_only", (
        "Autoregressive generation must use generation_positions="
        "'final_token_only' so old prompt/continuation tokens are not steered "
        "again on every forward pass."
    )
    eos_token_id = generation_config.get("eos_token_id")
    if eos_token_id is None and model.tokenizer is not None:
        eos_token_id = model.tokenizer.eos_token_id
    concept_threshold = float(generation_config.get("concept_threshold", 0.0))
    repetition_ngram = int(generation_config.get("repetition_ngram", 3))
    assert repetition_ngram > 0
    signature_data = {
        "pipeline_version": pipeline_version,
        "hook_name": hook_name,
        "steering_scale": "relative_activation_norm",
        "token_positions": generation_positions,
        "context_limit": context_limit,
        "prompt_truncation": "preserve_bos_and_recent_tokens",
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "do_sample": do_sample,
        "eos_token_id": eos_token_id,
        "incremental_steps": generation_config.get("incremental_steps"),
        "incremental_projected": generation_config.get("incremental_projected", False),
        "projected_beta": generation_config.get("projected_beta", 1.0),
        "repetition_ngram": repetition_ngram,
        "direction_split_hash": direction_split_hash(direction_split),
        "prompt_set_hash": hashlib.sha256(
            json.dumps(list(prompts), ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:20],
        "checkpoint_fingerprints": dict(checkpoint_fingerprints or {}),
        "sae": sae_identity or sae_hook_name,
    }
    generation_signature = json.dumps(signature_data, sort_keys=True)

    resolved_jsonl_path = Path(jsonl_path)
    resolved_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    completed_keys: set[tuple[Any, ...]] = set()
    if resolved_jsonl_path.exists():
        with resolved_jsonl_path.open("r", encoding="utf-8") as existing_file:
            for line in existing_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("pipeline_version") != pipeline_version:
                    continue
                if record.get("generation_signature") != generation_signature:
                    continue
                records.append(record)
                completed_keys.add(_generation_record_key(record))

    combinations = len(selected_methods) * len(requested_ids) * len(prompts) * len(alphas) * len(seeds)
    runtime = profiler if profiler is not None else RuntimeProfile()
    progress = tqdm(total=combinations, desc=f"Generation eval ({recorded_evaluation_split})")
    with resolved_jsonl_path.open("a", encoding="utf-8") as output_file:
        for method in selected_methods:
            for direction_id in requested_ids:
                direction = all_directions[direction_id]
                for prompt_id, prompt in enumerate(prompts):
                    prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
                    original_prompt_length = int(prompt_tokens.shape[1])
                    if original_prompt_length > max_prompt_tokens:
                        # Preserve GPT-2's BOS token and the most recent context.
                        if max_prompt_tokens == 1:
                            prompt_tokens = prompt_tokens[:, :1]
                        else:
                            prompt_tokens = torch.cat(
                                (
                                    prompt_tokens[:, :1],
                                    prompt_tokens[:, -(max_prompt_tokens - 1):],
                                ),
                                dim=1,
                            )
                    prompt_length = prompt_tokens.shape[1]
                    assert prompt_length + max_new_tokens <= context_limit
                    for alpha in alphas:
                        intervention_fn = _generation_intervention(
                            method,
                            direction,
                            float(alpha),
                            denoisers,
                            normalization_stats,
                            generation_config,
                            token_positions=generation_positions,
                        )
                        for seed in seeds:
                            key_record = {
                                "evaluation_split": recorded_evaluation_split,
                                "method": method,
                                "direction_id": direction_id,
                                "prompt_id": prompt_id,
                                "prompt": prompt,
                                "alpha": float(alpha),
                                "relative_strength": float(alpha),
                                "literal_alpha": None,
                                "steering_mode": "relative",
                                "seed": int(seed),
                                "generation_signature": generation_signature,
                                "original_prompt_tokens": original_prompt_length,
                                "prompt_tokens_used": int(prompt_length),
                                "prompt_was_truncated": (
                                    original_prompt_length != int(prompt_length)
                                ),
                            }
                            key = _generation_record_key(key_record)
                            if key in completed_keys:
                                progress.update(1)
                                continue

                            generation_start = time.perf_counter()
                            full_tokens = _generate_fixed_seed(
                                model,
                                prompt_tokens,
                                hook_name,
                                intervention_fn,
                                seed=int(seed),
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_k=top_k,
                                do_sample=do_sample,
                                eos_token_id=eos_token_id,
                            )
                            continuation_tokens = full_tokens[:, prompt_length:]
                            assert continuation_tokens.shape[1] > 0
                            runtime.add(
                                "generation", time.perf_counter() - generation_start,
                                modified_forwards=int(continuation_tokens.shape[1]),
                                configurations=1, examples=1,
                            )
                            generated_ids = continuation_tokens[0].tolist()
                            generated_text = model.tokenizer.decode(
                                generated_ids,
                                skip_special_tokens=True,
                            )
                            scoring_start = time.perf_counter()
                            clean_generation_nll = external_clean_lm_nll(
                                scoring_model,
                                full_tokens,
                                prefix_lengths=prompt_length,
                            )

                            clean_logits, cache = model.run_with_cache(
                                full_tokens,
                                names_filter=hook_name,
                            )
                            clean_h = cache[hook_name]
                            del cache
                            runtime.add(
                                "generation clean scoring",
                                time.perf_counter() - scoring_start,
                                clean_forwards=2, examples=1,
                            )
                            modified_h: Tensor | None = None

                            def capture_intervention(activation: Tensor) -> Tensor:
                                nonlocal modified_h
                                modified_h = intervention_fn(activation)
                                return modified_h

                            diagnostic_start = time.perf_counter()
                            modified_logits = get_logits_with_intervention(
                                model,
                                full_tokens,
                                hook_name,
                                capture_intervention,
                            )
                            runtime.add(
                                "generation modified diagnostic",
                                time.perf_counter() - diagnostic_start,
                                modified_forwards=1, configurations=1, examples=1,
                            )
                            assert modified_h is not None
                            metric_start = time.perf_counter()
                            continuation_mask = torch.zeros_like(
                                full_tokens, dtype=torch.bool
                            )
                            continuation_mask[:, prompt_length:] = True
                            final_position_mask = torch.zeros_like(
                                full_tokens, dtype=torch.bool
                            )
                            final_position_mask[:, -1] = True
                            kl = token_level_kl(
                                clean_logits,
                                modified_logits,
                                attention_mask=final_position_mask,
                            )
                            norm_ratio = activation_norm_ratio(
                                clean_h[:, -1:],
                                modified_h[:, -1:],
                            )
                            concept_metrics = sae_feature_activation_metrics(
                                sae,
                                clean_h,
                                feature_id=direction_id,
                                token_mask=continuation_mask,
                                threshold=concept_threshold,
                            )
                            runtime.add(
                                "generation metrics",
                                time.perf_counter() - metric_start,
                                configurations=1, examples=1,
                            )

                            record = {
                                **key_record,
                                "generated_text": generated_text,
                                "generated_token_ids": generated_ids,
                                "num_generated_tokens": len(generated_ids),
                                "n_steps": (
                                    int(generation_config["incremental_steps"])
                                    if method == "incremental_fluency"
                                    else None
                                ),
                                "clean_model_nll": clean_generation_nll.item(),
                                "dist_1": distinct_n(generated_ids, 1),
                                "dist_2": distinct_n(generated_ids, 2),
                                "dist_3": distinct_n(generated_ids, 3),
                                "repetition_rate": repetition_rate(
                                    generated_ids, n=repetition_ngram
                                ),
                                "kl": kl.item(),
                                "activation_norm_ratio": norm_ratio.item(),
                                "concept_score": concept_metrics["concept_score"].item(),
                                "target_sae_activation": concept_metrics["concept_score"].item(),
                                "max_feature_activation": concept_metrics[
                                    "max_feature_activation"
                                ].item(),
                                "feature_active_fraction": concept_metrics[
                                    "feature_active_fraction"
                                ].item(),
                                "steering_scale": "relative_activation_norm",
                                "token_positions": generation_positions,
                                "pipeline_version": pipeline_version,
                            }
                            is_identity_diagnostic = (
                                float(alpha) == 0.0
                                and method not in {"raw", "norm_preserving"}
                            )
                            activation_mse = (
                                (modified_h[:, -1:] - clean_h[:, -1:])
                                .float().square().mean().detach().item()
                            )
                            record["clean_identity_kl"] = (
                                record["kl"] if is_identity_diagnostic else None
                            )
                            record["clean_identity_delta_nll"] = None
                            record["clean_identity_activation_mse"] = (
                                activation_mse if is_identity_diagnostic else None
                            )
                            output_file.write(json.dumps(record) + "\n")
                            output_file.flush()
                            records.append(record)
                            completed_keys.add(key)
                            progress.update(1)
                            del clean_logits, modified_logits, clean_h, modified_h
    progress.close()

    results = pd.DataFrame(records)
    validate_result_metrics(results, generation=True)
    resolved_results_path = Path(results_path)
    resolved_results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(resolved_results_path, index=False)
    if aggregate_path is not None and not results.empty:
        metric_columns = [
            "clean_model_nll",
            "dist_1",
            "dist_2",
            "dist_3",
            "repetition_rate",
            "kl",
            "activation_norm_ratio",
            "concept_score",
            "max_feature_activation",
            "feature_active_fraction",
        ]
        aggregate = (
            results.groupby(
                ["evaluation_split", "method", "direction_id", "alpha"],
                as_index=False,
            )[metric_columns]
            .mean()
        )
        resolved_aggregate_path = Path(aggregate_path)
        resolved_aggregate_path.parent.mkdir(parents=True, exist_ok=True)
        aggregate.to_csv(resolved_aggregate_path, index=False)
    return results


def validate_result_metrics(frame: pd.DataFrame, generation: bool = False) -> None:
    """Fail on duplicate experiment keys, non-finite values, or invalid ranges."""
    if frame.empty:
        return
    strength_column = "alpha" if "alpha" in frame.columns else "strength"
    keys = ["method", "direction_id", strength_column]
    if "prompt_id" in frame.columns:
        keys.append("prompt_id")
    if generation and "seed" in frame.columns:
        keys.append("seed")
    if generation and "generation_signature" in frame.columns:
        keys.append("generation_signature")
    if "evaluation_split" in frame.columns:
        keys.append("evaluation_split")
    assert not frame.duplicated(keys).any(), f"Duplicate result keys detected: {keys}."
    numeric = [
        column for column in (
            "kl", "delta_nll", "activation_norm_ratio", "concept_score",
            "dist_1", "dist_2", "dist_3", "repetition_rate",
        ) if column in frame.columns
    ]
    for column in numeric:
        values = torch.as_tensor(frame[column].to_numpy(dtype=float))
        assert torch.isfinite(values).all(), f"Non-finite values in {column}."
    if "kl" in frame:
        assert (frame["kl"] >= -1e-6).all(), "KL contains materially negative values."
    if "activation_norm_ratio" in frame:
        assert (frame["activation_norm_ratio"] > 0).all()
    for column in ("dist_1", "dist_2", "dist_3", "repetition_rate"):
        if column in frame:
            assert frame[column].between(0.0, 1.0).all()


def freeze_test_config(
    config: Mapping[str, Any],
    methods: Sequence[str],
    path: str | Path = "outputs/frozen_test_config.json",
) -> Path:
    """Freeze validation-selected settings before any test-direction access."""
    assert methods and len(methods) == len(set(methods))
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "seed": int(config["seed"]),
        "model_name": config["model"]["name"],
        "hook_name": config["model"]["hook_name"],
        "strengths": list(config["steering"]["alphas"]),
        "methods": list(methods),
        "evaluation": dict(config["evaluation"]),
        "status": "frozen_after_validation",
    }
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        existing = json.loads(resolved.read_text(encoding="utf-8"))
        assert existing == payload, (
            "Frozen test configuration already exists with different settings."
        )
        return resolved
    resolved.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return resolved


def load_frozen_test_config(
    path: str | Path = "outputs/frozen_test_config.json",
) -> dict[str, Any]:
    """Load and strictly validate the only configuration allowed for test."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "pipeline_version", "seed", "model_name", "hook_name", "strengths",
        "methods", "evaluation", "status",
    }
    assert required == set(payload), "Frozen test config schema is invalid."
    assert payload["pipeline_version"] == PIPELINE_VERSION
    assert payload["status"] == "frozen_after_validation"
    assert payload["methods"] and payload["strengths"]
    return payload
