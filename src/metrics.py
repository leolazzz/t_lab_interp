"""Metrics for downstream disruption caused by activation interventions."""

from collections.abc import Sequence
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


def _validate_logits(logits: Tensor) -> None:
    assert logits.ndim == 3, (
        f"Expected logits with shape [batch, seq, vocab], got {logits.shape}."
    )
    assert logits.is_floating_point(), "Logits must be floating point."


def _validate_attention_mask(values: Tensor, attention_mask: Tensor) -> None:
    assert attention_mask.shape == values.shape, (
        f"Expected attention_mask shape {values.shape}, got {attention_mask.shape}."
    )
    assert attention_mask.device == values.device, (
        f"attention_mask is on {attention_mask.device}, but values are on {values.device}."
    )


def _masked_reduce(
    values: Tensor,
    attention_mask: Tensor | None,
    reduction: str,
) -> Tensor:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")

    if attention_mask is None:
        if reduction == "none":
            return values
        if reduction == "mean":
            return values.mean()
        return values.sum()

    _validate_attention_mask(values, attention_mask)
    mask = attention_mask.to(dtype=values.dtype)
    masked_values = values * mask
    if reduction == "none":
        return masked_values
    if reduction == "sum":
        return masked_values.sum()

    denominator = mask.sum()
    assert denominator.item() > 0, "attention_mask selects no tokens."
    return masked_values.sum() / denominator


def token_level_kl(
    clean_logits: Tensor,
    modified_logits: Tensor,
    attention_mask: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Compute ``KL(p_clean || p_modified)`` over next-token distributions."""
    _validate_logits(clean_logits)
    _validate_logits(modified_logits)
    assert clean_logits.shape == modified_logits.shape
    assert clean_logits.device == modified_logits.device

    clean_log_probs = F.log_softmax(clean_logits, dim=-1)
    modified_log_probs = F.log_softmax(modified_logits, dim=-1)
    clean_probs = clean_log_probs.exp()
    per_token_kl = (
        clean_probs * (clean_log_probs - modified_log_probs)
    ).sum(dim=-1)
    return _masked_reduce(per_token_kl, attention_mask, reduction)


def next_token_nll(
    logits: Tensor,
    tokens: Tensor,
    attention_mask: Tensor | None = None,
    return_per_token: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Compute autoregressive NLL, optionally returning ``[batch, seq - 1]``.

    If an attention mask is supplied, its positions ``[:, 1:]`` select target
    tokens. Masked entries in the returned per-token tensor are zero.
    """
    _validate_logits(logits)
    assert tokens.ndim == 2, f"Expected tokens [batch, seq], got {tokens.shape}."
    assert tokens.dtype == torch.long, f"Expected torch.long tokens, got {tokens.dtype}."
    assert tokens.shape == logits.shape[:2]
    assert tokens.device == logits.device
    assert tokens.shape[1] >= 2, "At least two sequence positions are required."

    target_logits = logits[:, :-1, :]
    target_tokens = tokens[:, 1:]
    per_token_nll = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_tokens.reshape(-1),
        reduction="none",
    ).reshape_as(target_tokens)

    target_mask = None
    if attention_mask is not None:
        assert attention_mask.shape == tokens.shape
        target_mask = attention_mask[:, 1:]
    mean_nll = _masked_reduce(per_token_nll, target_mask, reduction="mean")
    if return_per_token:
        masked_per_token = _masked_reduce(
            per_token_nll, target_mask, reduction="none"
        )
        return mean_nll, masked_per_token
    return mean_nll


def nll_increase(
    clean_logits: Tensor,
    modified_logits: Tensor,
    tokens: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Return modified mean next-token NLL minus clean mean NLL."""
    assert clean_logits.shape == modified_logits.shape
    clean_nll = next_token_nll(clean_logits, tokens, attention_mask)
    modified_nll = next_token_nll(modified_logits, tokens, attention_mask)
    assert isinstance(clean_nll, Tensor) and isinstance(modified_nll, Tensor)
    return modified_nll - clean_nll


def activation_norm_ratio(
    clean_h: Tensor,
    modified_h: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """Return the mean per-position ratio ``||modified_h|| / ||clean_h||``."""
    assert clean_h.shape == modified_h.shape
    assert clean_h.ndim >= 2
    assert clean_h.device == modified_h.device
    assert clean_h.dtype == modified_h.dtype
    assert clean_h.is_floating_point()

    stability_eps = max(eps, torch.finfo(clean_h.dtype).tiny)
    clean_norm = torch.linalg.vector_norm(clean_h, dim=-1)
    modified_norm = torch.linalg.vector_norm(modified_h, dim=-1)
    ratios = (modified_norm + stability_eps) / (clean_norm + stability_eps)
    return ratios.mean()


def _as_token_sequences(
    texts_or_token_sequences: str | Sequence[Any] | Tensor,
) -> list[list[Any]]:
    if isinstance(texts_or_token_sequences, str):
        return [texts_or_token_sequences.split()]
    if isinstance(texts_or_token_sequences, Tensor):
        tensor = texts_or_token_sequences.detach().cpu()
        if tensor.ndim == 1:
            return [tensor.tolist()]
        assert tensor.ndim == 2
        return tensor.tolist()

    items = list(texts_or_token_sequences)
    if not items:
        return []
    if all(isinstance(item, str) for item in items):
        return [item.split() for item in items]
    if all(isinstance(item, int) for item in items):
        return [items]
    return [
        sequence.detach().cpu().tolist()
        if isinstance(sequence, Tensor)
        else list(sequence)
        for sequence in items
    ]


def distinct_n(
    texts_or_token_sequences: str | Sequence[Any] | Tensor,
    n: int,
) -> float:
    """Return unique n-grams divided by all n-grams across the inputs.

    Strings are split on whitespace. Token sequences are used directly.
    ``n`` is restricted to 1, 2, or 3 for the project's generation report.
    """
    assert n in {1, 2, 3}
    ngrams: list[tuple[Any, ...]] = []
    for sequence in _as_token_sequences(texts_or_token_sequences):
        ngrams.extend(
            tuple(sequence[index : index + n])
            for index in range(max(0, len(sequence) - n + 1))
        )
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def repetition_rate(
    texts_or_token_sequences: str | Sequence[Any] | Tensor,
    n: int = 3,
) -> float:
    """Return the fraction of n-gram occurrences repeated within a sequence.

    Each sequence maintains its own history. An occurrence counts as repeated
    if the same n-gram appeared earlier in that generated sequence. Trigrams
    are used by default to capture phrase-level repetition.
    """
    assert n > 0
    repeated = 0
    total = 0
    for sequence in _as_token_sequences(texts_or_token_sequences):
        seen: set[tuple[Any, ...]] = set()
        for index in range(max(0, len(sequence) - n + 1)):
            ngram = tuple(sequence[index : index + n])
            repeated += int(ngram in seen)
            seen.add(ngram)
            total += 1
    return repeated / total if total else 0.0


@torch.inference_mode()
def external_clean_lm_nll(
    clean_model: Any,
    token_sequences: Tensor,
    attention_mask: Tensor | None = None,
    prefix_lengths: int | Sequence[int] | Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Score sequences autoregressively with an unmodified base model.

    ``prefix_lengths`` excludes prompt targets while retaining the prompt as
    conditioning context. The function asserts that no TransformerLens forward
    or backward hooks are active on the supplied scoring model.
    """
    assert token_sequences.ndim == 2 and token_sequences.dtype == torch.long
    assert token_sequences.shape[1] >= 2
    assert reduction in {"mean", "sum", "none"}
    if hasattr(clean_model, "hook_dict"):
        for hook_name, hook_point in clean_model.hook_dict.items():
            assert not getattr(hook_point, "fwd_hooks", []), (
                f"Clean scoring model has an active forward hook at {hook_name}."
            )
            assert not getattr(hook_point, "bwd_hooks", []), (
                f"Clean scoring model has an active backward hook at {hook_name}."
            )

    logits = clean_model(token_sequences, return_type="logits")
    _validate_logits(logits)
    assert logits.shape[:2] == token_sequences.shape
    per_token_nll = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        token_sequences[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(token_sequences.shape[0], token_sequences.shape[1] - 1)
    valid_mask = torch.ones_like(per_token_nll, dtype=torch.bool)
    if attention_mask is not None:
        assert attention_mask.shape == token_sequences.shape
        assert attention_mask.device == token_sequences.device
        valid_mask &= attention_mask[:, 1:].bool()

    if prefix_lengths is not None:
        if isinstance(prefix_lengths, int):
            prefix_tensor = torch.full(
                (token_sequences.shape[0],),
                prefix_lengths,
                device=token_sequences.device,
                dtype=torch.long,
            )
        elif isinstance(prefix_lengths, Tensor):
            assert prefix_lengths.device == token_sequences.device
            prefix_tensor = prefix_lengths.to(dtype=torch.long)
        else:
            prefix_tensor = torch.tensor(
                list(prefix_lengths),
                device=token_sequences.device,
                dtype=torch.long,
            )
        assert prefix_tensor.shape == (token_sequences.shape[0],)
        assert torch.all((0 <= prefix_tensor) & (prefix_tensor < token_sequences.shape[1])).item()
        target_positions = torch.arange(
            1, token_sequences.shape[1], device=token_sequences.device
        )
        valid_mask &= target_positions.unsqueeze(0) >= prefix_tensor.unsqueeze(1)

    assert valid_mask.any().item(), "No generation targets remain after masking."
    masked_nll = per_token_nll * valid_mask
    if reduction == "none":
        return masked_nll
    if reduction == "sum":
        return masked_nll.sum()
    return masked_nll.sum() / valid_mask.sum()


@torch.inference_mode()
def sae_feature_activation_metrics(
    sae: Any,
    activations: Tensor,
    feature_id: int,
    token_mask: Tensor,
    threshold: float = 0.0,
) -> dict[str, Tensor]:
    """Measure one SAE feature over selected generated-token activations."""
    assert activations.ndim == 3
    assert token_mask.shape == activations.shape[:2]
    assert token_mask.device == activations.device
    assert sae.W_dec.ndim == 2 and sae.W_dec.shape[1] == activations.shape[-1]
    assert 0 <= feature_id < sae.W_dec.shape[0]
    assert sae.W_dec.device == activations.device
    assert sae.W_dec.dtype == activations.dtype

    feature_activations = sae.encode(activations)
    assert feature_activations.shape[:2] == activations.shape[:2]
    assert feature_activations.shape[-1] == sae.W_dec.shape[0]
    selected = feature_activations[..., feature_id][token_mask.bool()]
    assert selected.numel() > 0, "Continuation mask selects no tokens."
    mean_activation = selected.mean()
    return {
        "concept_score": mean_activation,
        "mean_feature_activation": mean_activation,
        "max_feature_activation": selected.max(),
        "feature_active_fraction": (selected > threshold).to(selected.dtype).mean(),
    }


def _to_cpu_float32_2d(values: Tensor | np.ndarray) -> np.ndarray:
    """Convert explicitly to the CPU representation required by sklearn."""
    if isinstance(values, Tensor):
        assert values.is_floating_point()
        array = values.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        array = np.asarray(values, dtype=np.float32)
    assert array.ndim >= 2 and array.shape[-1] > 0
    array = array.reshape(-1, array.shape[-1])
    assert array.shape[0] > 0
    assert np.isfinite(array).all()
    return np.ascontiguousarray(array)


class NaturalNeighborIndex:
    """Small PCA + kNN model for clean residual-stream activations.

    sklearn runs on CPU, so tensor inputs are deliberately copied to CPU
    float32. By default at most 20,000 clean vectors are retained, which keeps
    the fitted index and its serialized copy practical in a Kaggle notebook.
    """

    def __init__(
        self,
        n_components: int = 64,
        n_neighbors: int = 10,
        max_fit_samples: int = 20_000,
        seed: int = 42,
        hook_name: str | None = None,
    ) -> None:
        assert n_components > 0 and n_neighbors > 0 and max_fit_samples > 1
        self.requested_n_components = n_components
        self.default_k = n_neighbors
        self.max_fit_samples = max_fit_samples
        self.seed = seed
        self.hook_name = hook_name
        self.pca: PCA | None = None
        self.neighbors: NearestNeighbors | None = None
        self.clean_activations: np.ndarray | None = None
        self.clean_pca: np.ndarray | None = None

    def fit(self, clean_activations: Tensor | np.ndarray) -> "NaturalNeighborIndex":
        clean = _to_cpu_float32_2d(clean_activations)
        if clean.shape[0] > self.max_fit_samples:
            generator = np.random.default_rng(self.seed)
            indices = generator.choice(
                clean.shape[0], self.max_fit_samples, replace=False
            )
            clean = clean[np.sort(indices)]
        assert clean.shape[0] > 1
        n_components = min(
            self.requested_n_components,
            clean.shape[1],
            clean.shape[0] - 1,
        )
        solver = "randomized" if n_components < min(clean.shape) else "full"
        self.pca = PCA(
            n_components=n_components,
            svd_solver=solver,
            random_state=self.seed,
        )
        clean_pca = self.pca.fit_transform(clean).astype(np.float32, copy=False)
        self.neighbors = NearestNeighbors(
            n_neighbors=min(self.default_k, clean.shape[0]),
            algorithm="auto",
        ).fit(clean_pca)
        self.clean_activations = clean.copy()
        self.clean_pca = clean_pca
        return self

    def _check_fitted(self) -> tuple[PCA, NearestNeighbors]:
        assert self.pca is not None and self.neighbors is not None, (
            "NaturalNeighborIndex.fit must be called before querying."
        )
        return self.pca, self.neighbors

    def transform(self, activations: Tensor | np.ndarray) -> np.ndarray:
        pca, _ = self._check_fitted()
        values = _to_cpu_float32_2d(activations)
        assert values.shape[1] == pca.n_features_in_
        return pca.transform(values).astype(np.float32, copy=False)

    def knn_distance(
        self,
        activations: Tensor | np.ndarray,
        k: int | None = None,
        reduction: str = "mean",
    ) -> float | np.ndarray:
        """Return mean distance to the k nearest clean PCA-space vectors."""
        _, neighbors = self._check_fitted()
        resolved_k = self.default_k if k is None else k
        assert resolved_k > 0
        assert self.clean_pca is not None
        resolved_k = min(resolved_k, self.clean_pca.shape[0])
        distances, _ = neighbors.kneighbors(
            self.transform(activations), n_neighbors=resolved_k
        )
        per_activation = distances.mean(axis=1)
        if reduction == "none":
            return per_activation
        if reduction != "mean":
            raise ValueError("reduction must be 'mean' or 'none'.")
        return float(per_activation.mean())

    def mahalanobis_distance(
        self,
        activations: Tensor | np.ndarray,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> float | np.ndarray:
        """Diagonal Mahalanobis distance using PCA explained variances."""
        pca, _ = self._check_fitted()
        transformed = self.transform(activations)
        variances = np.maximum(pca.explained_variance_, eps)
        distances = np.sqrt(np.sum(np.square(transformed) / variances, axis=1))
        if reduction == "none":
            return distances
        if reduction != "mean":
            raise ValueError("reduction must be 'mean' or 'none'.")
        return float(distances.mean())

    def nearest_clean_activations(
        self, activations: Tensor | np.ndarray
    ) -> np.ndarray:
        """Return the nearest indexed clean vector for every query vector."""
        _, neighbors = self._check_fitted()
        assert self.clean_activations is not None
        _, indices = neighbors.kneighbors(self.transform(activations), n_neighbors=1)
        return self.clean_activations[indices[:, 0]]

    def save(self, path: str | Path) -> Path:
        """Serialize the fitted sklearn objects; load only trusted files."""
        self._check_fitted()
        resolved_path = Path(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with resolved_path.open("wb") as output_file:
            pickle.dump(self, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        return resolved_path

    @classmethod
    def load(cls, path: str | Path) -> "NaturalNeighborIndex":
        """Load an index created by :meth:`save` from a trusted local file."""
        with Path(path).open("rb") as input_file:
            index = pickle.load(input_file)
        assert isinstance(index, cls)
        index._check_fitted()
        return index


def fit_natural_neighbor_index_from_shards(
    shard_directory: str | Path,
    n_components: int = 64,
    n_neighbors: int = 10,
    max_fit_samples: int = 20_000,
    seed: int = 42,
    expected_hook_name: str | None = None,
) -> NaturalNeighborIndex:
    """Fit from cached activation shards without loading the full bank."""
    shard_paths = sorted(Path(shard_directory).glob("shard_*.pt"))
    assert shard_paths, f"No activation shards found in {shard_directory}."
    stats_path = Path(shard_directory) / "stats.pt"
    assert stats_path.exists(), f"Missing activation-cache metadata: {stats_path}."
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    assert isinstance(stats, dict) and isinstance(stats.get("hook_name"), str)
    if expected_hook_name is not None:
        assert stats["hook_name"] == expected_hook_name, (
            f"Activation bank hook {stats['hook_name']!r} does not match "
            f"{expected_hook_name!r}."
        )
    chunks: list[Tensor] = []
    remaining = max_fit_samples
    for shard_path in shard_paths:
        if remaining <= 0:
            break
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        assert isinstance(shard, Tensor) and shard.ndim == 2
        chunk = shard[:remaining].to(dtype=torch.float32)
        chunks.append(chunk)
        remaining -= chunk.shape[0]
    clean = torch.cat(chunks, dim=0)
    return NaturalNeighborIndex(
        n_components=n_components,
        n_neighbors=n_neighbors,
        max_fit_samples=max_fit_samples,
        seed=seed,
        hook_name=stats["hook_name"],
    ).fit(clean)


def build_neighbor_diagnostic_frame(
    metadata: pd.DataFrame | Sequence[dict[str, Any]],
    modified_activations: Sequence[Tensor | np.ndarray],
    neighbor_index: NaturalNeighborIndex,
    k: int = 10,
) -> pd.DataFrame:
    """Attach one mean kNN distance to each experiment result row."""
    frame = metadata.copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)
    required = {"method", "alpha", "direction_id", "delta_nll", "kl"}
    missing = required - set(frame.columns)
    assert not missing, f"Missing diagnostic columns: {sorted(missing)}"
    assert len(frame) == len(modified_activations)
    frame = frame.copy()
    frame["knn_distance"] = [
        neighbor_index.knn_distance(activation, k=k)
        for activation in modified_activations
    ]
    return frame


def spearman_neighbor_correlations(frame: pd.DataFrame) -> dict[str, float]:
    """Compute tie-aware Spearman correlations without adding scipy."""
    required = {"knn_distance", "delta_nll", "kl"}
    assert required.issubset(frame.columns) and len(frame) >= 2

    def correlation(left: str, right: str) -> float:
        ranks = frame[[left, right]].dropna().rank(method="average")
        if len(ranks) < 2 or ranks[left].nunique() < 2 or ranks[right].nunique() < 2:
            return float("nan")
        return float(ranks[left].corr(ranks[right], method="pearson"))

    return {
        "knn_distance_vs_delta_nll": correlation("knn_distance", "delta_nll"),
        "knn_distance_vs_kl": correlation("knn_distance", "kl"),
    }


def denoiser_nearest_clean_cosine(
    corrupted: Tensor,
    denoised: Tensor,
    neighbor_index: NaturalNeighborIndex,
    eps: float = 1e-8,
) -> Tensor:
    """Cosine between denoiser correction and nearest-clean direction.

    For each ``x``, this compares ``D(x) - x`` with ``h_nearest - x`` in the
    original residual space. Arbitrary leading dimensions are flattened and
    restored in the returned tensor.
    """
    assert corrupted.shape == denoised.shape and corrupted.ndim >= 2
    assert corrupted.device == denoised.device and corrupted.dtype == denoised.dtype
    leading_shape = corrupted.shape[:-1]
    flat_corrupted = corrupted.reshape(-1, corrupted.shape[-1])
    flat_denoised = denoised.reshape_as(flat_corrupted)
    nearest_numpy = neighbor_index.nearest_clean_activations(flat_corrupted)
    nearest = torch.from_numpy(nearest_numpy).to(
        device=corrupted.device, dtype=corrupted.dtype
    )
    correction = flat_denoised - flat_corrupted
    natural_direction = nearest - flat_corrupted
    denominator = correction.norm(dim=-1) * natural_direction.norm(dim=-1)
    cosine = (correction * natural_direction).sum(dim=-1) / denominator.clamp_min(eps)
    return cosine.reshape(leading_shape)
