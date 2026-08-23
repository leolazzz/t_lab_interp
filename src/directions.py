"""SAE decoder directions and leakage-safe direction splits.

The project uses SAELens release ``gpt2-small-res-jb`` (the registered
``jbloom/GPT2-Small-SAEs-Reformatted`` repository) and SAE ID
``blocks.6.hook_resid_pre``. This exactly matches the TransformerLens hook used
by :mod:`src.model`: the residual stream entering zero-indexed GPT-2 block 6.
The SAELens registry also records ``center_writing_weights=True`` for the model
used with this release.

Only IDs in the ``train`` split may be used for denoiser training, direction
scoring used to construct training data, or hyperparameter fitting. Validation
and test IDs are evaluation-only and must not influence those procedures.
"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

if TYPE_CHECKING:
    from sae_lens import SAE


DEFAULT_SAE_RELEASE = "gpt2-small-res-jb"
DEFAULT_SAE_ID = "blocks.6.hook_resid_pre"
DEFAULT_SPLIT_PATH = Path("outputs/direction_split.json")


DIRECTION_USAGE_TO_SPLIT = {
    "training": "train",
    "damage_scoring": "train",
    "hyperparameter_selection": "val",
    "preliminary_evaluation": "val",
    "final_evaluation": "test",
}


def _metadata_value(metadata: Any, key: str, default: Any = None) -> Any:
    return metadata.get(key, default) if isinstance(metadata, Mapping) else getattr(metadata, key, default)


def get_sae_hook_name(sae: Any) -> str | None:
    """Read the hook name from mapping- or object-style SAELens metadata."""
    return _metadata_value(getattr(sae.cfg, "metadata", None), "hook_name")


def direction_split_hash(split: Mapping[str, Any] | str | Path) -> str:
    """Return a deterministic SHA-256 fingerprint for split provenance."""
    payload = (
        json.loads(Path(split).read_text(encoding="utf-8"))
        if isinstance(split, (str, Path)) else dict(split)
    )
    validate_direction_split(
        list(payload["train"]), list(payload["val"]), list(payload["test"]),
        int(payload["num_features"]),
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_sae(
    release: str = DEFAULT_SAE_RELEASE,
    sae_id: str = DEFAULT_SAE_ID,
    device: str = "cpu",
    dtype: str = "float32",
) -> SAE:
    """Load the matching pretrained SAE with the current SAELens API.

    Current SAELens versions return the SAE directly from
    ``SAE.from_pretrained`` (rather than the tuple returned by SAELens 5.x).
    The import is local so split utilities remain usable before SAELens is
    installed.
    """
    try:
        from sae_lens import SAE
    except ImportError as error:
        raise ImportError(
            "SAELens is required to load pretrained SAE directions. "
            "Install the packages in requirements.txt first."
        ) from error

    sae = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
        dtype=dtype,
    )
    hook_name = get_sae_hook_name(sae)
    assert hook_name == sae_id, (
        f"Loaded SAE hook {hook_name!r} does not match requested hook {sae_id!r}."
    )
    sae.eval()
    return sae


def extract_decoder_directions(sae: SAE, model_d_model: int) -> Tensor:
    """Return unit-normalized SAE decoder rows as ``[num_features, d_model]``."""
    decoder = sae.W_dec.detach()
    assert decoder.ndim == 2, f"Expected 2D W_dec, got shape {decoder.shape}."
    assert decoder.shape[1] == model_d_model, (
        f"SAE decoder dimension {decoder.shape[1]} does not match model d_model "
        f"{model_d_model}."
    )

    norms = torch.linalg.vector_norm(decoder, dim=-1, keepdim=True)
    stability_eps = torch.finfo(decoder.dtype).tiny
    assert torch.all(norms > stability_eps).item(), (
        "SAE decoder contains a zero or near-zero direction."
    )
    directions = (decoder / norms).clone()
    assert directions.shape == decoder.shape
    return directions


def load_sae_directions(
    model_d_model: int,
    release: str = DEFAULT_SAE_RELEASE,
    sae_id: str = DEFAULT_SAE_ID,
    device: str = "cpu",
    dtype: str = "float32",
) -> tuple[SAE, Tensor]:
    """Load the configured SAE and its normalized decoder directions."""
    sae = load_sae(release=release, sae_id=sae_id, device=device, dtype=dtype)
    return sae, extract_decoder_directions(sae, model_d_model=model_d_model)


def model_sae_compatibility_report(
    model: Any,
    sae: SAE,
    hook_name: str,
) -> dict[str, Any]:
    """Inspect SAELens preprocessing expectations against TransformerLens.

    This intentionally reports rather than suppresses SAELens' warning. Values
    are checked against either ``model.cfg`` or the explicit load provenance
    saved by :func:`src.model.load_model`. Unknown required preprocessing
    kwargs make the compatibility gate fail instead of being silently accepted.
    """
    metadata = getattr(sae.cfg, "metadata", None)
    expected = _metadata_value(metadata, "model_from_pretrained_kwargs", {}) or {}
    if not isinstance(expected, Mapping):
        expected = dict(expected)
    relevant_keys = (
        "center_writing_weights",
        "center_unembed",
        "fold_ln",
        "refactor_factored_attn_matrices",
    )
    compared: dict[str, dict[str, Any]] = {}
    unknown: dict[str, Any] = {}
    load_kwargs = getattr(model, "_steering_denoiser_load_kwargs", {})
    if not isinstance(load_kwargs, Mapping):
        load_kwargs = {}
    for key, expected_value in expected.items():
        if hasattr(model.cfg, key):
            actual_value = getattr(model.cfg, key)
            source = "model.cfg"
        elif key in load_kwargs:
            actual_value = load_kwargs[key]
            source = "load_kwargs"
        else:
            unknown[key] = expected_value
            continue
        compared[key] = {
            "expected": expected_value,
            "actual": actual_value,
            "matches": actual_value == expected_value,
            "source": source,
        }
    for key in relevant_keys:
        if key not in compared and hasattr(model.cfg, key):
            compared[key] = {
                "expected": None,
                "actual": getattr(model.cfg, key),
                "matches": None,
            }
    sae_hook = _metadata_value(metadata, "hook_name")
    dimensions_match = int(model.cfg.d_model) == int(sae.W_dec.shape[-1])
    hook_matches = sae_hook == hook_name
    explicit_mismatches = [
        key for key, item in compared.items() if item["matches"] is False
    ]
    unknown_required = [key for key in unknown if key in relevant_keys]
    report = {
        "hook_name": hook_name,
        "sae_hook_name": sae_hook,
        "hook_matches": hook_matches,
        "model_d_model": int(model.cfg.d_model),
        "sae_d_in": int(sae.W_dec.shape[-1]),
        "dimensions_match": dimensions_match,
        "expected_model_kwargs": dict(expected),
        "compared_model_cfg": compared,
        "unrecognized_expected_kwargs": unknown,
        "compatible": (
            hook_matches
            and dimensions_match
            and not explicit_mismatches
            and not unknown_required
        ),
    }
    assert hook_matches, f"SAE hook {sae_hook!r} != model hook {hook_name!r}."
    assert dimensions_match, "SAE decoder dimension does not match GPT-2 d_model."
    assert not explicit_mismatches, f"Model/SAE preprocessing mismatch: {explicit_mismatches}."
    assert not unknown_required, (
        "Cannot verify required model preprocessing kwargs: "
        f"{unknown_required}. Load GPT-2 through src.model.load_model."
    )
    return report


def validate_direction_split(
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    num_features: int,
) -> None:
    """Validate uniqueness, disjointness, type, and bounds of split IDs."""
    assert num_features > 0
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    for split_name, ids in split_ids.items():
        assert all(isinstance(feature_id, int) for feature_id in ids), (
            f"All {split_name} IDs must be integers."
        )
        assert len(ids) == len(set(ids)), f"Duplicate IDs found in {split_name} split."
        assert all(0 <= feature_id < num_features for feature_id in ids), (
            f"{split_name} split contains an ID outside [0, {num_features})."
        )

    train_set, val_set, test_set = map(set, (train_ids, val_ids, test_ids))
    assert train_set.isdisjoint(val_set), "Train and validation IDs overlap."
    assert train_set.isdisjoint(test_set), "Train and test IDs overlap."
    assert val_set.isdisjoint(test_set), "Validation and test IDs overlap."


def validate_direction_ids_for_usage(
    direction_ids: list[int],
    direction_split: Mapping[str, Any],
    usage: str,
    require_complete_split: bool = False,
) -> None:
    """Fail loudly when direction IDs are used outside their allowed split."""
    if usage not in DIRECTION_USAGE_TO_SPLIT:
        raise ValueError(
            f"Unknown direction usage {usage!r}; expected one of "
            f"{sorted(DIRECTION_USAGE_TO_SPLIT)}."
        )
    validate_direction_split(
        list(direction_split["train"]),
        list(direction_split["val"]),
        list(direction_split["test"]),
        int(direction_split["num_features"]),
    )
    assert direction_ids and len(direction_ids) == len(set(direction_ids))
    required_split = DIRECTION_USAGE_TO_SPLIT[usage]
    allowed_ids = set(direction_split[required_split])
    requested_ids = set(direction_ids)
    assert requested_ids.issubset(allowed_ids), (
        f"Usage {usage!r} is restricted to {required_split!r} directions; "
        f"invalid IDs: {sorted(requested_ids - allowed_ids)}."
    )
    if require_complete_split:
        assert requested_ids == allowed_ids, (
            f"Usage {usage!r} requires the complete {required_split!r} split."
        )


def split_direction_ids(
    num_features: int,
    num_train: int,
    num_val: int,
    num_test: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Create deterministic, random, disjoint feature-ID splits on CPU."""
    counts = (num_train, num_val, num_test)
    assert num_features > 0
    assert all(count >= 0 for count in counts)
    assert sum(counts) <= num_features, (
        "Requested direction counts exceed the number of SAE features."
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(num_features, generator=generator).tolist()
    train_end = num_train
    val_end = train_end + num_val
    test_end = val_end + num_test
    train_ids = permutation[:train_end]
    val_ids = permutation[train_end:val_end]
    test_ids = permutation[val_end:test_end]
    validate_direction_split(train_ids, val_ids, test_ids, num_features)
    return train_ids, val_ids, test_ids


def save_direction_split(
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    num_features: int,
    seed: int,
    path: str | Path = DEFAULT_SPLIT_PATH,
) -> Path:
    """Persist a validated direction split and its provenance as JSON."""
    validate_direction_split(train_ids, val_ids, test_ids, num_features)
    split_path = Path(path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_features": num_features,
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "num_test": len(test_ids),
        "seed": seed,
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }
    if split_path.exists():
        existing = load_direction_split(split_path)
        if existing != payload:
            raise FileExistsError(
                f"Refusing to replace the existing direction split at {split_path}. "
                "Use a different path deliberately."
            )
        return split_path
    with split_path.open("w", encoding="utf-8") as split_file:
        json.dump(payload, split_file, indent=2)
        split_file.write("\n")
    return split_path


def load_direction_split(
    path: str | Path = DEFAULT_SPLIT_PATH,
) -> dict[str, Any]:
    """Load and validate a persisted direction split."""
    split_path = Path(path)
    with split_path.open("r", encoding="utf-8") as split_file:
        payload = json.load(split_file)
    required_keys = {
        "num_features",
        "num_train",
        "num_val",
        "num_test",
        "seed",
        "train",
        "val",
        "test",
    }
    assert required_keys.issubset(payload), "Direction split JSON is incomplete."
    validate_direction_split(
        payload["train"], payload["val"], payload["test"], payload["num_features"]
    )
    assert len(payload["train"]) == payload["num_train"]
    assert len(payload["val"]) == payload["num_val"]
    assert len(payload["test"]) == payload["num_test"]
    return payload


def get_or_create_direction_split(
    num_features: int,
    num_train: int,
    num_val: int,
    num_test: int,
    seed: int,
    path: str | Path = DEFAULT_SPLIT_PATH,
) -> tuple[list[int], list[int], list[int]]:
    """Reuse an existing split, rejecting configuration drift, or create it."""
    split_path = Path(path)
    if split_path.exists():
        payload = load_direction_split(split_path)
        expected = {
            "num_features": num_features,
            "num_train": num_train,
            "num_val": num_val,
            "num_test": num_test,
            "seed": seed,
        }
        actual = {key: payload[key] for key in expected}
        assert actual == expected, (
            f"Existing direction split metadata {actual} does not match requested "
            f"configuration {expected}. Use a different path deliberately."
        )
        return payload["train"], payload["val"], payload["test"]

    train_ids, val_ids, test_ids = split_direction_ids(
        num_features, num_train, num_val, num_test, seed
    )
    save_direction_split(
        train_ids, val_ids, test_ids, num_features, seed, path=split_path
    )
    return train_ids, val_ids, test_ids


def select_directions(directions: Tensor, direction_ids: list[int]) -> Tensor:
    """Select unique direction rows by ID without moving or casting tensors."""
    assert directions.ndim == 2, (
        f"Expected directions [num_features, d_model], got {directions.shape}."
    )
    assert len(direction_ids) == len(set(direction_ids)), "Direction IDs are duplicated."
    assert all(0 <= direction_id < directions.shape[0] for direction_id in direction_ids)
    index = torch.tensor(direction_ids, dtype=torch.long, device=directions.device)
    selected = directions.index_select(0, index)
    assert selected.shape == (len(direction_ids), directions.shape[1])
    return selected
