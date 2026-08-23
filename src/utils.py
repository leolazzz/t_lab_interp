"""Configuration, reproducibility, and output-directory utilities."""

from pathlib import Path
from copy import deepcopy
import hashlib
import os
import random
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import yaml


OUTPUT_SUBDIRECTORIES = (
    "activations",
    "direction_scores",
    "checkpoints",
    "generations",
    "results",
    "figures",
    "validation",
    "analysis",
    "final_test",
    "final_figures",
)

PIPELINE_VERSION = "final_v2"
CHECKPOINT_SCHEMA_VERSION = 2


def normalize_activations(
    activations: Tensor,
    mean: Tensor,
    std: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Standardize ``[..., d_model]`` activations without mutating inputs."""
    assert activations.ndim >= 2
    assert mean.ndim == std.ndim == 1
    assert activations.shape[-1] == mean.numel() == std.numel()
    assert activations.device == mean.device == std.device
    assert activations.dtype == mean.dtype == std.dtype
    assert eps > 0.0
    return (activations - mean) / std.clamp_min(eps)


def denormalize_activations(
    standardized: Tensor,
    mean: Tensor,
    std: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Invert :func:`normalize_activations` without mutating inputs."""
    assert standardized.ndim >= 2
    assert mean.ndim == std.ndim == 1
    assert standardized.shape[-1] == mean.numel() == std.numel()
    assert standardized.device == mean.device == std.device
    assert standardized.dtype == mean.dtype == std.dtype
    assert eps > 0.0
    return standardized * std.clamp_min(eps) + mean


def load_config(
    path: str | Path = "config.yaml",
    debug: bool | None = None,
) -> dict[str, Any]:
    """Load a YAML configuration file into a plain dictionary."""
    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    debug_enabled = bool(config.get("debug", {}).get("enabled", False))
    if debug is True or (debug is None and debug_enabled):
        config = apply_debug_config(config)
    return config


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch, and all visible CUDA generators."""
    assert seed >= 0
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def seed_data_loader_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a device and fail explicitly when requested CUDA is unavailable."""
    resolved = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this runtime.")
    return resolved


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Resolve the small set of floating dtypes supported by this project."""
    if isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype {dtype!r}; expected {sorted(mapping)}.")
        resolved = mapping[dtype]
    assert resolved.is_floating_point
    return resolved


def _merge_mappings(base: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge_mappings(base[key], value)
        else:
            base[key] = deepcopy(value)


def apply_debug_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with the lightweight ``debug.overrides`` applied."""
    resolved = deepcopy(dict(config))
    debug_config = resolved.get("debug", {})
    overrides = debug_config.get("overrides", {})
    assert isinstance(overrides, Mapping), "debug.overrides must be a mapping."
    _merge_mappings(resolved, overrides)
    resolved.setdefault("debug", {})["enabled"] = True
    return resolved


def create_output_directories(
    root: str | Path = "outputs",
) -> Mapping[str, Path]:
    """Create standard output directories and return their resolved paths."""
    output_root = Path(root)
    paths = {name: output_root / name for name in OUTPUT_SUBDIRECTORIES}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def file_fingerprint(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a short SHA-256 fingerprint for a required checkpoint/artifact."""
    resolved = Path(path)
    assert resolved.is_file(), f"Cannot fingerprint missing file: {resolved}"
    assert chunk_size > 0
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()[:20]


def run_pipeline_audit(
    *,
    notebook_path: str | Path = "kaggle_final.ipynb",
    model: Any,
    sae: Any,
    hook_name: str,
    direction_split: Mapping[str, Any],
    smoke_results: pd.DataFrame,
    config: Mapping[str, Any],
    calibrated_sampler: Any | None = None,
    checkpoint_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, bool]:
    """Run the final lightweight correctness audit and stop on critical failure."""
    import inspect

    from src.directions import get_sae_hook_name, validate_direction_split
    from src.experiment import evaluate_generation_methods, positive_strength_concept_retention
    from src.train import calibrated_sampler_math_gate, correction_diagnostics

    notebook_text = Path(notebook_path).read_text(encoding="utf-8")
    statuses: dict[str, bool] = {}
    statuses["no legacy notebook execution"] = not any(
        marker in notebook_text for marker in ("_run_notebook_cells", "exec(", "nbconvert")
    )
    statuses["model/SAE dimensions match"] = (
        int(model.cfg.d_model) == int(sae.W_dec.shape[-1])
        and get_sae_hook_name(sae) == hook_name
    )
    try:
        validate_direction_split(
            list(direction_split["train"]), list(direction_split["val"]),
            list(direction_split["test"]), int(direction_split["num_features"]),
        )
        statuses["split disjoint"] = True
        statuses["no test leakage"] = set(direction_split["train"]).isdisjoint(
            set(direction_split["val"]) | set(direction_split["test"])
        )
    except (AssertionError, KeyError):
        statuses["split disjoint"] = False
        statuses["no test leakage"] = False

    d_model = int(model.cfg.d_model)
    sample = torch.randn(3, d_model)
    mean = torch.randn(d_model)
    std = torch.rand(d_model) + 0.1
    roundtrip = denormalize_activations(normalize_activations(sample, mean, std), mean, std)
    # The notebook gate uses these explicit float32 tolerances.  PyTorch's
    # default absolute tolerance (1e-8) is unnecessarily strict for the
    # subtract/divide/multiply/add round trip and can fail depending on the
    # random sample even when normalization is correct.
    statuses["normalization round trip"] = torch.allclose(
        roundtrip, sample, rtol=2e-5, atol=2e-5
    )
    statuses["calibrated standalone sampler math"] = False
    if calibrated_sampler is not None:
        component = (
            calibrated_sampler.samplers[0]
            if hasattr(calibrated_sampler, "samplers") else calibrated_sampler
        )
        gate = calibrated_sampler_math_gate(
            component, d_model, magnitudes=(1.0, 4.0), num_examples=4
        )
        statuses["calibrated standalone sampler math"] = bool(
            (gate["relative_error"] < 1e-3).all()
        )
    clean = torch.zeros(3, d_model)
    direction = torch.zeros(d_model)
    direction[0] = 1.0
    magnitude = 4.0
    corrupted = clean + magnitude * direction
    geometry = correction_diagnostics(clean, corrupted, clean)
    statuses["correction diagnostics invariant"] = torch.allclose(
        geometry["corruption_norm"].square(),
        geometry["corrupted_mse"] * d_model,
    )
    source = inspect.getsource(evaluate_generation_methods)
    statuses["evaluation uses inference mode"] = "@torch.inference_mode()" in source
    statuses["no multiprocessing DataLoader"] = (
        int(config.get("training", {}).get("num_workers", -1)) == 0
        and not bool(config.get("training", {}).get("persistent_workers", True))
    )
    core_numeric_columns = [
        column for column in (
            "kl", "clean_nll", "modified_nll", "delta_nll",
            "activation_norm_ratio", "target_sae_activation",
        ) if column in smoke_results.columns
    ]
    numeric = smoke_results[core_numeric_columns]
    statuses["no NaN/inf in smoke results"] = (
        not numeric.empty and np.isfinite(numeric.to_numpy()).all()
    )
    statuses["raw/raw concept retention"] = False
    if {"target_sae_activation", "strength"}.issubset(smoke_results.columns):
        retention = positive_strength_concept_retention(smoke_results)
        raw_values = retention.loc[retention.method == "raw", "concept_retention"]
        statuses["raw/raw concept retention"] = (
            not raw_values.empty and np.allclose(raw_values, 1.0, atol=1e-12, rtol=0)
        )
    statuses["checkpoint schema compatible"] = True
    statuses["calibrated REAL training batch math"] = True
    checkpoint_failures: list[str] = []
    if checkpoint_paths:
        from src.train import load_denoiser_checkpoint
        expected_model_name = config.get("model", {}).get("name")
        assert isinstance(expected_model_name, str) and expected_model_name, (
            "config.model.name is required for checkpoint audit."
        )
        for checkpoint_name, path in checkpoint_paths.items():
            if not Path(path).exists():
                statuses["checkpoint schema compatible"] = False
                checkpoint_failures.append(
                    f"{checkpoint_name}: checkpoint does not exist at {path}"
                )
                continue
            try:
                _, checkpoint = load_denoiser_checkpoint(
                    path, device="cpu", dtype="float32",
                    expected_hook_name=hook_name,
                    # TransformerLens may canonicalize cfg.model_name to
                    # ``gpt2`` while the project/checkpoint uses
                    # ``gpt2-small``. Compare against the same configured name
                    # that was saved during training.
                    expected_model_name=expected_model_name,
                )
            except (AssertionError, KeyError, RuntimeError) as error:
                statuses["checkpoint schema compatible"] = False
                checkpoint_failures.append(
                    f"{checkpoint_name} ({path}): {type(error).__name__}: {error}"
                )
                continue
            if checkpoint.get("corruption_mode") in {"sae_calibrated", "fluency_sensitive"}:
                statuses["calibrated REAL training batch math"] &= bool(
                    checkpoint.get("real_training_math_verified", False)
                )

    # Some NumPy predicates (notably ``np.allclose``) return ``numpy.bool_``.
    # They print exactly like Python booleans but are not serializable by the
    # standard JSON encoder.  The audit is a public, JSON-persisted result, so
    # normalize every value to the built-in type before printing and returning.
    statuses = {label: bool(ok) for label, ok in statuses.items()}

    print("PIPELINE AUDIT")
    for label, ok in statuses.items():
        print(f"[{'OK' if ok else 'FAIL'}] {label}")
    for failure in checkpoint_failures:
        print(f"[CHECKPOINT ERROR] {failure}")
    failed = [label for label, ok in statuses.items() if not ok]
    assert not failed, f"Critical pipeline audit failures: {failed}"
    return statuses


def _save_figure(figure: Any, path: str | Path) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(resolved_path, dpi=150)
    plt.close(figure)
    return resolved_path


def plot_pareto_fluency_concept(
    results: pd.DataFrame,
    path: str | Path = "outputs/figures/pareto_fluency_concept.png",
    fluency_cost_column: str = "delta_nll",
    concept_column: str = "concept_score",
) -> Path:
    """Plot concept strength against negative fluency cost (higher is better)."""
    required = {"method", "alpha", fluency_cost_column, concept_column}
    assert required.issubset(results.columns)
    grouped = (
        results.groupby(["method", "alpha"], as_index=False)[
            [fluency_cost_column, concept_column]
        ]
        .mean()
        .sort_values(["method", "alpha"])
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    for method, method_rows in grouped.groupby("method", sort=False):
        axis.plot(
            -method_rows[fluency_cost_column],
            method_rows[concept_column],
            marker="o",
            label=str(method),
        )
    axis.set_xlabel(f"Fluency (-{fluency_cost_column}; higher is better)")
    axis.set_ylabel(f"{concept_column} (higher is stronger concept)")
    axis.set_title("Concept-strength / fluency Pareto curves")
    axis.legend()
    return _save_figure(figure, path)


def plot_alpha_vs_metric(
    results: pd.DataFrame,
    metric: str,
    path: str | Path,
    ylabel: str,
) -> Path:
    """Plot a mean metric over alpha with one line per method."""
    assert {"method", "alpha", metric}.issubset(results.columns)
    grouped = (
        results.groupby(["method", "alpha"], as_index=False)[metric]
        .mean()
        .sort_values(["method", "alpha"])
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    for method, method_rows in grouped.groupby("method", sort=False):
        axis.plot(
            method_rows["alpha"],
            method_rows[metric],
            marker="o",
            label=str(method),
        )
    axis.set_xlabel("Steering strength alpha")
    axis.set_ylabel(ylabel)
    axis.set_title(f"Alpha vs {ylabel}")
    axis.legend()
    return _save_figure(figure, path)


def plot_alpha_vs_delta_nll(
    results: pd.DataFrame,
    path: str | Path = "outputs/figures/alpha_vs_delta_nll.png",
) -> Path:
    return plot_alpha_vs_metric(
        results,
        "delta_nll",
        path,
        "Delta NLL (lower is better fluency)",
    )


def plot_alpha_vs_activation_norm_ratio(
    results: pd.DataFrame,
    path: str | Path = "outputs/figures/alpha_vs_activation_norm_ratio.png",
) -> Path:
    return plot_alpha_vs_metric(
        results,
        "activation_norm_ratio",
        path,
        "Activation norm ratio (1 is clean norm)",
    )


def plot_direction_damage_histogram(
    scores: pd.DataFrame | torch.Tensor | np.ndarray,
    path: str | Path = "outputs/figures/direction_damage_histogram.png",
    score_column: str = "mean_kl",
) -> Path:
    """Plot the train-only fluency-damage score distribution."""
    if isinstance(scores, pd.DataFrame):
        assert score_column in scores.columns
        values = scores[score_column].to_numpy()
    elif isinstance(scores, torch.Tensor):
        values = scores.detach().cpu().numpy()
    else:
        values = np.asarray(scores)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    assert values.size > 0 and np.isfinite(values).all()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(values, bins=min(40, max(5, values.size // 2)))
    axis.set_xlabel("Mean KL(clean || modified); higher is more damaging")
    axis.set_ylabel("Training SAE directions")
    axis.set_title("Train-direction fluency damage")
    return _save_figure(figure, path)


def plot_knn_distance_vs_delta_nll(
    diagnostics: pd.DataFrame,
    path: str | Path = "outputs/figures/knn_distance_vs_delta_nll.png",
) -> Path:
    """Scatter natural-neighbor distance against fluency degradation."""
    assert {"knn_distance", "delta_nll"}.issubset(diagnostics.columns)
    from src.metrics import spearman_neighbor_correlations

    rho = spearman_neighbor_correlations(diagnostics)[
        "knn_distance_vs_delta_nll"
    ]
    figure, axis = plt.subplots(figsize=(6, 5))
    if "method" in diagnostics:
        for method, rows in diagnostics.groupby("method", sort=False):
            axis.scatter(rows["knn_distance"], rows["delta_nll"], label=str(method))
        axis.legend()
    else:
        axis.scatter(diagnostics["knn_distance"], diagnostics["delta_nll"])
    axis.set_xlabel("Mean kNN distance to clean activations (lower is more natural)")
    axis.set_ylabel("Delta NLL (lower is better fluency)")
    axis.set_title(f"Natural-neighbor distance vs fluency; Spearman rho={rho:.3f}")
    return _save_figure(figure, path)


def plot_concept_vs_nll(
    results: pd.DataFrame,
    path: str | Path = "outputs/figures/concept_vs_nll.png",
    nll_column: str = "delta_nll",
) -> Path:
    """Compare the four primary corruption distributions."""
    labels = {
        "raw": "raw",
        "gaussian_denoiser": "Gaussian",
        "sae_calibrated": "calibrated uniform SAE",
        "sae_denoiser": "calibrated uniform SAE",
        "fluency_denoiser": "fluency-sensitive SAE",
    }
    assert {"method", "concept_score", nll_column}.issubset(results.columns)
    selected = results[results["method"].isin(labels)].copy()
    assert not selected.empty, "No primary comparison methods are present."
    figure, axis = plt.subplots(figsize=(7, 5))
    for method, rows in selected.groupby("method", sort=False):
        axis.scatter(
            rows[nll_column],
            rows["concept_score"],
            label=labels[method],
            alpha=0.75,
        )
    axis.set_xlabel(f"{nll_column} (lower is better fluency)")
    axis.set_ylabel("Concept score (higher is stronger concept)")
    axis.set_title("Concept strength vs fluency")
    axis.legend()
    return _save_figure(figure, path)


def plot_incremental_steps_ablation(
    results: pd.DataFrame,
    path: str | Path = "outputs/figures/incremental_steps_ablation.png",
    metric: str = "delta_nll",
) -> Path | None:
    """Plot n_steps ablation, returning None when those rows do not exist."""
    if "n_steps" not in results.columns or metric not in results.columns:
        return None
    incremental = results[results["n_steps"].notna()]
    if incremental.empty:
        return None
    grouped = incremental.groupby("n_steps", as_index=False)[metric].mean()
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(grouped["n_steps"], grouped[metric], marker="o")
    axis.set_xlabel("Incremental steering steps")
    axis.set_ylabel(f"{metric} (lower is better fluency)")
    axis.set_title("Incremental steering ablation")
    axis.set_xticks(sorted(grouped["n_steps"].astype(int).unique()))
    return _save_figure(figure, path)


def generate_standard_figures(
    results: pd.DataFrame,
    output_directory: str | Path = "outputs/figures",
    damage_scores: pd.DataFrame | torch.Tensor | np.ndarray | None = None,
    neighbor_diagnostics: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Generate every applicable standard research figure."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    created: dict[str, Path] = {}
    fluency_cost_column = (
        "delta_nll" if "delta_nll" in results.columns else "clean_model_nll"
    )
    if {fluency_cost_column, "concept_score", "method", "alpha"}.issubset(results.columns):
        created["pareto"] = plot_pareto_fluency_concept(
            results,
            output / "pareto_fluency_concept.png",
            fluency_cost_column=fluency_cost_column,
        )
        created["concept_nll"] = plot_concept_vs_nll(
            results,
            output / "concept_vs_nll.png",
            nll_column=fluency_cost_column,
        )
    if {"delta_nll", "method", "alpha"}.issubset(results.columns):
        created["alpha_delta_nll"] = plot_alpha_vs_delta_nll(
            results, output / "alpha_vs_delta_nll.png"
        )
    if {"activation_norm_ratio", "method", "alpha"}.issubset(results.columns):
        created["alpha_norm"] = plot_alpha_vs_activation_norm_ratio(
            results, output / "alpha_vs_activation_norm_ratio.png"
        )
    if damage_scores is not None:
        created["damage_histogram"] = plot_direction_damage_histogram(
            damage_scores, output / "direction_damage_histogram.png"
        )
    if neighbor_diagnostics is not None:
        created["knn_delta_nll"] = plot_knn_distance_vs_delta_nll(
            neighbor_diagnostics, output / "knn_distance_vs_delta_nll.png"
        )
    incremental_path = plot_incremental_steps_ablation(
        results,
        output / "incremental_steps_ablation.png",
        metric=fluency_cost_column,
    )
    if incremental_path is not None:
        created["incremental_ablation"] = incremental_path
    return created
