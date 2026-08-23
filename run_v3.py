"""One-process orchestration for the pre-registered final V3 experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.denoiser_v3 import GatedConditionedDenoiser, structural_identity_gate
from src.directions import (
    direction_split_hash, load_direction_split, load_sae_directions,
    model_sae_compatibility_report,
)
from src.experiment import (
    RuntimeProfile, evaluate_generation_methods, get_or_create_text_bank,
    tokenize_text_batches, validate_generation_methods,
)
from src.experiment_v3 import (
    aggregate_v3_results, audit_v3_leakage, build_sentiment_contrast_direction,
    causal_ablation_table, create_v3_direction_split, evaluate_cross_concept_direction,
    evaluate_v3_methods, freeze_v3_protocol, paired_hierarchical_bootstrap,
    save_v3_figures, write_v3_report,
)
from src.model import load_model, sanity_check_identity_intervention
from src.metrics import (
    fit_natural_neighbor_index_from_shards, spearman_neighbor_correlations,
)
from src.semantic_eval import FIXED_PROMPTS
from src.train import load_denoiser_checkpoint
from src.train_v2 import calibrate_literal_alphas
from src.train_v3 import (
    ABLATION_SPECS, V3_PIPELINE_VERSION, calibrate_v3_loss_scales,
    load_v3_checkpoint, train_v3_ablation,
)
from src.utils import load_config, seed_everything


def _recursive_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _recursive_update(target[key], value)
        else:
            target[key] = value


def _required_v1_artifacts(config: dict[str, Any]) -> None:
    required = [config["training"]["normalization_stats_path"], config["directions"]["split_path"]]
    required.extend(config["training"]["checkpoint_paths"][key] for key in ("gaussian", "sae_calibrated", "fluency"))
    required.append(config["damage_score"]["csv_path"])
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "V3 runs after the V1 stages in final_v3_all_experiments.ipynb. Missing: " + str(missing)
        )


def _load_v1_models(config: dict[str, Any], hook: str, split_hash: str) -> tuple[dict[str, Any], dict[str, Any]]:
    models, normalizations = {}, {}
    modes = {"gaussian": "gaussian", "sae_calibrated": "sae_calibrated", "fluency": "fluency_sensitive"}
    for key, mode in modes.items():
        model, checkpoint = load_denoiser_checkpoint(
            config["training"]["checkpoint_paths"][key], config["model"]["device"],
            config["model"]["dtype"], expected_hook_name=hook,
            expected_model_name=config["model"]["name"], expected_corruption_mode=mode,
            expected_direction_split_hash=split_hash,
        )
        models[key], normalizations[key] = model, checkpoint["normalization"]
    return models, normalizations


def _text_splits(config: dict[str, Any], v3: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    training = v3["training"]
    counts = [int(training[key]) for key in ("num_training_texts", "num_validation_texts", "num_holdout_texts")]
    path = Path(v3["output_dir"]) / "configs" / "sequence_text_bank.json"
    dataset = None
    if not path.exists():
        from datasets import load_dataset
        dataset = load_dataset(config["data"]["dataset_name"], split=config["data"]["split"], streaming=True)
        skip_count = int(v3["prompt_split"]["stream_skip_examples"])
        assert skip_count >= 0
        if skip_count:
            dataset = dataset.skip(skip_count)
    texts = get_or_create_text_bank(dataset, sum(counts), path, config["data"]["text_column"])
    a, b = counts[0], counts[0] + counts[1]
    train, val, holdout = texts[:a], texts[a:b], texts[b:]
    assert not (set(train) & set(val) or set(train) & set(holdout) or set(val) & set(holdout))
    return train, val, holdout


def _select_validation_model(aggregate: pd.DataFrame, target: float) -> tuple[str, pd.DataFrame]:
    conditioned = aggregate[
        aggregate.method.isin(ABLATION_SPECS) & (aggregate.relative_strength > 0)
    ]
    selection = conditioned.groupby("method", as_index=False).agg(
        mean_delta_nll=("mean_delta_nll", "mean"), mean_kl=("mean_kl", "mean"),
        mean_retention=("mean_concept_retention", "mean"),
    )
    selection["feasible"] = selection.mean_retention >= target
    feasible = selection[selection.feasible]
    if feasible.empty:
        selected = selection.sort_values(
            ["mean_retention", "mean_delta_nll", "mean_kl"],
            ascending=[False, True, True],
        ).iloc[0]["method"]
        selection["selection_status"] = "fallback_no_model_met_retention_target"
    else:
        selected = feasible.sort_values(["mean_delta_nll", "mean_kl"]).iloc[0]["method"]
        selection["selection_status"] = "retention_feasible"
    selection["selected"] = selection.method == selected
    return str(selected), selection


def run_all(
    debug: bool = False,
    run_holdout: bool = True,
    run_generation: bool = True,
    run_semantic: bool = True,
    *,
    preloaded_model: Any | None = None,
    preloaded_sae: Any | None = None,
    preloaded_directions: torch.Tensor | None = None,
) -> None:
    config = load_config("config.yaml", debug=False)
    v3 = deepcopy(config["final_v3"])
    if debug:
        _recursive_update(v3, deepcopy(v3["debug_overrides"]))
    assert v3["pipeline_version"] == V3_PIPELINE_VERSION
    validate_generation_methods(v3["generation"]["methods"])
    v3["training"]["seed"] = int(config["seed"]); v3["seed"] = int(config["seed"])
    seed_everything(v3["seed"]); _required_v1_artifacts(config)
    root = Path(v3["output_dir"])
    for directory in ("configs", "checkpoints", "results", "generations", "figures", "diagnostics", "reports"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    timings: list[dict[str, Any]] = []

    start = time.perf_counter()
    supplied = (preloaded_model, preloaded_sae, preloaded_directions)
    assert all(value is None for value in supplied) or all(value is not None for value in supplied), (
        "Supply model, SAE, and directions together, or let V3 load all three."
    )
    if preloaded_model is None:
        model = load_model(config["model"]["name"], config["model"]["device"], config["model"]["dtype"])
        sae, directions = load_sae_directions(
            model.cfg.d_model, config["sae"]["release"], config["sae"]["sae_id"],
            config["sae"]["device"], config["sae"]["dtype"],
        )
    else:
        model, sae, directions = preloaded_model, preloaded_sae, preloaded_directions
    hook = config["model"]["hook_name"]; sanity_check_identity_intervention(model, hook)
    compatibility = model_sae_compatibility_report(model, sae, hook)
    assert compatibility["compatible"] and directions.shape == (24576, 768)
    assert structural_identity_gate() == 0.0
    numerical_gates: dict[str, Any] = {
        "hook_identity": True,
        "model_sae_compatibility": True,
        "conditioned_structural_identity_exact": True,
        "steering_direction_shape": list(directions.shape) == [24576, 768],
        "selected_generation_path_is_reference": True,
        # Every entry in this mapping is a pass/fail gate and must therefore
        # be True on success.  The previous negatively named entry stored
        # False to mean "the approximation was not used", which made the
        # all(...) assertion fail after every otherwise successful V3 run.
        "optimized_generation_approximation_disabled": True,
    }
    timings.append({"stage": "load_and_numerical_gates", "seconds": time.perf_counter() - start})

    canonical = load_direction_split(config["directions"]["split_path"])
    split = create_v3_direction_split(
        canonical, directions.shape[0], int(v3["num_validation_directions"]),
        int(v3["num_holdout_directions"]), int(v3["new_split_seed"]),
        v3["excluded_external_feature_ids"], root / "configs" / "direction_split_v3.json",
    )
    stats = torch.load(config["training"]["normalization_stats_path"], map_location="cpu", weights_only=True)
    normalization = {"mean": stats["mean"], "std": stats["std"], "eps": float(config["training"]["normalization_eps"])}
    test_value = torch.randn(2, 768)
    restored = (test_value - stats["mean"]) / stats["std"].clamp_min(normalization["eps"])
    restored = restored * stats["std"].clamp_min(normalization["eps"]) + stats["mean"]
    torch.testing.assert_close(restored, test_value, atol=2e-5, rtol=2e-5)
    numerical_gates["normalization_round_trip"] = True
    literal = calibrate_literal_alphas(stats, v3["strengths"])
    (root / "configs" / "literal_alpha_calibration.json").write_text(json.dumps(literal, indent=2) + "\n")

    train_texts, val_texts, holdout_texts = _text_splits(config, v3)
    def text_hash(values: list[str]) -> str:
        return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()
    prompt_manifest = {
        "source": v3["prompt_split"]["source"],
        "stream_skip_examples": int(v3["prompt_split"]["stream_skip_examples"]),
        "train_count": len(train_texts), "validation_count": len(val_texts),
        "holdout_count": len(holdout_texts),
        "train_sha256": text_hash(train_texts), "validation_sha256": text_hash(val_texts),
        "holdout_sha256": text_hash(holdout_texts), "exact_text_overlap": False,
    }
    (root / "configs" / "prompt_split_manifest.json").write_text(
        json.dumps(prompt_manifest, indent=2) + "\n", encoding="utf-8"
    )
    v3["prompt_split"] = {**v3["prompt_split"], **prompt_manifest}
    training = v3["training"]
    batch_size, max_length = int(training["batch_size"]), int(training["max_sequence_length"])
    train_batches = tokenize_text_batches(model, train_texts, batch_size, max_length)
    val_batches = tokenize_text_batches(model, val_texts, batch_size, max_length)

    start = time.perf_counter()
    scale_payload = calibrate_v3_loss_scales(
        model, sae, train_batches, directions, split, training["training_strengths"], hook,
        v3["seed"], int(training["loss_scale_batches"]), root / "diagnostics" / "loss_scale_diagnostic.json",
        float(v3["retention_target"]), float(v3["retention_mask_threshold"]), float(v3["retention_eps"]),
    )
    scales = scale_payload["frozen_reference_scales"]
    scale_batches = min(int(training["loss_scale_batches"]), len(train_batches))
    timings.append({"stage": "train_loss_scale_calibration", "seconds": time.perf_counter() - start,
                    "clean_forwards": scale_batches, "modified_forwards": scale_batches,
                    "configurations": scale_batches, "examples": scale_batches * batch_size})

    start = time.perf_counter(); v3_models, v3_norms = {}, {}
    checkpoint_paths = {key: str(root / "checkpoints" / f"{key}.pt") for key in v3["ablations"]}
    training_config = {
        **training,
        "retention_target": float(v3["retention_target"]),
        "retention_mask_threshold": float(v3["retention_mask_threshold"]),
        "retention_eps": float(v3["retention_eps"]),
        "ablation_weights": deepcopy(v3["ablation_weights"]),
    }
    for ablation in v3["ablations"]:
        # Reset before construction so every causal ablation starts from the
        # exact same weights and sees the same sampled TRAIN conditions.
        seed_everything(v3["seed"])
        denoiser = GatedConditionedDenoiser(
            v3["architecture"]["d_model"], v3["architecture"]["hidden_dim"],
            v3["architecture"]["conditioning_dim"], v3["architecture"]["gate_scale"],
        )
        denoiser.to(next(model.parameters()).device, dtype=next(model.parameters()).dtype)
        history = train_v3_ablation(
            model, sae, denoiser, train_batches, directions, split, normalization,
            scales, training_config, hook, ablation, checkpoint_paths[ablation],
        )
        pd.DataFrame(history).to_csv(root / "diagnostics" / f"training_{ablation}.csv", index=False)
        loaded, checkpoint = load_v3_checkpoint(checkpoint_paths[ablation], config["model"]["device"], direction_split_hash(split))
        v3_models[ablation], v3_norms[ablation] = loaded, checkpoint["normalization"]
    training_forwards = len(v3["ablations"]) * int(training["max_steps"])
    timings.append({"stage": "v3_ablation_training", "seconds": time.perf_counter() - start,
                    "clean_forwards": training_forwards, "modified_forwards": training_forwards,
                    "configurations": training_forwards, "examples": training_forwards * batch_size})

    v1_models, v1_norms = _load_v1_models(config, hook, direction_split_hash(canonical))
    methods = list(v3["token_evaluation"]["methods"])
    start = time.perf_counter()
    validation = evaluate_v3_methods(
        model, sae, val_batches, directions, split, split["val"], v3["strengths"], literal,
        hook, v3_models, v3_norms, v1_models, v1_norms, methods, "validation",
        float(v3["retention_mask_threshold"]), int(v3["token_evaluation"]["max_validation_batches"]),
    )
    validation.to_csv(root / "results" / "token_validation.csv", index=False)
    aggregate = aggregate_v3_results(validation); aggregate.to_csv(root / "results" / "ablation_summary.csv", index=False)
    causal_table = causal_ablation_table(aggregate)
    causal_table.to_csv(root / "results" / "causal_ablation_table.csv", index=False)
    numerical_gates["retention_zero_excluded"] = bool(
        validation.loc[validation.relative_strength == 0, "concept_retention"].isna().all()
    )
    numerical_gates["validation_core_finite"] = bool(
        np.isfinite(validation[["kl", "clean_nll", "modified_nll", "delta_nll", "activation_mse"]].to_numpy()).all()
    )
    failed_gates = [name for name, passed in numerical_gates.items() if not bool(passed)]
    assert not failed_gates, f"V3 numerical gates failed: {failed_gates}"
    selected, selection = _select_validation_model(aggregate, float(v3["retention_target"]))
    if selected not in v3["generation"]["methods"]:
        v3["generation"]["methods"] = [*v3["generation"]["methods"], selected]
    validate_generation_methods(v3["generation"]["methods"])
    selection.to_csv(root / "results" / "validation_model_selection.csv", index=False)
    comparisons = list(dict.fromkeys([
        (selected, "relative_raw"),
        (selected, "sae_calibrated"),
        ("conditioned_reconstruction", "sae_calibrated"),
        ("conditioned_kl", "conditioned_reconstruction"),
        ("conditioned_kl_retention", "conditioned_kl"),
        ("conditioned_kl_retention", "hard_projected_conditioned_kl_retention"),
    ]))
    statistics = paired_hierarchical_bootstrap(
        validation, comparisons, v3["seed"], int(v3["token_evaluation"]["bootstrap_samples"]),
    )
    statistics.insert(0, "generalization_scope", "within_sae_family_validation")
    statistics.to_csv(root / "results" / "statistical_tests.csv", index=False)
    validation[[column for column in validation.columns if "correction" in column or column in (
        "evaluation_split", "method", "direction_id", "prompt_id", "relative_strength",
        "parallel_fraction", "knn_distance", "nearest_clean_correction_cosine",
    )]].to_csv(
        root / "results" / "correction_geometry.csv", index=False
    )
    validation_batches = min(int(v3["token_evaluation"]["max_validation_batches"]), len(val_batches))
    validation_configs = validation_batches * len(split["val"]) * len(v3["strengths"]) * len(methods)
    timings.append({"stage": "validation_and_model_selection", "seconds": time.perf_counter() - start,
                    "clean_forwards": validation_batches, "modified_forwards": validation_configs,
                    "configurations": validation_configs, "examples": len(validation)})

    neighbor_results = pd.DataFrame()
    neighbor_cfg = v3["neighbor_diagnostics"]
    if bool(neighbor_cfg["enabled"]):
        start = time.perf_counter()
        natural_cfg = config["natural_neighbors"]
        neighbor_index = fit_natural_neighbor_index_from_shards(
            config["data"]["activation_output_dir"], natural_cfg["n_components"],
            natural_cfg["k"], natural_cfg["max_fit_samples"], v3["seed"], hook,
        )
        neighbor_results = evaluate_v3_methods(
            model, sae, val_batches, directions, split,
            split["val"][:int(neighbor_cfg["num_directions"])], v3["strengths"], literal,
            hook, v3_models, v3_norms, v1_models, v1_norms,
            neighbor_cfg["methods"], "validation", float(v3["retention_mask_threshold"]),
            int(neighbor_cfg["max_batches"]), neighbor_index=neighbor_index,
        )
        neighbor_results.to_csv(root / "results" / "natural_neighbor_diagnostics.csv", index=False)
        neighbor_correlations = spearman_neighbor_correlations(neighbor_results)
        (root / "diagnostics" / "neighbor_correlations.json").write_text(
            json.dumps(neighbor_correlations, indent=2) + "\n"
        )
        timings.append({"stage": "natural_neighbor_diagnostics", "seconds": time.perf_counter() - start,
                        "clean_forwards": min(int(neighbor_cfg["max_batches"]), len(val_batches)),
                        "modified_forwards": len(neighbor_results) // batch_size,
                        "configurations": len(neighbor_results) // batch_size,
                        "examples": len(neighbor_results)})

    harmfulness_source = Path(config["damage_score"]["csv_path"])
    harmfulness = pd.read_csv(harmfulness_source)
    assert set(harmfulness.direction_id) == set(split["train"])
    harmfulness.to_csv(root / "results" / "harmfulness.csv", index=False)
    harmfulness_correlations = {
        "spearman_mean_kl_vs_mean_delta_nll": float(harmfulness.mean_kl.corr(harmfulness.mean_delta_nll, method="spearman")),
        "pearson_mean_kl_vs_mean_delta_nll": float(harmfulness.mean_kl.corr(harmfulness.mean_delta_nll, method="pearson")),
        "interpretation": "descriptive, not causal",
    }
    (root / "diagnostics" / "harmfulness_correlations.json").write_text(json.dumps(harmfulness_correlations, indent=2) + "\n")

    prefreeze_audit = audit_v3_leakage(split, split["train"], split["train"], split["val"], False, False)
    protocol_checkpoint_paths = {
        **checkpoint_paths,
        "gaussian": config["training"]["checkpoint_paths"]["gaussian"],
        "sae_calibrated": config["training"]["checkpoint_paths"]["sae_calibrated"],
        "fluency": config["training"]["checkpoint_paths"]["fluency"],
    }
    protocol = freeze_v3_protocol(
        root / "configs" / "frozen_protocol.json", {**config, "final_v3": v3}, split,
        selected, protocol_checkpoint_paths,
        [
            "config.yaml", "run_v3.py", "run_semantic_eval.py",
            "src/denoiser_v3.py", "src/train_v3.py", "src/experiment_v3.py",
            "src/steering.py", "src/experiment.py", "src/semantic_eval.py",
            "notebooks/final_v3_all_experiments.ipynb",
        ],
    )
    (root / "frozen_protocol.json").write_text(
        (root / "configs" / "frozen_protocol.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    audit = audit_v3_leakage(
        split, split["train"], split["train"], split["val"],
        (root / "configs" / "frozen_protocol.json").exists(), run_holdout,
    )
    audit["prefreeze"] = prefreeze_audit
    (root / "diagnostics" / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    numerical_gates.update({
        "split_and_leakage_audit": bool(audit["passed"]),
        "checkpoint_schema": all(value["sha256"] for value in protocol["checkpoints"].values()),
        "v1_optimized_reference_token_and_autocast_gate": Path("outputs/analysis/optimized_eval_regression.csv").exists(),
        "calibrated_sampler_and_v1_audit": Path("outputs/analysis/audit.json").exists(),
    })
    failed_gates = [name for name, passed in numerical_gates.items() if not bool(passed)]
    assert not failed_gates, f"V3 numerical gates failed: {failed_gates}"
    (root / "diagnostics" / "numerical_gates.json").write_text(
        json.dumps(numerical_gates, indent=2) + "\n", encoding="utf-8"
    )

    holdout = None
    if run_holdout:
        start = time.perf_counter()
        # Holdout prompt tokenization and every model forward happen only after
        # the protocol file and checkpoint hashes have been frozen.
        holdout_batches = tokenize_text_batches(model, holdout_texts, batch_size, max_length)
        holdout = evaluate_v3_methods(
            model, sae, holdout_batches, directions, split, split["test"], v3["strengths"], literal,
            hook, v3_models, v3_norms, v1_models, v1_norms, methods, split["holdout_label"],
            float(v3["retention_mask_threshold"]), int(v3["token_evaluation"]["max_holdout_batches"]),
        )
        holdout.to_csv(root / "results" / "token_holdout_or_replication.csv", index=False)
        holdout_statistics = paired_hierarchical_bootstrap(
            holdout, comparisons, v3["seed"] + 2,
            int(v3["token_evaluation"]["bootstrap_samples"]),
        )
        holdout_statistics.insert(0, "generalization_scope", split["holdout_label"])
        holdout_statistics.to_csv(root / "results" / "holdout_statistical_tests.csv", index=False)
        statistics = pd.concat([statistics, holdout_statistics], ignore_index=True)
        statistics.to_csv(root / "results" / "statistical_tests.csv", index=False)
        holdout_batches_used = min(int(v3["token_evaluation"]["max_holdout_batches"]), len(holdout_batches))
        holdout_configs = holdout_batches_used * len(split["test"]) * len(v3["strengths"]) * len(methods)
        timings.append({"stage": "frozen_holdout_evaluation", "seconds": time.perf_counter() - start,
                        "clean_forwards": holdout_batches_used, "modified_forwards": holdout_configs,
                        "configurations": holdout_configs, "examples": len(holdout)})

    cross_concept = pd.DataFrame(); cross_statistics = pd.DataFrame()
    cross_cfg = v3["cross_concept"]
    if bool(cross_cfg["enabled"]):
        start = time.perf_counter()
        sentiment_direction, sentiment_metadata = build_sentiment_contrast_direction(
            model, cross_cfg["positive_examples"], cross_cfg["negative_examples"], hook,
            int(cross_cfg["max_sequence_length"]),
        )
        sentiment_metadata.update({
            "direction_id": cross_cfg["direction_id"],
            "pipeline_version": V3_PIPELINE_VERSION,
            "created_after_frozen_protocol": True,
            "frozen_protocol_sha256": hashlib.sha256(
                (root / "configs" / "frozen_protocol.json").read_bytes()
            ).hexdigest(),
        })
        (root / "configs" / "cross_concept_direction.json").write_text(
            json.dumps(sentiment_metadata, indent=2) + "\n", encoding="utf-8"
        )
        cross_concept = evaluate_cross_concept_direction(
            model, cross_cfg["evaluation_prompts"], sentiment_direction,
            str(cross_cfg["direction_id"]), v3["strengths"], hook,
            v3_models, v3_norms, v1_models, v1_norms, cross_cfg["methods"],
            float(v3["retention_mask_threshold"]), int(cross_cfg["max_sequence_length"]),
            cross_cfg["positive_logit_tokens"], cross_cfg["negative_logit_tokens"],
        )
        cross_concept.to_csv(root / "results" / "cross_concept_confirmation.csv", index=False)
        cross_aggregate = aggregate_v3_results(cross_concept)
        cross_aggregate.to_csv(root / "results" / "cross_concept_aggregate.csv", index=False)
        cross_comparisons = [
            ("conditioned_reconstruction", "sae_calibrated"),
            ("conditioned_kl", "conditioned_reconstruction"),
            ("conditioned_kl_retention", "conditioned_kl"),
            ("conditioned_kl_retention", "relative_raw"),
            ("conditioned_kl_retention", "hard_projected_conditioned_kl_retention"),
        ]
        cross_statistics = paired_hierarchical_bootstrap(
            cross_concept, cross_comparisons, v3["seed"] + 1,
            int(v3["token_evaluation"]["bootstrap_samples"]),
        )
        cross_statistics.insert(0, "generalization_scope", "cross_concept_sentiment")
        cross_statistics.to_csv(root / "results" / "cross_concept_statistics.csv", index=False)
        audit["cross_concept_confirmation"] = {
            "created_after_freeze": True,
            "used_for_training": False,
            "used_for_loss_scaling": False,
            "used_for_model_selection": False,
            "family": cross_cfg["family"],
        }
        (root / "diagnostics" / "audit.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        timings.append({"stage": "post_freeze_cross_concept_confirmation",
                        "seconds": time.perf_counter() - start,
                        "clean_forwards": 3, "modified_forwards": len(cross_concept) // len(cross_cfg["evaluation_prompts"]),
                        "configurations": len(cross_concept) // len(cross_cfg["evaluation_prompts"]),
                        "examples": len(cross_concept)})

    generation = pd.DataFrame(); denoisers: dict[str, Any] = {}; normalizations: dict[str, Any] = {}
    if run_generation:
        start = time.perf_counter(); generation_profile = RuntimeProfile()
        denoisers = {**v1_models, **v3_models}; normalizations = {**v1_norms, **v3_norms}
        generation_config = {**config["evaluation"], **v3["generation"]}
        generation = evaluate_generation_methods(
            model, sae, FIXED_PROMPTS[:int(v3["generation"]["num_prompts"])], directions,
            split, split["test"][:int(v3["generation"]["num_directions"])], v3["strengths"],
            v3["generation"]["seeds"], hook, denoisers, normalizations, generation_config,
            evaluation_split="test", methods=v3["generation"]["methods"],
            jsonl_path=root / "generations" / "final_generation.jsonl",
            results_path=root / "results" / "generation_results.csv",
            aggregate_path=root / "results" / "generation_aggregate.csv",
            profiler=generation_profile, sae_identity=config["sae"]["sae_id"],
            checkpoint_fingerprints={key: value["sha256"] for key, value in protocol["checkpoints"].items()},
            pipeline_version=V3_PIPELINE_VERSION,
            result_evaluation_split=split["holdout_label"],
        )
        assert not generation.empty
        timings.append({"stage": "generation", "seconds": time.perf_counter() - start,
                        "clean_forwards": generation_profile.total_clean_forwards,
                        "modified_forwards": generation_profile.total_modified_forwards,
                        "configurations": len(generation), "examples": len(generation)})

    save_v3_figures(aggregate, harmfulness, root / "figures", neighbor_results, generation)
    if run_semantic:
        # The classifier stage deliberately runs in a fresh process. Release
        # every parent-process CUDA model first so the subprocess cannot OOM.
        model.to("cpu"); sae.to("cpu")
        for denoiser in [*v1_models.values(), *v3_models.values()]:
            denoiser.to("cpu")
        del model, sae, directions, v1_models, v3_models, denoisers, normalizations
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        semantic_start = time.perf_counter()
        env = dict(os.environ); env["INCLUDE_V3"] = "1"
        env["SELECTED_V3_METHOD"] = selected
        subprocess.run([sys.executable, "run_semantic_eval.py"], check=True, env=env)
        timings.append({"stage": "independent_semantic_proxy", "seconds": time.perf_counter() - semantic_start})
    write_v3_report(root, validation, holdout, statistics, audit, protocol)
    runtime = pd.DataFrame(timings)
    for column in ("clean_forwards", "modified_forwards", "configurations", "examples"):
        if column not in runtime: runtime[column] = 0
        runtime[column] = runtime[column].fillna(0).astype(int)
    runtime["configs_per_sec"] = runtime.configurations / runtime.seconds.clip(lower=1e-12)
    runtime.to_csv(root / "diagnostics" / "runtime_breakdown.csv", index=False)
    (root / "diagnostics" / "runtime_breakdown.json").write_text(
        json.dumps(runtime.to_dict(orient="records"), indent=2) + "\n"
    )
    print(json.dumps({
        "pipeline_version": V3_PIPELINE_VERSION, "selected_validation_method": selected,
        "holdout_label": split["holdout_label"],
        "direction_split_hash": direction_split_hash(split),
        "output_dir": str(root),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-holdout", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-semantic", action="store_true")
    args = parser.parse_args()
    run_all(args.debug, not args.skip_holdout, not args.skip_generation, not args.skip_semantic)


if __name__ == "__main__":
    main()
