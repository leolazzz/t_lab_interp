import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd


if "transformer_lens" not in sys.modules and importlib.util.find_spec("transformer_lens") is None:
    fake_transformer_lens = types.ModuleType("transformer_lens")
    fake_transformer_lens.HookedTransformer = object
    sys.modules["transformer_lens"] = fake_transformer_lens

from src.denoiser_v3 import (  # noqa: E402
    GatedConditionedDenoiser, apply_gated_conditioned_denoiser,
)
from src.experiment_v3 import (  # noqa: E402
    aggregate_v3_results, causal_ablation_table, create_v3_direction_split,
    paired_hierarchical_bootstrap, retention_fields, write_v3_report,
)
from src.steering import apply_steering, relative_raw_steering  # noqa: E402
from src.train_v3 import ABLATION_SPECS, resolved_ablation_spec  # noqa: E402


def test_structural_identity_is_exact_after_training_perturbs_weights() -> None:
    denoiser = GatedConditionedDenoiser(8, 12, 4)
    with torch.no_grad():
        for parameter in denoiser.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.1)
    x = torch.randn(3, 5, 8)
    direction = F.normalize(torch.randn(3, 8), dim=-1)
    output = denoiser(x, direction, torch.zeros(3))
    assert torch.equal(output, x)


def test_raw_domain_identity_bypasses_normalization_rounding() -> None:
    denoiser = GatedConditionedDenoiser(8, 12, 4)
    x = torch.randn(2, 3, 8)
    direction = F.normalize(torch.randn(8), dim=-1)
    stats = {"mean": torch.randn(8), "std": torch.rand(8) + 0.2, "eps": 1e-6}
    output = apply_gated_conditioned_denoiser(denoiser, x, direction, 0.0, stats)
    assert torch.equal(output, x)


def test_retention_zero_and_small_denominator_are_invalid() -> None:
    zero = retention_fields(2.0, 2.0, 0.0, 1e-3)
    tiny = retention_fields(1e-5, 1.0, 0.5, 1e-3)
    valid = retention_fields(2.0, 1.6, 0.5, 1e-3)
    assert not zero["valid_retention_ratio"] and torch.isnan(torch.tensor(zero["concept_retention"]))
    assert not tiny["valid_retention_ratio"] and tiny["violates_80_percent"] is None
    assert valid["valid_retention_ratio"] and abs(valid["concept_retention"] - 0.8) < 1e-7


def test_v3_split_excludes_every_observed_canonical_direction() -> None:
    canonical = {
        "num_features": 30, "num_train": 3, "num_val": 2, "num_test": 2,
        "seed": 1, "train": [0, 1, 2], "val": [3, 4], "test": [5, 6],
    }
    with tempfile.TemporaryDirectory() as directory:
        first = create_v3_direction_split(
            canonical, 30, 3, 3, 99, [7], Path(directory) / "split.json"
        )
        second = create_v3_direction_split(
            canonical, 30, 3, 3, 99, [7], Path(directory) / "split.json"
        )
    assert first == second
    assert set(first["val"] + first["test"]).isdisjoint(set(range(8)))
    assert set(first["train"]) == {0, 1, 2}


def test_causal_ablation_metadata_and_weights_are_distinct() -> None:
    configured = {name: spec["weights"] for name, spec in ABLATION_SPECS.items()}
    config = {"ablation_weights": configured}
    specs = {name: resolved_ablation_spec(name, config) for name in ABLATION_SPECS}
    assert len({tuple(spec["weights"].values()) for spec in specs.values()}) == 4
    assert specs["conditioned_reconstruction"]["uses_downstream_kl"] is False
    assert specs["conditioned_kl"]["uses_retention_constraint"] is False
    assert specs["conditioned_kl_retention"]["uses_reconstruction"] is False
    assert all(spec["uses_structural_identity"] for spec in specs.values())


def test_hard_projection_removes_only_parallel_v3_correction() -> None:
    torch.manual_seed(2)
    denoiser = GatedConditionedDenoiser(8, 12, 4)
    with torch.no_grad():
        denoiser.output_projection.weight.normal_(0, 0.1)
        denoiser.output_projection.bias.normal_(0, 0.1)
    h = torch.randn(2, 3, 8)
    direction = F.normalize(torch.randn(8), dim=-1)
    stats = {"mean": torch.zeros(8), "std": torch.ones(8), "eps": 1e-6}
    steered = relative_raw_steering(h, direction, 0.5)
    projected = apply_steering(
        h, direction, 0.5, method="hard_projected_gated_conditioned_denoise",
        denoiser=denoiser, normalization_stats=stats,
    )
    projection = ((projected - steered) * direction).sum(-1)
    torch.testing.assert_close(projection, torch.zeros_like(projection), atol=2e-6, rtol=0)


def test_paired_bootstrap_contains_same_bounds_pareto_hv() -> None:
    rows = []
    for direction_id in (1, 2):
        for prompt_id in (0, 1, 2):
            for strength in (0.25, 0.5):
                for method in ("relative_raw", "conditioned_kl_retention"):
                    better = method == "conditioned_kl_retention"
                    rows.append({
                        "method": method, "direction_id": direction_id,
                        "prompt_id": prompt_id, "relative_strength": strength,
                        "delta_nll": strength * (1.0 if better else 2.0),
                        "kl": strength * (0.8 if better else 1.5),
                        "concept_retention": 1.1 if better else 1.0,
                        "valid_retention_ratio": True,
                    })
    result = paired_hierarchical_bootstrap(
        pd.DataFrame(rows), [("conditioned_kl_retention", "relative_raw")],
        seed=3, samples=50,
    )
    hv = result[result.metric == "pareto_hv"].iloc[0]
    assert hv.mean_recovery > 0
    assert hv.effect_definition == "paired_method_minus_reference_same_fixed_bounds"
    assert not bool(hv.hv_cross_study_comparable)


def test_causal_table_has_exact_five_identifiable_rows() -> None:
    methods = [
        "relative_raw", "sae_calibrated", "conditioned_reconstruction",
        "conditioned_kl", "conditioned_kl_retention",
    ]
    aggregate = pd.DataFrame([{
        "evaluation_split": "validation", "method": method,
        "steering_mode": "relative", "relative_strength": 0.5,
        "mean_delta_nll": float(index), "mean_kl": float(index),
        "mean_concept_retention": 1.0, "fraction_violating_80": 0.0,
    } for index, method in enumerate(methods)])
    table = causal_ablation_table(aggregate)
    assert table.method.tolist() == methods
    assert table.isolated_increment.tolist() == [
        "reference", "unconditional repair", "conditioning contribution",
        "downstream KL contribution", "retention contribution",
    ]


def test_report_is_generated_only_from_supplied_rows() -> None:
    methods = [
        "relative_raw", "literal_raw", "norm_preserving", "gaussian_denoiser",
        "sae_calibrated", "fluency_denoiser", "conditioned_reconstruction",
        "conditioned_kl", "conditioned_kl_retention", "conditioned_full",
        "hard_projected_conditioned_kl_retention",
    ]
    rows = []
    for method_index, method in enumerate(methods):
        for strength in (0.0, 0.5):
            rows.append({
                "pipeline_version": "final_v3", "evaluation_split": "validation",
                "method": method, "direction_id": 8, "prompt_id": 0,
                "steering_mode": "literal" if method == "literal_raw" else "relative",
                "relative_strength": strength, "literal_alpha": strength,
                "kl": strength * (1 + method_index / 100), "clean_nll": 2.0,
                "modified_nll": 2.0 + strength, "delta_nll": strength,
                "activation_norm_ratio": 1.0, "activation_mse": strength,
                "correction_norm": 0.0 if strength == 0 else 0.2,
                "correction_parallel_norm": 0.05, "correction_orthogonal_norm": 0.15,
                "parallel_fraction": 0.25, "correction_cosine_with_v": 0.1,
                "raw_target_sae_activation": strength,
                "denoised_target_sae_activation": strength,
                "concept_retention": float("nan") if strength == 0 else 1.0,
                "valid_retention_ratio": strength > 0,
                "violates_80_percent": None if strength == 0 else False,
                "semantic_proxy_delta_vs_clean": strength,
                "valid_semantic_retention_ratio": strength > 0,
                "semantic_proxy_retention": float("nan") if strength == 0 else 1.0,
                "token_positions": "all_tokens",
            })
    frame = pd.DataFrame(rows)
    statistics = pd.DataFrame([{"method": "conditioned_full", "reference": "relative_raw", "metric": "kl"}])
    protocol = {
        "holdout_label": "new_unseen_v3_holdout",
        "selected_method": "conditioned_kl_retention",
    }
    with tempfile.TemporaryDirectory() as directory:
        result_dir = Path(directory) / "results"
        semantic_dir = Path(directory) / "semantic"
        result_dir.mkdir(parents=True)
        semantic_dir.mkdir(parents=True)
        cross = frame.copy()
        cross["evaluation_split"] = "cross_concept_confirmation"
        cross.to_csv(result_dir / "cross_concept_confirmation.csv", index=False)
        pd.DataFrame([{
            "evaluation_split": "new_unseen_v3_holdout", "method": "relative_raw",
            "direction_id": 8, "alpha": 0.5, "clean_model_nll": 2.0,
            "concept_score": 1.0,
        }]).to_csv(result_dir / "generation_aggregate.csv", index=False)
        pd.DataFrame([{
            "method": "raw", "strength": 0.5,
            "mean_semantic_concept_score": 0.8,
        }]).to_csv(semantic_dir / "semantic_aggregate.csv", index=False)
        md_path, json_path = write_v3_report(
            directory, frame, frame.copy(), statistics, {"passed": True}, protocol
        )
        assert md_path.exists() and json_path.exists()
        assert "Hypothesis-by-hypothesis" in md_path.read_text(encoding="utf-8")
