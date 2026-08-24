"""Integrity checks for compact result tables committed with the release."""

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "released_results"


EXPECTED_ROWS = {
    "causal_ablation_table.csv": 5,
    "validation_model_selection.csv": 4,
    "holdout_statistical_tests.csv": 24,
    "cross_concept_statistics.csv": 20,
    "semantic_aggregate.csv": 30,
    "generation_aggregate.csv": 120,
    "runtime_breakdown.csv": 9,
}


def _read_rows(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_released_tables_exist_and_have_expected_row_counts() -> None:
    for name, expected_count in EXPECTED_ROWS.items():
        rows = _read_rows(name)
        assert len(rows) == expected_count, name


def test_released_primary_numbers_and_selection() -> None:
    causal = {row["method"]: row for row in _read_rows("causal_ablation_table.csv")}
    assert set(causal) == {
        "relative_raw",
        "sae_calibrated",
        "conditioned_reconstruction",
        "conditioned_kl",
        "conditioned_kl_retention",
    }
    assert abs(float(causal["relative_raw"]["mean_delta_nll"]) - 7.01033927731216) < 1e-12
    assert abs(float(causal["conditioned_kl_retention"]["mean_delta_nll"]) - 2.0626109179109333) < 1e-12

    selected = [
        row["method"]
        for row in _read_rows("validation_model_selection.csv")
        if row["selected"] == "True"
    ]
    assert selected == ["conditioned_kl_retention"]


def test_released_holdout_and_cross_concept_scopes() -> None:
    holdout = _read_rows("holdout_statistical_tests.csv")
    assert {row["generalization_scope"] for row in holdout} == {"new_unseen_v3_holdout"}
    assert {int(row["num_directions"]) for row in holdout} == {20}

    cross_concept = _read_rows("cross_concept_statistics.csv")
    assert {row["generalization_scope"] for row in cross_concept} == {
        "cross_concept_sentiment"
    }
    assert {int(row["num_directions"]) for row in cross_concept} == {1}
