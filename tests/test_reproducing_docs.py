"""Checks that the public reproduction contract remains discoverable and complete."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reproduction_guide_documents_canonical_full_run() -> None:
    text = (ROOT / "REPRODUCING.md").read_text(encoding="utf-8")

    required_snippets = (
        "notebooks/final_v3_all_experiments.ipynb",
        "DEBUG = False",
        "FORCE_REBUILD_ACTIVATIONS = True",
        "DEBUG_V3 = False",
        "RUN_V3_HOLDOUT = True",
        "RUN_V3_GENERATION = True",
        "RUN_V3_SEMANTIC_PROXY = True",
        "outputs/final_v3/checkpoints/conditioned_kl_retention.pt",
        "outputs/final_v3/results/causal_ablation_table.csv",
        "7.010339",
        "2.062611",
        "4.322",
        "1.401",
    )
    for snippet in required_snippets:
        assert snippet in text, f"REPRODUCING.md is missing {snippet!r}"


def test_readme_links_reproduction_guide() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[REPRODUCING.md](REPRODUCING.md)" in readme
