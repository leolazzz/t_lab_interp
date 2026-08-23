import pandas as pd

from src.experiment import positive_strength_concept_retention


def test_positive_strength_concept_retention_excludes_zero() -> None:
    frame = pd.DataFrame(
        [
            {"method": "raw", "prompt_id": 0, "direction_id": 1, "strength": 0.0, "target_sae_activation": 0.0},
            {"method": "raw", "prompt_id": 0, "direction_id": 1, "strength": 0.5, "target_sae_activation": 2.0},
            {"method": "sae", "prompt_id": 0, "direction_id": 1, "strength": 0.0, "target_sae_activation": 0.0},
            {"method": "sae", "prompt_id": 0, "direction_id": 1, "strength": 0.5, "target_sae_activation": 1.0},
        ]
    )
    result = positive_strength_concept_retention(frame)
    assert set(result.strength) == {0.5}
    assert abs(result[result.method == "raw"].concept_retention.iloc[0] - 1.0) < 1e-6


def test_concept_retention_rejects_duplicate_merge_keys() -> None:
    duplicated = pd.DataFrame([
        {"method": "raw", "prompt_id": 0, "direction_id": 1, "strength": 0.5, "target_sae_activation": 2.0},
        {"method": "raw", "prompt_id": 0, "direction_id": 1, "strength": 0.5, "target_sae_activation": 2.0},
    ])
    try:
        positive_strength_concept_retention(duplicated)
    except AssertionError:
        return
    raise AssertionError("Duplicate keys should have been rejected.")
