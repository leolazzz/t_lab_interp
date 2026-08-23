"""Independent text-only semantic evaluation for SAE activation steering.

The external evaluator in this module receives generated strings and an NLI
hypothesis only.  It never receives GPT-2 activations, SAE tensors, steering
directions, or denoiser states.  Generation and classifier scoring are two
separate cached stages so classifier changes never require text regeneration.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from src.directions import get_sae_hook_name, validate_direction_split
from src.experiment import (
    _generate_fixed_seed, _generation_intervention, validate_generation_methods,
)
from src.metrics import (
    distinct_n,
    external_clean_lm_nll,
    repetition_rate,
    sae_feature_activation_metrics,
)
from src.utils import PIPELINE_VERSION


CLASSIFIER_NAME = "typeform/distilbert-base-uncased-mnli"
PRIMARY_FEATURE_ID = 6516
POSITIVE_HYPOTHESIS = "This text is a greeting or welcoming interaction."
CONTROL_HYPOTHESIS = "This text is unrelated to greetings or welcoming interactions."
SEMANTIC_PIPELINE_VERSION = "semantic_eval_v1"
SEMANTIC_RETENTION_THRESHOLD = 0.01

PUBLIC_FEATURES: dict[int, dict[str, str]] = {
    6516: {
        "interpretation": "greeting-related words and phrases; greetings and welcoming interactions",
        "hypothesis": POSITIVE_HYPOTHESIS,
        "control_hypothesis": CONTROL_HYPOTHESIS,
        "source_url": "https://www.neuronpedia.org/gpt2-small/6-res-jb/6516",
    },
    9696: {
        "interpretation": "references to vacuums and vacuum-related language",
        "hypothesis": "This text is about vacuums or vacuum-related concepts.",
        "control_hypothesis": "This text is unrelated to vacuums or vacuum-related concepts.",
        "source_url": "https://www.neuronpedia.org/gpt2-small/6-res-jb/9696",
    },
    7672: {
        "interpretation": "CSS and HTML code snippets",
        "hypothesis": "This text is about HTML, CSS, or web page code.",
        "control_hypothesis": "This text is unrelated to HTML, CSS, or web page code.",
        "source_url": "https://www.neuronpedia.org/gpt2-small/6-res-jb/7672",
    },
}

FIXED_PROMPTS = [
    "The following passage begins:",
    "He looked around the room and said",
    "The next part of the story was",
    "She opened the door and",
    "The conversation continued:",
    "The speaker approached the group and",
    "At the beginning of the meeting,",
    "After a few moments,",
    "The scene changed when",
    "They paused before continuing:",
    "The written account states:",
    "As the discussion began,",
    "A person entered the building and",
    "The narrator then explained",
    "In the next paragraph,",
    "The group gathered near the entrance and",
]

GREETING_POSITIVE_SANITY = [
    "Hello! It is wonderful to meet you.",
    "Good morning, welcome to our event.",
    "Hi there! Welcome, and thanks for joining us.",
    "Greetings everyone, I hope you are doing well.",
    "It was a pleasure meeting you. Goodbye!",
]
GREETING_NEGATIVE_SANITY = [
    "The engine uses four cylinders.",
    "Water freezes at zero degrees Celsius.",
    "The report contains quarterly financial results.",
    "A tree was standing beside the road.",
    "The experiment measured the temperature every hour.",
]

_MASK_PATTERN = re.compile(
    r"\b(?:hello|hi|greetings?|greet(?:ed|ing|s)?|welcome(?:d|s|ing)?|goodbye)\b",
    flags=re.IGNORECASE,
)


def prompt_set_hash(prompts: Sequence[str]) -> str:
    """Return a deterministic SHA-256 hash for the ordered prompt list."""
    assert prompts and all(isinstance(prompt, str) for prompt in prompts)
    payload = json.dumps(list(prompts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def direction_split_membership(feature_id: int, split: Mapping[str, Any]) -> str:
    """Return train, validation, test, or outside for one SAE feature."""
    validate_direction_split(
        list(split["train"]), list(split["val"]), list(split["test"]),
        int(split["num_features"]),
    )
    assert 0 <= feature_id < int(split["num_features"])
    if feature_id in set(split["train"]):
        return "train"
    if feature_id in set(split["val"]):
        return "validation"
    if feature_id in set(split["test"]):
        return "test"
    return "outside"


def select_semantic_directions(
    split: Mapping[str, Any],
    output_path: str | Path | None = "outputs/semantic_eval/direction_selection.json",
) -> dict[str, Any]:
    """Apply the pre-registered split gate without inspecting classifier scores."""
    memberships = {
        feature_id: direction_split_membership(feature_id, split)
        for feature_id in PUBLIC_FEATURES
    }
    primary_membership = memberships[PRIMARY_FEATURE_ID]
    if primary_membership in {"outside", "validation"}:
        selected_id = PRIMARY_FEATURE_ID
        reason = (
            "preselected public feature outside the canonical sampled split"
            if primary_membership == "outside"
            else "preselected public feature used for validation-side semantic evaluation"
        )
    elif primary_membership == "test":
        selected_id = _select_fallback(memberships)
        reason = "feature 6516 is TEST and was rejected; used first legal predetermined fallback"
    else:
        selected_id = _select_fallback(memberships)
        reason = "feature 6516 is TRAIN and is not held-out evidence; used first legal predetermined fallback"

    control_candidates = [9696, 7672]
    control_id = next(
        (feature_id for feature_id in control_candidates
         if feature_id != selected_id and memberships[feature_id] != "test"),
        None,
    )
    if control_id is None:
        raise RuntimeError("No predetermined negative-control direction is legally available.")

    selected = PUBLIC_FEATURES[selected_id]
    payload = {
        "feature_id": selected_id,
        "split_membership": memberships[selected_id],
        "feature_6516_split_membership": primary_membership,
        "feature_6516_legally_usable": primary_membership in {"outside", "validation"},
        "selection_reason": reason,
        "public_interpretation": selected["interpretation"],
        "source_url": selected["source_url"],
        "hypothesis": selected["hypothesis"],
        "control_hypothesis": selected["control_hypothesis"],
        "selected_before_classifier_evaluation": True,
        "negative_control_feature_id": control_id,
        "negative_control_split_membership": memberships[control_id],
        "negative_control_public_interpretation": PUBLIC_FEATURES[control_id]["interpretation"],
        "candidate_memberships": {str(key): value for key, value in memberships.items()},
        "sae_release": "gpt2-small-res-jb",
        "hook_name": "blocks.6.hook_resid_pre",
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _select_fallback(memberships: Mapping[int, str]) -> int:
    preferred = [feature_id for feature_id in (9696, 7672)
                 if memberships[feature_id] in {"outside", "validation"}]
    if preferred:
        return preferred[0]
    diagnostic = [feature_id for feature_id in (9696, 7672)
                  if memberships[feature_id] == "train"]
    if diagnostic:
        return diagnostic[0]
    raise RuntimeError(
        "All predetermined semantic candidates are TEST-contaminated; stopping benchmark."
    )


def classifier_label_ids(classifier: nn.Module) -> dict[str, int]:
    """Resolve MNLI label IDs from config rather than assuming their order."""
    id2label = {
        int(key): str(value).upper()
        for key, value in dict(classifier.config.id2label).items()
    }
    label2id = {
        str(key).upper(): int(value)
        for key, value in dict(getattr(classifier.config, "label2id", {})).items()
    }
    resolved: dict[str, int] = {}
    for label in ("ENTAILMENT", "NEUTRAL", "CONTRADICTION"):
        matches = [index for index, value in id2label.items() if value == label]
        if label in label2id:
            matches.append(label2id[label])
        matches = sorted(set(matches))
        if len(matches) != 1:
            raise ValueError(
                f"Could not uniquely resolve {label!r} from id2label={id2label} "
                f"and label2id={label2id}."
            )
        resolved[label.lower()] = matches[0]
    assert len(set(resolved.values())) == 3
    return resolved


@torch.inference_mode()
def score_texts_nli(
    texts: Sequence[str],
    hypothesis: str,
    tokenizer: Any,
    classifier: nn.Module,
    batch_size: int = 32,
    max_length: int = 512,
) -> pd.DataFrame:
    """Batch text-only NLI scoring with automatic CUDA OOM batch fallback."""
    assert texts and all(isinstance(text, str) for text in texts)
    assert isinstance(hypothesis, str) and hypothesis
    assert batch_size > 0 and max_length > 0
    label_ids = classifier_label_ids(classifier)
    device = next(classifier.parameters()).device
    rows: list[dict[str, float]] = []
    start = 0
    current_batch_size = min(batch_size, len(texts))
    while start < len(texts):
        stop = min(start + current_batch_size, len(texts))
        batch_texts = list(texts[start:stop])
        try:
            encoded = tokenizer(
                batch_texts,
                [hypothesis] * len(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = classifier(**encoded).logits.float()
            probabilities = torch.softmax(logits, dim=-1).cpu()
        except torch.cuda.OutOfMemoryError:
            if current_batch_size == 1:
                raise
            torch.cuda.empty_cache()
            current_batch_size = max(1, current_batch_size // 2)
            continue
        for probability in probabilities:
            entail = float(probability[label_ids["entailment"]])
            neutral = float(probability[label_ids["neutral"]])
            contradiction = float(probability[label_ids["contradiction"]])
            binary = entail / (entail + contradiction + 1e-12)
            rows.append({
                "semantic_concept_score": entail,
                "p_entail": entail,
                "p_neutral": neutral,
                "p_contradiction": contradiction,
                "semantic_concept_score_binary": binary,
            })
        start = stop
    frame = pd.DataFrame(rows)
    assert len(frame) == len(texts)
    values = frame.to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert ((values >= 0.0) & (values <= 1.0 + 1e-6)).all()
    return frame


def load_external_classifier(
    classifier_name: str = CLASSIFIER_NAME,
    device: str | torch.device | None = None,
) -> tuple[Any, nn.Module, dict[str, Any]]:
    """Load the frozen MNLI evaluator and return reproducibility metadata."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import transformers

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(classifier_name)
    classifier = AutoModelForSequenceClassification.from_pretrained(classifier_name)
    classifier.to(resolved_device).eval().requires_grad_(False)
    labels = classifier_label_ids(classifier)
    commit_hash = getattr(classifier.config, "_commit_hash", None)
    metadata = {
        "classifier_name": classifier_name,
        "classifier_revision_or_commit_hash": commit_hash,
        "tokenizer_name": getattr(tokenizer, "name_or_path", classifier_name),
        "id2label": {str(key): value for key, value in classifier.config.id2label.items()},
        "label2id": dict(classifier.config.label2id),
        "resolved_label_ids": labels,
        "scoring_formula": "softmax(NLI_logits)[ENTAILMENT]",
        "transformers_version": transformers.__version__,
        "pipeline_version": SEMANTIC_PIPELINE_VERSION,
    }
    return tokenizer, classifier, metadata


def run_classifier_sanity(
    tokenizer: Any,
    classifier: nn.Module,
    hypothesis: str = POSITIVE_HYPOTHESIS,
    output_path: str | Path = "outputs/semantic_eval/classifier_sanity.csv",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run the fixed greeting sanity gate before steering evaluation."""
    texts = GREETING_POSITIVE_SANITY + GREETING_NEGATIVE_SANITY
    labels = np.array([1] * 5 + [0] * 5)
    scores = score_texts_nli(texts, hypothesis, tokenizer, classifier)
    frame = pd.concat([
        pd.DataFrame({"text": texts, "expected_positive": labels}), scores
    ], axis=1)
    positive_mean = float(frame.loc[frame.expected_positive == 1, "semantic_concept_score"].mean())
    negative_mean = float(frame.loc[frame.expected_positive == 0, "semantic_concept_score"].mean())
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(labels, frame["semantic_concept_score"]))
    metrics = {
        "mean_positive_score": positive_mean,
        "mean_negative_score": negative_mean,
        "roc_auc": auc,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if not positive_mean > negative_mean:
        raise RuntimeError(f"External classifier failed the fixed sanity gate: {metrics}")
    return frame, metrics


def mask_greeting_keywords(text: str) -> str:
    """Mask a fixed list of obvious greeting words for a lexical diagnostic."""
    return _MASK_PATTERN.sub("[MASKED]", text)


def _generation_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record["experiment_role"], record["method"], int(record["direction_id"]),
        int(record["prompt_id"]), float(record["strength"]), int(record["seed"]),
        record["generation_signature"],
    )


@torch.inference_mode()
def generate_semantic_cache(
    model: nn.Module,
    sae: nn.Module,
    all_directions: Tensor,
    split: Mapping[str, Any],
    selection: Mapping[str, Any],
    denoisers: Mapping[str, nn.Module],
    normalization_stats: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    prompts: Sequence[str] = FIXED_PROMPTS,
    output_path: str | Path = "outputs/semantic_eval/generations.jsonl",
) -> pd.DataFrame:
    """Generate each fixed semantic-evaluation text exactly once and cache it."""
    primary_id = int(selection["feature_id"])
    control_id = int(selection["negative_control_feature_id"])
    for feature_id in (primary_id, control_id):
        membership = direction_split_membership(feature_id, split)
        if membership == "test":
            raise AssertionError(f"Refusing to evaluate TEST direction {feature_id}.")
    hook_name = str(config.get("hook_name", "blocks.6.hook_resid_pre"))
    assert get_sae_hook_name(sae) == hook_name
    assert all_directions.shape == (int(split["num_features"]), int(model.cfg.d_model))
    assert all_directions.device == next(model.parameters()).device
    strengths = [float(value) for value in config["strengths"]]
    methods = list(config["methods"])
    validate_generation_methods(methods)
    seed = int(config.get("seed", 0))
    max_new_tokens = int(config.get("max_new_tokens", 40))
    generation_cfg = {
        "incremental_steps": config.get("incremental_steps", 4),
        "projected_beta": config.get("projected_beta", 1.0),
    }
    signature_payload = {
        "pipeline": SEMANTIC_PIPELINE_VERSION,
        "prompt_hash": prompt_set_hash(prompts),
        "primary_id": primary_id,
        "control_id": control_id,
        "strengths": strengths,
        "methods": methods,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "temperature": float(config.get("temperature", 1.0)),
        "top_k": config.get("top_k", 50),
        "do_sample": bool(config.get("do_sample", True)),
        "hook_name": hook_name,
        "steering_scale": "relative_activation_norm",
        "generation_positions": "final_token_only",
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    completed: set[tuple[Any, ...]] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("generation_signature") == signature:
                records.append(record)
                completed.add(_generation_key(record))

    jobs = [("primary", primary_id, method) for method in methods]
    jobs.append(("negative_control", control_id, "raw"))
    total = len(jobs) * len(prompts) * len(strengths)
    progress = tqdm(total=total, initial=len(completed), desc="Semantic generations")
    with path.open("a", encoding="utf-8") as handle:
        for role, direction_id, method in jobs:
            direction = all_directions[direction_id]
            for prompt_id, prompt in enumerate(prompts):
                prompt_tokens = model.to_tokens(prompt, prepend_bos=True)
                prompt_length = int(prompt_tokens.shape[1])
                assert prompt_length + max_new_tokens <= int(model.cfg.n_ctx)
                for strength in strengths:
                    base = {
                        "experiment_role": role,
                        "method": method,
                        "direction_id": direction_id,
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "strength": strength,
                        "seed": seed,
                        "generation_signature": signature,
                    }
                    if _generation_key(base) in completed:
                        continue
                    intervention = _generation_intervention(
                        method, direction, strength, denoisers,
                        normalization_stats, generation_cfg,
                        token_positions="final_token_only",
                    )
                    full_tokens = _generate_fixed_seed(
                        model, prompt_tokens, hook_name, intervention, seed,
                        max_new_tokens, float(config.get("temperature", 1.0)),
                        config.get("top_k", 50), bool(config.get("do_sample", True)),
                        getattr(model.tokenizer, "eos_token_id", None),
                    )
                    continuation = full_tokens[:, prompt_length:]
                    generated_ids = continuation[0].tolist()
                    generated_text = model.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    _, cache = model.run_with_cache(
                        full_tokens, names_filter=hook_name, return_type=None
                    )
                    clean_h = cache[hook_name]
                    token_mask = torch.zeros_like(full_tokens, dtype=torch.bool)
                    token_mask[:, prompt_length:] = True
                    feature_metrics = sae_feature_activation_metrics(
                        sae, clean_h, feature_id=direction_id,
                        token_mask=token_mask, threshold=0.0,
                    )
                    nll = external_clean_lm_nll(
                        model, full_tokens, prefix_lengths=prompt_length
                    )
                    record = {
                        **base,
                        "generated_text": generated_text,
                        "generated_token_ids": generated_ids,
                        "num_generated_tokens": len(generated_ids),
                        "target_sae_activation": float(feature_metrics["concept_score"]),
                        "max_feature_activation": float(feature_metrics["max_feature_activation"]),
                        "feature_active_fraction": float(feature_metrics["feature_active_fraction"]),
                        "clean_model_continuation_nll": float(nll),
                        "dist_1": distinct_n(generated_ids, 1),
                        "dist_2": distinct_n(generated_ids, 2),
                        "dist_3": distinct_n(generated_ids, 3),
                        "repetition_rate": repetition_rate(generated_ids, n=3),
                        "pipeline_version": SEMANTIC_PIPELINE_VERSION,
                    }
                    numeric = [value for value in record.values()
                               if isinstance(value, float)]
                    assert all(math.isfinite(value) for value in numeric)
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    records.append(record)
                    completed.add(_generation_key(record))
                    progress.update(1)
                    del cache, clean_h, full_tokens
    progress.close()
    frame = pd.DataFrame(records)
    expected = total
    assert len(frame) == expected, f"Expected {expected} cached generations, got {len(frame)}."
    assert not frame.duplicated([
        "experiment_role", "method", "direction_id", "prompt_id", "strength", "seed"
    ]).any()
    return frame


def score_cached_generations(
    generation_frame: pd.DataFrame,
    selection: Mapping[str, Any],
    tokenizer: Any,
    classifier: nn.Module,
    batch_size: int = 32,
    max_length: int = 512,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach primary and keyword-masked text-only scores to cached generations."""
    required = {"generated_text", "experiment_role", "method", "strength"}
    assert required.issubset(generation_frame.columns)
    hypothesis = str(selection["hypothesis"])
    texts = generation_frame["generated_text"].astype(str).tolist()
    scores = score_texts_nli(
        texts, hypothesis, tokenizer, classifier, batch_size, max_length
    )
    masked_texts = [mask_greeting_keywords(text) for text in texts]
    masked_scores = score_texts_nli(
        masked_texts, hypothesis, tokenizer, classifier, batch_size, max_length
    )
    result = pd.concat([generation_frame.reset_index(drop=True), scores], axis=1)
    result["semantic_concept_score_masked"] = masked_scores["semantic_concept_score"]
    masked = result[[
        "experiment_role", "method", "direction_id", "prompt_id", "strength",
        "seed", "generated_text", "semantic_concept_score",
        "semantic_concept_score_masked",
    ]].copy()
    masked["masked_text"] = masked_texts
    numeric = result.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(numeric).all()
    return result, masked


def aggregate_semantic_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the fixed grid and add matched-strength comparison to raw."""
    primary = scores[scores["experiment_role"] == "primary"].copy()
    expected_rows = (
        primary["method"].nunique() * primary["strength"].nunique()
        * primary["prompt_id"].nunique() * primary["seed"].nunique()
    )
    assert len(primary) == expected_rows
    aggregate = primary.groupby(["method", "strength"], as_index=False).agg(
        mean_semantic_concept_score=("semantic_concept_score", "mean"),
        median_semantic_concept_score=("semantic_concept_score", "median"),
        std_semantic_concept_score=("semantic_concept_score", "std"),
        mean_target_sae_activation=("target_sae_activation", "mean"),
        mean_clean_model_continuation_nll=("clean_model_continuation_nll", "mean"),
        mean_dist_1=("dist_1", "mean"),
        mean_dist_2=("dist_2", "mean"),
        mean_dist_3=("dist_3", "mean"),
        mean_repetition_rate=("repetition_rate", "mean"),
        num_samples=("semantic_concept_score", "size"),
    )
    raw = aggregate[aggregate.method == "raw"][[
        "strength", "mean_semantic_concept_score"
    ]].rename(columns={"mean_semantic_concept_score": "raw_semantic_concept_score"})
    aggregate = aggregate.merge(raw, on="strength", how="left", validate="many_to_one")
    assert aggregate["raw_semantic_concept_score"].notna().all()
    aggregate["semantic_concept_delta_vs_raw"] = (
        aggregate["mean_semantic_concept_score"] - aggregate["raw_semantic_concept_score"]
    )
    aggregate["valid_semantic_retention_ratio"] = (
        aggregate["raw_semantic_concept_score"].abs() >= SEMANTIC_RETENTION_THRESHOLD
    )
    aggregate["semantic_retention_ratio"] = np.where(
        aggregate["valid_semantic_retention_ratio"],
        aggregate["mean_semantic_concept_score"] / aggregate["raw_semantic_concept_score"],
        np.nan,
    )
    required_finite = aggregate.drop(columns=["semantic_retention_ratio"]).select_dtypes(include=[np.number])
    assert np.isfinite(required_finite.to_numpy()).all()
    return aggregate


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman correlation via average ranks, avoiding a SciPy dependency."""
    x_rank = pd.Series(x, dtype=float).rank(method="average").to_numpy()
    y_rank = pd.Series(y, dtype=float).rank(method="average").to_numpy()
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def internal_external_correlations(scores: pd.DataFrame) -> dict[str, float]:
    """Correlate the mechanistic proxy with external semantics without causality claims."""
    primary = scores[scores.experiment_role == "primary"]
    raw = primary[primary.method == "raw"]
    return {
        "raw_strength_vs_semantic_spearman": _spearman(
            raw["strength"], raw["semantic_concept_score"]
        ),
        "raw_sae_activation_vs_semantic_spearman": _spearman(
            raw["target_sae_activation"], raw["semantic_concept_score"]
        ),
        "all_methods_sae_activation_vs_semantic_spearman": _spearman(
            primary["target_sae_activation"], primary["semantic_concept_score"]
        ),
    }


def save_semantic_plots(
    scores: pd.DataFrame,
    aggregate: pd.DataFrame,
    figure_dir: str | Path = "outputs/final_figures",
) -> None:
    """Save the four pre-registered matplotlib semantic diagnostics."""
    import matplotlib.pyplot as plt

    root = Path(figure_dir)
    root.mkdir(parents=True, exist_ok=True)
    primary = scores[scores.experiment_role == "primary"]
    raw = primary[primary.method == "raw"]

    fig, ax = plt.subplots(figsize=(8, 5))
    raw.groupby("strength")["semantic_concept_score"].mean().plot(marker="o", ax=ax)
    ax.set(xlabel="Relative steering strength", ylabel="Mean NLI entailment probability",
           title="Independent semantic score vs raw steering strength")
    fig.tight_layout(); fig.savefig(root / "semantic_score_vs_strength.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for method, group in aggregate.groupby("method"):
        group = group.sort_values("strength")
        x = -group["mean_clean_model_continuation_nll"]
        y = group["mean_semantic_concept_score"]
        ax.plot(x, y, marker="o", label=method)
        for x_value, y_value, strength in zip(x, y, group["strength"]):
            ax.annotate(f"{strength:g}", (x_value, y_value), fontsize=7)
    ax.set(xlabel="Negative clean-model continuation NLL (higher is better)",
           ylabel="Mean NLI entailment probability (higher is better)",
           title="External semantic Pareto comparison")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(root / "semantic_pareto_classifier.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(primary["target_sae_activation"], primary["semantic_concept_score"], alpha=0.6)
    ax.set(xlabel="Target SAE feature activation", ylabel="NLI entailment probability",
           title="Internal mechanistic proxy vs external semantic proxy")
    fig.tight_layout(); fig.savefig(root / "sae_vs_external_semantic_score.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    control = scores[scores.experiment_role == "negative_control"]
    for label, group in (("selected semantic direction", raw), ("negative control", control)):
        grouped = group.groupby("strength")["semantic_concept_score"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", label=label)
    ax.set(xlabel="Relative steering strength", ylabel="Mean NLI entailment probability",
           title="Selected semantic direction vs unrelated control")
    ax.legend(); fig.tight_layout()
    fig.savefig(root / "greeting_vs_control_direction.png", dpi=160); plt.close(fig)


def save_prompt_manifest(
    prompts: Sequence[str] = FIXED_PROMPTS,
    path: str | Path = "outputs/semantic_eval/prompts.json",
) -> dict[str, Any]:
    forbidden = re.compile(r"\b(?:hello|hi|greeting|welcome|goodbye)\b", re.IGNORECASE)
    assert len(prompts) == 16 and not any(forbidden.search(prompt) for prompt in prompts)
    payload = {"prompts": list(prompts), "prompt_hash": prompt_set_hash(prompts)}
    resolved = Path(path); resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def finalize_semantic_outputs(
    scores: pd.DataFrame,
    selection: Mapping[str, Any],
    sanity_metrics: Mapping[str, float],
    prompt_scores: pd.DataFrame,
    output_dir: str | Path = "outputs/semantic_eval",
    figure_dir: str | Path = "outputs/final_figures",
) -> dict[str, Any]:
    """Write tables, correlations, plots, and a factual machine-readable summary."""
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_semantic_scores(scores)
    correlations = internal_external_correlations(scores)
    scores.to_csv(root / "semantic_scores.csv", index=False)
    aggregate.to_csv(root / "semantic_aggregate.csv", index=False)
    scores[(scores.experiment_role == "primary") & (scores.method == "raw")].to_csv(
        root / "raw_direction_validation.csv", index=False
    )
    scores[scores.experiment_role == "negative_control"].to_csv(
        root / "negative_control.csv", index=False
    )
    masking_frame = scores[[
        "experiment_role", "method", "direction_id", "prompt_id", "strength",
        "seed", "generated_text", "semantic_concept_score",
        "semantic_concept_score_masked",
    ]].copy()
    masking_frame["masked_text"] = masking_frame["generated_text"].map(
        mask_greeting_keywords
    )
    masking_frame.to_csv(root / "masked_keyword_diagnostic.csv", index=False)
    (root / "internal_external_correlation.json").write_text(
        json.dumps(correlations, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    raw_by_strength = (
        aggregate[aggregate.method == "raw"]
        .set_index("strength")["mean_semantic_concept_score"].to_dict()
    )
    comparison_columns = [
        "method", "strength", "mean_semantic_concept_score",
        "mean_clean_model_continuation_nll", "semantic_concept_delta_vs_raw",
        "semantic_retention_ratio",
    ]
    requested_comparisons = aggregate[
        aggregate.method.isin(["raw", "sae_calibrated", "fluency_denoiser"])
    ][comparison_columns].to_dict(orient="records")
    raw_strength_spearman = correlations["raw_strength_vs_semantic_spearman"]
    raw_effect_statement = (
        "The publicly interpreted feature showed a positive rank association "
        "between steering strength and the external semantic proxy."
        if math.isfinite(raw_strength_spearman) and raw_strength_spearman > 0
        else "The publicly interpreted feature did not yield a reliable external "
             "semantic steering effect under this setup."
    )
    summary = {
        "pipeline_version": SEMANTIC_PIPELINE_VERSION,
        "selected_direction": dict(selection),
        "classifier_sanity": dict(sanity_metrics),
        "prompt_mean_semantic_score": float(prompt_scores.semantic_concept_score.mean()),
        "raw_semantic_score_by_strength": {str(key): value for key, value in raw_by_strength.items()},
        "correlations": correlations,
        "raw_direction_validation_statement": raw_effect_statement,
        "raw_sae_calibrated_fluency_comparison": requested_comparisons,
        "negative_control_mean_by_strength": {
            str(key): float(value) for key, value in
            scores[scores.experiment_role == "negative_control"]
            .groupby("strength")["semantic_concept_score"].mean().items()
        },
        "keyword_masking_mean_delta": float(
            (scores["semantic_concept_score_masked"]
             - scores["semantic_concept_score"]).mean()
        ),
        "interpretation_note": (
            "The classifier score is an independent external semantic proxy, not a perfect human evaluator."
        ),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    save_semantic_plots(scores, aggregate, figure_dir)
    return summary
