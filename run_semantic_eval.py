"""Run only the frozen, post-hoc semantic evaluation on Kaggle."""

from pathlib import Path
import gc
import json
import os

import torch

from src.directions import load_direction_split, load_sae_directions, model_sae_compatibility_report
from src.experiment import validate_generation_methods
from src.model import load_model, sanity_check_identity_intervention
from src.semantic_eval import (
    FIXED_PROMPTS, finalize_semantic_outputs, generate_semantic_cache,
    load_external_classifier, run_classifier_sanity, save_prompt_manifest,
    score_cached_generations, score_texts_nli, select_semantic_directions,
)
from src.train import load_denoiser_checkpoint
from src.train_v2 import load_conditioned_v2_checkpoint
from src.train_v3 import load_v3_checkpoint
from src.utils import load_config, seed_everything


def main() -> None:
    config = load_config("config.yaml", debug=False)
    semantic = dict(config["semantic_evaluation"])
    if os.environ.get("INCLUDE_V3", "0") == "1":
        semantic["output_dir"] = "outputs/final_v3/semantic"
        semantic["figure_dir"] = "outputs/final_v3/figures"
        # Keep the expensive secondary generation focused on the causal V3
        # comparison; projected/incremental variants remain token diagnostics.
        semantic["methods"] = [
            "raw", "sae_calibrated", "conditioned_reconstruction",
            "conditioned_kl", "conditioned_kl_retention",
        ]
        selected_v3 = os.environ.get("SELECTED_V3_METHOD")
        if selected_v3 and selected_v3 not in semantic["methods"]:
            semantic["methods"].append(selected_v3)
    # Validate before loading GPT-2, the SAE, or any checkpoints so a config
    # typo cannot waste another long Kaggle run.
    validate_generation_methods(semantic["methods"])
    output_dir = Path(semantic["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config["seed"]))
    model = load_model(
        config["model"]["name"], config["model"]["device"],
        config["model"]["dtype"],
    )
    hook_name = config["model"]["hook_name"]
    sanity_check_identity_intervention(model, hook_name=hook_name)
    sae, directions = load_sae_directions(
        model.cfg.d_model, config["sae"]["release"], config["sae"]["sae_id"],
        config["sae"]["device"], config["sae"]["dtype"],
    )
    model_sae_compatibility_report(model, sae, hook_name)
    split = load_direction_split(config["directions"]["split_path"])
    selection = select_semantic_directions(split, output_dir / "direction_selection.json")
    prompt_manifest = save_prompt_manifest(FIXED_PROMPTS, output_dir / "prompts.json")

    denoisers, normalization_stats = {}, {}
    checkpoint_modes = {
        "gaussian": "gaussian",
        "sae_calibrated": "sae_calibrated",
        "fluency": "fluency_sensitive",
    }
    for key, mode in checkpoint_modes.items():
        path = config["training"]["checkpoint_paths"][key]
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Required frozen checkpoint {path!r} is missing. This script will not retrain it."
            )
        denoiser, checkpoint = load_denoiser_checkpoint(
            path, config["model"]["device"], config["model"]["dtype"],
            expected_hook_name=hook_name,
            expected_model_name=config["model"]["name"],
            expected_corruption_mode=mode,
        )
        denoisers[key] = denoiser
        normalization_stats[key] = checkpoint["normalization"]

    if os.environ.get("INCLUDE_CONDITIONED_V2", "0") == "1":
        v2_path = config["conditioned_v2"]["best_checkpoint_path"]
        conditioned, checkpoint = load_conditioned_v2_checkpoint(
            v2_path, config["model"]["device"]
        )
        denoisers["conditioned_v2"] = conditioned
        normalization_stats["conditioned_v2"] = checkpoint["normalization"]
        if "conditioned_kl_denoiser" not in semantic["methods"]:
            semantic["methods"] = [*semantic["methods"], "conditioned_kl_denoiser"]

    if os.environ.get("INCLUDE_V3", "0") == "1":
        for method in [value for value in semantic["methods"] if value.startswith("conditioned_")]:
            v3_model, checkpoint = load_v3_checkpoint(
                config["final_v3"]["checkpoints"][method], config["model"]["device"]
            )
            denoisers[method] = v3_model
            normalization_stats[method] = checkpoint["normalization"]
            if method not in semantic["methods"]:
                semantic["methods"] = [*semantic["methods"], method]

    generation_config = {
        **semantic,
        "hook_name": hook_name,
        "projected_beta": config["evaluation"]["projected_beta"],
    }
    generations = generate_semantic_cache(
        model, sae, directions, split, selection, denoisers,
        normalization_stats, generation_config, FIXED_PROMPTS,
        output_dir / "generations.jsonl",
    )

    # Classifier is loaded only after the expensive GPT-2 texts are safely cached.
    model.to("cpu"); sae.to("cpu")
    for denoiser in denoisers.values():
        denoiser.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer, classifier, metadata = load_external_classifier(
        semantic["classifier_name"], config["model"]["device"]
    )
    metadata.update({
        "hypothesis": selection["hypothesis"],
        "control_hypothesis": selection["control_hypothesis"],
        "batch_size": int(semantic["classifier_batch_size"]),
        "prompt_hash": prompt_manifest["prompt_hash"],
        "selected_direction": int(selection["feature_id"]),
        "selected_direction_public_source": selection["source_url"],
    })
    (output_dir / "classifier_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    _, sanity_metrics = run_classifier_sanity(
        tokenizer, classifier, output_path=output_dir / "classifier_sanity.csv"
    )
    prompt_scores = score_texts_nli(
        FIXED_PROMPTS, selection["hypothesis"], tokenizer, classifier,
        int(semantic["classifier_batch_size"]), int(semantic["classifier_max_length"]),
    )
    prompt_scores.insert(0, "prompt", FIXED_PROMPTS)
    prompt_scores.to_csv(output_dir / "prompt_scores.csv", index=False)
    if float((prompt_scores.semantic_concept_score > 0.5).mean()) > 0.25:
        raise RuntimeError("More than 25% of the fixed prompts have greeting score > 0.5.")
    scores, _ = score_cached_generations(
        generations, selection, tokenizer, classifier,
        int(semantic["classifier_batch_size"]), int(semantic["classifier_max_length"]),
    )
    summary = finalize_semantic_outputs(
        scores, selection, sanity_metrics, prompt_scores,
        output_dir, semantic["figure_dir"],
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
