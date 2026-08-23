---
library_name: pytorch
license: mit
pipeline_tag: feature-extraction
tags:
  - mechanistic-interpretability
  - activation-steering
  - transformer-lens
  - sparse-autoencoder
  - gpt2
---

# GPT-2 Small intervention-aware steering denoiser

This model card describes the paper-selected `conditioned_kl_retention`
checkpoint from *Repairing Activation Steering with Downstream-Aware Denoising*.
The checkpoint is a small residual correction network for GPT-2 Small activations;
it is not a standalone language model.

## Model details

- Base language model: TransformerLens `gpt2-small`
- Intervention hook: `blocks.6.hook_resid_pre`
- Matching SAE release: `gpt2-small-res-jb`
- Matching SAE ID: `blocks.6.hook_resid_pre`
- Residual dimension: 768
- Architecture: `src.denoiser_v3.GatedConditionedDenoiser`
- Inputs: steered activation `x`, unit steering direction `v`, relative strength `s`
- Structural gate: `D(x, v, 0) = x`
- Training objective: downstream KL plus a soft target-feature retention constraint
- Checkpoint schema: V3 schema 2 with normalization statistics and split metadata

The exact architecture parameters are stored under the checkpoint's `architecture`
key. Inference must use the normalization payload saved in the same checkpoint.

## Intended use

The checkpoint is intended for research on activation steering and representation
interventions at the exact model, layer, and hook listed above. It can be used to
reproduce the paper's conditioned KL plus retention method or to study correction
geometry and target-feature retention.

It must not be treated as a general text-safety system, a universal fluency repair
model, or a denoiser for arbitrary models and layers.

## Training and evaluation boundaries

Training uses clean GPT-2 residual activations and training-only SAE decoder
directions. Validation directions are used for selection. The frozen unseen holdout
is opened only after the protocol is fixed. One independently constructed sentiment
direction is evaluated post-selection and is never used for training or selection.

Main causal-ablation means:

| Method | Delta NLL | KL | Retention |
|---|---:|---:|---:|
| Raw steering | 7.010339 | 7.047667 | 1.000000 |
| Conditioned reconstruction | 6.905518 | 6.942378 | 1.161900 |
| Conditioned KL | 2.068376 | 2.125058 | 1.010862 |
| Conditioned KL + retention | 2.062611 | 2.119669 | 1.115480 |

On the frozen 20-direction SAE holdout, the selected method recovers 4.322 mean
Delta NLL relative to raw steering (95% CI `[2.927, 5.795]`) and improves all 20
directions.

## Limitations

- Results are limited to GPT-2 Small at one residual-stream hook.
- The main holdout measures unseen directions from the same SAE family.
- Cross-concept evidence contains one sentiment direction.
- Clean-model NLL is a language-model degradation proxy, not human fluency.
- SAE activation is an intervention-signal proxy, not semantic ground truth.
- The external semantic classifier is also a proxy rather than human evaluation.

## Files to publish in the Hugging Face model repository

```text
conditioned_kl_retention.pt
README.md                      # copy this model card as README.md
config.yaml
src/denoiser_v2.py
src/denoiser_v3.py
src/train_v3.py
src/utils.py
```

The source files are included so the custom PyTorch module and exact normalization
path are auditable. The full experiment repository remains the authoritative source.

## Loading

From a checkout of the experiment repository:

```python
from huggingface_hub import hf_hub_download
from src.train_v3 import load_v3_checkpoint

checkpoint_path = hf_hub_download(
    repo_id="leolazz/steering-denoiser-gpt2-small",
    filename="conditioned_kl_retention.pt",
)
denoiser, checkpoint = load_v3_checkpoint(checkpoint_path, device="cuda")
normalization = checkpoint["normalization"]
```

Use `src.denoiser_v3.apply_gated_conditioned_denoiser` for inference so the
training-time normalization and direction standardization are applied exactly.

## Citation

> leolazz. *Repairing Activation Steering with Downstream-Aware Denoising*.
> T-Lab Mechanistic Interpretability Research Project, 2026.

## License

The released code and checkpoint are licensed under the MIT License.
