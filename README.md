# Repairing Activation Steering with Downstream-Aware Denoising

Research code for the T-Lab Mechanistic Interpretability Research Project.

This repository studies a narrow question: can a denoiser repair the language-model
degradation caused by activation steering without erasing the intended intervention?
The central comparison is unconditional activation reconstruction, `D(x)`, versus an
intervention-aware denoiser, `D(x, v, s)`, optimized through the frozen downstream
transformer.

The complete paper is available at [`paper/paper.pdf`](paper/paper.pdf).
Exact from-scratch Kaggle instructions, expected artifacts, and numerical
verification checks are in [REPRODUCING.md](REPRODUCING.md).

## Main result

Experiments use GPT-2 Small and an SAE direction at
`blocks.6.hook_resid_pre`. The causal ablation separates conditioning,
reconstruction, downstream KL, and target-feature retention.

| Method | Mean Delta NLL | Mean KL | Retention |
|---|---:|---:|---:|
| Raw steering | 7.010339 | 7.047667 | 1.000000 |
| SAE reconstruction | 6.201735 | 6.245004 | 1.241633 |
| Conditioned reconstruction | 6.905518 | 6.942378 | 1.161900 |
| Conditioned KL | 2.068376 | 2.125058 | 1.010862 |
| Conditioned KL + retention | **2.062611** | **2.119669** | 1.115480 |

Conditioning alone is insufficient. The dominant improvement comes from optimizing
downstream KL. The retention constraint materially improves preservation of the
target SAE feature while providing essentially no additional downstream NLL benefit.

On the frozen 20-direction SAE holdout, the selected method recovers 4.322 mean
Delta NLL relative to raw steering (95% CI `[2.927, 5.795]`) and improves all 20
directions. A separate post-selection sentiment direction confirms the central
KL-over-reconstruction result, but it is only one cross-concept direction and does
not establish universal transfer.

## Scientific scope

- Clean-model continuation NLL is an LM degradation/fluency proxy, not human fluency.
- SAE feature activation is a mechanistic intervention-signal proxy, not semantic
  ground truth.
- The 20-direction result establishes within-SAE-family generalization only.
- Cross-concept evidence contains one independent sentiment direction.
- Low kNN distance describes proximity to cached clean activations; it does not prove
  semantic correctness.
- Pareto hypervolume is compared only through paired differences under identical
  bounds within one experiment.

## Exact setup

- Base model: TransformerLens `gpt2-small`
- Intervention hook: `blocks.6.hook_resid_pre`
- Hook meaning: residual stream entering transformer block 6, after block 5
- Residual width: 768
- SAE release: `gpt2-small-res-jb`
- SAE ID: `blocks.6.hook_resid_pre`
- Direction `i`: unit-normalized row `sae.W_dec[i]`
- Primary relative intervention:

  ```text
  u = v / ||v||
  h_modified = h + strength * ||h|| * u
  ```

At generation time, steering is applied only to the current final token position.
Earlier prefix positions are not repeatedly steered.

## Repository layout

```text
.
├── config.yaml                         # all experiment and debug parameters
├── requirements.txt                    # runtime dependencies
├── requirements-dev.txt                # local test dependency
├── run_v3.py                           # final end-to-end V3 runner
├── run_semantic_eval.py                # independent post-hoc semantic evaluation
├── notebooks/
│   └── final_v3_all_experiments.ipynb  # canonical one-notebook Kaggle run
├── src/
│   ├── model.py                        # GPT-2 loading and TransformerLens hooks
│   ├── directions.py                   # SAE loading and leakage-safe splits
│   ├── steering.py                     # raw, projected, and denoised steering
│   ├── denoiser.py                     # unconditional residual MLP
│   ├── denoiser_v3.py                  # intervention-aware denoiser
│   ├── train.py / train_v3.py          # corruption and optimization
│   ├── metrics.py                      # downstream, generation, and geometry metrics
│   ├── experiment.py                   # shared cache/evaluation utilities
│   ├── experiment_v3.py                # final protocol and statistical analysis
│   ├── semantic_eval.py                # independent output-level semantic proxy
│   └── utils.py                        # config, seeds, normalization, and audit helpers
├── tests/                              # CPU-friendly unit and orchestration tests
├── released_results/                   # compact frozen tables from the paper run
├── paper/
│   ├── main.tex
│   ├── references.bib
│   ├── paper.pdf
│   ├── make_publication_figures.py
│   └── figures/
└── MODEL_CARD.md                       # checkpoint contract and limitations
```

`outputs/`, temporary render files, local logs, and the executed local notebook are
excluded from Git. A fresh run creates outputs automatically. The checked-in paper
and publication figures are lightweight frozen summaries of the completed run.

## Run the complete experiment on Kaggle

The canonical entry point is
[`notebooks/final_v3_all_experiments.ipynb`](notebooks/final_v3_all_experiments.ipynb).
It runs the full pipeline from clean activation caching through training, frozen
holdout evaluation, semantic confirmation, figures, and final reports.

For the exact paper-scale flags, expected runtime, stage-by-stage gates, and a
post-run verification cell, follow [REPRODUCING.md](REPRODUCING.md).

### Option A: import from GitHub

1. Open the public GitHub repository `leolazzz/t_lab_interp`.
2. In Kaggle, create a notebook with one T4 GPU and Internet enabled.
3. Import `notebooks/final_v3_all_experiments.ipynb` from GitHub.
4. Run all cells. The bootstrap uses the current checkout when present and otherwise
   clones `https://github.com/leolazzz/t_lab_interp.git`.

### Option B: attach as a Kaggle Dataset

1. Upload the unpacked repository as a Kaggle Dataset.
2. Open the canonical notebook and attach that Dataset with **Add Input**.
3. Enable one GPU and Internet, then select **Run All**.

Internet is needed on the first run to download GPT-2, the SAE, datasets, and the
external classifier. A prior output Dataset is not required. The full profile builds
200,000 clean token activations and trains every causal ablation from scratch.

The notebook uses `DEBUG=False` and `DEBUG_V3=False` for the production run. For a
short path check only, set both flags to `True`; debug results are not paper results.

## Pipeline stages

1. Seed Python, NumPy, PyTorch, and CUDA.
2. Load GPT-2 Small and verify an identity forward hook.
3. Load the matching SAE and validate hook and dimensional compatibility.
4. Build or load the immutable train/validation/test direction split.
5. Cache clean residual activations and streaming normalization statistics.
6. Validate standardized corruption geometry on analytical and real batches.
7. Train unconditional and intervention-aware denoisers.
8. Score harmfulness using training directions only.
9. Evaluate on validation directions and freeze the selected protocol.
10. Open the new unseen SAE holdout only after the freeze gate.
11. Run the independent sentiment confirmation and semantic proxy evaluation.
12. Compute hierarchical paired confidence intervals and diagnostics.
13. Write results, figures, manifests, audit records, and the final report.

## Leakage policy

| Split | Permitted use |
|---|---|
| Train | structured corruption, harmfulness scoring, sampling, loss calibration |
| Validation | model and hyperparameter selection, preliminary comparisons |
| Test/holdout | evaluation only after the protocol is frozen |

The samplers require the complete training split. Direction-score files must match
the training IDs, hook name, split hash, and pipeline version. Misuse fails with an
assertion before the relevant model forward.

## Reproducibility and checkpoints

All important parameters live in `config.yaml`. Checkpoint schema 2 includes:

- architecture configuration and state dict;
- training step and random seed;
- normalization mean/std required at inference;
- model, hook, and SAE identities;
- corruption configuration and split hash;
- pipeline version and calibrated real-batch gate metadata.

The paper-selected checkpoint is:

```text
outputs/final_v3/checkpoints/conditioned_kl_retention.pt
```

It is intentionally not committed to Git. The public checkpoint is hosted at
[`leolaz/steering-denoiser-gpt2`](https://huggingface.co/leolaz/steering-denoiser-gpt2)
as `conditioned_kl_retention.pt`. Loading instructions and the model's scope are
documented in `MODEL_CARD.md`.

## Released result tables

The main compact CSV tables from the successful full run are committed under
[`released_results/`](released_results/README.md). This includes the causal
ablation, frozen holdout uncertainty, cross-concept statistics, semantic and
generation aggregates, model-selection record, and runtime breakdown. Large raw
token-level outputs and checkpoints remain excluded from Git.

## Local validation

Python 3.10+ is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q src run_v3.py run_semantic_eval.py
```

The unit tests use synthetic/fake models where practical. Passing local tests does
not substitute for a real GPT-2/SAE GPU run; the empirical claims come from the
frozen Kaggle artifacts summarized in the paper.

## Rebuild the paper

The committed `paper/paper.pdf` is the release document. To regenerate its figures
from a completed local run:

```bash
python paper/make_publication_figures.py
cd paper
tectonic main.tex
```

The figure script reads frozen CSV files under `outputs/final_v3/` and writes only
publication copies under `paper/figures/`.

## Citation

Citation metadata is provided in `CITATION.cff`. The accompanying paper is:

> leolazz. *Repairing Activation Steering with Downstream-Aware Denoising*.
> T-Lab Mechanistic Interpretability Research Project, 2026.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
