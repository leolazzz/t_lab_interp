# Reproducing the reported results

This document describes the supported path for reproducing the results reported in
`paper/paper.pdf`. The canonical entry point is the single notebook
[`notebooks/final_v3_all_experiments.ipynb`](notebooks/final_v3_all_experiments.ipynb).
It creates the activation cache, trains all models, evaluates the frozen holdout,
runs the cross-concept confirmation, and writes the final tables and figures. No
outputs from an earlier Kaggle version are required.

## 1. Recommended environment

The reported run used:

- Kaggle's latest container image at the time of the run;
- a Kaggle `GPU T4 x2` session, although the code uses one CUDA device and does not
  use distributed training;
- Internet enabled, because the first run downloads GPT-2 Small, the SAE, text
  datasets, and the independent semantic classifier;
- seed `42`;
- the configuration committed in [`config.yaml`](config.yaml).

One 16 GB T4 is sufficient for the implemented single-device pipeline. The recorded
end-to-end V3 stages took about 16,300 seconds, and the complete Kaggle run took
approximately 4 hours 40 minutes. Allow a full Kaggle session and at least 2--3 GB
of free working storage for outputs plus downloaded model and dataset caches. The
completed local output tree used about 638 MiB, including about 372 MiB of cached
activations and 243 MiB of V3 artifacts.

## 2. Start a fresh Kaggle run

1. Open Kaggle and choose **Create > New Notebook**.
2. In **Notebook options**, select a T4 GPU and turn **Internet** on.
3. Import
   `https://github.com/leolazzz/t_lab_interp/blob/main/notebooks/final_v3_all_experiments.ipynb`
   as the notebook.
4. Do not attach an output Dataset from an older run. The notebook can run from an
   empty `/kaggle/working` directory.
5. Select **Run All** or **Save Version > Save & Run All**.

The first cell uses an existing project checkout when one is present. Otherwise it
clones `https://github.com/leolazzz/t_lab_interp.git` into
`/kaggle/working/steering-denoiser-github`. Attaching an unpacked copy of the
repository with **Add Input** is also supported, but is not required.

For the paper-scale run, leave the production flags in the notebook unchanged:

```python
DEBUG = False
FORCE_REBUILD_ACTIVATIONS = True
FORCE_RETRAIN_GAUSSIAN = True
FORCE_RETRAIN_CALIBRATED = True
FORCE_RECOMPUTE_HARMFULNESS = True
FORCE_RETRAIN_FLUENCY = True
RUN_PROJECTED = True
RUN_INCREMENTAL = True
RUN_QUICK_VALIDATION = True
RUN_FULL_VALIDATION = True
RUN_NEIGHBOR_DIAGNOSTICS = True
RUN_GENERATION = False
RUN_TEST = False
EVAL_INTERVENTION_BATCH = 8
TOKEN_EVAL_PROMPT_BATCH_SIZE = 16
USE_INFERENCE_AUTOCAST = True
HARMFULNESS_NUM_DIRECTIONS = 100
HARMFULNESS_NUM_CONTEXTS = 32
HARMFULNESS_STRENGTHS = [0.25, 0.5]
```

The final V3 cell must remain:

```python
DEBUG_V3 = False
RUN_V3_HOLDOUT = True
RUN_V3_GENERATION = True
RUN_V3_SEMANTIC_PROXY = True
```

`RUN_GENERATION=False` refers to the older supporting V1 generation grid. The final
V3 generation study is still enabled by `RUN_V3_GENERATION=True`.

## 3. What the notebook executes

The notebook performs these stages in order:

1. installs missing packages from `requirements.txt`;
2. seeds Python, NumPy, PyTorch, and CUDA;
3. loads TransformerLens `gpt2-small` and checks an identity intervention at
   `blocks.6.hook_resid_pre`;
4. loads SAE Lens release `gpt2-small-res-jb`, SAE ID
   `blocks.6.hook_resid_pre`, and verifies its `24576 x 768` decoder matrix;
5. creates the leakage-safe train/validation/test direction split;
6. caches 200,000 clean token activations and normalization statistics;
7. trains the Gaussian, calibrated SAE, and fluency-sensitive supporting baselines;
8. trains the V3 causal ablations, including the selected
   `conditioned_kl_retention` model for 1,000 steps;
9. selects on validation directions and freezes the protocol;
10. evaluates the previously unopened 20-direction SAE holdout;
11. runs the independent post-selection sentiment confirmation;
12. runs generation and independent semantic-proxy evaluation;
13. computes paired hierarchical-bootstrap uncertainty, diagnostics, tables,
    figures, the audit, and the final report.

Important early gates should report the following invariants:

```text
model: gpt2-small
hook: blocks.6.hook_resid_pre
d_model: 768
SAE decoder shape: (24576, 768)
pipeline_version: final_v3
seed: 42
```

The final leakage audit must contain `"passed": true`.

## 4. Expected artifacts

The main output directory is `outputs/final_v3/`. A successful complete run writes:

```text
outputs/final_v3/
├── checkpoints/
│   ├── conditioned_reconstruction.pt
│   ├── conditioned_kl.pt
│   ├── conditioned_kl_retention.pt
│   └── conditioned_full.pt
├── results/
│   ├── causal_ablation_table.csv
│   ├── token_validation.csv
│   ├── token_holdout_or_replication.csv
│   ├── statistical_tests.csv
│   ├── holdout_statistical_tests.csv
│   ├── cross_concept_confirmation.csv
│   ├── cross_concept_statistics.csv
│   ├── correction_geometry.csv
│   ├── natural_neighbor_diagnostics.csv
│   ├── generation_results.csv
│   └── generation_aggregate.csv
├── generations/final_generation.jsonl
├── diagnostics/
│   ├── audit.json
│   ├── numerical_gates.json
│   ├── loss_scale_diagnostic.json
│   └── runtime_breakdown.json
├── configs/frozen_protocol.json
├── figures/
├── final_report.md
└── final_summary.json
```

Copies of `final_report.md` and `final_summary.json` are also placed under
`outputs/final_v3/reports/`.

The primary result table is
`outputs/final_v3/results/causal_ablation_table.csv`.

The paper-selected checkpoint is
`outputs/final_v3/checkpoints/conditioned_kl_retention.pt`. It includes the
architecture, model state, optimizer state, normalization statistics, training
configuration, hook name, split hash, pipeline version, and completed step.

## 5. Verify a completed run

Run this cell after the notebook finishes:

```python
from pathlib import Path
import json
import pandas as pd

root = Path("outputs/final_v3")
required = [
    root / "checkpoints/conditioned_kl_retention.pt",
    root / "results/causal_ablation_table.csv",
    root / "results/token_holdout_or_replication.csv",
    root / "results/cross_concept_confirmation.csv",
    root / "generations/final_generation.jsonl",
    root / "configs/frozen_protocol.json",
    root / "diagnostics/audit.json",
    root / "diagnostics/runtime_breakdown.json",
    root / "final_report.md",
    root / "final_summary.json",
]
missing = [str(path) for path in required if not path.exists()]
assert not missing, f"Missing final artifacts: {missing}"

audit = json.loads((root / "diagnostics/audit.json").read_text())
assert audit["passed"] is True

table = pd.read_csv(root / "results/causal_ablation_table.csv")
print(table[["method", "mean_delta_nll", "mean_kl", "mean_concept_retention"]])
print("All final artifact and leakage gates passed.")
```

The released causal-ablation table is the primary numerical checksum:

| Method | Mean Delta NLL | Mean KL | Mean retention |
|---|---:|---:|---:|
| `relative_raw` | 7.010339 | 7.047667 | 1.000000 |
| `sae_calibrated` | 6.201735 | 6.245004 | 1.241633 |
| `conditioned_reconstruction` | 6.905518 | 6.942378 | 1.161900 |
| `conditioned_kl` | 2.068376 | 2.125058 | 1.010862 |
| `conditioned_kl_retention` | 2.062611 | 2.119669 | 1.115480 |

The frozen 20-direction SAE holdout should show a selected-versus-raw mean Delta NLL
recovery of `4.322` with hierarchical-bootstrap 95% CI `[2.927, 5.795]`. The
post-selection sentiment confirmation contains 40 matched examples and a mean Delta
NLL recovery of `1.401`, 95% CI `[1.082, 1.692]`.

## 6. Local execution

Kaggle is the reference environment. A local CUDA machine can execute the same
notebook from the repository root:

```powershell
git clone https://github.com/leolazzz/t_lab_interp.git
cd t_lab_interp
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyterlab
jupyter lab notebooks/final_v3_all_experiments.ipynb
```

Choose **Run All** in Jupyter. The supported fresh end-to-end entry point is the
notebook, not a direct invocation of `run_v3.py`: the early notebook cells build the
activation cache and baseline prerequisites consumed by the final runner.

For a genuinely fresh rerun, use a new clone or rename the existing `outputs/`
directory before starting. Setting `FORCE_*` flags to `False` intentionally resumes
from existing artifacts and is useful operationally, but is not the clean
from-scratch reproduction described here.

## 7. Reusing the published checkpoint

If only inference or evaluation is required, download
`conditioned_kl_retention.pt` from
[`leolaz/steering-denoiser-gpt2`](https://huggingface.co/leolaz/steering-denoiser-gpt2)
and follow [`MODEL_CARD.md`](MODEL_CARD.md). This bypasses training and therefore is
not a reproduction of the training experiment.

## 8. Determinism boundary

The code fixes all available random seeds and records the split and protocol hashes.
Nevertheless, `requirements.txt` intentionally lists compatible packages rather
than exact build hashes, Kaggle's latest image changes over time, and CUDA/autocast
kernels are not guaranteed to be bitwise deterministic across library and hardware
versions. A rerun should reproduce the qualitative conclusions and closely match
the reported aggregates and confidence intervals; exact last-decimal or bytewise
identity is not guaranteed. Treat the leakage audit, artifact manifest, paired
direction-level conclusions, and numerical tolerances as the reproducibility gates.
