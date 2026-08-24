# Released result tables

This directory contains compact, frozen CSV exports from the successful full
`final_v3` Kaggle run used in the paper. They are committed so that the central
empirical claims can be inspected without downloading the large token-level
artifacts or rerunning GPT-2.

| File | Purpose |
|---|---|
| `causal_ablation_table.csv` | Primary comparison isolating unconditional repair, conditioning, downstream KL, and retention. |
| `validation_model_selection.csv` | Frozen validation-stage selection record. |
| `holdout_statistical_tests.csv` | Paired hierarchical-bootstrap results on 20 unseen SAE directions. |
| `cross_concept_statistics.csv` | Post-selection confirmation on one independent sentiment direction. |
| `semantic_aggregate.csv` | Independent output-level semantic-proxy aggregates by method and strength. |
| `generation_aggregate.csv` | Generation-level NLL, diversity, repetition, KL, and SAE-feature summaries. |
| `runtime_breakdown.csv` | Measured runtime and forward-pass counts by final V3 stage. |

These are result artifacts, not hand-entered tables. Their source paths in a
completed run are under `outputs/final_v3/`, as documented in
[`../REPRODUCING.md`](../REPRODUCING.md).

## Scope

- `holdout_statistical_tests.csv` is the headline within-SAE-family generalization
  result. Its confidence intervals use the paired hierarchy direction, then prompt;
  token rows are not treated as independent samples.
- `cross_concept_statistics.csv` covers one qualitatively different sentiment
  direction. It is confirmation beyond the SAE family, not evidence of universal
  cross-concept generalization.
- `semantic_aggregate.csv` uses an independent output-level classifier proxy. SAE
  activation is a mechanistic intervention-signal proxy and is not semantic ground
  truth.
- Clean-model continuation NLL is an LM degradation/fluency proxy, not a human
  fluency judgment.
- Pareto hypervolume differences are meaningful only for paired comparisons with
  the identical bounds stored in the same table. Absolute hypervolume values must
  not be compared across studies.

Large raw artifacts remain intentionally excluded from Git:

- `token_validation.csv` and `token_holdout_or_replication.csv`;
- per-example generation and semantic-score tables;
- activation caches, checkpoints, and training traces.

The selected checkpoint is published separately at
[`leolaz/steering-denoiser-gpt2`](https://huggingface.co/leolaz/steering-denoiser-gpt2).
