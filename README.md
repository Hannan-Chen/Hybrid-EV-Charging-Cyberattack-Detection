# Hybrid EV Charging Cyberattack Detection

Public code, de-identified benchmark inputs, frozen models, and results for studying how known attack examples can improve EV charging cyberattack detection while retaining detection of other attack types.

**No manuscript, manuscript LaTeX source, submission package, or full-paper PDF is included.** Figure files are standalone experiment plots. The four main comparisons are the normal-only detector **U**, **Random Forest (RF)**, the supervised dual-branch transition network **SDTN**, and the hybrid **U+SDTN**. The complete experiment also includes Deep SAD, a joint semi-supervised detector, U+RF, and matched ablations.

## Data and Privacy

The original source is [ACN-Data](https://ev.caltech.edu/dataset), Caltech 2021. Recorded benign user updates are changes in `kWhRequested` and/or `requestedDeparture` relative to the preceding input. The benchmark contains 1,213 source sessions, 1,264 normal states, and 51 recorded benign user updates from 45 sources.

| Role | Source sessions | Normal states | Benign user updates |
|---|---:|---:|---:|
| Training | 606 | 630 | 24 |
| Validation | 364 | 382 | 18 |
| Test | 243 | 252 | 9 |

Raw ACN JSON is **not redistributed**. Raw session identifiers are replaced with pseudonymous numeric/source identifiers, and calendar dates are shifted per source to a synthetic anchor. Time of day, within-session timing, every numerical model feature, source membership, and the saved attack arrays are preserved. These transformations do not create additional observations or guarantee irreversible anonymity. See [data documentation](data/README.md), [data terms](DATA_TERMS.md), and the machine-readable [release provenance](data/public_release_provenance.json).

## Frozen Experiment

- Fixed attacks: `Catt=0.15`, `Chav=30 kW`, `Pl=24 kW`, five-minute power grid, attack seed 66. Attacks are simulated, not verified real intrusions.
- Six training tasks: each Type-i group contains both Type-i and Type-i-ta. Each detector is evaluated against all twelve attack settings without receiving the true test attack type.
- Per-model attack counts: 2, 5, 10, 20, 50, and All, combined over both variants and with one attack per source. Five paired repeats use seeds 66, 166, 266, 366, and 466.
- Twenty validation attacks per task, ten per variant. At 20 training attacks, configuration selection across repeats uses 106--115 distinct attack labels per group, not only 40 labels in total.
- Fixed training/validation/test source roles; no cross-validation or training-plus-validation refit. Validation selects configurations. Normal validation scores set thresholds with empirical normal TNR >=95% and benign-user-update TNR >=90%.
- U+SDTN uses independently trained branches, training-normal score scales, maximum-score fusion, and **one threshold for the combined score**, not an OR of two separately thresholded decisions.
- The primary F1 equally weights the benign-update/attack classes, then both variants and the six tasks. Other-type recall equally weights the ten variants excluded from that task's attack-based training and selection.

Internal frozen IDs such as `Mechanism-1` are retained in original result files to preserve identities and hashes; they mean the Type-1/Type-1-ta group, not an additional attack class. The upstream split-plan file preserves earlier source metadata; **the roles in `results/frozen_design.json` are the authoritative roles for this experiment**.

## Main Results

At 20 training attacks per model, averaged across six tasks and five paired repeats:

| Method | Known-type F1 | Other-type recall | Overall normal FPR | Mean benign-update FP / 9 |
|---|---:|---:|---:|---:|
| U | 0.606 | 0.663 | 4.84% | 0.40 |
| RF | 0.932 | 0.525 | 0.17% | 0.43 |
| SDTN | 0.860 | 0.327 | 3.08% | 0.17 |
| U+SDTN | 0.878 | 0.691 | 3.88% | 0.37 |

RF is stronger for known attacks; U+SDTN offers a known/other-type tradeoff, not universal superiority. The same nine test benign user updates are reused across tasks and repeats. These results do not establish field performance or zero-day guarantees. See the [full result browser](reports/full_results_zh.html), [summary CSV](results/case_summary.csv), and [label-exposure table](results/cross_repeat_label_exposure.csv).

## Quick Start: Inspect and Audit Saved Results

Use a fresh virtual environment. Audit-only checks do not require PyTorch:

```bash
python -m pip install -r environment/audit_requirements.txt
python tools/verify_public_release.py
python tools/verify_download.py
python -m unittest discover -s tests -p "test_*.py"
```

Open `notebooks/00_public_results.ipynb` for result tables and plots. `juno_executed/` contains historical executed experiment notebooks, with private output strings removed where needed. The public audit explicitly handles documented de-identification; it does not pretend that redacted input files have their original hashes.

## Reproduce Training

The original execution used Python 3.10.18 and the versions in `environment/executed_requirements.txt`. The 980 frozen models preserve their numerical payloads; five U models have only embedded source-location strings redacted, with original/public hash mappings recorded. Cached predictions are unchanged. **Only unpickle trusted files**, and use the original Python/library versions for model loading. Saved numerical audits can run without loading model pickles.

```bash
python -m pip install -r environment/requirements.txt
python tools/run_fresh_experiment.py --output ../ev_cma_fresh_run --actions freeze smoke
python tools/run_fresh_experiment.py --output ../ev_cma_fresh_run --resume --actions baseline search select test --workers 16
```

Fresh runs write to a separate output directory and produce a new protocol signature for the public input snapshot; they never overwrite the released frozen results. A full run is substantial: 4,320 candidate fits plus 180 matched-ablation fits, in addition to the U fits. Numerical-library threads are capped at one and worker count at 16. Hardware and library differences may affect bitwise training reproducibility.

## Generate Figures

```bash
python figure_sources/build_assets.py
python figure_sources/build_attack_power_profiles.py
```

The attack plot uses the unchanged reference power functions and a cyclic 04:00-to-next-day-04:00 display. Its endpoint repeats the first sample for display only. It is a time-of-day aggregate, not one chronological station day. These commands may generate local figure PDFs; the repository distributes SVG/PNG plots, not the paper.

## Files

- `data/processed/`: de-identified state catalog, fixed attack pool, upstream source plan.
- `src/`, `tests/`, `reference/`: model/protocol code, focused tests, reference attack code.
- `results/`: frozen selections, candidate tables, models, prediction arrays, all budgets and ablations.
- `notebooks/`, `juno_executed/`: interactive entrypoints and historical executed outputs.
- `figure_sources/`, `paper_figures/`, `evidence/`: figure code, standalone plots, derived tables and checks.
- `release_manifest.json`: exact public-file inventory and SHA-256 hashes.

## Attribution, Rights, and AI Assistance

Please cite ACN-Data: Z. J. Lee, T. Li, and S. H. Low, *ACN-Data: Analysis and Applications of an Open EV Charging Dataset*, ACM e-Energy, 2019, DOI [10.1145/3307772.3328313](https://doi.org/10.1145/3307772.3328313). Dataset access is governed by its provider's terms; this repository does not relicense underlying ACN data. No blanket software or data license is assigned by this release; see `DATA_TERMS.md`.

OpenAI Codex assisted with coding, writing, figures, and checks. The researchers are responsible for the experiment and interpretation. This repository does not include or assert publication of a manuscript.

The CMA motivation follows H. Jahangir, S. Lakshminarayana, and H. V. Poor, *Charge Manipulation Attacks Against Smart Electric Vehicle Charging Stations and Deep Learning-Based Detection Mechanisms*, IEEE Transactions on Smart Grid, 2024, DOI [10.1109/TSG.2024.3401090](https://doi.org/10.1109/TSG.2024.3401090). The released reference notebook documents the executed transformations rather than claiming that the underlying attack concept is new.
