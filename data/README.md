# Public Experimental Database

`processed/fixed_multi_input_attack_pool_seed66.joblib` is the fixed benchmark used by the released detectors, including current-state features, transition features, source indices, twelve attack variants, and profiles needed by the original power functions. Attack parameters and feature arrays are not changed by this release.

`processed/multi_input_state_catalog.csv` has 1,264 normal states, including 51 recorded benign user updates. `source_uid` now contains only `source-NNNN` pseudonyms. Numeric source/state IDs and row order are unchanged. Each source's original calendar date is shifted to a synthetic anchor, preserving its times of day and all within-session durations and transitions. Other identifiers from the raw API are not distributed.

`processed/multi_input_source_split_plan.json` is upstream provenance. The experiment's actual 606/364/243 roles and exact attack-label subsets are stored in `results/frozen_design.json`, not inferred from that upstream file's historical training or calibration fields.

`public_release_provenance.json` maps original and released hashes for the few transformed files. `input_manifest.json` retains original source hashes for traceability but does not provide original raw records. The verification tools distinguish omitted raw files and documented transformations from genuinely unchanged artifacts.

The source dataset has 1,213 valid sessions and 45 sources with qualifying updates. An update changes requested energy, requested departure, or both relative to the preceding input. Recorded updates are treated as normal; no verified intrusion labels exist in the source data.

For raw data, register at the [official ACN-Data portal](https://ev.caltech.edu/dataset) and use its documented Caltech session API for the 2021 export window. Keep any API token private and store raw downloads outside the repository. The service may change, and a later export need not be byte-identical to the archived export hash. The supplied derived snapshot is the fixed reproducibility input.
