# Six-mechanism few-attack-label EV-CMA results

Catt=0.15; fixed source holdout; simulated and previously studied benchmark.
Each mechanism jointly trains Type-i and Type-i-ta. The primary F1 equally averages the two variants' change-only binary class-macro F1s.
N+20 attack labels are used per task. Every method uses the same subsets and normal-validation constraints.

N is the combined attack count over both variants. Finite budgets are balanced within one record; All covers all distinct eligible sources and may be imbalanced.
## Results by training-label budget (equal weight over six training mechanisms)

### N=2

|Method|Known-mechanism change F1|Known recall|Other-five-mechanism recall|Mean edit FP / 9|
|---|---:|---:|---:|---:|
|U|0.6056|0.6627|0.6627|0.40|
|RF|0.7699|0.8340|0.5500|0.50|
|SDTN|0.7264|0.7782|0.3128|0.00|
|DeepSAD|0.6950|0.7076|0.5381|0.00|
|SS|0.5414|0.6051|0.3228|0.10|
|U+RF|0.7713|0.8344|0.5518|0.50|
|U+SDTN|0.7852|0.8397|0.6892|0.30|

### N=5

|Method|Known-mechanism change F1|Known recall|Other-five-mechanism recall|Mean edit FP / 9|
|---|---:|---:|---:|---:|
|U|0.6056|0.6627|0.6627|0.40|
|RF|0.8398|0.8926|0.5273|0.43|
|SDTN|0.7897|0.8496|0.3904|0.07|
|DeepSAD|0.6849|0.7010|0.5352|0.03|
|SS|0.6505|0.7231|0.4076|0.03|
|U+RF|0.8398|0.8926|0.5273|0.43|
|U+SDTN|0.8308|0.8854|0.6993|0.37|

### N=10

|Method|Known-mechanism change F1|Known recall|Other-five-mechanism recall|Mean edit FP / 9|
|---|---:|---:|---:|---:|
|U|0.6056|0.6627|0.6627|0.40|
|RF|0.8887|0.9470|0.4772|0.40|
|SDTN|0.8414|0.9040|0.3170|0.20|
|DeepSAD|0.6966|0.7184|0.5352|0.03|
|SS|0.6710|0.7389|0.4087|0.03|
|U+RF|0.8887|0.9470|0.4774|0.40|
|U+SDTN|0.8731|0.9304|0.7043|0.57|

### N=20

|Method|Known-mechanism change F1|Known recall|Other-five-mechanism recall|Mean edit FP / 9|
|---|---:|---:|---:|---:|
|U|0.6056|0.6627|0.6627|0.40|
|RF|0.9319|0.9774|0.5252|0.43|
|SDTN|0.8595|0.9026|0.3265|0.17|
|DeepSAD|0.6997|0.7211|0.5380|0.00|
|SS|0.7359|0.7903|0.4509|0.10|
|U+RF|0.9319|0.9774|0.5255|0.43|
|U+SDTN|0.8783|0.9180|0.6909|0.37|

### N=50

|Method|Known-mechanism change F1|Known recall|Other-five-mechanism recall|Mean edit FP / 9|
|---|---:|---:|---:|---:|
|U|0.6056|0.6627|0.6627|0.40|
|RF|0.9441|0.9799|0.4659|0.33|
|SDTN|0.9198|0.9512|0.3109|0.13|
|DeepSAD|0.7084|0.7276|0.5374|0.00|
|SS|0.7875|0.8364|0.4742|0.03|
|U+RF|0.9441|0.9799|0.4671|0.33|
|U+SDTN|0.9028|0.9348|0.6955|0.37|

### N=All

|Method|Known-mechanism change F1|Known recall|Other-five-mechanism recall|Mean edit FP / 9|
|---|---:|---:|---:|---:|
|U|0.6056|0.6627|0.6627|0.40|
|RF|0.9863|0.9989|0.5363|0.33|
|SDTN|0.9397|0.9594|0.3413|0.23|
|DeepSAD|0.7149|0.7371|0.5363|0.00|
|SS|0.8084|0.8481|0.5078|0.03|
|U+RF|0.9863|0.9989|0.5364|0.33|
|U+SDTN|0.9148|0.9374|0.7137|0.33|

## Interpretation limits

- Nine test edits are nine observations, not 45 observations after five repeats. Report exact per-repeat errors.
- Other settings were withheld from the current training and selection. Architectures and this benchmark were studied previously; this is not proof of genuinely unforeseen attacks.
- Normal validation constraints are empirical and do not guarantee population or test acceptance.
- Preset U uses no current attack labels for fitting or selection; its historical architecture is not evidence of zero historical research-label cost.
- Original generator fingerprints remain unchanged. Start-time manipulation is an easy cue in these data; feature-group results are diagnostic.
- Source bootstrap is conditional on fitted models. Attack-burst dependence is not fully resolved by source grouping, and intervals are not multiplicity adjusted.
- Parallel methods include both component inference costs. Training times include cached-work timing and are not a clean hardware-normalized benchmark.
- The two class F1s use the observed, attack-heavy test mix. Balanced accuracy and edit FP are essential companions.

## Controls, matched ablations, and label accounting

Budget scope: N+20 is exposure for one fitted model and its validation. Shared configuration selection over five different training subsets uses their label union; `cross_repeat_label_exposure.csv` reports that larger full research-selection cost. These are per-fit learning curves, not a strict N+20-label end-to-end selection guarantee.
`actual_attack_label_costs.csv` distinguishes the comparator's task budget from labels actually consumed: U uses zero current attack fitting/selection labels. Historical architecture-development cost is not erased.
`shortcut_controls.csv` includes all-normal and every-change controls; the latter has change-only balanced accuracy 0.5 and falsely flags all nine real benign edits.
`gains_and_label_costs.csv` reports both known variants separately and transfer to the other five mechanisms (ten variants).
`attack_source_and_launch_group_counts.csv` reports coordinated launch-day groups as well as source records. Distinct charging sessions need not be independent attack incidents.
`matched_ablation_comparison.csv` compares each loss removal to the same selected full SS structure, subset, and seed; no test retuning was performed.
