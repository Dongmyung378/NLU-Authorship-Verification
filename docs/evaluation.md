# Evaluation protocol

[한국어](./evaluation.ko.md) | [Project overview](../README.md) | [Architecture](./architecture.md) | [Measured results](./results.md)

The experiment has four separate roles: train fits task weights, selection chooses checkpoints and model policy, calibration fits the output decisions, and locked holdout evaluates the frozen system. The target proportions are 70 / 10 / 5 / 15. All reported counts and labels must come from the saved split manifest and source CSV hashes.

## Model selection

CE and custom binary ASL share the pinned RoBERTa-Large revision, initial weights, tokenizer, seed 378, training records, seeded pair-swap schedules, microbatch 2, effective batch 32 and optimizer settings. Two independent workers train one loss each on their own assigned T4. A shared epoch barrier guarantees equal completed epochs up to the six-epoch cap frozen before task training, with patience three when neither loss improves raw AB selection macro F1 at threshold 0.5. An improvement by either loss resets the common patience. Each loss saves its own best checkpoint; exact ties retain the earlier epoch. Best-checkpoint epochs may differ, while completed epochs remain equal. Both saved checkpoints are reloaded before inference.

A completed train-only throughput probe retained microbatch 2. Candidate 8's speedup was 1.0244, below the predeclared 1.05 adoption gate, while 32 was slower. The maximum six equal epochs remain within the conservative nine-hour wall/18-worker-hour protocol. The probe performed no prediction-based validation, calibration or holdout inference and discarded every updated weight. Its measured receipt is [`throughput-profile.json`](../reports/capability/throughput-profile.json), with a separate [`summary`](../reports/capability/throughput-summary.json). Resource measurements are not task-performance results.

The parent enforces a nine-hour wall limit and a conservative 18 assigned-worker-hour limit. A deadline, failed worker, mismatched initial-state hash, incomplete epoch barrier or nonfinite result prevents promotion of either run to a successful matched release. Partial artifacts remain failure evidence. FP16 overflow-skipped updates are disclosed separately from successful updates, so the report does not equate equal attempted schedules with identical update counts.

The raw AB CE-versus-ASL comparison holds orientation and threshold 0.5 fixed. The report separately compares each model's AB and symmetric mean-logit policies. Symmetry is selected if it gains at least 0.002 macro F1, or improves exact input-order invariance while losing at most 0.002 macro F1. A provisional raw winner is recorded at threshold 0.5. All orientation choices and raw comparisons use selection only.

After independently fitting calibration for CE, custom ASL and the stylometric ensemble, deployment selection reuses the saved selection logits with each model's fixed temperature and calibration-fitted threshold. The deployed Large is custom ASL only if this complete pipeline gains at least 0.002 selection macro F1 over calibrated CE; otherwise CE remains selected. The raw and calibrated-pipeline winners are reported separately. Calibration parameters are not refitted on selection, input-order policies are not revisited, and no additional GPU inference is required. This gain includes loss, orientation and calibration, so it cannot be attributed solely to the loss. A non-improving ASL pipeline remains in the record.

Raw CE and custom-ASL training losses have different scales. Comparing their numerical size does not establish performance or reduced overfitting. The new experiment retains Large but changes data separation, task initialization, training and output calibration. The historical Large development score and the new frozen holdout score are therefore not a controlled before-and-after performance comparison. The matched new CE and ASL runs provide the controlled comparison.

## Stylometric study

All TF-IDF vocabularies and IDF values are fitted on train only. The CPU study predeclares four XGBoost-LightGBM recipes, at most eight component fits and a 120-minute budget. Three bounded pilot recipes compare full, TF-IDF-only and style-only features; the fourth preserves the original larger full-feature tree ceilings. Native selection logloss chooses component checkpoints, and selection macro F1 at threshold 0.5 chooses the ensemble recipe and weight.

The reference is the full-feature pilot with XGBoost weight 0.5. A candidate must gain at least 0.002 macro F1 to replace it. Final XGBoost weights are restricted to 0.25, 0.5 or 0.75; weights 0 and 1 are reported only as component diagnostics. Time-limited candidates are identified, and an unfinished required reference prevents a completed study. Saved-model reload scores and actual AB/BA calls must agree with the selected ensemble. Calibration and locked holdout are excluded from this study.

## Calibration

Temperature scaling minimizes negative log likelihood on calibration data only, using the class-margin logits. The bounded positive temperature is accepted only if calibration NLL does not worsen against the identity temperature under the declared rule. The decision threshold is then chosen by calibration macro F1. Candidates include distinct calibrated scores, 1.0 and 0.5; exact ties prefer the threshold closest to 0.5, then the lower threshold. The final evaluation does not refit temperature or thresholds.

Calibration diagnostics include NLL, binary Brier score and expected calibration error with 10 fixed, equal-width bins of same-author probability. ECE is the count-weighted absolute difference between each bin's mean probability and observed same-author label frequency; empty bins contribute zero. This is a positive-class probability convention, not predicted-class confidence ECE. Raw and calibrated metrics use the same records, and a lower NLL does not necessarily mean better classification accuracy.

A separate table groups the confidence assigned to the threshold-selected class into 10 equal-width bins. It reports count, correct count, observed accuracy and mean confidence, with null accuracy for empty bins. Raw scores use threshold 0.5; calibrated scores use the frozen calibration threshold. This accuracy table is distinct from the positive-label frequency used in ECE, and a threshold other than 0.5 can assign a class with confidence below 0.5.

## Abstention

The optional confidence threshold is selected on calibration data. Confidence is the calibrated probability assigned to the threshold-selected label: `p` for a same-author decision and `1-p` for a different-author decision. Its declared target is at least 80% accepted coverage and at least 90% accepted accuracy. The rule chooses the greatest coverage meeting both targets. If that is infeasible, it retains the best calibration accuracy among cutoffs with the required coverage, breaks ties toward greater coverage, and reports the unmet target. Locked-holdout accuracy cannot be used to adjust that threshold until it meets the target.

The final report includes full-coverage accuracy and macro F1, accepted count, total count, accepted coverage, accepted accuracy and the number of abstentions. Full-coverage metrics and the confusion matrix use the frozen binary prediction for every pair, including pairs whose user-facing decision is `uncertain`. Accepted accuracy excludes abstentions and must be reported with coverage. Abstention is an undecided user-facing result; it earns no additional correct classification.

## Freeze and final reports

After calibration-aware deployment selection and before any locked-holdout inference, bind all three model artifacts, tokenizer files, inventories, data manifest, training configuration, source snapshot, raw comparison, deployed choice and calibrated decisions into a joint freeze. Verify their hashes again when loading cached results and evaluating copied artifacts. Final reports identify the freeze, model, manifest and source hashes and retain record-order IDs, integer correct count, class support, confusion matrix and predictions.

Once final metrics are accessed, do not automatically change the selected loss, input order, calibration or abstention policy. A later experiment requires a clearly versioned protocol and cannot reclassify the already inspected holdout as untouched selection-free evidence.

The supplied unlabeled `test.csv` and `live.csv` can produce prediction files but no accuracy or F1. The supplied corpus was historically used during earlier development, and the source lacks author and domain identifiers. The verified scope is this experiment's document-disjoint split, not independent author or domain generalization.

## Diagnostic slices

Length bins use the shorter cleaned passage's whitespace-separated word count before truncation: under 50, 50 to 199, and at least 200 words. Include each slice's support before reporting its metrics. Also describe truncation rate, AB/BA score disagreement and prediction disagreement, symmetric score invariance, and actual truncated-token collisions under the fixed tokenizer.

The completed [input-only token-length audit](../reports/token-length-audit.json) finds 197 of 4,708 holdout pairs (4.184%) above the combined 512-token limit. Token length and shorter-passage word count describe different properties: the at-least-200-word bin contains only five holdout pairs. Its final metrics must retain that support and cannot establish reliable long-passage generalization. This audit used no labels, model weights or predictions and did not change the split or model choices.

These are descriptive analyses of the same frozen evaluation. They do not prove a causal reason for model errors or authorize retuning from final labels. Small slices may fluctuate substantially. The single seed 378 does not provide multi-run uncertainty estimates or a universal conclusion that one loss is better.

## Conditional uncertainty

The frozen plan predeclares [`scripts/bootstrap_results.py`](../scripts/bootstrap_results.py): 1,000 paired whole-document-component bootstrap draws, seed 378 and 95% percentile intervals. Each draw samples the original number of locked-holdout groups with replacement. Every pair in a sampled group repeats by that group's multiplicity, so sample pair counts may vary. All models use the same draws, while metrics weight evaluated pairs equally.

The report gives accuracy and macro F1 intervals for all three fixed pipelines and a paired calibrated custom-ASL-minus-CE macro F1 interval. It includes every final pair irrespective of abstention, checks hashes and exact IDs against the freeze and release reports, and performs no model inference, parameter fitting or reselection. Raw loss comparison remains separately reported; the calibrated interval includes orientation and calibration effects.

These are conditional intervals for the provided corpus, its observed document groups and one trained seed. They do not estimate variation across training seeds, new authors or new domains. Group exchangeability remains an assumption, and bootstrap does not erase historical corpus exposure or unmeasured near-duplicate dependence. The output reports uncertainty after evaluation and cannot change the deployed model or decision rules.

## Runtime and reproducibility evidence

Synthetic tests check implementation properties such as grouping, gamma-zero equivalence to CE, gradients, calibration and artifact integrity. Saved-model replay checks actual released weights on labeled records. Retraining is a separate and more expensive procedure; passing unit tests does not imply that another environment reproduces model predictions.

The completed Large capability probe attempted 40 effective-batch updates on one T4, with 39 successful updates and one FP16 overflow skip. Its median attempt took 5.136 seconds and peak allocated memory was 6.09 GiB, with no OOM retry. All probe weights were discarded and no calibration or final inference was performed. This is feasibility evidence, not accuracy, F1 or a promise about the full-run duration. Its earlier three-epoch serial projection is recorded separately from the subsequent parallel protocol in [`reports/capability/summary.json`](../reports/capability/summary.json).

State the actual operating system, Python and pinned package versions for each observed replay. Two operating systems on the same physical computer remain the same host. CPU timing should use identical examples, token-length policy, batch size and thread count, report warmup and repetitions, and distinguish AB from the two-forward-pass symmetric policy. Model file size and latency are measured properties, not a substitute for final accuracy.
