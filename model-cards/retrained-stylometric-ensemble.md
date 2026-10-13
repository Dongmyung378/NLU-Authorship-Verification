---
language: en
tags:
  - text-classification
  - authorship-verification
  - stylometry
  - xgboost
  - lightgbm
repo: https://github.com/Dongmyung378/NLU-Authorship-Verification
---

# Retrained stylometric ensemble

[한국어](./retrained-stylometric-ensemble.ko.md) | [Project overview](../README.md) | [Historical model card](./stylometric-ensemble.md)

This card describes the completed CPU training and selection study on the document-disjoint manifest and its separate frozen final evaluation. Selection tables use 3,139 pairs at threshold 0.5 before calibration; the final section uses all 4,708 heldout pairs after the joint freeze. The historical development score is documented in its own card and is not a controlled comparator for this study.

## Model and data

The model preserves the original joint XGBoost and LightGBM ensemble architecture by Dongmyung Park and Juho Kim. The original project development period remains February to May 2026. Both tree learners and the three TF-IDF transforms were fitted afresh; historical task weights were not reused.

| Item | Observed configuration |
|---|---|
| Task | English text-pair classification: different authors or same author |
| Training pairs | 21,972 |
| Selection pairs | 3,139 |
| Trained seed | 378, one seed |
| Selected features | Full representation, 28,512 dimensions |
| Ensemble weights | XGBoost 0.25, LightGBM 0.75 |
| Selection threshold | 0.5, before calibration |
| Saved XGBoost trees | 261; best iteration 260, zero-based |
| Saved LightGBM trees | 493; best iteration 493, one-based |
| Serialized bundle | Seven files, 6,066,073 bytes, including `artifact-inventory.json` |

Word and character TF-IDF each retain 7,000 features; the function-word vocabulary has 219 entries. Pair features combine absolute differences and element-wise products, 72 transformations of 24 per-document style measurements, and two cosine distances. Each vectorizer uses only training passages. The independent [IDF proof](../reports/stylometric-study/train-only-tfidf-proof.json) checks the stored values against document frequencies from 43,944 training-passage occurrences. These are occurrences in training pairs, not a count of unique authors or unique documents.

The [manifest](../reports/split-manifest.json) separates connected components of shared exact and normalized document identities. Author IDs and domain labels are unavailable. Historical use of the original labeled files and the limits of document separation are described in the [evaluation protocol](../docs/evaluation.md).

## Bounded selection study

The [predeclared plan](../reports/stylometric-study/predeclared-plan.json) fixes four recipes, up to eight component fits, two CPU threads and a 120-minute total budget. The pilots share 500-tree ceilings, learning rate 0.03 and native selection-logloss checkpointing. Full features use XGBoost depth 6 and LightGBM 63 leaves. Each pilot component has a 600-second callback limit.

A single larger full-feature recipe uses the original-style ceilings of 7,000 XGBoost and 2,000 LightGBM trees, learning rate 0.01, depth 7 and 127 leaves, with 1,800 seconds per component. Ceilings are upper limits: native early stopping and callback limits determine the retained checkpoint. Fit limits are checked at iteration boundaries, and the reports preserve actual durations and time-stop flags.

Final XGBoost weight candidates are 0.25, 0.5 and 0.75, keeping both learners active. Weights 0 and 1 are single-component diagnostics. A candidate must improve selection macro F1 by at least 0.002 over the full-feature 50/50 pilot reference. Ties prefer full features, then a weight closer to 0.5 and the earlier recipe. Calibration and final evaluation are excluded from these decisions.

| Selection recipe | Dimensions | XGBoost / LightGBM | Macro F1 | Accuracy | Correct |
|---|---:|---:|---:|---:|---:|
| Full-feature pilot reference | 28,512 | 0.50 / 0.50 | 0.737298 | 73.85% | 2,318 / 3,139 |
| Selected full-feature pilot | 28,512 | 0.25 / 0.75 | 0.742355 | 74.32% | 2,333 / 3,139 |
| TF-IDF-only pilot, including cosine distances | 28,440 | 0.25 / 0.75 | 0.730061 | 73.11% | 2,295 / 3,139 |
| Style-only pilot | 72 | 0.25 / 0.75 | 0.643756 | 64.38% | 2,021 / 3,139 |
| Larger full-feature recipe | 28,512 | 0.25 / 0.75 | 0.739923 | 74.10% | 2,326 / 3,139 |

The selected blend gains 0.005056 macro F1, or 0.5056 percentage points, and 15 correct selection pairs over the 50/50 reference. The larger recipe did not beat the selected pilot, so its expansion was not retained and no further fits were launched. These are selection observations from one trained seed. The ablation table compares complete fitted recipes under the declared time limits; retained tree counts can differ. It is not an estimate of final generalization or a repeated-seed significance result.

The [machine-readable ablation report](../reports/stylometric-study/feature-ablation.json) includes every eligible weight and both single-component diagnostics. Full checkpoint curves and original evidence remain in the [completion report](../reports/stylometric-study/completion.json).

## Saved-model and environment evidence

The entire CPU study and saved-model verification took 5,703.518 seconds, or 95.06 minutes. Peak process working set was 10,805,551,104 bytes, or 10.06 GiB, across the complete study. The selected components fitted for 600.728 seconds and 367.317 seconds respectively. XGBoost reached its time limit and retained its best 261-tree checkpoint; LightGBM retained 493 trees within its 500-tree ceiling. The unselected larger components saved 681 and 1,389 trees and both reached their 30-minute callback limits.

Training ran in an isolated Windows CPU environment with Python 3.12.15, NumPy 1.26.4, SciPy 1.16.3, scikit-learn 1.6.1, joblib 1.5.3, XGBoost 3.2.0 and LightGBM 4.6.0. See the exact [environment receipt](../reports/stylometric-study/cpu-environment.json).

The saved ensemble was reloaded and predicted all 3,139 selection pairs. Its maximum difference from the recorded component blend was 0.0, and physically swapping the passages produced a maximum probability gap of 0.0. Every selected file passed byte-size and SHA256 verification; the ten training-source and dependency bindings remained unchanged. This is actual saved-model verification on selection data. Separate Windows and Ubuntu WSL2 replay completed all 4,708 locked-holdout pairs and passed the original comparison tolerances. The [CPU summary](../reports/cpu-reproduction.json) also records completed latency and whole-process memory measurements on these two environments on the same physical host.

The [study summary](../reports/stylometric-study/summary.json), [artifact inventory](../reports/stylometric-study/artifact-inventory.json), [selected configuration](../reports/stylometric-study/selected-configuration.json) and [public evidence inventory](../reports/stylometric-study/public-evidence-inventory.json) provide exact measured values and hashes. These reports contain metadata rather than model binaries, private absolute paths or personal work journals. The original external proofs were preserved without editing.

## Frozen final evaluation

After all model and decision parameters were jointly frozen, the ensemble classified 3,491 of 4,708 locked-holdout pairs correctly: accuracy 74.1504% and macro F1 0.741498. Its uncalibrated AB result at threshold 0.5 was F1 0.743335 with 3,503 correct pairs. The independently fitted temperature 0.834865 and threshold 0.453506 reduced NLL from 0.481949 to 0.477097 and ECE from 0.032128 to 0.019479, but reduced the correct count by twelve. Improved probability calibration does not establish improved classification.

The fixed abstention rule accepts 3,858 pairs (81.9456% coverage) at 79.3675% accepted accuracy and abstains on 850 pairs. It misses the declared 90% accuracy target at at least 80% coverage and is not retuned from holdout. Both the raw and frozen pipelines have zero AB/BA probability gap. The conditional 95% macro F1 interval is 0.723662–0.758584 from 1,000 paired document-component draws for this corpus and one seed.

See [measured results](../docs/results.md), [release summary](../reports/release-summary.json) and [bootstrap evidence](../reports/evaluation-bootstrap.json). Full fresh CPU replay and public download verification are separate delivery checks in the [reproduction guide](../docs/reproduction.md).

## Scope

The task predicts supplied same-author labels from English passage pairs. Document-disjoint evaluation does not establish new-author or new-domain generalization, and historical corpus exposure remains disclosed. The one-seed selection gain does not quantify training-seed uncertainty. Final heldout outcomes come from the separate frozen release evidence and are not inferred from selection scores. The historical development F1 0.7702 is not a controlled before/after comparator.
