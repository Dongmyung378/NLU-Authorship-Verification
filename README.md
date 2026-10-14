# Authorship Verification

**A joint NLP project by Dongmyung Park and Juho Kim, developed in February to May 2026.**

[한국어](./README.ko.md) | [Reproduction](./docs/reproduction.md) | [Architecture](./docs/architecture.md) | [Evaluation](./docs/evaluation.md) | [Training](./docs/training.md)

This project compares the original two model families for English passage-pair authorship verification: a stylometric XGBoost-LightGBM ensemble and RoBERTa-Large. The new pipeline adds document-disjoint data splits, matched CE versus custom binary asymmetric focal training, independent calibration, immutable evaluation and hash-verified CPU replay. It predicts different-author (`0`) or same-author (`1`) labels; it does not establish a person's identity on arbitrary new text.

The model named **custom focal** in the reports uses the project's **custom class-dependent binary focal loss**, historically called `ASL` or `ASLoss`. Its runtime and model identifier remains `custom_asl` for compatibility. It reduces the contribution of already-correct, easy examples. It is a directly implemented binary ASL adaptation rather than an exact reproduction of the official defaults or paper experiment; both multi-label and single-label official variants exist. The [loss terminology and implementation](./docs/loss-terminology.md) explain the difference from those methods and the numerical-stability change from the original notebook.

## Current release status

Fresh RoBERTa-Large CE and custom focal training, the bounded train-only stylometric study and the frozen Kaggle evaluation have completed. Both Large models completed six matched epochs and retained epoch-five checkpoints. English and Korean [measured results](./docs/results.md) report the classification, calibration, order, length and abstention outcomes.

| Frozen pipeline, all 4,708 pairs | Macro F1 | Accuracy | Correct |
|---|---:|---:|---:|
| Stylometric XGBoost-LightGBM ensemble | 0.741498 | 74.1504% | 3,491 |
| RoBERTa-Large CE, deployed | 0.808383 | 80.8624% | 3,807 |
| RoBERTa-Large custom ASL (class-dependent binary focal), retained comparator | 0.815170 | 81.5208% | 3,838 |

CE was selected before holdout using the fixed calibrated pipelines on selection data. The later focal holdout advantage does not change that choice. Historical development F1 0.8348 uses different evaluation conditions and is not a before/after comparator. Calibration reduced ECE/NLL for all three models, but F1 did not improve uniformly and the 90% accepted-accuracy abstention goal was unmet.

The [joint freeze](./reports/evaluation-freeze.json) and ordered predictions were validated against source/data/model inventories. Implementation checks passed 109/109 on Windows 11 and 109/109 on Ubuntu 24.04 WSL2 with the same 35 executable/config files. All three saved models completed every one of the 4,708 pairs in both environments. All six runs matched binary predictions, abstention decisions, F1 and accuracy exactly. Stylometric replay passed the fixed comparison tolerances in both environments. Both Large models failed the predeclared GPU-reference probability tolerance of `1e-5`; maximum calibrated differences were `1.79e-5` and `8.81e-6` for CE on Windows and Linux, and `1.80e-4` and `1.10e-4` for custom focal. Linux CE passed the calibrated-probability check but failed the raw-probability check. The strict failures are preserved without changing the tolerance, reference or calibration. Matched CPU resource measurements have completed and are linked from the [CPU summary](./reports/cpu-reproduction.json) and [measured results](./docs/results.md); public model publication and full anonymous download verification have completed separately. The local environments share one physical host and are not independent external review.

Actual paired training took 6.3959 wall hours, with 12.7758 summed assigned-worker hours. The full wrapper through frozen evaluation took 7.5475 wall hours. The nine-hour paired and ten-hour complete-kernel limits were respected. Earlier timing probes discarded all updated weights; no further GPU quality experiment was launched after the bounded comparison. Local assigned-device scope and account quota readings are distinct, and no official T4x2 billing multiplier is asserted.

## Run the CPU interface

Automatic model download uses [version-1 raw files in the public Kaggle dataset](https://www.kaggle.com/datasets/dongmyungpark/nlu-authorship-frozen-large-20261012). All 21 files, totaling 2,856,189,574 bytes, were downloaded anonymously through the unchanged public CLI and matched their frozen byte counts and SHA256 hashes. The separate [download verification](./reports/model-download-verification.json) establishes artifact delivery, not another saved-model inference run. From the repository root in Windows PowerShell:

```powershell
.\launch-nlu.bat doctor
.\launch-nlu.bat verify-data
.\launch-nlu.bat download
.\launch-nlu.bat download --verify-only
.\launch-nlu.bat predict
```

Linux uses `sh ./launch-nlu.sh` with the same commands. Launchers create an isolated project-local Python 3.12 environment, install pinned CPU packages and hide local GPUs. Docker and API keys are unnecessary. Initial setup and model download require Internet; saved-model inference uses local weights and tokenizer files.

The default model is the CPU stylometric ensemble. Optional Large inference uses the pipeline chosen on selection data with independently calibration-fitted parameters:

```powershell
.\launch-nlu.bat download --model transformer
.\launch-nlu.bat predict --model transformer --text-file-1 passage-a.txt --text-file-2 passage-b.txt
```

The verified release descriptor, [reports/artifact-download.json](./reports/artifact-download.json), pins anonymous version-1 raw URLs, byte counts and SHA256 for every checkpoint, tokenizer, transform and configuration file. Flat remote files map directly to the original `models/ce/`, `models/custom_asl/` and `models/stylometric/` runtime paths using `kind=file`. Full anonymous HTTP 200/hash verification and the CLI verify-only check completed successfully. The historical Drive files below cannot substitute for this new release's inventories and freeze.

The automatic download and verification CLI continues to pin the existing version-1 raw-file URLs and SHA256 hashes. The [complete ZIP download](https://www.kaggle.com/api/v1/datasets/download/dongmyungpark/nlu-authorship-frozen-large-20261012?datasetVersionNumber=2) is provided by dataset version 2, which repackages the same 21 frozen model files. The separate [archive verification](./reports/model-archive-verification.json) records the downloaded ZIP's byte counts and SHA256 hashes. Models and evaluation results are unchanged. The complete ZIP preserves Kaggle's flat filenames. To install files under `artifacts/runtime/models/<model-name>/`, use the automatic download commands in this guide.

If model-only ZIPs are supplied separately, extract their `models/<model-name>/` contents into `artifacts/runtime` in this matching source checkout. The runtime verifies installed bytes against the joint freeze before prediction or evaluation; the automatic-download descriptor is unnecessary for a manual installation. The public automatic route above is the verified release distribution.

`doctor` verifies dependencies, environment isolation and a small CPU tensor operation. `verify-data` checks source CSV hashes and document isolation. Neither command measures real model accuracy. Prediction output distinguishes the class, calibrated same-author score and an optional `uncertain` decision after abstention. See the [CPU runtime guide](./docs/runtime.md) for input limits and all options.

## Replay the frozen final evaluation

Once the release artifacts are available, use a fresh output directory:

```powershell
.\launch-nlu.bat evaluate --model stylometric --model-name stylometric --model-dir artifacts/runtime/models/stylometric --output .local/replay-stylometric --batch-size 4 --threads 4
```

For both Large checkpoints:

```powershell
.\launch-nlu.bat download --model ce
.\launch-nlu.bat download --model custom_asl
.\launch-nlu.bat evaluate --model transformer --model-name ce --model-dir artifacts/runtime/models/ce --output .local/replay-ce --batch-size 4 --threads 4
.\launch-nlu.bat evaluate --model transformer --model-name custom_asl --model-dir artifacts/runtime/models/custom_asl --output .local/replay-custom-asl --batch-size 4 --threads 4
```

These full-replay examples explicitly use batch size 4 and four CPU threads. The setting was chosen from a small historical-model timing probe for practical CPU execution, before new-model quality outputs; it does not select a model or alter calibration. CLI defaults remain batch size 8 and one thread. Actual final-model CPU costs are reported in the [measured results](./docs/results.md); the separate benchmark uses batch size 1 and one thread.

The default evaluation covers all 4,708 locked-holdout pairs. It verifies source, data, model and calibration bindings, then saves metrics, confusion and ordered predictions without refitting. Both Large FP32 checkpoints are retained even when only one is selected for deployment. A small prediction or unit-test pass does not replace full saved-model replay. Clean operating systems on the same physical computer are reported as separate environments on the same host, not independent external hosts.

## Controlled experiment

The supplied 27,643 train and 5,993 development pairs are audited together. Cleanup excludes 86 empty-document rows and removes 2,162 consistent duplicate pairs, retaining 31,388 pairs. Connected components of shared document identity remain wholly inside one split:

| Role | Pairs | Use |
|---|---:|---|
| Train | 21,972 | Task fitting and train-only TF-IDF |
| Selection | 3,139 | Checkpoints, feature/weight choices and deployment policy |
| Calibration | 1,569 | Temperature, threshold and abstention fitting |
| Locked holdout | 4,708 | Final evaluation after the joint freeze |

Two independently assigned T4 workers train fresh pinned RoBERTa-Large with matched initialization, seed, records and epoch schedules. CE is compared with the custom two-class softmax focal objective: `-(1-p_y)^gamma_y * log(p_y)`, with `gamma_0=2` and `gamma_1=1`, calculated using FP32 log-softmax. The original notebook used an explicit `1e-8` probability floor before its logarithm; the current stable calculation retains the focal mechanism but can differ below that floor. This study trains the direct binary ASL adaptation, not an exact reproduction of the official default recipe or paper experiment.

A separate CPU study fits every TF-IDF transform on train only and compares four bounded XGBoost-LightGBM recipes within 120 minutes. Final ensemble weights keep both components active. A new recipe/weight must gain at least 0.002 selection macro F1 over the reference to be adopted.

Raw AB loss comparison stays at threshold 0.5. Each input-order policy is chosen on selection; all three models are calibrated independently. Deployment then compares the fixed calibrated Large pipelines on stored selection logits, adopting custom focal only with at least 0.002 macro F1 gain over calibrated CE. Raw and deployed winners are reported separately. The joint freeze precedes every heldout inference, and heldout outcomes never tune decisions.

The plan also declares 1,000 paired whole-document-group bootstrap draws with seed 378. Conditional 95% percentile intervals describe the frozen corpus and one trained seed, not repeated-training variation or new-author/domain generalization. Details are in the [evaluation protocol](./docs/evaluation.md).

## Historical development work

The original February to May 2026 notebooks, poster, saved predictions and model cards remain available with joint credit. These results use the original development split and are separate from the new frozen evaluation:

| Historical model | Reported development macro F1 | Historical artifact download |
|---|---:|---|
| [Stylometric ensemble](./model-cards/stylometric-ensemble.md) | 0.7702 | [Original ensemble and vectorizers on Drive](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing) |
| [RoBERTa-Large custom asymmetric focal](./model-cards/roberta-asymmetric-loss.md) | 0.8348 | [Original checkpoint and tokenizer on Drive](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing) |

The 5,993 development pairs include 630 exact training-pair duplicates. In total, 2,738 development pairs contain at least one document already present in train, including those duplicate pairs: 1,658 reuse one and 1,080 reuse both. The original stylometric notebook fits TF-IDF on combined train and development text, so development passages affect vocabulary and IDF. Development also selects checkpoints, ensemble weights and thresholds. The historical scores and new holdout results therefore cannot establish a controlled before-and-after gain. The new split separates document components and fits TF-IDF on train only.

[Original notebooks](./notebooks/) and [poster preview](./docs/authorship-verification-poster-preview.png) ([PDF](./docs/authorship-verification-poster.pdf?raw=1)) document the joint work. Claims about reduced cross-domain overfitting are not verified by an independent domain-separated evaluation. The 5,985-pair `test.csv` and 20-pair `live.csv` are unlabeled inference inputs and provide no measured accuracy.

## Repository layout

```text
nlu_authorship/             Data, models, training, calibration and frozen evaluation
configs/                   Predeclared experiment plan
reports/                   Split manifest, frozen measurements and reproduction evidence
scripts/                   Isolated setup, downloads, Kaggle packaging and reporting
launch-nlu.bat / .sh        Public CPU entrypoints
docs/                      English guides with Korean partners and historical poster
model-cards/               Clearly labeled original development artifacts
notebooks/                 Original joint training and demo notebooks
results/                   Historical unlabeled predictions
artifacts/                 Downloaded runtime models, excluded from Git
```

Author IDs and domain labels are absent. The documented scope is document separation in this supplied corpus; historical corpus exposure and unmeasured semantic near duplicates remain limitations. Long transformer inputs are truncated to 512 combined tokens, and short or unfamiliar passages may be unreliable.
