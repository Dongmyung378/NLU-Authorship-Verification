# Reproduction guide

[한국어](./reproduction.ko.md) | [Project overview](../README.md) | [Runtime](./runtime.md) | [Evaluation](./evaluation.md) | [Measured results](./results.md)

This guide replays the new frozen document-disjoint experiment from saved artifacts. The default model is the CPU stylometric ensemble; both RoBERTa-Large checkpoints are optional downloads. Windows and Linux use isolated pinned Python 3.12 CPU environments without Docker, API keys or local GPU setup. Retraining is a separate Kaggle procedure.

**Release status:** capability and throughput probes are complete, with microbatch 2 and a maximum of six matched epochs retained. Implementation checks passed 109/109 on Windows 11 and 109/109 on Ubuntu 24.04 WSL2 against the same 35 executable/config files and isolated Python 3.12.15 CPU environments. They share one physical host and do not represent external third-party review. Both Large losses completed six matched epochs, and all three models completed frozen evaluation on 4,708 locked-holdout pairs. [Measured results](./results.md) and [execution timing](../reports/execution-timing.json) record the completed run. All six full saved-model CPU inference executions using exact local frozen artifacts are complete; resource-cost observations are complete and linked in the [CPU summary](../reports/cpu-reproduction.json). They measure whole fresh CLI processes on one shared physical host; successful cost execution does not change strict numerical failures. All 21 model files are publicly available in Kaggle dataset version 1 and passed full anonymous HTTP 200, byte-count and SHA256 verification through the unchanged public CLI; see the [transport receipt](../reports/model-download-verification.json). The following commands describe the finalized interface; passing synthetic tests is not a full saved-model evaluation.

All six saved-model CPU inference executions are complete: all three models ran on all 4,708 locked-holdout pairs on Windows and Linux. Ordered IDs, class and acceptance decisions, counts, confusion matrices, F1, accuracy and frozen bindings matched the original references. The original selected-main floating metrics stayed within `1e-6`; this does not claim every diagnostic is bit-identical. The stylometric CPU-to-CPU checks passed. Both Large models failed the unchanged `1e-5` GPU-to-CPU probability checks. Custom-focal calibrated maximum differences were `1.80045428e-4` on Windows and `1.09622536e-4` on Linux; the CE failures are also preserved. Full strict numerical reproduction therefore remains failed despite classification agreement. Shared eager attention/FP32 inference is unchanged; device arithmetic drift is plausible, not a proven cause. CPU resource-cost observations are complete and linked in the [CPU summary](../reports/cpu-reproduction.json). They measure whole fresh CLI processes on one shared physical host; successful cost execution does not change strict numerical failures. The separate anonymous transport verification for all 21 public model files is in the [delivery receipt](../reports/model-download-verification.json).

Measured CPU costs use the same first 32 selection records, one CPU thread, batch size one, five warmup predictions and three timed repetitions, sequentially on Windows and then Linux WSL2. The [Windows raw CE record](../reports/cpu-replay/windows/costs/ce.json) and [Linux raw CE record](../reports/cpu-replay/linux/costs/ce.json) include preprocessing, both frozen AB/BA forwards and calibration; model loading is separate. Whole-process resident peaks use Windows PeakWorkingSetSize and Linux GNU-time maxRSS. Windows PeakPagefileUsage is peak process commit charge, not measured disk swapping. These 32 nonrandom cases, OS memory accounting and one shared physical host limit performance comparisons. Both Large Windows-to-Linux CPU probability comparisons also failed the unchanged tolerance, with all classifications and acceptance decisions identical; the [CPU summary](../reports/cpu-reproduction.json) preserves those results separately from the original Kaggle comparisons.

## 1. Preserve the released source

Use the exact released source snapshot and its `reports/split-manifest.json`. Keep executable files, pinned requirements and the experiment plan intact: the evaluation freeze binds them, and changing them invalidates that release's replay. Do not regenerate splits or fit new thresholds to reproduce the recorded result.

The labeled `data/train.csv` and `data/dev.csv` are source inputs to the new manifest, not its current train/selection partitions. The manifest retains 31,388 pairs: 21,972 train, 3,139 selection, 1,569 calibration and 4,708 locked holdout. Identity-connected document groups are disjoint under all declared normalization views. The source has no author IDs or domain labels; this does not establish unseen-author or cross-domain generalization.

## 2. Set up a clean CPU environment

Run from the repository root in Windows PowerShell:

```powershell
.\launch-nlu.bat doctor
.\launch-nlu.bat verify-data
```

Linux equivalents:

```bash
sh ./launch-nlu.sh doctor
sh ./launch-nlu.sh verify-data
```

The launcher obtains a suitable 64-bit Python 3.12 runtime, creating a managed one if necessary, then installs `requirements-runtime.txt` in the local `.venv`. It uses no system-site packages and changes no global Python installation. An incompatible existing environment is preserved under `.local/environment-backups/` before replacement. Initial setup needs Internet; it hides local GPUs and uses the pinned CPU PyTorch build.

`doctor` checks package pins, isolation and an actual small CPU tensor operation. It does not check model bytes or accuracy. `verify-data` checks CSV hashes, original rows, split roles and actual normalized-document isolation. Keep the resulting environment report when recording a reproduction.

## 3. Install the selected artifacts

### Manual installation from an already obtained verified package

Three model-only ZIPs are prepared locally: `nlu-frozen-stylometric.zip`, `nlu-frozen-ce.zip` and `nlu-frozen-custom_asl.zip`. These optional manual ZIPs have no separate published links; automatic raw-file download is available. If you already have these verified packages, place them in the matching project root. Each contains seven immutable files under `models/<model-name>/`; extract it into `artifacts/runtime` after launcher setup:

```powershell
.\launch-nlu.bat doctor
.\.venv\Scripts\python.exe -I -m zipfile -e nlu-frozen-stylometric.zip artifacts/runtime
.\.venv\Scripts\python.exe -I -m zipfile -e nlu-frozen-ce.zip artifacts/runtime
.\.venv\Scripts\python.exe -I -m zipfile -e nlu-frozen-custom_asl.zip artifacts/runtime
```

Only the stylometric package is required for default use. The two Large packages are optional comparisons. Linux uses `.venv/bin/python -I -m zipfile -e` with the same archive and destination. This route does not require `reports/artifact-download.json`: `predict` and `evaluate` directly verify all model files against the joint freeze before loading them. The matching strengthened source and freeze are required; historical Drive models use a separate format. Public automatic download is available through the version-pinned descriptor.



Use the verified public descriptor included in this release:

```powershell
.\launch-nlu.bat download
.\launch-nlu.bat download --verify-only
```

The default installs `artifacts/runtime/models/stylometric/`. [`reports/artifact-download.json`](../reports/artifact-download.json) uses `kind=file` entries with anonymous, version-pinned raw URLs in a public model-only Kaggle dataset. Flat remote names such as `ce--model.safetensors` map to their original local runtime paths such as `models/ce/model.safetensors`. All 21 files passed full anonymous HTTP 200, byte-count and SHA256 checks; the [delivery receipt](../reports/model-download-verification.json) preserves the actual CLI evidence.

The automatic download and verification CLI continues to pin the existing version-1 raw-file URLs and SHA256 hashes. The [complete ZIP download](https://www.kaggle.com/api/v1/datasets/download/dongmyungpark/nlu-authorship-frozen-large-20261012?datasetVersionNumber=2) is provided by dataset version 2, which repackages the same 21 frozen model files. The separate [archive verification](../reports/model-archive-verification.json) records the downloaded ZIP's byte counts and SHA256 hashes. Models and evaluation results are unchanged. The complete ZIP preserves Kaggle's flat filenames. To install files under `artifacts/runtime/models/<model-name>/`, use the automatic download commands in this guide.

Every model, transform, tokenizer, configuration and inventory file must match its exact byte count and SHA256. The downloader stages verified files before installation, preserves existing destinations and rejects unsafe paths or symlinks. A model's own inventory and the trusted evaluation freeze are checked again before inference, including before decoding stylometric pickle files.

Optional groups:

```powershell
.\launch-nlu.bat download --model transformer
.\launch-nlu.bat download --model ce
.\launch-nlu.bat download --model custom_asl
.\launch-nlu.bat download --model all
.\launch-nlu.bat download --model all --verify-only
```

`transformer` resolves the calibrated pipeline chosen on selection and bound in the freeze. Both Large FP32 checkpoints are available for comparative replay; together their weights are about 2.84 GB before auxiliary files. Actual sizes come from the descriptor. Downloading the default ensemble does not install either transformer.

## 4. Predict two passages

```powershell
.\launch-nlu.bat predict
.\launch-nlu.bat predict --text-file-1 passage-a.txt --text-file-2 passage-b.txt
.\launch-nlu.bat predict --model transformer --text-file-1 passage-a.txt --text-file-2 passage-b.txt
```

Linux uses `sh ./launch-nlu.sh` with the same arguments. Supply both passages or both UTF-8 files for unattended use. Inputs are limited to 2 MiB per passage and 1,000,000 characters by the normalizer; NUL and empty-after-cleanup text are rejected. RoBERTa caps the combined pair at 512 tokens with longest-first truncation.

The output reports the numeric class, calibrated same-author score, frozen temperature, threshold and inference policy. `decision=uncertain` means the frozen abstention rule declined to decide. A calibrated score from this corpus is not proof of identity on new text. All installed model and tokenizer files are loaded locally.

## 5. Replay all fixed final records

Use an absent output directory for each model:

```powershell
.\launch-nlu.bat evaluate --model stylometric --model-name stylometric --model-dir artifacts/runtime/models/stylometric --output .local/replay-stylometric --batch-size 4 --threads 4
.\launch-nlu.bat evaluate --model transformer --model-name ce --model-dir artifacts/runtime/models/ce --output .local/replay-ce --batch-size 4 --threads 4
.\launch-nlu.bat evaluate --model transformer --model-name custom_asl --model-dir artifacts/runtime/models/custom_asl --output .local/replay-custom-asl --batch-size 4 --threads 4
```

These full-replay examples explicitly use batch size 4 and four CPU threads. The setting was chosen from a small historical-model timing probe for practical CPU execution, before new-model quality outputs; it does not select a model or alter calibration. CLI defaults remain batch size 8 and one thread. Actual final-model CPU costs are available in the [CPU summary](../reports/cpu-reproduction.json) and [measured results](./results.md); the separate benchmark uses batch size 1 and one thread.

The default split is the entire 4,708-pair `locked_holdout`. Defaults bind `reports/split-manifest.json` and `reports/evaluation-freeze.json`. Evaluation reloads fixed weights and calibration without training or decision fitting, saving `metrics.json`, `confusion.json` and ordered `predictions.jsonl`. Compare support, correct counts, confusion, calibrated metrics, coverage and ordered predictions against the release reports. Retain model, source, data and freeze hashes with the observed OS/Python/package versions.

After evaluation, automate the comparison with the [report-only helper](./reference-code/compare_frozen_predictions.py). The commands use the project-local `.venv` interpreter created by the launcher; no global `python` command is needed. This standard-library checker can also use an existing Python 3.12 interpreter. Use a new comparison JSON path each time:

```powershell
.\.venv\Scripts\python.exe -I docs/reference-code/compare_frozen_predictions.py --model-name stylometric --evaluation-dir .local/replay-stylometric --output .local/compare-stylometric.json
.\.venv\Scripts\python.exe -I docs/reference-code/compare_frozen_predictions.py --model-name ce --evaluation-dir .local/replay-ce --output .local/compare-ce.json
.\.venv\Scripts\python.exe -I docs/reference-code/compare_frozen_predictions.py --model-name custom_asl --evaluation-dir .local/replay-custom-asl --output .local/compare-custom-asl.json
```

On Linux, use `.venv/bin/python -I` with the same helper and arguments. It checks every ordered ID, prediction-file SHA256 and manifest, freeze, executable source, model inventory, calibration and class binding. Labels, predictions, acceptance, groups, lengths, counts, F1 and accuracy must match exactly. Probability differences allow at most `1e-5`; other floating metrics allow at most `1e-6`. These fixed tolerances cannot be changed through arguments. Exit code 0 means pass; 1 means failure, with an auditable JSON receipt. It does not overwrite an existing receipt or refit any decision. The Kaggle reference used CPU for the stylometric ensemble and CUDA/T4 for Large; numerical comparison does not require identical JSON bytes across devices. Report comparison performs no inference and does not establish public download availability. A numerical failure remains a failure even if classifications and accepted cases agree; retain the failed receipt and original reference without changing the tolerances.

The raw AB comparison at threshold 0.5 and the deployed calibrated-pipeline choice are separate reports. Both are fixed before final evaluation; evaluating the alternative Large checkpoint does not authorize choosing a different model from its heldout score.

All six full saved-model CPU inference executions in fresh Windows and Linux environments are complete using exact local frozen artifacts. This local-origin inference evidence remains separate from the completed public anonymous transport verification linked above. When both environments share one physical computer, describe them as clean environments on the same host, not independent third-party machines. CPU Large evaluation has hardware-dependent time and memory requirements; the T4 probe does not benchmark local CPU inference.

## 6. Check correctness, timing and uncertainty

Synthetic tests after setup:

```powershell
.\.venv\Scripts\python.exe -I -m unittest discover -s tests -v
```

Linux uses `.venv/bin/python -I -m unittest discover -s tests -v`. These cover grouping, loss/calibration behavior, artifact integrity and small round trips. Actual-model replay above is a distinct check.

```powershell
.\launch-nlu.bat benchmark --model stylometric --model-name stylometric --model-dir artifacts/runtime/models/stylometric --output .local/benchmark-stylometric.json
.\launch-nlu.bat benchmark --model transformer --model-name ce --model-dir artifacts/runtime/models/ce --output .local/benchmark-ce.json
```

Default timing uses 32 fixed selection examples, one CPU thread, batch size 1, five warmups and three repetitions. It includes preprocessing, tokenization/features, selected orientation and calibration; loading is reported separately. Symmetric Large inference makes two forward passes. Compare fresh processes on the same hardware and examples.

The predeclared [bootstrap helper](../scripts/bootstrap_results.py) takes `--manifest`, `--freeze`, `--summary`, `--prediction-root` and a new `--output` path. Point it at the completed release summary and the directory containing all three final prediction reports. It verifies their exact IDs/hashes, then uses 1,000 paired whole-document-component draws with seed 378 for conditional 95% percentile intervals. No model is loaded, no decision is refitted and no winner is reselected. These intervals condition on this corpus and one seed, not new authors, domains or training-seed variation.

Figure generation is optional and uses Matplotlib in a separate environment; it is not installed by the model launcher. After the completed release report exists, Windows can run:

```powershell
.\.venv\Scripts\python.exe -m venv .local/plotting-env
.\.local\plotting-env\Scripts\python.exe -m pip install -r docs/reference-code/requirements-plotting.txt
.\.local\plotting-env\Scripts\python.exe docs/reference-code/plot_frozen_results.py --summary reports/release-summary.json --output .local/frozen-evaluation.png
```

On Linux, use `.venv/bin/python` to create the environment and `.local/plotting-env/bin/python` for the remaining commands. Plotting reads the saved report without loading models, recalibrating or changing their CPU packages. The release report and model evaluation must already be complete; the figure does not substitute for them.

For a new matched training experiment, use the [Kaggle training guide](./training.md). It preserves RoBERTa-Large, runs the bounded stylometric study separately on CPU and packages its verified model for the two assigned T4 loss workers.

## Historical notebooks and Drive artifacts

The original joint work from February to May 2026 is preserved under `notebooks/`, with its [poster](./authorship-verification-poster-preview.png) and [historical model cards](../model-cards/). These files describe the earlier development procedure and are not the new runtime release or evaluation protocol. Original notebooks are unchanged; notebook dependencies and historical demos should not replace the new pinned CPU environment.

| Historical artifact | Original download | Historical local folder |
|---|---|---|
| Stylometric ensemble | [Original models and vectorizers](https://drive.google.com/drive/folders/1mF0bMpzfs3XODpQA2Hd7H_JvXLk5AGgp?usp=sharing) | `models/stylometric-ensemble/` or `models/Models_ML/` |
| RoBERTa-Large custom asymmetric focal | [Original checkpoint and tokenizer](https://drive.google.com/drive/folders/1Ty9-EpOLAgzvTWPuqrHBTbehmZZuDs3V?usp=sharing) | `models/roberta-asymmetric-loss/` or `models/Models_ASL/` |

The original 5,993 development pairs contain 630 exact training-pair duplicates. Including those duplicates, 2,738 pairs reuse at least one train document (1,658 one-document and 1,080 two-document reuse). The historical stylometric notebook fits TF-IDF on combined train and development passages, influencing vocabulary and IDF. Checkpoints, weights and thresholds also use development. Its scores are development results, not an independent comparator for new holdout performance.

The original custom binary focal was historically named ASL or ASLoss. It directly implements a binary ASL adaptation using two-class softmax focal weighting with gamma 2/1; it does not reproduce the official default recipes or paper experiment. The [terminology guide](./loss-terminology.md) describes the numerical-stability change and compatibility names. The saved 5,985 test predictions and 20 live pairs have no labels, so they provide no accuracy or F1. Document separation, train-only TF-IDF and frozen evaluation improve the new protocol's validity; they do not erase historical corpus exposure or prove unseen-author/domain generalization.
