# CPU runtime and saved-model verification

[한국어](./runtime.ko.md) | [Project overview](../README.md) | [Evaluation protocol](./evaluation.md) | [Measured results](./results.md)

Windows and Linux launchers support CPU inference and saved-model evaluation without Docker, a local GPU or API keys. They use a project-local 64-bit Python 3.12 environment with [`requirements-runtime.txt`](../requirements-runtime.txt). The default model is the smaller stylometric ensemble. A transformer request uses the deployed RoBERTa-Large pipeline selected on selection data with independently calibration-fitted parameters and recorded in the freeze. It may differ from the raw threshold-0.5 winner; final holdout scores do not choose either model.

Large capability and throughput probes are complete. The retained training recipe is microbatch 2, effective batch 32, maximum six matched epochs and common patience three, under the nine-hour paired wall/18-hour assigned-worker sum caps. Both losses completed six matched epochs. Final model artifacts and the [joint evaluation freeze](../reports/evaluation-freeze.json) are validated, and all three models completed frozen evaluation on 4,708 pairs; see the [measured results](./results.md). All 21 model files are publicly available in Kaggle dataset version 1 and passed full anonymous HTTP 200, byte-count and SHA256 verification through the unchanged public CLI; see the [transport receipt](../reports/model-download-verification.json). All six full saved-model CPU inference executions are complete in fresh Windows and Linux environments; resource-cost observations are complete and linked in the [CPU summary](../reports/cpu-reproduction.json). They measure whole fresh CLI processes on one shared physical host; successful cost execution does not change strict numerical failures. The commands below describe the supported release interface; setup or synthetic-test success alone does not establish that the final models have been replayed on either operating system.

All six saved-model CPU inference executions are complete: all three models ran on all 4,708 locked-holdout pairs on Windows and Linux. Ordered IDs, class and acceptance decisions, counts, confusion matrices, F1, accuracy and frozen bindings matched the original references. The original selected-main floating metrics stayed within `1e-6`; this does not claim every diagnostic is bit-identical. The stylometric CPU-to-CPU checks passed. Both Large models failed the unchanged `1e-5` GPU-to-CPU probability checks. Custom-focal calibrated maximum differences were `1.80045428e-4` on Windows and `1.09622536e-4` on Linux; the CE failures are also preserved. Full strict numerical reproduction therefore remains failed despite classification agreement. Shared eager attention/FP32 inference is unchanged; device arithmetic drift is plausible, not a proven cause. CPU resource-cost observations are complete and linked in the [CPU summary](../reports/cpu-reproduction.json). They measure whole fresh CLI processes on one shared physical host; successful cost execution does not change strict numerical failures. The separate anonymous transport verification for all 21 public model files is in the [delivery receipt](../reports/model-download-verification.json).

Measured CPU costs use the same first 32 selection records, one CPU thread, batch size one, five warmup predictions and three timed repetitions, sequentially on Windows and then Linux WSL2. The [Windows raw CE record](../reports/cpu-replay/windows/costs/ce.json) and [Linux raw CE record](../reports/cpu-replay/linux/costs/ce.json) include preprocessing, both frozen AB/BA forwards and calibration; model loading is separate. Whole-process resident peaks use Windows PeakWorkingSetSize and Linux GNU-time maxRSS. Windows PeakPagefileUsage is peak process commit charge, not measured disk swapping. These 32 nonrandom cases, OS memory accounting and one shared physical host limit performance comparisons. Both Large Windows-to-Linux CPU probability comparisons also failed the unchanged tolerance, with all classifications and acceptance decisions identical; the [CPU summary](../reports/cpu-reproduction.json) preserves those results separately from the original Kaggle comparisons.

## Set up and download

Public automatic model download is available without Kaggle credentials. The descriptor pins dataset version 1, and every downloaded file must match the frozen size and SHA256. Run these commands with the matching release source and descriptor.

The automatic download and verification CLI continues to pin the existing version-1 raw-file URLs and SHA256 hashes. The [complete ZIP download](https://www.kaggle.com/api/v1/datasets/download/dongmyungpark/nlu-authorship-frozen-large-20261012?datasetVersionNumber=2) is provided by dataset version 2, which repackages the same 21 frozen model files. The separate [archive verification](../reports/model-archive-verification.json) records the downloaded ZIP's byte counts and SHA256 hashes. Models and evaluation results are unchanged. The complete ZIP preserves Kaggle's flat filenames. To install files under `artifacts/runtime/models/<model-name>/`, use the automatic download commands in this guide.

Run these commands from the repository root in Windows PowerShell:

```powershell
.\launch-nlu.bat doctor
.\launch-nlu.bat download
.\launch-nlu.bat download --verify-only
.\launch-nlu.bat verify-data
```

The equivalent Linux commands are:

```bash
sh ./launch-nlu.sh doctor
sh ./launch-nlu.sh download
sh ./launch-nlu.sh download --verify-only
sh ./launch-nlu.sh verify-data
```

The first setup can download a managed Python runtime when no suitable Python 3.12 is found, then install pinned packages inside the project. It does not change global Python packages. A conflicting project environment is preserved under `.local/environment-backups/` before replacement. Setup requires Internet access; subsequent saved-model inference uses local tokenizer and model files. The launcher hides local GPUs and the runtime uses CPU explicitly.

`doctor` checks pinned package versions, isolation and an actual small CPU tensor operation. It does not load real weights or reproduce task accuracy. `download` verifies credential-free HTTPS files or ZIP bundles against declared byte counts, SHA256 and complete file inventories before installation. Multipart files additionally require every part hash and the reconstructed whole-file hash. `download --verify-only` checks installed files without fetching them again. `verify-data` verifies CSV bytes, source rows, split membership and executed normalized-document isolation. These checks have different purposes.

The release descriptor is [`reports/artifact-download.json`](../reports/artifact-download.json); it must bind the actual public URLs, complete model and tokenizer files, byte counts and SHA256 values. Delivery uses direct `kind=file` entries for anonymous raw files in a public model-only Kaggle dataset with an explicit version. Flat remote names map to the original runtime paths; model groups do not depend on extracting a nested archive. The current URLs and hashes passed actual publication and full anonymous HTTP 200/SHA256 verification. The private training dataset is not a public model endpoint. A missing descriptor or URL means that release is unavailable.

Files are installed under `artifacts/runtime/models/stylometric`, `ce` or `custom_asl`. Models are downloaded separately by group:

```powershell
.\launch-nlu.bat download --model transformer
.\launch-nlu.bat download --model ce
.\launch-nlu.bat download --model custom_asl
.\launch-nlu.bat download --model all
```

Use the same `--model` value with `--verify-only` to check a chosen group. `transformer` resolves the selection decision recorded in `reports/evaluation-freeze.json`. Complete evaluation needs the corresponding model group; downloading only the default ensemble does not install both Large transformers. Both FP32 checkpoints and their tokenizer/configuration files are delivered, even when only one loss is selected as the transformer default. Actual file sizes come from the descriptor; the two Large weight files are roughly 2.84 GB together before auxiliary files.

### Manual installation from an already obtained verified package

Three model-only ZIPs are prepared locally: `nlu-frozen-stylometric.zip`, `nlu-frozen-ce.zip` and `nlu-frozen-custom_asl.zip`. These optional manual ZIPs have no separate published links; automatic raw-file download is available. If you already have these verified packages, place them in the matching project root. Each contains seven immutable files under `models/<model-name>/`; extract it into `artifacts/runtime` after launcher setup:

```powershell
.\launch-nlu.bat doctor
.\.venv\Scripts\python.exe -I -m zipfile -e nlu-frozen-stylometric.zip artifacts/runtime
.\.venv\Scripts\python.exe -I -m zipfile -e nlu-frozen-ce.zip artifacts/runtime
.\.venv\Scripts\python.exe -I -m zipfile -e nlu-frozen-custom_asl.zip artifacts/runtime
```

Only the stylometric package is required for default use. The two Large packages are optional comparisons. Linux uses `.venv/bin/python -I -m zipfile -e` with the same archive and destination. This route does not require `reports/artifact-download.json`: `predict` and `evaluate` directly verify all model files against the joint freeze before loading them. The matching strengthened source and freeze are required; historical Drive models use a separate format. Public automatic download is available through the version-pinned descriptor.

## Predict a pair

```powershell
.\launch-nlu.bat predict
.\launch-nlu.bat predict --model transformer
.\launch-nlu.bat predict --text-file-1 passage-a.txt --text-file-2 passage-b.txt
.\launch-nlu.bat predict --model transformer --model-name ce --model-dir artifacts/runtime/models/ce --text-file-1 passage-a.txt --text-file-2 passage-b.txt
```

Interactive commands prompt for missing passages. For unattended use, provide both passages or both UTF-8 files. Each input file is limited to 2 MiB; the shared normalizer also rejects more than 1,000,000 characters, NUL characters and text that becomes empty after cleanup. Very short passages can be accepted by the program while still giving unreliable classifications. Transformer pairs are truncated to a combined maximum of 512 tokens.

JSON output distinguishes the numerical class prediction from `decision`, which is `same_author`, `different_authors` or `uncertain`. `abstained=true` makes the operational decision uncertain. `same_author_score` is the calibrated class-1 score; `confidence_for_predicted_class` is the score assigned to the threshold-selected class. The output includes the frozen threshold, temperature and inference method. Calibration on the supplied corpus does not turn a score on new authors into identity evidence.

The runtime checks the model directory and executable source against the evaluation freeze before loading saved artifacts, including before decoding the stylometric pickle files. Calibration stays in the frozen report, outside the immutable model inventory. Model and tokenizer loading are local after installation. Original RoBERTa-Large and stylometric Drive artifacts belong to the historical notebooks and are not interchangeable with this runtime's new frozen models.

## Repeat the final evaluation

Use a fresh output directory for each replay:

```powershell
.\launch-nlu.bat evaluate --model stylometric --model-name stylometric --model-dir artifacts/runtime/models/stylometric --output .local/replay-stylometric --batch-size 4 --threads 4
.\launch-nlu.bat evaluate --model transformer --model-name ce --model-dir artifacts/runtime/models/ce --output .local/replay-ce --batch-size 4 --threads 4
.\launch-nlu.bat evaluate --model transformer --model-name custom_asl --model-dir artifacts/runtime/models/custom_asl --output .local/replay-custom-asl --batch-size 4 --threads 4
```

These full-replay examples explicitly use batch size 4 and four CPU threads. The setting was chosen from a small historical-model timing probe for practical CPU execution, before new-model quality outputs; it does not select a model or alter calibration. CLI defaults remain batch size 8 and one thread. Actual final-model CPU costs are available in the [CPU summary](../reports/cpu-reproduction.json) and [measured results](./results.md); the separate benchmark uses batch size 1 and one thread.

The default split is `locked_holdout`, containing 4,708 document-disjoint pairs in the saved manifest. The default manifest and freeze are `reports/split-manifest.json` and `reports/evaluation-freeze.json`. Evaluation verifies source, data, artifact and calibration bindings and saves metrics, a confusion matrix and ordered predictions. It reloads released weights without fitting a temperature, threshold or model. Changing executable files or pinned configuration after the freeze intentionally invalidates that release's replay. Keep the released source snapshot intact when reproducing it.

Compare the saved outputs with the [report-only helper](./reference-code/compare_frozen_predictions.py). These commands use the launcher-created project-local `.venv` interpreter, without relying on a global `python` command. An existing Python 3.12 interpreter can also run this standard-library checker:

```powershell
.\.venv\Scripts\python.exe -I docs/reference-code/compare_frozen_predictions.py --model-name stylometric --evaluation-dir .local/replay-stylometric --output .local/compare-stylometric.json
.\.venv\Scripts\python.exe -I docs/reference-code/compare_frozen_predictions.py --model-name ce --evaluation-dir .local/replay-ce --output .local/compare-ce.json
.\.venv\Scripts\python.exe -I docs/reference-code/compare_frozen_predictions.py --model-name custom_asl --evaluation-dir .local/replay-custom-asl --output .local/compare-custom-asl.json
```

Linux uses `.venv/bin/python -I` with the same arguments. The helper verifies all 4,708 ordered IDs, prediction-file hashes and frozen source, manifest, model, calibration and class bindings. It requires exact discrete predictions, acceptance, groups, lengths, counts, F1 and accuracy, while using fixed absolute tolerances of `1e-5` for probabilities and `1e-6` for other floating metrics. It writes a new comparison JSON and exits 0 on pass or 1 on failure. The reference stylometric evaluation ran on Kaggle CPU and Large on CUDA/T4; JSON byte identity is not required. The helper loads no model and neither adjusts a threshold nor verifies public downloading. The saved comparison can fail its numerical limits even when every class and acceptance decision matches. Preserve the failure receipt and the original reference; agreement of task scores does not override `status=failed`.

Full replay must execute the actual saved models on all fixed records and compare metrics and ordered outputs against the release reports. A successful `doctor`, verified download, single prediction or tiny model test does not replace that check. CPU evaluation of both Large checkpoints is substantial work, with hardware-dependent time and memory use; no speed or memory guarantee is inferred from the GPU capability probe.

After all three frozen prediction reports exist, [`scripts/bootstrap_results.py`](../scripts/bootstrap_results.py) can replay the predeclared uncertainty calculation without loading models. It requires the manifest, freeze, completed release summary and three ordered prediction files, checks their bindings, and performs 1,000 paired whole-document-group draws with seed 378. Its 95% percentile intervals concern this corpus and one trained seed; they do not verify new authors/domains or repeated-training stability. The calculation does not fit thresholds or choose the deployed model. All six full saved-model CPU inference executions are complete using exact local frozen artifacts. Both environments share one physical computer, so they are separate clean environments on the same host rather than independent third-party machines.

## Tests and timing

After setup, run synthetic correctness tests in Windows:

```powershell
.\.venv\Scripts\python.exe -I -m unittest discover -s tests -v
```

On Linux, use `.venv/bin/python -I -m unittest discover -s tests -v`. These tests check implementation properties and tiny saved-model round trips, not the real dataset's final accuracy.

```powershell
.\launch-nlu.bat benchmark --model stylometric --model-name stylometric --model-dir artifacts/runtime/models/stylometric --output .local/benchmark-stylometric.json
.\launch-nlu.bat benchmark --model transformer --model-name ce --model-dir artifacts/runtime/models/ce --output .local/benchmark-ce.json
```

Timing uses declared selection examples with batch size 1, five warmup calls, three repetitions and one CPU thread by default. It includes normalization, tokenization or feature extraction, the frozen inference policy and calibration. Model loading is reported separately. Compare fresh processes on the same hardware and samples. A symmetric policy contains two forward passes; compare that cost explicitly. Only observed environment and actual saved-model replay reports support a claim that a release was verified there.
