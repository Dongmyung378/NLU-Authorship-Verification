# Controlled training on Kaggle

[한국어](./training.ko.md) | [Project overview](../README.md) | [Evaluation protocol](./evaluation.md) | [Measured results](./results.md)

The public CPU runtime replays saved artifacts. Transformer task training uses Kaggle GPUs; the stylometric study runs separately on an isolated local CPU environment. The original February to May 2026 RoBERTa-Large and stylometric notebooks remain historical joint work by Dongmyung Park and Juho Kim. The new protocol preserves both model families and refits them on the document-disjoint manifest. The completed Large capability probe is available under [`reports/capability/`](../reports/capability/). Both losses completed six matched epochs and saved epoch-five checkpoints; frozen final evaluation is complete for all three models. See the [measured results](./results.md) and [execution timing](../reports/execution-timing.json). All 21 model files are publicly available in Kaggle dataset version 1 and passed full anonymous HTTP 200, byte-count and SHA256 verification through the unchanged public CLI; see the [transport receipt](../reports/model-download-verification.json). All six full saved-model CPU inference executions in fresh Windows and Linux environments are complete; resource-cost observations are complete and linked in the [CPU summary](../reports/cpu-reproduction.json). They measure whole fresh CLI processes on one shared physical host; successful cost execution does not change strict numerical failures.

All six saved-model CPU inference executions are complete: all three models ran on all 4,708 locked-holdout pairs on Windows and Linux. Ordered IDs, class and acceptance decisions, counts, confusion matrices, F1, accuracy and frozen bindings matched the original references. The original selected-main floating metrics stayed within `1e-6`; this does not claim every diagnostic is bit-identical. The stylometric CPU-to-CPU checks passed. Both Large models failed the unchanged `1e-5` GPU-to-CPU probability checks. Custom-focal calibrated maximum differences were `1.80045428e-4` on Windows and `1.09622536e-4` on Linux; the CE failures are also preserved. Full strict numerical reproduction therefore remains failed despite classification agreement. Shared eager attention/FP32 inference is unchanged; device arithmetic drift is plausible, not a proven cause. CPU resource-cost observations are complete and linked in the [CPU summary](../reports/cpu-reproduction.json). They measure whole fresh CLI processes on one shared physical host; successful cost execution does not change strict numerical failures.

## Fixed recipe

The trained alternative uses [custom class-dependent binary focal](./loss-terminology.md), historically called ASL or ASLoss. The preserved `custom_asl` runtime name is a compatibility identifier, not an official-ASL reproduction claim.

[`configs/experiment-plan.json`](../configs/experiment-plan.json) declares the full experiment before task training. It pins `FacebookAI/roberta-large` revision `722cf37b1afa9454edce342e7895e588b6ff1d59` and pretrained safetensors SHA256 `047c85f0b96269cd62e6f732644f067004eebd95af5b5d35965ae2528f13bf38`. Its binary classifier has 355,361,794 parameters. Both losses start from the same fresh task initialization, rather than the historical task checkpoint.

The fixed shared recipe uses seed 378, combined length 512, microbatch 2, effective batch 32, maximum six matched epochs, learning rate `5e-6`, weight decay `0.01`, warmup ratio `0.1`, gradient norm cap `1.0` and gradient checkpointing. The completed throughput probe did not justify a larger microbatch; six epochs preserve the conservative budget for setup, selection, checkpointing, calibration, fixed evaluation and failure reserve. This cap was set without validation scores and is frozen before either loss starts. CE and custom focal use identical initial-state hashes and epoch schedules. The intended objective difference is the original class-dependent focal weighting, with gamma 2 for class 0 and gamma 1 for class 1.

Each epoch uses a deterministic shuffled order and swaps the two passages for approximately half of pairs without doubling rows. The swap decision is the low bit of the first SHA256 byte of `seed:epoch:pair_id`. Gradient accumulation weights microbatches by their actual count, including the final smaller effective batch. Mixed-precision forward passes use FP32 loss calculations. The report records overflow-skipped updates instead of treating them as successful optimizer steps.

## Observed Large capability

The separate private Kaggle profile completed on a T4 with Python 3.12.15, PyTorch 2.5.1+cu124 and CUDA 12.4. It attempted 40 effective-batch updates: 39 succeeded and one was skipped by the FP16 scaler. Median attempt duration was 5.136 seconds, peak allocated memory 6.09 GiB and peak reserved memory 6.37 GiB. There was no OOM retry. All profile weights were discarded; no model selection, calibration or final inference was performed.

That completed profile used an earlier three-epoch serial projection. Its 7.65-hour estimate with 30% margin is preserved as that projection, not reported as the measured duration of the later full protocol. Actual full-run time remains an observed result, not a guarantee from the profile.

## Bounded throughput probe

Before task training, a separate timing-only probe compared microbatch candidates 2, 8 and 32 while retaining effective batch 32. Each candidate attempted 20 optimizer updates on the same seeded train-record schedule. The predeclared protocol allowed one half-size OOM retry per candidate, required at least 1.05 timing speedup for adoption and capped reserved memory at 90% of device capacity. Every candidate used the same pinned fresh initialization, and every updated weight was discarded. No validation inference, model-performance comparison, calibration or locked-holdout inference was performed.

| Microbatch | Median attempt | Peak allocated | Successful updates | Overflow skips | OOM retries |
|---|---:|---:|---:|---:|---:|
| 2 | 4.9123 s | 6.09 GiB | 19 / 20 | 1 | 0 |
| 8 | 4.7953 s | 6.66 GiB | 19 / 20 | 1 | 0 |
| 32 | 5.4535 s | 7.94 GiB | 19 / 20 | 1 | 0 |

The probe completed in 316.29 seconds. Microbatch 8's 1.0244 speedup did not meet the 1.05 adoption gate, and 32 was slower, so microbatch 2 was retained. The measured bytes are preserved in [`throughput-profile.json`](../reports/capability/throughput-profile.json) and its [`summary`](../reports/capability/throughput-summary.json). These train-only resource measurements do not provide accuracy or select a loss. The cleaned-data manifest and source snapshot remain bound, and the completed v3 capability evidence remains unchanged.

## Matched parallel training and budget

The parent starts two independent loss workers on the two T4s already assigned to its Kaggle kernel. Each child sees only its assigned GPU. This is parallel training of separate CE and custom focal models, not data-parallel training of one model. No other kernel or GPU assignment is changed.

Workers wait at each completed epoch. The common patience is three consecutive epochs in which neither loss improves its raw AB selection macro F1 at threshold 0.5. An improvement by either loss resets the common patience. Both stop together at that rule or the frozen six-epoch cap, keeping equal completed epochs. Each loss reloads its own best checkpoint, with exact ties favoring the earlier epoch, to generate AB and BA selection logits. The training function does not fit calibration or infer on locked holdout.

Paired training is limited to nine wall hours and 18 hours summed across the two assigned workers. The summed worker duration includes preparation and barrier waiting while a GPU is assigned. Setup and release also consume session time, so the parent limits alone are not a full-session budget. This is a conservative local budget, not a verified claim about Kaggle's official T4x2 quota multiplier. The parent enforces both training limits. An incomplete or failed worker causes the pair to fail; partial outputs are preserved but no successful comparison or release is promoted.

The actual full kernel is also submitted with a 36,000-second session timeout. This bounds setup, paired training and release together to ten wall hours, or at most 20 hours summed across two assigned devices. The earlier completed probes have a separate conservative allowance of two assigned-device hours within the 22-hour ceiling. An interrupted release is a failure, not a completed evaluation.

## CPU stylometric study

`nlu_authorship.stylometric.STUDY_PLAN` fixes four recipes, at most eight XGBoost/LightGBM component fits, seed 378, two CPU threads and 120 minutes total. Word, character and function-word TF-IDF are fitted on train only. The three pilot recipes compare full, TF-IDF-only and style-only features with 500-tree ceilings. A full-feature recipe also retains the original larger ceilings of 7,000 XGBoost and 2,000 LightGBM trees. Selection-logloss early stopping and fit-time callbacks bound the fits.

The full-feature 50/50 pilot is the reference. Recipe/weight candidates need at least 0.002 selection macro F1 gain at threshold 0.5 to replace it. Final XGBoost weights are 0.25, 0.5 or 0.75; weights 0 and 1 remain single-component diagnostics. Calibration and locked holdout are not used. The chosen ensemble is saved with all TF-IDF transforms, configuration and file hashes, then reloaded to verify its selected scores and actual swapped-input invariance.

After CPU setup, run a new study from the project root. Initialize the two-thread CPU runtime before the study. On minimal Linux, this loads the OpenMP runtime bundled with the pinned Torch wheel before LightGBM, avoiding a global system-package installation.

Windows:

```powershell
.\.venv\Scripts\python.exe -I -c "from nlu_authorship.runtime import configure_cpu; configure_cpu(2); from nlu_authorship.stylometric import train_stylometric; train_stylometric('.', 'reports/split-manifest.json', '.local/stylometric')"
```

Linux:

```bash
.venv/bin/python -I -c "from nlu_authorship.runtime import configure_cpu; configure_cpu(2); from nlu_authorship.stylometric import train_stylometric; train_stylometric('.', 'reports/split-manifest.json', '.local/stylometric')"
```

Use absent model and sibling study directories. The saved CPU model is supplied to the full Kaggle package, so the GPU job does not refit this study.

## Package a run

The helper prepares exact byte inventories and private Kaggle metadata. It does not launch a GPU job. Use IDs belonging to your own account and a new output directory:

```powershell
.\.venv\Scripts\python.exe -I -m scripts.package_kaggle --output .local/kaggle-full-package --dataset-id YOURNAME/nlu-large-source --kernel-id YOURNAME/nlu-large-paired --stylometric-model .local/stylometric
```

Upload the generated `dataset/` as a private dataset and run the generated `kernel/` script as a private GPU kernel with Internet enabled and two T4s assigned. Authenticate through your own Kaggle tools; credentials do not belong in source archives. Kernel titles derive from the requested kernel slug so creation does not silently change the run identity.

A capability-only package uses `--profile-only`; a throughput-only package uses `--throughput-profile`. These modes are mutually exclusive and both reject `--stylometric-model`. Neither generated worker requests `--complete-release`. For example, this prepares only timing-probe input and does not upload or launch it:

```powershell
.\.venv\Scripts\python.exe -I -m scripts.package_kaggle --output .local/kaggle-throughput-package --dataset-id YOURNAME/nlu-throughput-source --kernel-id YOURNAME/nlu-large-throughput --throughput-profile
```

`nlu-source.zip` includes the exact executable snapshot, plan, split manifest and labeled CSVs. It excludes historical transformer weights, environments, Git metadata, historical notebooks, private logs and unlabeled inference inputs. The completed capability package was 23,780,198 bytes; each subsequent package records its own actual size and hash. `stylometric-model.zip` is a separate small, verified seven-file CPU model bundle under `NLU-Stylometric-Model`.

Kaggle may automatically expand an uploaded archive. The outer script accepts either the exact ZIP or one uniquely matching expanded directory, verifies every declared size and SHA256, rejects symlinks and unsafe paths, and copies only declared files. It then uses [`scripts/setup_kaggle_runtime.py`](../scripts/setup_kaggle_runtime.py) to obtain an isolated Python 3.12.15 CUDA runtime. Existing global Kaggle or other project environments are not reused or modified.

## Training and release stages

The generated worker invokes:

```bash
python -m scripts.run_kaggle_experiment \
  --repo-root /kaggle/working/nlu-source/NLU-Authorship-Verification \
  --manifest /kaggle/working/nlu-source/NLU-Authorship-Verification/reports/split-manifest.json \
  --output /kaggle/working/nlu-results \
  --plan /kaggle/working/nlu-source/NLU-Authorship-Verification/configs/experiment-plan.json \
  --stylometric-model-dir /kaggle/working/nlu-stylometric-model/NLU-Stylometric-Model \
  --complete-release
```

Here `python` means the isolated CUDA interpreter selected by the bootstrap. The worker verifies the CPU ensemble's artifact hashes, split binding, study protocol and executable source binding, then runs the matched transformers. `--complete-release` preserves raw loss comparison at threshold 0.5, chooses each input-order policy on selection and fits calibration independently for all three models. It then reuses saved selection logits with the fixed calibration parameters to choose the deployed Large pipeline, requiring at least 0.002 macro F1 gain over calibrated CE to adopt custom focal. Raw and calibrated-pipeline winners are reported separately; a calibrated gain includes loss, orientation and calibration.

This final choice needs no additional GPU inference and does not refit calibration on selection. All three models and final decisions are jointly frozen before any locked-holdout inference. Final results never reselect the loss, orientation, temperature or thresholds. This release protocol is declared before any measured full-run Large validation result is observed.

Preserve `experiment-completion.json`, `training-report.json`, worker receipts and logs, common stop decisions, per-model checkpoint reports, discarded profiles, runtime metadata, both input receipts, the CPU study reports and complete release evidence. Completion requires `release_completed=true`, matching evidence and no `experiment-failure.json`. A successful kernel exit alone is insufficient. CPU replay and anonymous public model-download verification remain separate checks after training.

The source package also includes the predeclared [`bootstrap helper`](../scripts/bootstrap_results.py). After frozen final predictions exist, it reports 1,000 paired whole-document-group draws with seed 378 and conditional 95% percentile intervals for accuracy, macro F1 and the calibrated custom-focal-minus-CE macro F1 difference. It loads no model and cannot refit or reselect the release. It measures corpus-level uncertainty for this one trained seed, not repeated-training or unseen-author/domain generalization.
