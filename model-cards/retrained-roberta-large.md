# Retrained RoBERTa-Large

[한국어](./retrained-roberta-large.ko.md) | [Measured results](../docs/results.md) | [Training](../docs/training.md) | [Historical model](./roberta-asymmetric-loss.md)

This release retains the original RoBERTa-Large architecture from Dongmyung Park and Juho Kim's February to May 2026 joint project. Two new task checkpoints start from the same pinned pretrained initialization and compare cross entropy with the original custom binary asymmetric focal objective. Historical fine-tuned weights are not reused.

## Architecture and matched training

| Item | Actual configuration |
|---|---|
| Base | `FacebookAI/roberta-large` |
| Pretrained revision | `722cf37b1afa9454edce342e7895e588b6ff1d59` |
| Task | English passage-pair same-author classification, two classes |
| Parameters | 355,361,794, saved as FP32 |
| Backbone | 24 layers, width 1,024, 16 attention heads |
| Maximum input | 512 combined tokens, longest-first truncation |
| Train / selection / calibration / holdout | 21,972 / 3,139 / 1,569 / 4,708 pairs |
| Training seed | 378, one seed |
| Microbatch / effective batch | 2 / 32 |
| Learning rate / weight decay | 0.000005 / 0.01 |
| Completed epochs | Six for each loss, common matched-epoch cap |
| Saved best checkpoint | Epoch five for each loss |
| Training precision | FP16 forward, FP32 loss, gradient checkpointing |
| Overflow-skipped updates | CE seven; custom focal six |
| Checkpoint weights | 1,421,495,360 bytes each, approximately 1.42 GB / 1.32 GiB |

Both workers share initialization hash `ffe7ba5e9107a846e6c9303dc914893535a441412c12fee323111b17a754acd3`, the same six recorded schedule fingerprints and optimizer recipe. Each uses a separately assigned T4. Shared epoch barriers and patience three ensure equal completed epochs; each loss keeps its own best selection checkpoint. Equal attempted schedules do not imply identical successful update counts when FP16 scaler overflow occurs.

Custom focal is `-(1-p_y)^gamma_y * log(p_y)` with `gamma_0=2` and `gamma_1=1` over two-class softmax probabilities. The runtime preserves the historical `custom_asl` model identifier. This implementation differs from the published multi-label sigmoid ASL with clipping, and comparing loss values of different objectives does not demonstrate improved accuracy.

## Selection and frozen evaluation

Before holdout, CE's fixed calibrated pipeline scored selection macro F1 0.801199 versus 0.799145 for custom focal. The alternative did not reach the required +0.002 gain, so the deployed transformer remains CE. Both models use selection-chosen symmetric mean-logit inference with independently fitted calibration parameters.

| Model | Raw AB F1 at 0.5 | Frozen macro F1 | Accuracy | Correct |
|---|---:|---:|---:|---:|
| CE, deployed | 0.807724 | 0.808383 | 80.8624% | 3,807 / 4,708 |
| Custom focal, retained comparator | 0.808153 | 0.815170 | 81.5208% | 3,838 / 4,708 |

These results cover every locked-holdout pair, including the binary predictions underlying abstentions. The later focal advantage does not change the frozen deployment choice. Neither score is a controlled before/after comparison with the original development F1 0.8348. Detailed orientation, calibration, slices and conditional uncertainty are in the [results](../docs/results.md) and [protocol](../docs/evaluation.md).

Calibration reduces CE ECE from 0.045364 to 0.024756 and focal ECE from 0.057828 to 0.033591. CE's accepted coverage is 82.3067% with accepted accuracy 86.3742%; focal's is 82.8802% with 86.6735%. Neither meets the declared 90% accepted-accuracy target at at least 80% coverage. Thresholds are not refitted from that result.

## Saved artifacts and resources

Each bundle contains seven frozen files: FP32 weights, model configuration, tokenizer, tokenizer configuration, training configuration, checkpoint report and artifact inventory. CE bundle size is 1,425,061,716 bytes and focal bundle size is 1,425,061,785 bytes. Their exact content inventories are bound in the [joint freeze](../reports/evaluation-freeze.json).

| Artifact | SHA256 |
|---|---|
| CE weights | `39c875dd559e076c24ed645da55d723cf4686898d175ecb3131f8f31766f45ab` |
| Focal weights | `647fe0e7c12d8c7b03dc4946cb8bf105fb866ccb1c41530da7947bc617e040a4` |
| CE bundle content | `a5f0c30f03f44bfc52acd8436a58594aac556bdf0fc28f1b285d345ad9970a5d` |
| Focal bundle content | `10ffa6bd384121062ab9acea49ecc5702bbd7bd15733dd2d4b7002e32f5b478e` |

The paired-training parent took 23,025.254 seconds, or 6.3959 wall hours. Summed assigned-worker time was 45,993.036 seconds, or 12.7758 hours, including preparation and synchronization waits. The full wrapper through frozen evaluation took 7.5475 wall hours. These local measurements do not attribute the account's official quota delta to this run. The bounded full-run GPU profile recorded 6.09 GiB allocated memory per worker; it does not measure the maximum across all epochs or CPU process memory. See the [training report](../reports/training-report.json) and [resource profiles](../reports/training/gpu-profiles.json).

Public model delivery and actual clean CPU replay were verified separately in the [reproduction guide](../docs/reproduction.md). Both Large checkpoints completed all 4,708 pairs on Windows and Ubuntu WSL2 with exact class, abstention, F1 and accuracy agreement. The unchanged `1e-5` GPU-reference probability comparison failed for both checkpoints on both platforms; the [CPU summary](../reports/cpu-reproduction.json) preserves these failures alongside completed latency and whole-process memory observations. The two local platforms share one physical host. Large symmetric inference requires two forward passes; the [measured results](../docs/results.md) compare its actual resource costs with the stylometric ensemble.

## Intended use and limits

The model predicts supplied English pair labels for research and portfolio demonstrations. Author and domain IDs are unavailable, historical corpus exposure remains disclosed, and semantic near-duplicate separation is unverified. The shorter-than-50-word CE slice has F1 0.741795; the ≥200-word slice has only five cases. Longest-first truncation affects 197 of 4,708 holdout pairs. One trained seed does not estimate training stability or support forensic identity claims on arbitrary writing.
