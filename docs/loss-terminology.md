# Loss terminology and implementation

[한국어](./loss-terminology.ko.md) | [Project overview](../README.md) | [Measured results](./results.md) | [Architecture](./architecture.md)

The trained alternative to cross entropy is a **directly implemented binary ASL adaptation using class-dependent focal weighting**, named `ASL` and `ASLoss` in the original project and labeled custom focal in the reports. The public runtime keeps `custom_asl` for compatibility. This is an ASL-family adaptation; it is not an exact reproduction of the official library's default recipe or the paper's experiment.

## What the project implements

For different-author class `0` and same-author class `1`, let `p_y` be the true-class probability from a two-class softmax. The project uses `gamma_0=2` and `gamma_1=1`:

`L = mean(-(1 - p_y)^gamma_y * log(p_y))`

The focal factor reduces the contribution of easy examples, with a stronger reduction for class `0`. Gamma also appears in ordinary focal loss, as documented by [Torchvision's sigmoid focal loss](https://docs.pytorch.org/vision/main/generated/torchvision.ops.sigmoid_focal_loss.html). Gamma alone therefore does not identify ASL. Using different exponents for the binary positive and negative classes is the asymmetric-focusing part of this adaptation. It is not presented as a new loss method or a reproduction of the paper's experiment.

The [original notebook](../notebooks/roberta-asymmetric-loss/training.ipynb) names the class `ASLoss` and computes softmax probabilities followed by `log(max(p_y, 1e-8))`. The [maintained implementation](../nlu_authorship/losses.py) computes the true-class log probability directly with FP32 log-softmax. It retains the same focal mechanism and class exponents, while removing the original logarithm floor. Values below that floor can therefore differ; the implementations are not claimed to be numerically identical at extreme probabilities or in mixed precision.

## Relationship to official ASL variants

The authors' [official implementation](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py) provides both multi-label and single-label variants. ASL is not restricted to multi-label classification.

| Implementation | Probability and weighting |
|---|---|
| Official multi-label `AsymmetricLoss` | Independent sigmoid outputs, separate positive and negative focusing, and configurable negative-probability shifting |
| Official `ASLSingleLabel` | Softmax with positive and negative target masks and configurable label smoothing |
| This project's custom binary focal | Two-class softmax, weighting determined by the true class (`gamma_0=2`, `gamma_1=1`), and no negative-probability shifting or label smoothing |

Negative-probability shifting is configurable in official ASL, so its absence alone does not exclude an ASL variant. For a binary task, let `d = z_1 - z_0`. Then `sigmoid(d) = softmax([z_0, z_1])_1`. Configuring official `AsymmetricLoss` with `clip=0`, `gamma_neg=2`, `gamma_pos=1` and `disable_torch_grad_focal_loss=False` yields the same class-dependent focal objective for a single binary output, apart from its sum reduction versus this project's mean and logarithm floors at extreme probabilities. With matched normalization and outside the floors, these mathematical objectives coincide. This observation follows from the displayed formula and official code; it is not a new training experiment or a claim of bit-identical execution.

The project implements that binary adaptation directly rather than importing the official loss. Its settings and experiment differ from the official defaults and from `ASLSingleLabel`'s target-mask and label-smoothing recipe. The [ASL paper](https://openaccess.thecvf.com/content/ICCV2021/html/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.html) is method background. Neither a two-class softmax nor the presence of gamma alone establishes exact reproduction of a published recipe.

## What was trained and selected

The matched RoBERTa-Large study compared CE with this custom binary focal objective. Both completed six epochs and retained epoch-five checkpoints. On the fixed selection pipelines, CE scored macro F1 `0.801199` and custom focal `0.799145`; custom focal did not meet the predeclared `+0.002` replacement gate, so deployment remained CE.

On all 4,708 final pairs, the retained focal pipeline scored macro F1 `0.815170` and accuracy `81.5208%` (`3,838 / 4,708`). The frozen CE pipeline scored `0.808383` and `80.8624%` (`3,807 / 4,708`). The later holdout result did not change deployment selection. This full-pipeline difference includes inference-order and calibration choices and does not isolate a causal loss effect or demonstrate universal superiority.

## Preserved names and evidence

Historical notebook names, `ASLoss`, paths such as `Models_ASL`, and the runtime/model identifier `custom_asl` remain for compatibility and provenance. Frozen configurations, weights, predictions and metric reports remain unchanged. Reader-facing descriptions use **custom binary focal (direct binary ASL adaptation)** to connect the ASL name with the implemented objective without claiming an exact official-default reproduction. This terminology correction does not involve retraining or recalculating results.
