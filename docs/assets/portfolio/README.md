# Portfolio visuals

[English](README.md) | [한국어](README.ko.md)

The figures below use English labels. Each is a presentation derivative of the retained Korean SVG with the same data geometry, palette and numerical values. In both language copies of the conceptual thumbnail, the illustrative connector now joins Text A and Text B before the model families, so both inputs are explicit. That declared annotation exception does not change any chart, measurement or retained source SVG. The underlying experiment records and checksums remain in `visual-data.json`. Models, data splits, selection decisions and scientific reports are unchanged.

The localization renderer uses Python's standard library, the source SVGs and explicit labels in `localization.json`. It does not run models or repeat evaluation. The historical joint notebooks and original English poster retain their original language and provenance.

The model labelled **Large custom ASL** uses the original project loss named `ASLoss`: the class-dependent binary softmax focal objective `-(1-p_y)^gamma_y * log(p_y)`, with `gamma_0=2` for different authors and `gamma_1=1` for the same author. **It is a custom binary focal variant using ASL's asymmetric focusing; the experiment is distinct from reproducing the official ASL or `ASLSingleLabel` code with its default settings.** A single sigmoid of the two-logit difference represents the same binary probability, so softmax alone does not place this objective outside the ASL family. The runtime identifier `custom_asl` is preserved for compatibility. See the [official loss implementations](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py) and [loss terminology](../../loss-terminology.md) for the configuration and numerical details. Only terminology and explanatory text have changed; the plotted measurements, geometry and palette are unchanged.

```bash
python docs/assets/portfolio/render_localized_visuals.py
python docs/assets/portfolio/render_localized_visuals.py --check
```

`localization-receipt.json` binds each English derivative to the retained Korean source SVG and SHA-256. The original Korean renderer and `visual-receipt.json` remain unchanged. The [Korean results](../../results.ko.md) use a separate Korean-label derivative of the English frozen-results PNG. Website assets and existing ZIPs are not overwritten by localization.

## Figures

- [Style and context model families](nlu-thumbnail.en.svg)
- [Frozen classification results](nlu-results.en.svg)
- [Probability calibration](nlu-calibration.en.svg)
