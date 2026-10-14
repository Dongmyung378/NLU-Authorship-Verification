# Portfolio visuals

[English](README.md) | [한국어](README.ko.md)

These Korean-labelled SVGs are the presentation assets used on the portfolio site. Their values come from the retained experiment records referenced, with checksums, in `visual-data.json`. They do not change trained models, data splits, selection decisions or scientific reports. Read the main project documentation for scope and limitations.

The renderer uses Python's standard library and the embedded display data only. Regenerating these visuals does not rerun a model or verify its scientific evaluation.

The model labelled **Large custom ASL** uses the original project loss named `ASLoss`: the class-dependent binary softmax focal objective `-(1-p_y)^gamma_y * log(p_y)`, with `gamma_0=2` for different authors and `gamma_1=1` for the same author. **It is a custom binary focal variant using ASL's asymmetric focusing; the experiment is distinct from reproducing the official ASL or `ASLSingleLabel` code with its default settings.** A single sigmoid of the two-logit difference represents the same binary probability, so softmax alone does not place this objective outside the ASL family. The runtime identifier `custom_asl` is preserved for compatibility. See the [official loss implementations](https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py) and [loss terminology](../../loss-terminology.md) for the configuration and numerical details. Only terminology and explanatory text have changed; the plotted measurements, geometry and palette are unchanged.

```bash
python docs/assets/portfolio/render_visuals.py --input docs/assets/portfolio/visual-data.json --output docs/assets/portfolio
```

`visual-receipt.json` binds the generated SVG bytes to the display data. The source-download ZIP includes these exact site assets. The website's technical diagrams describe implemented processing steps; they are not new benchmark results.

## Figures

- [nlu-thumbnail.svg](nlu-thumbnail.svg)
- [nlu-results.svg](nlu-results.svg)
- [nlu-calibration.svg](nlu-calibration.svg)
