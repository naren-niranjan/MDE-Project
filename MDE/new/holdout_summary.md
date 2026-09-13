# Two holdout regimes

| file | camera | uncorrected | deployed | change |
|---|---|---|---|---|
| deployed_ltr.json | left | 221.40 | 223.06 | +1.66 |
| deployed_ltr.json | top | 187.12 | 181.06 | -6.06 |
| deployed_ltr.json | right | 195.02 | 202.66 | +7.64 |
| deployed_ltr_0831.json | left | 231.23 | 235.70 | +4.47 |
| deployed_ltr_0831.json | top | 237.82 | 243.02 | +5.20 |
| deployed_ltr_0831.json | right | 147.66 | 154.95 | +7.29 |
| depth_correction_fitted_deployed.json | center | 239.98 | 245.96 | +5.98 |
| depth_correction_fitted_deployed.json | left | 211.00 | 213.45 | +2.45 |
| depth_correction_fitted_deployed.json | top | 151.36 | 141.31 | -10.05 |
| depth_correction_fitted_deployed.json | right | 209.01 | 223.75 | +14.74 |

- records: 10, worsened in 8
- uncorrected held-out RMS spans 147.7 to 240.0 mm
- the correction changes it by -10.1 to +14.7 mm
- the height folds sit at 15.8 to 37.0 mm

Sentence for the text: on belt geometry the held-out RMS is 148 to 240\,mm before correction and the deployed map changes it by -10.1 to +14.7\,mm, worsening it in 8 of 10 camera-runs. That is an order of magnitude above the 15.8 to 37.0\,mm the same correction achieves on the board planes it was fitted against.
