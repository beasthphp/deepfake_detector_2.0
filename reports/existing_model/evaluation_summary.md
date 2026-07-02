# Existing Model Evaluation Summary

## Model Load

- Model path: `D:\ddp\model\deepfake_detector_93acc.h5`
- File size: 178,116,384 bytes
- TensorFlow version: `2.20.0`
- Loaded with compile metadata: `True`
- Input shape: `[None, 256, 256, 3]`
- Output shape: `[None, 1]`
- Output activation: `sigmoid`
- Optimizer: `Adam`
- Loss: `binary_crossentropy`
- Stored metrics: `['compile_metrics', 'loss']`

## Metrics On Test Split

- Accuracy: 0.939700
- Balanced accuracy: 0.939700
- Precision fake: 0.929982
- Recall fake: 0.951000
- F1 fake: 0.940374
- Precision real: 0.949867
- Recall real: 0.928400
- F1 real: 0.939011
- ROC-AUC using fake score: 0.984720
- False positive rate: 0.071600
- False negative rate: 0.049000
- Confusion matrix labels: `['fake', 'real']`
- Confusion matrix: `[[9510, 490], [716, 9284]]`

False positive means a real face was incorrectly labelled fake. False negative means a fake face was incorrectly labelled real.

## Deterministic Individual Predictions

| Path | True label | Raw real_score | Fake score | Predicted label | Correct |
| --- | --- | ---: | ---: | --- | --- |
| test/real/30872.jpg | real | 0.999211 | 0.000789 | real | True |
| test/real/27074.jpg | real | 0.994382 | 0.005618 | real | True |
| test/real/17420.jpg | real | 0.998855 | 0.001145 | real | True |
| test/real/45130.jpg | real | 0.593680 | 0.406320 | real | True |
| test/real/48312.jpg | real | 0.987238 | 0.012762 | real | True |
| test/fake/6MJFQXMTQ1.jpg | fake | 0.161938 | 0.838062 | fake | True |
| test/fake/KDVBU1XEB9.jpg | fake | 0.000045 | 0.999955 | fake | True |
| test/fake/ADJWQO1GRK.jpg | fake | 0.622778 | 0.377222 | real | False |
| test/fake/YA9YTPHW20.jpg | fake | 0.000513 | 0.999487 | fake | True |
| test/fake/YQN0ECMOSQ.jpg | fake | 0.098759 | 0.901241 | fake | True |

## Score Distributions

```json
{
  "correctly_classified_real": {
    "count": 9284,
    "min": 0.500699520111084,
    "p05": 0.6763564586639405,
    "p25": 0.9087462425231934,
    "median": 0.979189395904541,
    "p75": 0.9969967305660248,
    "p95": 0.9998779445886612,
    "max": 1.0
  },
  "incorrectly_classified_real": {
    "count": 716,
    "min": 1.1011574088115594e-06,
    "p05": 0.008957857498899102,
    "p25": 0.10583329014480114,
    "median": 0.25305910408496857,
    "p75": 0.39307792484760284,
    "p95": 0.47891830652952194,
    "max": 0.49987706542015076
  },
  "correctly_classified_fake": {
    "count": 9510,
    "min": 1.3654386693686468e-19,
    "p05": 2.08394500544884e-09,
    "p25": 3.2863457590792677e-06,
    "median": 0.00025694641226436943,
    "p75": 0.009119651047512889,
    "p95": 0.20865921676158847,
    "max": 0.4984263777732849
  },
  "incorrectly_classified_fake": {
    "count": 490,
    "min": 0.5001067519187927,
    "p05": 0.5249419331550598,
    "p25": 0.6291656196117401,
    "median": 0.7435419857501984,
    "p75": 0.8864268064498901,
    "p95": 0.9815392166376113,
    "max": 0.9996793270111084
  }
}
```

## Validation-Based Uncertainty Probe

This is an analysis probe only; no permanent threshold is adopted here.

```json
{
  "basis": "validation split score distributions, not test-set threshold tuning",
  "candidate_uncertainty_ranges": [
    {
      "real_score_range": [
        0.45,
        0.55
      ],
      "uncertain_count": 303,
      "uncertain_fraction": 0.01515,
      "certain_count": 19697,
      "certain_fraction": 0.98485,
      "accuracy_on_certain": 0.945017007666142
    },
    {
      "real_score_range": [
        0.4,
        0.6
      ],
      "uncertain_count": 638,
      "uncertain_fraction": 0.0319,
      "certain_count": 19362,
      "certain_fraction": 0.9681,
      "accuracy_on_certain": 0.9517611816961058
    },
    {
      "real_score_range": [
        0.35,
        0.65
      ],
      "uncertain_count": 962,
      "uncertain_fraction": 0.0481,
      "certain_count": 19038,
      "certain_fraction": 0.9519,
      "accuracy_on_certain": 0.9567181426620444
    }
  ],
  "score_distributions": {
    "real_images_real_score": {
      "count": 10000,
      "min": 1.3179065838642146e-08,
      "p05": 0.37372613996267323,
      "p25": 0.8730768859386444,
      "median": 0.9736284613609314,
      "p75": 0.9962271153926849,
      "p95": 0.9998669743537902,
      "max": 1.0
    },
    "fake_images_real_score": {
      "count": 10000,
      "min": 9.956060623998233e-19,
      "p05": 2.0223466901114776e-09,
      "p25": 4.641148052542121e-06,
      "median": 0.00039703576476313174,
      "p75": 0.017193172127008438,
      "p95": 0.5226491957902903,
      "max": 0.9998692870140076
    }
  }
}
```
