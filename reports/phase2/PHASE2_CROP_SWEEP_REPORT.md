# Phase 2.1 Crop Sweep Report

## A. Strategies Tested

- `original_reference`: complete original image, reference only.
- `square_m20_current`: Phase 2 baseline square crop with 20% margin.
- `square_m30`, `square_m40`, `square_m50`, `square_m65`, `square_m80`: larger square margins.
- `context_full_head_l40_r40_t70_b35_square`: contextual square crop with extra top area.
- `context_l40_r40_t70_b35_stretch`: contextual non-square crop stretched to 256x256.
- `padded_context_black`, `padded_context_reflect`, `padded_context_edge`: contextual crop resized with aspect preservation and padding.
- `preserve_or_context_full_head_l40_r40_t70_b35_square`: prepared-face bypass, otherwise `context_full_head_l40_r40_t70_b35_square`.

The prepared-face bypass requires a single detected face, near-square image, centered face, high face occupancy, and a box inside image boundaries. A 256x256 image is only one supporting reason, never sufficient by itself.

## B. Fifty-Image Sweep

The first sweep reused the exact 50 records from `reports/phase2/crop_consistency.json`.

| Rank | Strategy | F->R flips | Class changes | Fake recall | Median diff | Failure rate | Avg ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `context_full_head_l40_r40_t70_b35_square` | 0 | 0 | 0.960000 | 0.000000 | 0.000000 | 27.159000 |
| 2 | `preserve_or_context_full_head_l40_r40_t70_b35_square` | 0 | 0 | 0.960000 | 0.000000 | 0.000000 | 27.323000 |
| 3 | `square_m40` | 0 | 0 | 0.960000 | 0.000000 | 0.000000 | 27.482000 |
| 4 | `square_m50` | 0 | 0 | 0.960000 | 0.000000 | 0.000000 | 27.516000 |
| 5 | `square_m65` | 0 | 0 | 0.960000 | 0.000000 | 0.000000 | 27.541000 |
| 6 | `square_m80` | 0 | 0 | 0.960000 | 0.000000 | 0.000000 | 27.887000 |
| 7 | `original_reference` | 0 | 0 | 0.960000 | 0.000000 | 0.000000 | 28.208000 |
| 8 | `padded_context_edge` | 0 | 0 | 0.960000 | 0.000000 | 0.000000 | 28.096000 |
| 9 | `padded_context_reflect` | 0 | 0 | 0.960000 | 0.000000 | 0.000000 | 27.455000 |
| 10 | `context_l40_r40_t70_b35_stretch` | 0 | 0 | 0.960000 | 0.000000 | 0.000000 | 27.339000 |
| 11 | `padded_context_black` | 0 | 0 | 0.960000 | 0.000000 | 0.000000 | 27.593000 |
| 12 | `square_m30` | 0 | 0 | 0.960000 | 0.000000 | 0.000000 | 27.334000 |
| 13 | `square_m20_current` | 14 | 16 | 0.400000 | 0.057793 | 0.000000 | 28.435000 |

Best non-reference fallback from the 50-image sweep: `context_full_head_l40_r40_t70_b35_square`.
Top ranked strategy including the reference: `context_full_head_l40_r40_t70_b35_square`.

Row-level predictions are saved in `reports/phase2/crop_sweep/crop_sweep_predictions.csv`.

## C. Expanded Validation

The top contextual crop and the prepared-face conditional candidate from the 50-image sweep were evaluated on 100 real and 100 fake validation images selected with seed 69.

| Rank | Strategy | F->R flips | Class changes | Fake recall | Median diff | Failure rate | Avg ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `preserve_or_context_full_head_l40_r40_t70_b35_square` | 0 | 0 | 0.959596 | 0.000000 | 0.050000 | 27.050000 |
| 2 | `context_full_head_l40_r40_t70_b35_square` | 0 | 0 | 0.959596 | 0.000000 | 0.050000 | 27.714000 |

Expanded row-level predictions are saved in `reports/phase2/crop_sweep/expanded_predictions.csv`.

## D. Failure Analysis

No fake-to-real flips occurred in the top three ranked strategies, so no failure-example folders were required.

Failure examples are saved under `phase2_samples/crop_sweep/` with original, detected-box, crop, model-input, and JSON result files.

## E. Detector Assessment

Haar was sufficient for this prepared-face crop-sweep sample. The dominant issue is crop/model sensitivity, not clear detector-box failure. YuNet should not be added yet.

## F. Selected Production Candidate

Selected candidate: `preserve_or_context_full_head_l40_r40_t70_b35_square`.

```json
{
  "name": "preserve_or_context_full_head_l40_r40_t70_b35_square",
  "family": "prepared_face_conditional",
  "description": "Use the original image when the prepared-face bypass fires; otherwise use `context_full_head_l40_r40_t70_b35_square`.",
  "margin": null,
  "expansion": null,
  "padding_method": null,
  "fallback": {
    "name": "context_full_head_l40_r40_t70_b35_square",
    "family": "context_square",
    "description": "Full-head contextual square crop with 40/40/70/35% left/right/top/bottom expansion.",
    "margin": null,
    "expansion": {
      "left": 0.4,
      "right": 0.4,
      "top": 0.7,
      "bottom": 0.35
    },
    "padding_method": null,
    "fallback": null
  }
}
```

Internal MVP gate on expanded validation:

- Fake-to-real flip rate: 0.000000
- Total class-change rate: 0.000000
- Detection/crop failure rate: 0.050000
- Fake recall after crop: 0.959596
- Gate result: fail

## G. Phase 2.1 Verdict

ready for API prototype with documented limitations

## H. Exact Next Action

Integrate the selected crop candidate into a local API prototype with the Phase 2.1 limitations documented.
