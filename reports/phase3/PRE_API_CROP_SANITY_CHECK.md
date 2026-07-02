# Pre-API Crop Sanity Check

## Method

- Strategy checked: `preserve_or_context_full_head_l40_r40_t70_b35_square`.
- Deterministic source: first 25 successful records from `reports/phase2/crop_consistency.json`.
- For each image, the check records original dimensions, Haar box, expanded boxes, final crop box, hashes, and model-input identity.

## Summary

- Images checked: 25
- Images with a detected face: 25
- No-face records: 0
- Prepared-face bypass fired: 24 / 25
- Final crop equals complete original: 25 / 25
- Expanded rectangle clamped to full image: 25 / 25
- Final model-input pixels identical to original model-input pixels: 25 / 25
- Conclusion: `expected_because_prepared_faces_occupy_full_frame`

## Interpretation

The identical predictions are explained by prepared-face geometry. Most rows fired the prepared-face bypass, and every contextual expansion still clamped to the complete original frame. The final model-input pixels were identical for every checked image, so this is not an implementation bug.

## Sample Rows

| Image | Face box | Final crop | Bypass | Full original | Identical model input |
| --- | --- | --- | ---: | ---: | ---: |
| `test/real/16619.jpg` | `{"x1": 35, "x2": 219, "y1": 45, "y2": 229}` | `{"x1": 0, "x2": 256, "y1": 0, "y2": 256}` | True | True | True |
| `test/real/29984.jpg` | `{"x1": 38, "x2": 215, "y1": 53, "y2": 230}` | `{"x1": 0, "x2": 256, "y1": 0, "y2": 256}` | True | True | True |
| `test/real/46844.jpg` | `{"x1": 39, "x2": 219, "y1": 49, "y2": 229}` | `{"x1": 0, "x2": 256, "y1": 0, "y2": 256}` | True | True | True |
| `test/real/05920.jpg` | `{"x1": 28, "x2": 218, "y1": 45, "y2": 235}` | `{"x1": 0, "x2": 256, "y1": 0, "y2": 256}` | True | True | True |
| `test/real/02858.jpg` | `{"x1": 46, "x2": 212, "y1": 61, "y2": 227}` | `{"x1": 0, "x2": 256, "y1": 0, "y2": 256}` | True | True | True |
| `test/real/17461.jpg` | `{"x1": 43, "x2": 213, "y1": 55, "y2": 225}` | `{"x1": 0, "x2": 256, "y1": 0, "y2": 256}` | True | True | True |
| `test/real/21130.jpg` | `{"x1": 43, "x2": 214, "y1": 55, "y2": 226}` | `{"x1": 0, "x2": 256, "y1": 0, "y2": 256}` | True | True | True |
| `test/real/32827.jpg` | `{"x1": 54, "x2": 225, "y1": 53, "y2": 224}` | `{"x1": 0, "x2": 256, "y1": 0, "y2": 256}` | True | True | True |
| `test/real/61096.jpg` | `{"x1": 35, "x2": 219, "y1": 52, "y2": 236}` | `{"x1": 0, "x2": 256, "y1": 0, "y2": 256}` | True | True | True |
| `test/real/34125.jpg` | `{"x1": 52, "x2": 208, "y1": 63, "y2": 219}` | `{"x1": 0, "x2": 256, "y1": 0, "y2": 256}` | False | True | True |

Full row-level data is saved in `reports/phase3/pre_api_crop_sanity.json`.
