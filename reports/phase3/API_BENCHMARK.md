# API Benchmark

## Cold Startup

- Model and detector load: 10690.011 ms
- Memory before load: n/a MB
- Memory after load: n/a MB
- Memory delta: n/a MB

## Warm Requests

| Case | HTTP | API status | Faces | Decode ms | Detection ms | Crop ms | Classification ms | Serialization ms | Response total ms | Client wall ms |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| one_face | 200 | completed | 1 | 0.744 | 22.566 | 0.134 | 160.412 | 0.060 | 183.949 | 188.944 |
| multiple_faces | 200 | completed | 2 | 1.838 | 44.261 | 0.732 | 294.626 | 0.056 | 341.547 | 347.947 |
| no_face | 200 | no_face_detected | 0 | 0.780 | 9.352 | 0.000 | 0.000 | 0.000 | 10.148 | 16.208 |

## Ten Sequential Warm Requests

- Count: 10
- Failed: 0
- Average client wall time: 205.549 ms
- Median client wall time: 205.183 ms
- Min client wall time: 197.510 ms
- Max client wall time: 215.503 ms
- Average response total time: 199.963 ms
- Median response total time: 200.316 ms

Full benchmark JSON is saved in `reports/phase3/api_benchmark.json`.
