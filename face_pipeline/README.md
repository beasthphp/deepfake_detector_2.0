# Face preprocessing pipeline

This package isolates face detection and crop preparation from the API and model runtime.

The pipeline is responsible for detecting candidate human faces, producing consistent model-ready crops, and returning structured face metadata to the inference layer. Keeping preprocessing separate makes detector changes easier to test and prevents browser/API code from depending on model-specific image logic.

Current work remains CPU-friendly and is designed for the local MVP rather than high-throughput production inference.
