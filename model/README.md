# Legacy model workspace

This directory contains legacy model-analysis code retained for traceability while the application uses the newer `model_runtime/` provider boundary.

New detector integrations should go through `model_runtime/` and `models/registry.json` rather than coupling application code directly to this legacy workspace.
