# AGENTS.md

## Project Scope

This repository is for human-face deepfake detection only. It is not a general media-forensics system or proof-of-authenticity product.

## Completed Architecture

- Face detection and crop pipeline.
- Local FastAPI inference API.
- Manifest V3 browser extension.
- User-controlled visible webpage scanning.
- Responsive per-face webpage overlays.

## Current Limitations

- The current model is an experimental placeholder.
- Predictions are unreliable on real-world web images.
- There is no external model validation.
- Haar face detection has known limitations.
- The browser extension requires a running local API unless mock mode is enabled.

## Rules For Codex

- Inspect before editing.
- Use small focused changes.
- Never fabricate test results.
- Never commit datasets, model binaries, environments, or secrets.
- Never change the `/predict` contract without migration notes.
- Keep model-specific logic inside model providers.
- Run relevant tests before completion.
- Report files changed.
- Report commands run.
- Do not push or merge without explicit approval.
- Do not modify completed reports unless correcting a verified error.
- Do not adjust thresholds to conceal model failures.

## Required Validation

For Python changes:

- API tests.
- Face-pipeline tests.
- Syntax and import checks.

For extension changes:

- Extension automated tests.
- `node --check`.
- Manifest validation.
- Unsafe-rendering checks.

## Git Expectations

- One focused branch per task.
- Use logical commits.
- Use conventional commit messages.
- Do not commit directly to `main`.
- Review the diff before committing.
- Use a pull request before merge.
