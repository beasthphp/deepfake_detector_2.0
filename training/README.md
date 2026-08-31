# Training workspace

This directory contains training-related code and notes for detector replacement work.

The production-facing API and browser extension do not depend on notebook-specific training logic. New models should be evaluated independently, documented in `MODEL_CARD.md`, registered under `models/registry.json`, and integrated through the replaceable model-provider interface.

The current repository treats model training and application integration as separate concerns so a stronger detector can be introduced without changing the extension/API contract.
