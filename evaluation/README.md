# Evaluation

This directory contains evaluation utilities and artifacts for measuring the detector outside the browser-extension workflow.

Reported metrics should be interpreted together with `MODEL_CARD.md`: the current CNN has reproduced results on its prepared local test split, but it has not completed external validation across unrelated generators, identities, compression pipelines, or real-world webpage imagery.

The evaluation layer exists to make model replacement measurable rather than relying on visual spot checks alone.
