# Contributing

Use one focused branch per task. Do not commit directly to `main`.

Before editing, inspect the relevant source, tests, and reports. Keep changes small and scoped to the task.

## Workflow

1. Create a branch with a descriptive name.
2. Make focused changes.
3. Run relevant tests.
4. Review the diff before committing.
5. Use conventional commit messages.
6. Open a pull request before merging.

## Tests

Python changes should run:

```powershell
python -m compileall api face_pipeline model_runtime scripts
pytest api/tests face_pipeline/test_pipeline.py
```

Extension changes should run:

```powershell
npm run test:extension
node --check extension/content-script.js
python scripts/scan_extension_safety.py extension
```

Actual-model or dataset-heavy tests are local-only unless a task explicitly provisions those assets.

## Pull Requests

Every pull request should describe scope, files changed, tests performed, screenshots for extension UI changes, API contract impact, privacy/security impact, and model-version impact.

Do not push, merge, tag, or publish releases without explicit project-owner approval.
