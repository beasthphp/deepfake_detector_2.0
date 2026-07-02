# Security

## Local API

The API defaults to `127.0.0.1:8000`. Do not expose it on a public interface without authentication, rate limits, logging review, and explicit deployment hardening.

## Image Privacy

With default settings, browser-extension requests go to the local API. The extension does not store raw image bytes or face crops in its cache.

If a remote API origin is configured, eligible images are transmitted to that remote origin.

## Vulnerability Reporting

Report suspected vulnerabilities privately to the repository owner. Do not include secrets, private images, model files, credentials, or exploit details in public issues.

## Dependencies

Report dependency vulnerabilities with package name, version, advisory link, and affected surface. Do not modify lockfiles or upgrade broad dependency groups without a focused reason.

## Model Integrity

Model artifacts must be verified before use. The provider rejects missing files, Git LFS pointers, unexpected file types, configured hash mismatches, and input/output contract mismatches.

Do not commit datasets, model binaries, environments, `.env` files, Kaggle credentials, API keys, or secret JSON files.
