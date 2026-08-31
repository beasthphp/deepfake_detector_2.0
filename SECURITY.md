# Security

This project is designed as a **local experimental MVP**. Its default security posture assumes the API and browser extension run on the same trusted machine.

## Local API

The API defaults to `127.0.0.1:8000`. Do not expose it on a public interface without authentication, rate limits, request logging, resource controls, and explicit deployment hardening.

## Image Privacy

With default settings, browser-extension requests go to the local API. The extension does not store raw image bytes or face crops in its prediction cache.

If a remote API origin is configured, eligible images are transmitted to that configured remote origin. Users should treat that as a change in the privacy boundary.

## Input Handling

Image uploads are decoded and validated before the face/model pipeline runs. Invalid images, detector failures, and crop failures should fail explicitly rather than being converted into a model prediction.

## Vulnerability Reporting

Report suspected vulnerabilities privately to the repository owner. Do not include secrets, private images, model files, credentials, or exploit details in public issues.

## Dependencies

Report dependency vulnerabilities with package name, version, advisory link, and affected surface. Avoid broad dependency upgrades without a focused compatibility or security reason.

## Model Integrity

Model artifacts must be verified before use. The provider rejects missing files, Git LFS pointers, unexpected file types, configured hash mismatches, and input/output contract mismatches.

Do not commit datasets, model binaries, environments, `.env` files, Kaggle credentials, API keys, or secret JSON files.
