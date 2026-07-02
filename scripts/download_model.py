from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_runtime.registry import get_registry_model


CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a configured model artifact.")
    parser.add_argument("--model-id", default=os.getenv("ACTIVE_MODEL_ID"), help="Model ID from models/registry.json.")
    parser.add_argument("--registry", default=os.getenv("MODEL_REGISTRY_PATH", "models/registry.json"), help="Registry JSON path.")
    parser.add_argument("--artifact-dir", default=os.getenv("MODEL_ARTIFACT_DIR", "models/artifacts"), help="Local model artifact directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing invalid artifact.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    entry = get_registry_model(args.model_id, args.registry)
    if not entry.download_url:
        print(f"Model {entry.id} has no download URL. Place {entry.local_filename} manually in {args.artifact_dir}.", file=sys.stderr)
        return 2

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / entry.local_filename
    if target.suffix.lower() not in entry.file_types:
        print(f"Refusing unexpected file type: {target.suffix}", file=sys.stderr)
        return 2

    if target.exists():
        if entry.sha256 and sha256_file(target).lower() == entry.sha256.lower():
            print(f"Valid model already exists: {target}")
            return 0
        if not args.overwrite:
            print(f"Refusing to overwrite existing model without --overwrite: {target}", file=sys.stderr)
            return 2

    tmp_path = target.with_suffix(target.suffix + ".tmp")
    try:
        with urllib.request.urlopen(entry.download_url, timeout=60) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
        if entry.sha256:
            actual = sha256_file(tmp_path)
            if actual.lower() != entry.sha256.lower():
                tmp_path.unlink(missing_ok=True)
                print("Downloaded model SHA-256 mismatch.", file=sys.stderr)
                return 1
        tmp_path.replace(target)
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"Downloaded model: {target}")
    return 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
