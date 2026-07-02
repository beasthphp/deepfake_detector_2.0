from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_runtime.registry import create_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify configured model artifact and provider contract.")
    parser.add_argument("--model-id", default=os.getenv("ACTIVE_MODEL_ID"), help="Model ID from models/registry.json.")
    parser.add_argument("--registry", default=os.getenv("MODEL_REGISTRY_PATH", "models/registry.json"), help="Registry JSON path.")
    parser.add_argument("--artifact-dir", default=os.getenv("MODEL_ARTIFACT_DIR", "models/artifacts"), help="Local model artifact directory.")
    parser.add_argument("--model-path", default=os.getenv("DEEPFAKE_MODEL_PATH"), help="Explicit model artifact path override.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        provider = create_provider(
            active_model_id=args.model_id,
            registry_path=args.registry,
            artifact_dir=Path(args.artifact_dir),
            explicit_model_path=args.model_path,
        )
        provider.validate()
    except Exception as exc:
        print(f"Model verification failed: {exc}", file=sys.stderr)
        return 1

    metadata = provider.metadata
    print(f"Model verified: {metadata.id} v{metadata.version}")
    print(f"Provider: {metadata.provider}")
    print(f"Input shape: {metadata.input_shape}")
    print(f"Output shape: {metadata.output_shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
