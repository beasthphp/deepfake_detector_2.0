from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PATTERNS = {
    "eval": re.compile(r"\beval\s*\("),
    "document.write": re.compile(r"\bdocument\.write\s*\("),
    "innerHTML": re.compile(r"\.innerHTML\s*="),
    "outerHTML": re.compile(r"\.outerHTML\s*="),
    "insertAdjacentHTML": re.compile(r"\.insertAdjacentHTML\s*\("),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan extension files for unsafe rendering APIs.")
    parser.add_argument("root", nargs="?", default="extension", help="Extension source directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    hits: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".js", ".mjs", ".html"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{path}:{line}: unsafe API candidate: {name}")
    if hits:
        print("\n".join(hits), file=sys.stderr)
        return 1
    print("No unsafe rendering API candidates found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
