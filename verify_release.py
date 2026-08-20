#!/usr/bin/env python3
"""Verify the anonymous artifact against release_manifest.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "release_manifest.json").read_text())
expected = manifest["files"]
actual = {}
for path in sorted(ROOT.rglob("*")):
    if (
        path.is_file()
        and ".git" not in path.parts
        and path.name != "release_manifest.json"
    ):
        actual[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
if actual != expected:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(
        name for name in set(actual) & set(expected) if actual[name] != expected[name]
    )
    raise SystemExit(
        f"release verification failed: missing={missing}, extra={extra}, changed={changed}"
    )
print(f"verified {len(actual)} release files")
