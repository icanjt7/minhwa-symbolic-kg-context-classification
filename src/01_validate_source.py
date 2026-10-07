#!/usr/bin/env python3
"""Validate local normalized inputs without publishing or modifying them."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


EXPECTED_METADATA_COLUMNS = {
    "file_name", "subject_type", "theme_type", "painting_type",
    "ground_material", "purpose", "objects", "keywords", "era",
}
EXPECTED_MANIFEST_COLUMNS = {"image_id", "image_stem", "file_path", "rel_path"}
EXPECTED_METADATA_SHA256 = "7b8648ebfecd94cb201e6354d991b80f07533ccb5a82d50b4006103753c09d9c"
EXPECTED_MANIFEST_SHA256 = "ed46d2ca98e7213ca89710c4888f6d5f17da5aa8d236515a925789c91cd14178"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_csv(path: Path, required: set[str]) -> tuple[int, list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = sorted(required - set(columns))
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        rows = sum(1 for _ in reader)
    return rows, columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--allow-hash-mismatch", action="store_true")
    args = parser.parse_args()

    metadata_rows, metadata_columns = inspect_csv(args.metadata, EXPECTED_METADATA_COLUMNS)
    manifest_rows, manifest_columns = inspect_csv(args.manifest, EXPECTED_MANIFEST_COLUMNS)
    metadata_hash = sha256(args.metadata)
    manifest_hash = sha256(args.manifest)
    errors = []
    if metadata_rows != 9119:
        errors.append(f"metadata rows: expected 9119, got {metadata_rows}")
    if manifest_rows != 9119:
        errors.append(f"manifest rows: expected 9119, got {manifest_rows}")
    if not args.allow_hash_mismatch and metadata_hash != EXPECTED_METADATA_SHA256:
        errors.append("metadata SHA-256 differs from the paper input")
    if not args.allow_hash_mismatch and manifest_hash != EXPECTED_MANIFEST_SHA256:
        errors.append("manifest SHA-256 differs from the paper input")
    if errors:
        raise SystemExit("FAIL\n- " + "\n- ".join(errors))

    print(f"PASS: metadata rows={metadata_rows:,}; columns={len(metadata_columns)}; sha256={metadata_hash}")
    print(f"PASS: manifest rows={manifest_rows:,}; columns={len(manifest_columns)}; sha256={manifest_hash}")


if __name__ == "__main__":
    main()
