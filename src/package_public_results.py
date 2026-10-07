#!/usr/bin/env python3
"""Create privacy-minimized OOF and fixed-fold files from the full local OOF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import unicodedata
from pathlib import Path


NAMESPACE = "aihub-minhwa-v1.1\0"


def pseudonym(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    return hashlib.sha256(f"{NAMESPACE}{normalized}".encode("utf-8")).hexdigest()[:24]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Full private A/B/C/D OOF CSV")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    parser.add_argument("--metadata-dir", type=Path, default=root / "metadata")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    oof_path = args.output_dir / "oof_predictions_minimal.csv"
    fold_path = args.metadata_dir / "fold_assignment_hashed.csv"
    source_to_public = {
        "pred_motif_only": "pred_A",
        "pred_metadata_context": "pred_B",
        "pred_kg_context": "pred_C",
        "pred_kg_context_graph_embedding": "pred_D",
    }
    fold_by_artwork: dict[str, int] = {}
    count = 0
    with args.input.open(encoding="utf-8-sig", newline="") as src, oof_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.DictReader(src)
        required = {"fold", "group_id", "y_true", *source_to_public}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing OOF columns: {sorted(missing)}")
        fields = ["record_id", "artwork_id_hash", "fold", "y_true", *source_to_public.values()]
        writer = csv.DictWriter(dst, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for count, row in enumerate(reader, start=1):
            artwork_hash = pseudonym(row["group_id"])
            fold = int(row["fold"])
            prior = fold_by_artwork.setdefault(artwork_hash, fold)
            if prior != fold:
                raise ValueError(f"Artwork crosses folds: {artwork_hash}")
            public = {
                "record_id": f"R{count:05d}",
                "artwork_id_hash": artwork_hash,
                "fold": fold,
                "y_true": row["y_true"],
            }
            public.update({target: row[source] for source, target in source_to_public.items()})
            writer.writerow(public)

    with fold_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst, fieldnames=["artwork_id_hash", "fold"], lineterminator="\n"
        )
        writer.writeheader()
        for artwork_hash, fold in sorted(fold_by_artwork.items()):
            writer.writerow({"artwork_id_hash": artwork_hash, "fold": fold})

    print(f"Wrote {count:,} OOF rows to {oof_path}")
    print(f"Wrote {len(fold_by_artwork):,} fixed artwork folds to {fold_path}")


if __name__ == "__main__":
    main()
