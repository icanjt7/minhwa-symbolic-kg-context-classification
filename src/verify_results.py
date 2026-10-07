#!/usr/bin/env python3
"""Verify public OOF structure, reported metrics, and file hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


LABELS = ["기타", "길상", "다산", "벽사", "부귀", "장수", "출세", "화합"]
PREDICTIONS = {
    "motif_only": "pred_A",
    "metadata_context": "pred_B",
    "kg_context": "pred_C",
    "kg_context_graph_embedding": "pred_D",
}
EXPECTED_BOOTSTRAP = {
    "f1_macro": (0.00040059289720161895, -0.001609248183242322, 0.002492288481674001),
    "f1_micro": (0.0005728823959090847, -0.0013511225879666868, 0.0025182610206013114),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f1_scores(rows: list[dict[str, str]], prediction: str) -> tuple[float, float, float]:
    true_counts = Counter(row["y_true"] for row in rows)
    pred_counts = Counter(row[prediction] for row in rows)
    true_positive = Counter(
        row["y_true"] for row in rows if row["y_true"] == row[prediction]
    )
    per_class = []
    for label in LABELS:
        tp = true_positive[label]
        fp = pred_counts[label] - tp
        fn = true_counts[label] - tp
        denominator = 2 * tp + fp + fn
        per_class.append((2 * tp / denominator) if denominator else 0.0)
    accuracy = sum(true_positive.values()) / len(rows)
    return sum(per_class) / len(LABELS), accuracy, accuracy


def read_oof(root: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    path = root / "results" / "oof_predictions_minimal.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = ["record_id", "artwork_id_hash", "fold", "y_true", "pred_A", "pred_B", "pred_C", "pred_D"]
        if reader.fieldnames != expected_columns:
            raise AssertionError(f"Unexpected OOF columns: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 34522:
        raise AssertionError(f"Expected 34,522 OOF rows, got {len(rows):,}")
    artwork_folds: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        if row["record_id"] != f"R{index:05d}":
            raise AssertionError(f"Non-sequential record_id at row {index}")
        if not re.fullmatch(r"[0-9a-f]{24}", row["artwork_id_hash"]):
            raise AssertionError(f"Invalid artwork pseudonym at row {index}")
        fold = int(row["fold"])
        if fold not in range(1, 6):
            raise AssertionError(f"Invalid fold {fold}")
        prior = artwork_folds.setdefault(row["artwork_id_hash"], fold)
        if prior != fold:
            raise AssertionError(f"Artwork crosses folds: {row['artwork_id_hash']}")
        for column in ("y_true", *PREDICTIONS.values()):
            if row[column] not in LABELS:
                raise AssertionError(f"Unexpected label {row[column]!r} in {column}")
    if len(artwork_folds) != 3779:
        raise AssertionError(f"Expected 3,779 artworks, got {len(artwork_folds):,}")
    if {row["y_true"] for row in rows} != set(LABELS):
        raise AssertionError("The OOF file does not contain all eight classes")
    return rows, artwork_folds


def verify_fold_file(root: Path, artwork_folds: dict[str, int]) -> None:
    path = root / "metadata" / "fold_assignment_hashed.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        stored = {row["artwork_id_hash"]: int(row["fold"]) for row in csv.DictReader(handle)}
    if stored != artwork_folds:
        raise AssertionError("Hashed fold assignment does not match minimal OOF")


def verify_metrics(root: Path, rows: list[dict[str, str]]) -> None:
    path = root / "results" / "reference" / "fold_and_pooled_metrics.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reference = {
            row["condition"]: row for row in csv.DictReader(handle) if row["scope"] == "pooled_oof"
        }
    for condition, column in PREDICTIONS.items():
        macro, micro, accuracy = f1_scores(rows, column)
        target = reference[condition]
        for name, actual in (("f1_macro", macro), ("f1_micro", micro), ("accuracy", accuracy)):
            expected = float(target[name])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(f"{condition} {name}: expected {expected}, calculated {actual}")


def verify_bootstrap(root: Path) -> None:
    path = root / "results" / "reference" / "paired_group_bootstrap_summary.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["metric"]: row for row in csv.DictReader(handle)}
    for metric, (mean, low, high) in EXPECTED_BOOTSTRAP.items():
        row = rows[metric]
        values = (float(row["mean"]), float(row["ci95_low"]), float(row["ci95_high"]))
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15) for a, b in zip(values, (mean, low, high))):
            raise AssertionError(f"Bootstrap summary mismatch for {metric}")


def verify_hashes(root: Path) -> None:
    manifest = root / "metadata" / "hash_manifest.txt"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        actual = sha256(root / relative)
        if actual != expected:
            raise AssertionError(f"SHA-256 mismatch: {relative}")


def verify_local_sources_if_present(root: Path) -> None:
    manifest = root / "metadata" / "source_hashes.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = root / relative
        if path.exists() and sha256(path) != expected:
            raise AssertionError(f"Local source SHA-256 mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", action="store_true", help="Also require generated result directories")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows, artwork_folds = read_oof(root)
    verify_fold_file(root, artwork_folds)
    verify_metrics(root, rows)
    verify_bootstrap(root)
    verify_hashes(root)
    verify_local_sources_if_present(root)
    if args.generated:
        required = [
            root / "results" / "generated" / "abc" / "grouped_ablation_predictions.csv",
            root / "results" / "generated" / "graph_embedding" / "fold_and_pooled_metrics.csv",
            root / "results" / "generated" / "other_sensitivity" / "named7_oof_metrics.csv",
        ]
        missing = [str(path.relative_to(root)) for path in required if not path.exists()]
        if missing:
            raise AssertionError("Missing generated outputs: " + ", ".join(missing))
    print("PASS: 34,522 rows; 3,779 artworks; 8 classes; 5 folds")
    print("PASS: pooled A/B/C/D metrics and D-C bootstrap intervals match")
    print("PASS: public result-file SHA-256 hashes match")


if __name__ == "__main__":
    main()
