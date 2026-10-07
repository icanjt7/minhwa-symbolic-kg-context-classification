from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "private" / "minhwa_raw_combined.csv"
MANIFEST_PATH = ROOT / "data" / "private" / "manifest.csv"
OUTPUT_DIR = ROOT / "results" / "generated" / "abc"
SEED = 42
N_SPLITS = 5

SUPER_PATTERNS = [
    ("장수", [r"장수", r"불로", r"수명", r"학", r"거북", r"소나무", r"복숭아", r"영지"]),
    ("부귀", [r"부귀", r"영화", r"재물", r"모란", r"금전", r"보화"]),
    ("다산", [r"다산", r"자손", r"번창", r"석류", r"포도", r"잉어", r"개구리"]),
    ("출세", [r"출세", r"입신", r"공명", r"과거", r"합격", r"문방", r"책"]),
    ("벽사", [r"벽사", r"액막", r"잡귀", r"수호", r"호랑이", r"해태"]),
    ("화합", [r"화합", r"화목", r"부부", r"원앙", r"평안", r"조화"]),
    ("길상", [r"길상", r"복", r"행운", r"상서", r"경사"]),
]
OTHER_TERMS = {
    "한국전통민화", "민화", "전통", "기타", "그림", "도상", "동물", "식물",
    "화훼", "화목", "문자", "장식", "행사", "교육", "생활", "용도", "한지",
    "비단", "일필", "공필", "채색", "수묵",
}
CONDITIONS = ("motif_only", "metadata_context", "kg_context")


def split_terms(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").replace(";", ",").split(",") if part.strip()]


def super_label(symbol: str) -> str:
    for label, patterns in SUPER_PATTERNS:
        if any(re.search(pattern, symbol) for pattern in patterns):
            return label
    return "기타"


def base_artwork_id(file_name: str, rel_path: str) -> str:
    normalized_path = unicodedata.normalize("NFC", rel_path)
    if "상세묘사데이터" in normalized_path:
        return re.sub(r"_\d+$", "", file_name)
    return file_name


def read_samples() -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, set[str]]]:
    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig").fillna("")
    manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig").fillna("")
    stem_to_path = dict(zip(manifest["image_stem"].astype(str), manifest["rel_path"].astype(str)))
    raw["group_id"] = raw["file_name"].astype(str).map(
        lambda stem: base_artwork_id(stem, stem_to_path.get(stem, ""))
    )
    if raw["file_name"].map(lambda stem: stem in stem_to_path).sum() != len(raw):
        raise ValueError("Some metadata records could not be joined to the image manifest.")

    group_motifs: dict[str, set[str]] = defaultdict(set)
    group_subjects: dict[str, set[str]] = defaultdict(set)
    for row in raw.itertuples(index=False):
        group_motifs[row.group_id].update(split_terms(row.objects))
        group_subjects[row.group_id].update(split_terms(row.subject_type))

    relation_rows: set[tuple[str, str, str]] = set()
    for row in raw.itertuples(index=False):
        group_id = row.group_id
        motifs = split_terms(row.objects)
        blocked = (
            set(motifs)
            | group_motifs[group_id]
            | {
                str(getattr(row, column)).strip()
                for column in (
                    "subject_type", "theme_type", "painting_type", "purpose", "era", "ground_material"
                )
            }
            | OTHER_TERMS
        )
        labels = [term for term in split_terms(row.keywords) if term not in blocked and len(term) <= 80]
        for motif in motifs:
            for label in labels:
                relation_rows.add((group_id, motif, super_label(label)))

    samples = pd.DataFrame(sorted(relation_rows), columns=["group_id", "motif", "label"])
    if samples.empty:
        raise ValueError("No group-motif-symbol rows could be reconstructed.")
    return samples, group_motifs, group_subjects


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
            ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, class_weight="balanced", random_state=SEED)),
        ]
    )


def main() -> None:
    samples, group_motifs, group_subjects = read_samples()
    labels = sorted(samples["label"].unique())
    groups_per_label = samples.groupby("label")["group_id"].nunique()
    if groups_per_label.min() < N_SPLITS:
        raise ValueError(f"5-fold grouped CV is unsupported; smallest class has {groups_per_label.min()} groups.")

    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    group_values = samples["group_id"].to_numpy()
    target_values = samples["label"].to_numpy()
    all_predictions: list[dict[str, object]] = []
    fold_metrics: list[dict[str, object]] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(samples["motif"], target_values, group_values), start=1
    ):
        train = samples.iloc[train_idx]
        test = samples.iloc[test_idx]
        train_groups = set(train["group_id"])
        test_groups = set(test["group_id"])
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(f"Artwork groups crossed folds: {len(overlap)}")

        motif_symbol_groups: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for row in train.itertuples(index=False):
            motif_symbol_groups[row.motif][row.label].add(row.group_id)

        train_targets = train["label"].to_numpy()
        fold_predictions: dict[str, list[str]] = {}
        fold_kg_coverage: list[bool] = []
        for condition in CONDITIONS:
            texts: list[str] = []
            for row in test.itertuples(index=False):
                neighbors = sorted(group_motifs[row.group_id] - {row.motif})
                subject = " ".join(sorted(group_subjects[row.group_id]))
                if condition == "motif_only":
                    text = f"motif {row.motif}"
                elif condition == "metadata_context":
                    text = f"motif {row.motif} subject {subject} neighboring_motifs {' '.join(neighbors)}"
                else:
                    related_symbols = graph_symbols(neighbors, row.group_id, motif_symbol_groups)
                    fold_kg_coverage.append(bool(related_symbols))
                    text = (
                        f"motif {row.motif} subject {subject} neighboring_motifs {' '.join(neighbors)} "
                        f"kg_neighbor_symbols {' '.join(related_symbols)}"
                    )
                texts.append(text)

            model = make_pipeline()
            train_texts = (
                train["motif"].map(lambda motif: f"motif {motif}").tolist()
                if condition == "motif_only"
                else _train_texts(
                    train, condition, group_motifs, group_subjects, motif_symbol_groups
                )
            )
            model.fit(train_texts, train_targets)
            predictions = model.predict(texts).tolist()
            fold_predictions[condition] = predictions

        for condition, predictions in fold_predictions.items():
            fold_metrics.append(
                {
                    "fold": fold,
                    "condition": condition,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_train_groups": len(train_groups),
                    "n_test_groups": len(test_groups),
                    "f1_macro": f1_score(test["label"], predictions, labels=labels, average="macro", zero_division=0),
                    "f1_micro": f1_score(test["label"], predictions, labels=labels, average="micro", zero_division=0),
                    "accuracy": accuracy_score(test["label"], predictions),
                    "kg_context_coverage": (
                        sum(fold_kg_coverage) / len(fold_kg_coverage) if condition == "kg_context" else None
                    ),
                }
            )
        for local_idx, row in enumerate(test.itertuples(index=False)):
            result = {"fold": fold, "group_id": row.group_id, "motif": row.motif, "y_true": row.label}
            result.update({f"pred_{condition}": fold_predictions[condition][local_idx] for condition in CONDITIONS})
            all_predictions.append(result)

    predictions = pd.DataFrame(all_predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(fold_metrics)
    metrics.to_csv(OUTPUT_DIR / "grouped_ablation_fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_DIR / "grouped_ablation_predictions.csv", index=False, encoding="utf-8-sig")

    summary = metrics.groupby("condition").agg(
        f1_macro_mean=("f1_macro", "mean"),
        f1_macro_std=("f1_macro", "std"),
        f1_micro_mean=("f1_micro", "mean"),
        accuracy_mean=("accuracy", "mean"),
        kg_context_coverage_mean=("kg_context_coverage", "mean"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "grouped_ablation_summary.csv", index=False, encoding="utf-8-sig")

    for condition in CONDITIONS:
        report = classification_report(
            predictions["y_true"], predictions[f"pred_{condition}"], labels=labels,
            output_dict=True, zero_division=0,
        )
        pd.DataFrame(report).transpose().rename_axis("class").to_csv(
            OUTPUT_DIR / f"grouped_ablation_class_report_{condition}.csv", encoding="utf-8-sig"
        )
        cm = confusion_matrix(predictions["y_true"], predictions[f"pred_{condition}"], labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).rename_axis("actual").to_csv(
            OUTPUT_DIR / f"grouped_ablation_confusion_{condition}.csv", encoding="utf-8-sig"
        )

    config = {
        "analysis": "Artwork-grouped 5-fold CV with three context ablations",
        "seed": SEED,
        "n_splits": N_SPLITS,
        "n_samples": len(samples),
        "n_artwork_groups": int(samples["group_id"].nunique()),
        "n_classes": len(labels),
        "classes": labels,
        "min_groups_per_class": int(groups_per_label.min()),
        "label_rule": "First matching keyword regex in SUPER_PATTERNS; otherwise 기타",
        "metadata_context": "Target motif + group-level subject types and other motifs; no keywords",
        "kg_context": "Metadata context plus symbol classes linked in the training-fold motif-symbol graph for other motifs only",
        "kg_leakage_control": "Graph relations are fitted within each training fold; the target motif is excluded from KG context lookup",
        "vectorizer": {"type": "TF-IDF", "ngram_range": [1, 2], "min_df": 2, "max_df": 0.95, "sublinear_tf": True},
        "classifier": {"type": "LogisticRegression", "C": 1.0, "solver": "lbfgs", "max_iter": 2000, "class_weight": "balanced", "random_state": SEED},
        "input_raw_sha256": hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
        "input_manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "caveat": "This is a metadata-reconstructed supplementary reanalysis, not an exact reproduction of the original 15,901-row pair table.",
    }
    (OUTPUT_DIR / "grouped_ablation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"Saved grouped-CV reports to {OUTPUT_DIR}")


def _train_texts(
    train: pd.DataFrame,
    condition: str,
    group_motifs: dict[str, set[str]],
    group_subjects: dict[str, set[str]],
    motif_symbol_groups: dict[str, dict[str, set[str]]],
) -> list[str]:
    texts = []
    for row in train.itertuples(index=False):
        neighbors = sorted(group_motifs[row.group_id] - {row.motif})
        subject = " ".join(sorted(group_subjects[row.group_id]))
        text = f"motif {row.motif} subject {subject} neighboring_motifs {' '.join(neighbors)}"
        if condition == "kg_context":
            related_symbols = graph_symbols(neighbors, row.group_id, motif_symbol_groups)
            text += f" kg_neighbor_symbols {' '.join(related_symbols)}"
        texts.append(text)
    return texts


def graph_symbols(
    neighbors: list[str],
    excluded_group: str,
    motif_symbol_groups: dict[str, dict[str, set[str]]],
) -> list[str]:
    return sorted(
        {
            symbol
            for neighbor in neighbors
            for symbol, groups in motif_symbol_groups.get(neighbor, {}).items()
            if any(group_id != excluded_group for group_id in groups)
        }
    )


if __name__ == "__main__":
    main()
