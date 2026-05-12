"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.utils.csv_io import (
    check_audio_exists,
    ensure_columns,
    read_csv,
    save_dataframe,
    save_json,
)
from src.utils.labels import normalize_l2_label
from src.utils.plots import save_bar_chart, save_label_distribution


BASELINE_LABELS = [
    "Arabic",
    "Hindi",
    "Korean",
    "Mandarin",
    "Spanish",
    "Vietnamese",
]

DEFAULT_NEW_LABELS = [
    "OOD_Singaporean_English",
    "OOD_Malaysian_English",
    "OOD_British_Isles_English",
]

DEFAULT_EXTERNAL_METADATA = Path("data/processed/external_metadata.csv")
DEFAULT_SPLIT_ROOT = Path("data/processed/splits")
DEFAULT_OUTPUT_ROOT = Path("data/processed/round2_balanced")
DEFAULT_REPORT_DIR = Path("results/round2_balanced")

STANDARD_COLUMNS = [
    "split",
    "index",
    "utterance_id",
    "audio_path",
    "speaker",
    "label",
    "gender",
    "text",
    "dataset",
    "source_config",
    "original_label",
    "is_mapped_to_baseline",
]


def bool_like(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}

def standardize_external(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_columns(df, STANDARD_COLUMNS)
    out = out[STANDARD_COLUMNS].copy()

    out["label"] = out["label"].astype(str).str.strip()

    # Fill dataset field.
    out["dataset"] = out["dataset"].fillna("").astype(str)
    out.loc[out["dataset"].str.strip() == "", "dataset"] = "external"

    # Fill original_label with label when original_label is missing/empty.
    out["original_label"] = out["original_label"].fillna("").astype(str)
    missing_original = out["original_label"].str.strip() == ""
    out.loc[missing_original, "original_label"] = out.loc[missing_original, "label"]

    # Normalize baseline mapping flag.
    out["is_mapped_to_baseline"] = out["is_mapped_to_baseline"].apply(bool_like)

    return out


def standardize_l2(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = ensure_columns(df, STANDARD_COLUMNS)
    out = out[STANDARD_COLUMNS].copy()

    out["split"] = split_name
    out["label"] = out["label"].apply(normalize_l2_label)
    out["dataset"] = "l2_arctic"
    out["source_config"] = ""
    out["original_label"] = out["label"]
    out["is_mapped_to_baseline"] = True

    return out


def sample_one_label(
    df: pd.DataFrame,
    label: str,
    target_total: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    group = df[df["label"] == label].copy()
    available = len(group)

    if available == 0:
        print(f"Warning: no samples found for label={label}")
        return group, {
            "available": 0,
            "used": 0,
        }

    used = min(available, target_total)

    if available < target_total:
        print(
            f"Warning: label={label} has only {available} samples, "
            f"less than target_total={target_total}. Using all available samples."
        )

    sampled = group.sample(n=used, random_state=seed).reset_index(drop=True)

    return sampled, {
        "available": int(available),
        "used": int(used),
    }


def split_train_dev_test(
    df: pd.DataFrame,
    train_ratio: float,
    dev_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)

    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train = df.iloc[:n_train].copy()
    dev = df.iloc[n_train:n_train + n_dev].copy()
    test = df.iloc[n_train + n_dev:].copy()

    train["split"] = "round2_external_train"
    dev["split"] = "round2_external_dev"
    test["split"] = "round2_external_test"

    return train, dev, test


def make_balanced_external_splits(
    external_df: pd.DataFrame,
    new_labels: List[str],
    target_total_per_new_label: int,
    seed: int,
    train_ratio: float,
    dev_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    train_parts = []
    dev_parts = []
    test_parts = []
    report = {}

    for label_id, label in enumerate(new_labels):
        sampled, label_report = sample_one_label(
            df=external_df,
            label=label,
            target_total=target_total_per_new_label,
            seed=seed + label_id,
        )

        if len(sampled) == 0:
            report[label] = {
                **label_report,
                "train": 0,
                "dev": 0,
                "test": 0,
            }
            continue

        train, dev, test = split_train_dev_test(
            sampled,
            train_ratio=train_ratio,
            dev_ratio=dev_ratio,
        )

        train_parts.append(train)
        dev_parts.append(dev)
        test_parts.append(test)

        report[label] = {
            **label_report,
            "train": int(len(train)),
            "dev": int(len(dev)),
            "test": int(len(test)),
        }

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=external_df.columns)
    dev_df = pd.concat(dev_parts, ignore_index=True) if dev_parts else pd.DataFrame(columns=external_df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=external_df.columns)

    return train_df, dev_df, test_df, report


def save_distribution_tables_and_figures(
    selected_before: pd.DataFrame,
    ext_train: pd.DataFrame,
    ext_dev: pd.DataFrame,
    ext_test: pd.DataFrame,
    report_dir: Path,
):
    table_dir = report_dir / "tables"
    figure_dir = report_dir / "figures"

    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    selected_before["label"].value_counts().sort_index().to_csv(
        table_dir / "selected_external_before_balancing.csv",
        header=["count"],
    )

    ext_train["label"].value_counts().sort_index().to_csv(
        table_dir / "external_train_after_balancing.csv",
        header=["count"],
    )

    ext_dev["label"].value_counts().sort_index().to_csv(
        table_dir / "external_dev_after_balancing.csv",
        header=["count"],
    )

    ext_test["label"].value_counts().sort_index().to_csv(
        table_dir / "external_test_after_balancing.csv",
        header=["count"],
    )

    save_label_distribution(
        selected_before,
        figure_dir / "selected_external_before_balancing.png",
    )

    save_label_distribution(
        ext_train,
        figure_dir / "external_train_after_balancing.png",
    )

    save_label_distribution(
        ext_dev,
        figure_dir / "external_dev_after_balancing.png",
    )

    save_label_distribution(
        ext_test,
        figure_dir / "external_test_after_balancing.png",
    )


def make_round2_for_fold(
    fold: int,
    output_root: Path,
    split_root: Path,
    ext_train: pd.DataFrame,
    ext_dev: pd.DataFrame,
    ext_test: pd.DataFrame,
):
    fold_input_dir = split_root / f"split2_speaker_disjoint_fold{fold}"
    fold_output_dir = output_root / f"fold{fold}"

    fold_output_dir.mkdir(parents=True, exist_ok=True)

    l2_train_path = fold_input_dir / "train.csv"
    l2_dev_path = fold_input_dir / "dev.csv"
    l2_test_path = fold_input_dir / "test.csv"

    if not l2_train_path.exists():
        raise FileNotFoundError(f"Missing split file: {l2_train_path}")

    l2_train = read_csv(l2_train_path)
    l2_dev = read_csv(l2_dev_path)
    l2_test = read_csv(l2_test_path)

    l2_train = check_audio_exists(l2_train, f"fold{fold} L2 train")
    l2_dev = check_audio_exists(l2_dev, f"fold{fold} L2 dev")
    l2_test = check_audio_exists(l2_test, f"fold{fold} L2 test")

    l2_train = standardize_l2(l2_train, f"fold{fold}_l2_train")
    l2_dev = standardize_l2(l2_dev, f"fold{fold}_l2_dev")
    l2_test = standardize_l2(l2_test, f"fold{fold}_l2_test")

    # Main Round2 expanded setting:
    # original six L2 classes + balanced selected external OOD labels.
    train_expanded = pd.concat([l2_train, ext_train], ignore_index=True)
    dev_expanded = pd.concat([l2_dev, ext_dev], ignore_index=True)
    test_combined = pd.concat([l2_test, ext_test], ignore_index=True)

    save_dataframe(
        train_expanded,
        fold_output_dir / "train_round2_expanded_balanced.csv",
        reset_index=True,
    )
    save_dataframe(
        dev_expanded,
        fold_output_dir / "dev_round2_expanded_balanced.csv",
        reset_index=True,
    )
    save_dataframe(
        test_combined,
        fold_output_dir / "test_round2_expanded_combined.csv",
        reset_index=True,
    )

    # Separate test files for clearer reporting.
    save_dataframe(
        l2_test,
        fold_output_dir / "test_l2_only.csv",
        reset_index=True,
    )
    save_dataframe(
        ext_test,
        fold_output_dir / "test_external_new_labels_only.csv",
        reset_index=True,
    )

    # Useful controlled baseline files.
    save_dataframe(
        l2_train,
        fold_output_dir / "train_l2_only.csv",
        reset_index=True,
    )
    save_dataframe(
        l2_dev,
        fold_output_dir / "dev_l2_only.csv",
        reset_index=True,
    )

    fold_report = {
        "fold": fold,
        "train_round2_expanded_balanced_rows": int(len(train_expanded)),
        "dev_round2_expanded_balanced_rows": int(len(dev_expanded)),
        "test_round2_expanded_combined_rows": int(len(test_combined)),
        "train_label_counts": train_expanded["label"].value_counts().sort_index().to_dict(),
        "dev_label_counts": dev_expanded["label"].value_counts().sort_index().to_dict(),
        "test_label_counts": test_combined["label"].value_counts().sort_index().to_dict(),
    }

    save_json(
        fold_report,
        fold_output_dir / "round2_fold_report.json",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--external-metadata", type=str, default=str(DEFAULT_EXTERNAL_METADATA))
    parser.add_argument("--split-root", type=str, default=str(DEFAULT_SPLIT_ROOT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--report-dir", type=str, default=str(DEFAULT_REPORT_DIR))

    parser.add_argument(
        "--new-labels",
        nargs="+",
        default=DEFAULT_NEW_LABELS,
        help="External OOD labels to add as new accent classes.",
    )

    parser.add_argument(
        "--target-total-per-new-label",
        type=int,
        default=3750,
        help=(
            "Total external samples per new label before train/dev/test split. "
            "3750 gives around 3000 train samples with 80/10/10 split."
        ),
    )

    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])

    args = parser.parse_args()

    external_metadata = Path(args.external_metadata)
    split_root = Path(args.split_root)
    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Making Round2 balanced expanded datasets")
    print("=" * 80)
    print("External metadata:", external_metadata)
    print("Split root:", split_root)
    print("Output root:", output_root)
    print("Report dir:", report_dir)
    print("New labels:", args.new_labels)
    print("Target total per new label:", args.target_total_per_new_label)
    print("Train/dev/test ratios:", args.train_ratio, args.dev_ratio, 1 - args.train_ratio - args.dev_ratio)
    print("Seed:", args.seed)
    print("Folds:", args.folds)

    if not external_metadata.exists():
        raise FileNotFoundError(f"Missing external metadata: {external_metadata}")

    ext = read_csv(external_metadata)
    ext = standardize_external(ext)
    ext = check_audio_exists(ext, "external metadata")

    print("\nExternal label counts before filtering:")
    print(ext["label"].value_counts())

    selected = ext[ext["label"].isin(args.new_labels)].copy()

    if len(selected) == 0:
        raise ValueError(
            "No selected external labels found. "
            "Check --new-labels and data/processed/external_metadata.csv."
        )

    print("\nSelected external label counts before balancing:")
    print(selected["label"].value_counts().sort_index())

    ext_train, ext_dev, ext_test, balance_report = make_balanced_external_splits(
        external_df=selected,
        new_labels=args.new_labels,
        target_total_per_new_label=args.target_total_per_new_label,
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
    )

    print("\nBalance report:")
    print(json.dumps(balance_report, indent=2, ensure_ascii=False))

    save_json(
        balance_report,
        report_dir / "round2_balance_report.json",
    )

    print("\nExternal train label counts after balancing:")
    print(ext_train["label"].value_counts().sort_index())

    print("\nExternal dev label counts after balancing:")
    print(ext_dev["label"].value_counts().sort_index())

    print("\nExternal test label counts after balancing:")
    print(ext_test["label"].value_counts().sort_index())

    save_dataframe(
        ext_train,
        output_root / "external_new_labels_train_balanced.csv",
        reset_index=True,
    )
    save_dataframe(
        ext_dev,
        output_root / "external_new_labels_dev_balanced.csv",
        reset_index=True,
    )
    save_dataframe(
        ext_test,
        output_root / "external_new_labels_test_balanced.csv",
        reset_index=True,
    )

    save_distribution_tables_and_figures(
        selected_before=selected,
        ext_train=ext_train,
        ext_dev=ext_dev,
        ext_test=ext_test,
        report_dir=report_dir,
    )

    for fold in args.folds:
        print("\n" + "=" * 80)
        print(f"Making Round2 CSVs for fold {fold}")
        print("=" * 80)

        make_round2_for_fold(
            fold=fold,
            output_root=output_root,
            split_root=split_root,
            ext_train=ext_train,
            ext_dev=ext_dev,
            ext_test=ext_test,
        )

    global_report = {
        "external_metadata": str(external_metadata),
        "new_labels": args.new_labels,
        "target_total_per_new_label": args.target_total_per_new_label,
        "train_ratio": args.train_ratio,
        "dev_ratio": args.dev_ratio,
        "test_ratio": 1 - args.train_ratio - args.dev_ratio,
        "seed": args.seed,
        "folds": args.folds,
        "balance_report": balance_report,
        "external_train_rows": int(len(ext_train)),
        "external_dev_rows": int(len(ext_dev)),
        "external_test_rows": int(len(ext_test)),
        "external_train_label_counts": ext_train["label"].value_counts().sort_index().to_dict(),
        "external_dev_label_counts": ext_dev["label"].value_counts().sort_index().to_dict(),
        "external_test_label_counts": ext_test["label"].value_counts().sort_index().to_dict(),
    }

    save_json(
        global_report,
        report_dir / "round2_global_report.json",
    )

    print("\nDone.")
    print(f"Round2 balanced data saved under: {output_root}")
    print(f"Reports and figures saved under: {report_dir}")


if __name__ == "__main__":
    main()