"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

from pathlib import Path
from typing import List

import pandas as pd


BASELINE_LABELS = [
    "Arabic",
    "Hindi",
    "Korean",
    "Mandarin",
    "Spanish",
    "Vietnamese",
]

PROCESSED_DIR = Path("data/processed")
SPLIT_DIR = PROCESSED_DIR / "splits"
AUG_DIR = PROCESSED_DIR / "augmented"

EXTERNAL_METADATA = PROCESSED_DIR / "external_metadata.csv"


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


def normalize_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df


def standardize_l2(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = ensure_columns(df, STANDARD_COLUMNS)
    df = df[STANDARD_COLUMNS].copy()

    df["split"] = split_name
    df["dataset"] = "l2_arctic"
    df["source_config"] = ""
    df["original_label"] = df["label"]
    df["is_mapped_to_baseline"] = True

    return df


def standardize_external(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = ensure_columns(df, STANDARD_COLUMNS)
    df = df[STANDARD_COLUMNS].copy()
    df["split"] = split_name
    return df


def check_audio_exists(df: pd.DataFrame, name: str) -> pd.DataFrame:
    before = len(df)
    df = df.copy()
    df["audio_path"] = df["audio_path"].astype(str)

    mask = df["audio_path"].apply(lambda p: Path(p).exists())
    missing = before - int(mask.sum())

    if missing > 0:
        print(f"Warning: {name}: dropping {missing} rows with missing audio files.")

    return df[mask].reset_index(drop=True)


def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["index"] = range(len(df))
    df.to_csv(path, index=False, encoding="utf-8")

    print(f"Saved {len(df):7d} rows -> {path}")
    print(df["label"].value_counts().sort_index())


def load_external() -> pd.DataFrame:
    if not EXTERNAL_METADATA.exists():
        raise FileNotFoundError(
            f"Missing {EXTERNAL_METADATA}. Run prepare_external_datasets first."
        )

    ext = pd.read_csv(EXTERNAL_METADATA)
    ext = ensure_columns(ext, STANDARD_COLUMNS)

    if "is_mapped_to_baseline" not in ext.columns:
        raise ValueError("external_metadata.csv missing is_mapped_to_baseline")

    ext = check_audio_exists(ext, "external")
    return ext


def make_external_train_sets(ext: pd.DataFrame):
    ext = ext.copy()

    mapped_mask = normalize_bool_series(ext["is_mapped_to_baseline"])
    ext_six = ext[mapped_mask & ext["label"].isin(BASELINE_LABELS)].copy()
    ext_six = standardize_external(ext_six, "external_six_train")

    ext_expanded = ext.copy()
    ext_expanded = ext_expanded[ext_expanded["label"].astype(str).str.len() > 0]
    ext_expanded = standardize_external(ext_expanded, "external_expanded_train")

    save_csv(ext_six, AUG_DIR / "external_six_train.csv")
    save_csv(ext_expanded, AUG_DIR / "external_expanded_train.csv")

    return ext_six, ext_expanded


def make_fold_augmented_sets(ext_six: pd.DataFrame, ext_expanded: pd.DataFrame):
    for fold_id in range(4):
        fold_dir = SPLIT_DIR / f"split2_speaker_disjoint_fold{fold_id}"

        train_path = fold_dir / "train.csv"
        dev_path = fold_dir / "dev.csv"
        test_path = fold_dir / "test.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"Missing fold train file: {train_path}")

        l2_train = pd.read_csv(train_path)
        l2_train = check_audio_exists(l2_train, f"fold{fold_id} l2 train")
        l2_train = standardize_l2(l2_train, f"fold{fold_id}_l2_train")

        # A1: L2 + external, six original labels only.
        six_plus = pd.concat([l2_train, ext_six], ignore_index=True)
        save_csv(
            six_plus,
            AUG_DIR / f"fold{fold_id}" / "train_six_l2_plus_external.csv",
        )

        # A2: external only, six original labels only.
        save_csv(
            ext_six,
            AUG_DIR / f"fold{fold_id}" / "train_six_external_only.csv",
        )

        # B1: L2 + external, expanded label space.
        expanded_plus = pd.concat([l2_train, ext_expanded], ignore_index=True)
        save_csv(
            expanded_plus,
            AUG_DIR / f"fold{fold_id}" / "train_expanded_l2_plus_external.csv",
        )

        # B2: external only, expanded label space.
        save_csv(
            ext_expanded,
            AUG_DIR / f"fold{fold_id}" / "train_expanded_external_only.csv",
        )

        # Copy dev/test references as small helper files.
        dev = pd.read_csv(dev_path)
        test = pd.read_csv(test_path)

        save_csv(
            standardize_l2(dev, f"fold{fold_id}_dev"),
            AUG_DIR / f"fold{fold_id}" / "dev.csv",
        )
        save_csv(
            standardize_l2(test, f"fold{fold_id}_test"),
            AUG_DIR / f"fold{fold_id}" / "test.csv",
        )


def main():
    print("=" * 80)
    print("Making external augmented train sets")
    print("=" * 80)

    AUG_DIR.mkdir(parents=True, exist_ok=True)

    ext = load_external()

    print("\nExternal metadata:")
    print("Rows:", len(ext))
    print("Label counts:")
    print(ext["label"].value_counts().sort_index())

    ext_six, ext_expanded = make_external_train_sets(ext)
    make_fold_augmented_sets(ext_six, ext_expanded)

    print("\nDone.")
    print(f"Augmented CSVs saved under: {AUG_DIR}")


if __name__ == "__main__":
    main()