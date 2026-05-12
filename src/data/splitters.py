"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.constants import BASELINE_LABELS


STANDARD_COLUMNS = [
    "split",
    "index",
    "utterance_id",
    "audio_path",
    "speaker",
    "label",
    "gender",
    "text",
]


def load_and_clean_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    required = ["audio_path", "speaker", "label"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column in metadata: {col}")

    df = df.copy()

    df["audio_path"] = df["audio_path"].astype(str)
    df["speaker"] = df["speaker"].astype(str).str.upper()
    df["label"] = df["label"].astype(str)

    df = df[df["audio_path"].str.len() > 0]
    df = df[df["speaker"].str.len() > 0]
    df = df[df["label"].str.len() > 0]
    df = df[df["label"].isin(BASELINE_LABELS)]

    exists_mask = df["audio_path"].apply(lambda p: Path(p).exists())
    missing_count = len(df) - int(exists_mask.sum())

    if missing_count > 0:
        print(f"Warning: dropping {missing_count} rows with missing audio files.")

    df = df[exists_mask].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid rows after cleaning metadata.")

    if "utterance_id" not in df.columns:
        df["utterance_id"] = df.index.map(lambda i: f"utt_{i:06d}")

    if "gender" not in df.columns:
        df["gender"] = ""

    if "text" not in df.columns:
        df["text"] = ""

    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = [c for c in STANDARD_COLUMNS if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def save_split(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = reorder_columns(df)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved {len(df):6d} rows -> {path}")


def print_summary(name: str, df: pd.DataFrame):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    print(f"Rows:     {len(df)}")
    print(f"Speakers: {df['speaker'].nunique()}")

    print("\nLabel counts:")
    print(df["label"].value_counts().sort_index())

    print("\nSpeaker counts:")
    print(df["speaker"].value_counts().sort_index())


def check_no_utterance_overlap(parts: Dict[str, pd.DataFrame], key: str = "audio_path"):
    names = list(parts.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name = names[i]
            b_name = names[j]

            a = set(parts[a_name][key].astype(str))
            b = set(parts[b_name][key].astype(str))

            overlap = a & b
            if overlap:
                raise ValueError(
                    f"Utterance overlap found between {a_name} and {b_name}: "
                    f"{len(overlap)} overlapping rows."
                )


def check_no_speaker_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_speakers = set(train_df["speaker"].astype(str))
    test_speakers = set(test_df["speaker"].astype(str))

    overlap = train_speakers & test_speakers

    if overlap:
        raise ValueError(f"Speaker overlap found: {sorted(overlap)}")


def make_split1_file_level(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Paper-inspired Split-1.

    Source idea:
    Split-1 in the referenced L2-ARCTIC split protocol is a
    speaker-dependent, multi-accent split. Speakers can appear
    in Train, Dev, and Test, but utterances do not overlap.

    Adaptation here:
    We use an utterance-level 80/10/10 split stratified by accent label.
    """
    total = train_ratio + dev_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + dev_ratio + test_ratio must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(dev_ratio + test_ratio),
        random_state=seed,
        stratify=df["label"],
    )

    relative_test_ratio = test_ratio / (dev_ratio + test_ratio)

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=temp_df["label"],
    )

    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df["split"] = "split1_file_train"
    dev_df["split"] = "split1_file_dev"
    test_df["split"] = "split1_file_test"

    check_no_utterance_overlap(
        {
            "train": train_df,
            "dev": dev_df,
            "test": test_df,
        }
    )

    return train_df, dev_df, test_df


def get_speakers_by_label(df: pd.DataFrame) -> Dict[str, List[str]]:
    speaker_table = (
        df[["speaker", "label"]]
        .drop_duplicates()
        .sort_values(["label", "speaker"])
        .reset_index(drop=True)
    )

    speakers_by_label = {}

    for label, group in speaker_table.groupby("label"):
        speakers = sorted(group["speaker"].unique())
        speakers_by_label[label] = speakers

    return speakers_by_label


def validate_l2arctic_speaker_structure(speakers_by_label: Dict[str, List[str]]):
    print("\nSpeakers by label:")

    for label in BASELINE_LABELS:
        speakers = speakers_by_label.get(label, [])
        print(f"  {label:12s}: {speakers}")

        if len(speakers) < 2:
            raise ValueError(f"Label {label} has fewer than 2 speakers.")

    missing = set(BASELINE_LABELS) - set(speakers_by_label.keys())
    if missing:
        raise ValueError(f"Missing labels in metadata: {sorted(missing)}")


def make_split2_speaker_disjoint_folds(
    df: pd.DataFrame,
    seed: int,
    dev_ratio_within_train_speakers: float = 0.1,
) -> List[Dict]:
    """
    Paper-inspired Split-2.

    Source idea:
    Split-2 in the referenced L2-ARCTIC split protocol is a
    speaker-independent cross-validation split with multiple accents.
    In each fold, one speaker from each accent is removed from Train/Dev
    and used as Test. Other speakers with the same accent remain in Train/Dev.

    Adaptation here:
    Since L2-ARCTIC has four speakers per accent, we create four folds.
    In each fold:
      - one speaker per accent is held out for Test
      - the remaining speakers are split into Train/Dev by utterance
      - Dev is 10% of utterances from training speakers
    """
    speakers_by_label = get_speakers_by_label(df)
    validate_l2arctic_speaker_structure(speakers_by_label)

    min_speakers_per_label = min(len(v) for v in speakers_by_label.values())
    n_folds = min_speakers_per_label

    folds = []

    for fold_id in range(n_folds):
        test_speakers = []
        train_speakers = []

        for label in BASELINE_LABELS:
            speakers = speakers_by_label[label]

            test_speaker = speakers[fold_id]
            train_speaker_list = [s for s in speakers if s != test_speaker]

            test_speakers.append(test_speaker)
            train_speakers.extend(train_speaker_list)

        train_dev_df = df[df["speaker"].isin(train_speakers)].reset_index(drop=True)
        test_df = df[df["speaker"].isin(test_speakers)].reset_index(drop=True)

        train_df, dev_df = train_test_split(
            train_dev_df,
            test_size=dev_ratio_within_train_speakers,
            random_state=seed + fold_id,
            stratify=train_dev_df["label"],
        )

        train_df = train_df.reset_index(drop=True)
        dev_df = dev_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df["split"] = f"split2_fold{fold_id}_train"
        dev_df["split"] = f"split2_fold{fold_id}_dev"
        test_df["split"] = f"split2_fold{fold_id}_test"

        check_no_utterance_overlap(
            {
                "train": train_df,
                "dev": dev_df,
                "test": test_df,
            }
        )

        check_no_speaker_overlap(train_df, test_df)
        check_no_speaker_overlap(dev_df, test_df)

        folds.append(
            {
                "fold_id": fold_id,
                "train": train_df,
                "dev": dev_df,
                "test": test_df,
                "train_speakers": sorted(train_speakers),
                "test_speakers": sorted(test_speakers),
            }
        )

    return folds


def write_split1_outputs(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    split_dir: Path,
):
    save_split(train_df, split_dir / "split1_file_level_train.csv")
    save_split(dev_df, split_dir / "split1_file_level_dev.csv")
    save_split(test_df, split_dir / "split1_file_level_test.csv")

    save_split(train_df, output_dir / "train_file_level.csv")
    save_split(dev_df, output_dir / "val_file_level.csv")
    save_split(test_df, output_dir / "test_file_level.csv")


def write_split2_outputs(folds: List[Dict], output_dir: Path, split_dir: Path):
    for fold in folds:
        fold_id = fold["fold_id"]
        fold_dir = split_dir / f"split2_speaker_disjoint_fold{fold_id}"

        save_split(fold["train"], fold_dir / "train.csv")
        save_split(fold["dev"], fold_dir / "dev.csv")
        save_split(fold["test"], fold_dir / "test.csv")

        speaker_info_path = fold_dir / "speakers.txt"
        speaker_info_path.write_text(
            "\n".join(
                [
                    f"fold_id: {fold_id}",
                    f"train_speakers: {fold['train_speakers']}",
                    f"test_speakers: {fold['test_speakers']}",
                ]
            ),
            encoding="utf-8",
        )
        print(f"Saved speaker info -> {speaker_info_path}")

    fold0 = folds[0]

    save_split(fold0["train"], output_dir / "train_speaker_disjoint.csv")
    save_split(fold0["dev"], output_dir / "dev_speaker_disjoint.csv")
    save_split(fold0["test"], output_dir / "val_speaker_disjoint.csv")
    save_split(fold0["test"], output_dir / "test_speaker_disjoint.csv")


def print_split2_fold_summary(folds: List[Dict]):
    print("\n" + "=" * 80)
    print("Split-2 speaker-disjoint fold summary")
    print("=" * 80)

    for fold in folds:
        fold_id = fold["fold_id"]

        print(f"\nFold {fold_id}")
        print("-" * 80)
        print("Held-out test speakers:", fold["test_speakers"])
        print("Train rows:", len(fold["train"]))
        print("Dev rows:  ", len(fold["dev"]))
        print("Test rows: ", len(fold["test"]))

        print("Test label counts:")
        print(fold["test"]["label"].value_counts().sort_index())