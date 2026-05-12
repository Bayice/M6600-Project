"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


STANDARD_FIRST = [
    "split",
    "index",
    "utterance_id",
    "audio_path",
    "speaker",
    "label",
    "gender",
    "text",
]


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        if col not in out.columns:
            out[col] = ""

    return out


def check_audio_exists(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()

    if "audio_path" not in out.columns:
        raise ValueError(f"{name} has no audio_path column.")

    out["audio_path"] = out["audio_path"].astype(str)
    mask = out["audio_path"].apply(lambda p: Path(p).exists())
    missing = int((~mask).sum())

    if missing > 0:
        print(f"Warning: {name}: dropping {missing} rows with missing audio files.")

    return out[mask].reset_index(drop=True)


def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()
    out["index"] = range(len(out))

    keys = list(out.columns)
    first = [c for c in STANDARD_FIRST if c in keys]
    rest = [c for c in keys if c not in first]
    out = out[first + rest]

    out.to_csv(path, index=False, encoding="utf-8")

    print(f"Saved {len(out)} rows -> {path}")
    print(out["label"].value_counts().sort_index())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-train-csv", type=str, required=True)
    parser.add_argument("--pseudo-train-csv", type=str, required=True)
    parser.add_argument("--output-train-csv", type=str, required=True)

    parser.add_argument(
        "--allowed-labels",
        type=str,
        required=True,
        help="Comma-separated label list. Rows outside these labels are dropped.",
    )

    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    allowed_labels = [x.strip() for x in args.allowed_labels.split(",") if x.strip()]
    allowed = set(allowed_labels)

    base_path = Path(args.base_train_csv)
    pseudo_path = Path(args.pseudo_train_csv)
    output_path = Path(args.output_train_csv)

    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if not pseudo_path.exists():
        raise FileNotFoundError(pseudo_path)

    print("=" * 80)
    print("Making semi-supervised train CSV")
    print("=" * 80)
    print("Base train:", base_path)
    print("Pseudo train:", pseudo_path)
    print("Output:", output_path)
    print("Allowed labels:", allowed_labels)

    base = pd.read_csv(base_path)
    pseudo = pd.read_csv(pseudo_path)

    base = ensure_columns(base, STANDARD_FIRST)
    pseudo = ensure_columns(pseudo, STANDARD_FIRST)

    base = check_audio_exists(base, "base train")
    pseudo = check_audio_exists(pseudo, "pseudo train")

    before_base = len(base)
    before_pseudo = len(pseudo)

    base = base[base["label"].astype(str).isin(allowed)].copy()
    pseudo = pseudo[pseudo["label"].astype(str).isin(allowed)].copy()

    print(f"Base rows before/after label filter: {before_base} -> {len(base)}")
    print(f"Pseudo rows before/after label filter: {before_pseudo} -> {len(pseudo)}")

    base["is_pseudo_labeled"] = base.get("is_pseudo_labeled", False)
    pseudo["is_pseudo_labeled"] = True

    combined = pd.concat([base, pseudo], ignore_index=True)

    if args.shuffle:
        combined = combined.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    save_df(combined, output_path)

    summary = {
        "base_train_csv": str(base_path),
        "pseudo_train_csv": str(pseudo_path),
        "output_train_csv": str(output_path),
        "base_rows": int(len(base)),
        "pseudo_rows": int(len(pseudo)),
        "combined_rows": int(len(combined)),
        "label_counts": combined["label"].value_counts().sort_index().to_dict(),
        "pseudo_label_counts": pseudo["label"].value_counts().sort_index().to_dict(),
        "allowed_labels": allowed_labels,
    }

    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()