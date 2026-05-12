"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import csv
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

def save_dataframe(df: pd.DataFrame, path: Path, reset_index: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df.copy()

    if reset_index:
        out["index"] = range(len(out))

    out.to_csv(path, index=False, encoding="utf-8")

    print(f"Saved {len(out):7d} rows -> {path}")

    if "label" in out.columns and len(out) > 0:
        print(out["label"].value_counts().sort_index())


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()

    for col in columns:
        if col not in out.columns:
            out[col] = ""

    return out


def check_audio_exists(df: pd.DataFrame, name: str = "metadata") -> pd.DataFrame:
    if "audio_path" not in df.columns:
        raise ValueError(f"{name} does not contain audio_path column.")

    out = df.copy()
    before = len(out)

    out["audio_path"] = out["audio_path"].astype(str)
    mask = out["audio_path"].apply(lambda p: Path(p).exists())

    missing = before - int(mask.sum())

    if missing > 0:
        print(f"Warning: {name}: dropping {missing} rows with missing audio files.")

    return out[mask].reset_index(drop=True)

def save_csv(rows: List[Dict], path: Path, standard_first=None):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"No rows to save for {path}")
        return

    keys = sorted(set().union(*[set(r.keys()) for r in rows]))

    if standard_first is None:
        standard_first = []

    remaining = [k for k in keys if k not in standard_first]
    fieldnames = standard_first + remaining

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {path}")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)