"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from datasets.features import ClassLabel

from src.utils.constants import (
    FIGURE_DIR,
    L2ARCTIC_HF_DIR,
    L2ARCTIC_HF_METADATA,
    L2ARCTIC_HF_NAME,
    LOG_DIR,
    SPEAKER_TO_L1,
)
from src.utils.hf_datasets import (
    disable_audio_decoding,
    get_audio_column,
    get_split,
    get_split_names,
    load_or_download_dataset,
)
from src.utils.labels import normalize_l2_label
from src.utils.plots import generate_basic_metadata_figures


DEFAULT_REPORT_PATH = LOG_DIR / "l2arctic_inspection.txt"


SPEAKER_CANDIDATES = [
    "speaker_code",
    "speaker",
    "speaker_id",
    "spk_id",
    "speaker_name",
    "client_id",
    "user_id",
    "id",
]

LABEL_CANDIDATES = [
    "speaker_native_language",
    "accent",
    "l1",
    "L1",
    "first_language",
    "native_language",
    "language",
    "label",
    "class",
    "category",
]


def log_line(lines, text=""):
    print(text)
    lines.append(str(text))


def get_feature_label_name(ds_split, column: str, value: Any) -> str:
    feature = ds_split.features.get(column)

    if isinstance(feature, ClassLabel):
        try:
            return feature.int2str(int(value))
        except Exception:
            return str(value)

    return str(value)


def safe_get_path_from_audio(audio_value: Any) -> str:
    if isinstance(audio_value, dict):
        path = audio_value.get("path", "")
        return str(path) if path is not None else ""

    if isinstance(audio_value, str):
        return audio_value

    return ""


def infer_speaker_from_path(path_str: str) -> str:
    if not path_str:
        return ""

    parts = Path(path_str).parts

    for part in parts:
        p = part.upper()
        if p in SPEAKER_TO_L1:
            return p

    return ""


def infer_speaker(row: Dict[str, Any], audio_path: str) -> str:
    for col in SPEAKER_CANDIDATES:
        if col in row and row[col] is not None:
            value = str(row[col]).strip()
            if value:
                upper_value = value.upper()
                if upper_value in SPEAKER_TO_L1:
                    return upper_value
                return value

    return infer_speaker_from_path(audio_path)


def infer_label(row: Dict[str, Any], ds_split, speaker: str) -> Tuple[str, str]:
    for col in LABEL_CANDIDATES:
        if col in row and row[col] is not None:
            raw_value = row[col]
            label = get_feature_label_name(ds_split, col, raw_value).strip()

            if label:
                return normalize_l2_label(label), f"{col}={raw_value}"

    speaker_upper = str(speaker).upper()
    if speaker_upper in SPEAKER_TO_L1:
        return SPEAKER_TO_L1[speaker_upper], f"speaker_map={speaker_upper}"

    return "", ""


def try_get_duration_seconds(audio_value: Any) -> str:
    """
    Duration is optional. With Audio(decode=False), duration is usually unavailable.
    We leave it blank during metadata preparation.
    """
    try:
        if isinstance(audio_value, dict):
            array = audio_value.get("array", None)
            sampling_rate = audio_value.get("sampling_rate", None)

            if array is not None and sampling_rate:
                return f"{len(array) / float(sampling_rate):.3f}"
    except Exception:
        pass

    return ""


def inspect_dataset(ds, report_path: Path, max_examples: int = 2):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    log_line(lines, "=" * 80)
    log_line(lines, "L2-ARCTIC DATASET INSPECTION")
    log_line(lines, "=" * 80)
    log_line(lines, "")
    log_line(lines, f"Dataset object:\n{ds}")
    log_line(lines, "")

    for split in get_split_names(ds):
        ds_split = get_split(ds, split)

        log_line(lines, "-" * 80)
        log_line(lines, f"Split: {split}")
        log_line(lines, f"Number of rows: {len(ds_split)}")
        log_line(lines, f"Columns: {ds_split.column_names}")
        log_line(lines, "")
        log_line(lines, "Features:")
        log_line(lines, str(ds_split.features))
        log_line(lines, "")

        audio_col = get_audio_column(ds_split)
        log_line(lines, f"Detected audio column: {audio_col}")
        log_line(lines, "")

        n = min(max_examples, len(ds_split))
        for i in range(n):
            log_line(lines, f"Example {i}:")
            row = ds_split[i]

            for key, value in row.items():
                if key == audio_col:
                    if isinstance(value, dict):
                        summary = {
                            "path": value.get("path", ""),
                            "sampling_rate": value.get("sampling_rate", ""),
                            "array_shape": getattr(value.get("array", None), "shape", ""),
                        }
                        log_line(lines, f"  {key}: {summary}")
                    else:
                        log_line(lines, f"  {key}: {type(value)} {value}")
                else:
                    text = str(value)
                    if len(text) > 300:
                        text = text[:300] + "..."
                    log_line(lines, f"  {key}: {text}")

            log_line(lines, "")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nInspection report saved to: {report_path}")


def build_metadata(ds, metadata_path: Path) -> Path:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for split in get_split_names(ds):
        ds_split = get_split(ds, split)
        audio_col = get_audio_column(ds_split)

        if audio_col is None:
            print(f"Warning: no audio column detected for split={split}")

        for idx in range(len(ds_split)):
            row = ds_split[idx]

            audio_value = row.get(audio_col, None) if audio_col else None
            audio_path = safe_get_path_from_audio(audio_value)

            speaker = infer_speaker(row, audio_path)
            label, raw_label = infer_label(row, ds_split, speaker)
            duration = try_get_duration_seconds(audio_value)

            rows.append(
                {
                    "split": split,
                    "index": idx,
                    "audio_column": audio_col or "",
                    "audio_path": audio_path,
                    "speaker": speaker,
                    "label": label,
                    "raw_label_source": raw_label,
                    "duration_seconds": duration,
                }
            )

    fieldnames = [
        "split",
        "index",
        "audio_column",
        "audio_path",
        "speaker",
        "label",
        "raw_label_source",
        "duration_seconds",
    ]

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMetadata saved to: {metadata_path}")
    print(f"Total metadata rows: {len(rows)}")

    label_count = {}
    speaker_count = {}
    split_count = {}

    for r in rows:
        label = r["label"] or "UNKNOWN"
        speaker = r["speaker"] or "UNKNOWN"
        split = r["split"] or "UNKNOWN"

        label_count[label] = label_count.get(label, 0) + 1
        speaker_count[speaker] = speaker_count.get(speaker, 0) + 1
        split_count[split] = split_count.get(split, 0) + 1

    print("\nLabel counts:")
    for label, count in sorted(label_count.items()):
        print(f"  {label:12s}: {count}")

    print("\nSplit counts:")
    for split, count in sorted(split_count.items()):
        print(f"  {split:12s}: {count}")

    print("\nSpeaker counts, first 30:")
    for speaker, count in sorted(speaker_count.items())[:30]:
        print(f"  {speaker:12s}: {count}")

    summary_path = metadata_path.with_suffix(".summary.json")
    summary = {
        "total_rows": len(rows),
        "label_count": label_count,
        "speaker_count": speaker_count,
        "split_count": split_count,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")

    return metadata_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default=L2ARCTIC_HF_NAME)
    parser.add_argument("--save-dir", type=str, default=str(L2ARCTIC_HF_DIR))
    parser.add_argument("--metadata-path", type=str, default=str(L2ARCTIC_HF_METADATA))
    parser.add_argument("--report-path", type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--figure-dir", type=str, default=str(FIGURE_DIR))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    metadata_path = Path(args.metadata_path)
    report_path = Path(args.report_path)
    figure_dir = Path(args.figure_dir)

    print("=" * 80)
    print("Preparing Hugging Face L2-ARCTIC dataset")
    print("=" * 80)
    print(f"Dataset name: {args.dataset_name}")
    print(f"Save dir:     {save_dir}")
    print(f"Metadata:     {metadata_path}")
    print(f"Report:       {report_path}")
    print(f"Figure dir:   {figure_dir}")
    print("")

    ds = load_or_download_dataset(
        dataset_name=args.dataset_name,
        save_dir=save_dir,
        force_download=args.force_download,
    )

    ds = disable_audio_decoding(ds)

    inspect_dataset(ds, report_path=report_path)
    build_metadata(ds, metadata_path=metadata_path)

    if not args.skip_visualization:
        print("\nGenerating basic data visualizations...")
        df = pd.read_csv(metadata_path)
        generate_basic_metadata_figures(df, figure_dir, prefix="l2arctic")
        print("Visualization generation finished.")

    print("\nDone.")
    print("Generated files:")
    print(f"  1. {metadata_path}")
    print(f"  2. {metadata_path.with_suffix('.summary.json')}")
    print(f"  3. {report_path}")
    print(f"  4. {figure_dir / 'l2arctic_label_distribution.png'}")
    print(f"  5. {figure_dir / 'l2arctic_speaker_distribution.png'}")
    print(f"  6. {figure_dir / 'l2arctic_split_distribution.png'}")
    print(f"  7. {figure_dir / 'l2arctic_label_speaker_heatmap.png'}")


if __name__ == "__main__":
    main()