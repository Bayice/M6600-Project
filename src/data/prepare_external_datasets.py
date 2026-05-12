"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from src.utils.audio_io import (
    load_audio_from_hf_value,
    load_audio_file_mono,
    resample_array_to_16k,
    resample_waveform,
    save_wav,
)
from src.utils.baseline_model import build_baseline_pipeline, predict_audio_file
from src.utils.constants import EXTERNAL_DIR, EXTERNAL_METADATA, LOG_DIR
from src.utils.csv_io import save_csv, save_json
from src.utils.hf_datasets import iter_rows, load_hf_split
from src.utils.labels import normalize_external_label_for_baseline, safe_name


PREDICTION_PATH = LOG_DIR / "external_baseline_predictions.csv"
SUMMARY_PATH = LOG_DIR / "external_baseline_summary.json"


DATASET_PRESETS = {
    "common_accent": {
        "hf_name": "DTU54DL/common-accent",
        "configs": [None],
        "split": "train",
        "label_candidates": ["accent", "label", "class", "category"],
        "speaker_candidates": ["speaker", "speaker_id", "client_id", "id"],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": ["sentence", "text", "transcription"],
        "audio_candidates": ["audio", "file", "path", "audio_path"],
    },
    "common_native": {
        "hf_name": "DTU54DL/common-native",
        "configs": [None],
        "split": "train",
        "label_candidates": ["accent", "label", "class", "category"],
        "speaker_candidates": ["speaker", "speaker_id", "client_id", "id"],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": ["sentence", "text", "transcription"],
        "audio_candidates": ["audio", "file", "path", "audio_path"],
    },
    "commonvoice_accent_test": {
        "hf_name": "DTU54DL/commonvoice_accent_test",
        "configs": [None],
        "split": "train",
        "label_candidates": ["accent", "label", "class", "category"],
        "speaker_candidates": ["speaker", "speaker_id", "client_id", "id"],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": ["sentence", "text", "transcription"],
        "audio_candidates": ["audio", "file", "path", "audio_path"],
    },
    "english_dialects": {
        "hf_name": "ylacombe/english_dialects",
        "configs": [
            "irish_male",
            "midlands_female",
            "midlands_male",
            "northern_female",
            "northern_male",
            "scottish_female",
            "scottish_male",
            "southern_female",
            "southern_male",
            "welsh_female",
            "welsh_male",
        ],
        "split": "train",
        "label_candidates": ["accent", "dialect", "region", "label", "class", "speaker_region"],
        "speaker_candidates": ["speaker", "speaker_id", "client_id", "id"],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": ["sentence", "text", "transcription"],
        "audio_candidates": ["audio", "file", "path", "audio_path"],
    },
    "svarah": {
        "hf_name": "ai4bharat/Svarah",
        "configs": [None],
        "split": "train",
        "label_candidates": [
            "accent",
            "native_language",
            "mother_tongue",
            "language",
            "state",
            "region",
            "district",
            "label",
        ],
        "speaker_candidates": ["speaker", "speaker_id", "client_id", "id"],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": ["sentence", "text", "transcription", "transcript"],
        "audio_candidates": ["audio", "file", "path", "audio_path"],
    },
}


STANDARD_EXTERNAL_COLUMNS = [
    "split",
    "index",
    "utterance_id",
    "audio_path",
    "speaker",
    "label",
    "gender",
    "text",
]


def find_first_existing_key(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in row and row[key] is not None:
            return key
    return None


def get_value(row: Dict[str, Any], candidates: List[str], default: str = "") -> str:
    key = find_first_existing_key(row, candidates)
    if key is None:
        return default

    value = row.get(key, default)
    return default if value is None else str(value)


def get_audio_value(row: Dict[str, Any], candidates: List[str]):
    key = find_first_existing_key(row, candidates)
    if key is None:
        return None, None
    return key, row.get(key)


def load_external_audio(audio_value):
    """
    Load audio from common HF audio formats and return a 16 kHz numpy array.
    """
    if audio_value is None:
        raise ValueError("audio_value is None")

    if isinstance(audio_value, dict):
        return load_audio_from_hf_value(audio_value)

    if isinstance(audio_value, str):
        path = Path(audio_value)
        if path.exists():
            waveform, sample_rate = load_audio_file_mono(str(path))
            waveform = resample_waveform(waveform, sample_rate, target_rate=16000)
            return waveform.squeeze().numpy()

    raise ValueError(f"Unsupported audio value type/content: {type(audio_value)}")


def prepare_one_dataset_config(
    preset_name: str,
    preset: Dict[str, Any],
    config: Optional[str],
    max_per_label: int,
    max_total: int,
    max_scan: int,
    streaming: bool,
    global_start_index: int,
) -> List[Dict[str, Any]]:
    hf_name = preset["hf_name"]
    split = preset["split"]
    config_name = config or "default"

    dataset_out_dir = EXTERNAL_DIR / preset_name / safe_name(config_name)
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_hf_split(
        hf_name=hf_name,
        config=config,
        split=split,
        streaming=streaming,
    )

    rows = []
    per_label_count = {}
    total_saved = 0
    skipped = 0

    print("\n" + "=" * 80)
    print(f"Preparing external dataset: {preset_name}, config={config_name}")
    print(f"HF name: {hf_name}")
    print("=" * 80)

    for row_id, row in enumerate(tqdm(iter_rows(ds, max_scan=max_scan), desc=f"{preset_name}:{config_name}")):
        if max_total > 0 and total_saved >= max_total:
            break

        original_label = get_value(row, preset["label_candidates"], default=config_name)
        if original_label == "UNKNOWN" and preset_name == "english_dialects":
            original_label = config_name

        target_label, is_mapped = normalize_external_label_for_baseline(original_label)

        current_count = per_label_count.get(target_label, 0)
        if current_count >= max_per_label:
            continue

        audio_key, audio_value = get_audio_value(row, preset["audio_candidates"])

        if audio_key is None:
            skipped += 1
            if skipped <= 5:
                print(f"Warning: no audio column found. Row keys={list(row.keys())}")
            continue

        try:
            audio_array = load_external_audio(audio_value)
            audio_array = resample_array_to_16k(audio_array, 16000)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"Warning: failed to load audio row={row_id}: {e}")
            continue

        utterance_id = f"{safe_name(preset_name)}_{safe_name(config_name)}_{row_id:06d}"
        output_wav = dataset_out_dir / safe_name(target_label) / f"{utterance_id}.wav"

        try:
            save_wav(audio_array, 16000, output_wav)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"Warning: failed to save wav row={row_id}: {e}")
            continue

        speaker = get_value(
            row,
            preset["speaker_candidates"],
            default=f"{preset_name}_{config_name}_speaker_unknown",
        )
        gender = get_value(row, preset["gender_candidates"], default="")
        text = get_value(row, preset["text_candidates"], default="")

        standard_index = global_start_index + len(rows)

        rows.append(
            {
                # L2-ARCTIC-like standard fields.
                "split": "external",
                "index": standard_index,
                "utterance_id": utterance_id,
                "audio_path": str(output_wav),
                "speaker": safe_name(speaker),
                "label": target_label,
                "gender": gender,
                "text": text,

                # Extra analysis fields.
                "dataset": preset_name,
                "hf_name": hf_name,
                "source_config": config_name,
                "source_split": split,
                "source_row_id": row_id,
                "original_label": original_label,
                "target_label": target_label,
                "is_mapped_to_baseline": is_mapped,
            }
        )

        per_label_count[target_label] = current_count + 1
        total_saved += 1

    print(f"\nPrepared {len(rows)} wav files for {preset_name}/{config_name}. Skipped: {skipped}")
    if rows:
        print("Counts by label:")
        print(pd.Series([r["label"] for r in rows]).value_counts())

    return rows


def prepare_external_datasets(
    dataset_names: List[str],
    max_per_label: int,
    max_total_per_config: int,
    max_scan: int,
    streaming: bool,
) -> List[Dict[str, Any]]:
    all_rows = []

    for name in dataset_names:
        preset = DATASET_PRESETS[name]

        for config in preset["configs"]:
            try:
                rows = prepare_one_dataset_config(
                    preset_name=name,
                    preset=preset,
                    config=config,
                    max_per_label=max_per_label,
                    max_total=max_total_per_config,
                    max_scan=max_scan,
                    streaming=streaming,
                    global_start_index=len(all_rows),
                )
                all_rows.extend(rows)
            except Exception as e:
                print("\n" + "!" * 80)
                print(f"Failed to prepare dataset {name}, config={config}: {e}")
                print("Continuing with other configs/datasets.")
                print("!" * 80)

    return all_rows


def run_baseline_on_external(metadata_rows: List[Dict[str, Any]], top_k: int, force_remote: bool):
    classifier = build_baseline_pipeline(force_remote=force_remote)

    pred_rows = []

    for row in tqdm(metadata_rows, desc="baseline"):
        audio_path = row["audio_path"]
        pred = predict_audio_file(classifier, audio_path, top_k=top_k)

        label = row["label"]
        is_mapped = bool(row["is_mapped_to_baseline"])

        correct = ""
        if is_mapped:
            correct = pred["pred_label"] == label

        out = dict(row)
        out.update(
            {
                "pred_label": pred["pred_label"],
                "pred_score": pred["pred_score"],
                "correct": correct,
            }
        )

        for item in pred["all_results"]:
            out[f"score_{item['label']}"] = float(item["score"])

        pred_rows.append(out)

    return pred_rows


def make_summary(pred_rows: List[Dict[str, Any]]):
    df = pd.DataFrame(pred_rows)

    summary = {}

    if len(df) == 0:
        return summary

    summary["total_rows"] = int(len(df))
    summary["datasets"] = df["dataset"].value_counts().to_dict()
    summary["label_counts"] = df["label"].value_counts().to_dict()
    summary["prediction_counts"] = df["pred_label"].value_counts().to_dict()

    mapped = df[df["is_mapped_to_baseline"] == True].copy()
    if len(mapped) > 0:
        mapped["correct"] = mapped["correct"].astype(bool)
        summary["mapped_rows"] = int(len(mapped))
        summary["mapped_accuracy"] = float(mapped["correct"].mean())
        summary["mapped_confusion"] = pd.crosstab(mapped["label"], mapped["pred_label"]).to_dict()

    ood = df[df["is_mapped_to_baseline"] == False].copy()
    if len(ood) > 0:
        summary["ood_rows"] = int(len(ood))
        summary["ood_prediction_counts"] = ood["pred_label"].value_counts().to_dict()
        summary["ood_by_label"] = pd.crosstab(ood["label"], ood["pred_label"]).to_dict()

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["common_accent"],
        choices=list(DATASET_PRESETS.keys()),
    )
    parser.add_argument("--max-per-label", type=int, default=30)
    parser.add_argument("--max-total-per-config", type=int, default=300)
    parser.add_argument("--max-scan", type=int, default=5000)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--force-remote", action="store_true")
    args = parser.parse_args()

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_METADATA.parent.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Preparing external accent datasets")
    print("=" * 80)
    print("Datasets:", args.datasets)
    print("Max per label:", args.max_per_label)
    print("Max total per config:", args.max_total_per_config)
    print("Max scan:", args.max_scan)
    print("Streaming:", not args.no_streaming)
    print("")

    all_rows = prepare_external_datasets(
        dataset_names=args.datasets,
        max_per_label=args.max_per_label,
        max_total_per_config=args.max_total_per_config,
        max_scan=args.max_scan,
        streaming=not args.no_streaming,
    )

    save_csv(all_rows, EXTERNAL_METADATA, standard_first=STANDARD_EXTERNAL_COLUMNS)

    if args.eval and all_rows:
        pred_rows = run_baseline_on_external(
            metadata_rows=all_rows,
            top_k=args.top_k,
            force_remote=args.force_remote,
        )

        save_csv(pred_rows, PREDICTION_PATH, standard_first=STANDARD_EXTERNAL_COLUMNS)

        summary = make_summary(pred_rows)
        save_json(summary, SUMMARY_PATH)

        print("\n" + "=" * 80)
        print("External baseline summary")
        print("=" * 80)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\nSummary saved to: {SUMMARY_PATH}")

    print("\nDone.")
    print(f"External metadata: {EXTERNAL_METADATA}")
    if args.eval:
        print(f"Predictions:        {PREDICTION_PATH}")


if __name__ == "__main__":
    main()