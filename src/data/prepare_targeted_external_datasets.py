"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import argparse
import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, get_dataset_config_names, get_dataset_split_names, load_dataset
from tqdm import tqdm


TARGET_ROOT = Path("data/targeted_external")
OUTPUT_METADATA = Path("data/processed/targeted_external_metadata.csv")
REPORT_PATH = Path("results/logs/targeted_external_report.json")


TARGET_LABELS = {
    "OOD_Singaporean_English",
    "OOD_Malaysian_English",
    "OOD_British_Isles_English",
}


DATASET_PRESETS = {
    # Dedicated Singapore / Singlish sources.
    # MNSC is derived from IMDA National Speech Corpus and targets Singapore local accent / Singlish tasks.
    "mnsc_v1": {
        "hf_name": "MERaLiON/Multitask-National-Speech-Corpus-v1",
        "configs": "auto",
        "splits": "auto",
        "target_label_from_preset": "OOD_Singaporean_English",
        "label_candidates": ["accent", "label", "language", "locale", "speaker_accent", "country"],
        "speaker_candidates": ["speaker", "speaker_id", "client_id", "id", "part1_id", "part2_id"],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": ["sentence", "text", "transcription", "transcript", "answer", "instruction"],
        "audio_candidates": ["audio", "context", "file", "path", "audio_path", "mp3", "wav"],
    },

    "mnsc_v1_extend": {
        "hf_name": "AudioLLMs/Multitask-National-Speech-Corpus-v1-extend",
        "configs": "auto",
        "splits": "auto",
        "target_label_from_preset": "OOD_Singaporean_English",
        "label_candidates": ["accent", "label", "language", "locale", "speaker_accent", "country"],
        "speaker_candidates": ["speaker", "speaker_id", "client_id", "id", "part1_id", "part2_id"],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": ["sentence", "text", "transcription", "transcript", "answer", "instruction"],
        "audio_candidates": ["audio", "context", "file", "path", "audio_path", "mp3", "wav"],
    },

    # IMDA National Speech Corpus mirror. Depending on HF structure, this may or may not load directly.
    "imda_stt": {
        "hf_name": "mesolitica/IMDA-STT",
        "configs": "auto",
        "splits": "auto",
        "target_label_from_preset": "OOD_Singaporean_English",
        "label_candidates": ["accent", "label", "language", "locale", "speaker_accent", "country"],
        "speaker_candidates": ["speaker", "speaker_id", "client_id", "id"],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": ["sentence", "text", "transcription", "transcript"],
        "audio_candidates": ["audio", "context", "file", "path", "audio_path", "mp3", "wav"],
    },

    # Small Singlish dataset. Useful if it exposes audio through datasets.
    "singlish_speaker2050": {
        "hf_name": "cesinsingapore/singlish-speaker2050",
        "configs": "auto",
        "splits": "auto",
        "target_label_from_preset": "OOD_Singaporean_English",
        "label_candidates": ["accent", "label", "language", "locale", "speaker_accent", "country"],
        "speaker_candidates": ["speaker", "speaker_id", "client_id", "id"],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": ["sentence", "text", "transcription", "transcript"],
        "audio_candidates": ["audio", "context", "file", "path", "audio_path", "mp3", "wav"],
    },

    # Keep this as optional fallback only. Not used by default.
    "common_accent": {
        "hf_name": "DTU54DL/common-accent",
        "configs": [None],
        "splits": ["train"],
        "target_label_from_preset": None,
        "label_candidates": ["accent", "label", "class", "category"],
        "speaker_candidates": ["speaker", "speaker_id", "client_id", "id"],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": ["sentence", "text", "transcription"],
        "audio_candidates": ["audio", "file", "path", "audio_path"],
    },
}


def safe_name(text: str, max_len: int = 120) -> str:
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    return text.strip("_")[:max_len] or "unknown"


def normalize_target_label(label: str) -> str:
    x = str(label).strip().lower()

    if "singapore" in x or "singlish" in x:
        return "OOD_Singaporean_English"

    if "malaysia" in x or "malaysian" in x or "malay english" in x:
        return "OOD_Malaysian_English"

    if any(k in x for k in ["irish", "scottish", "welsh", "northern", "southern", "midlands", "england", "british"]):
        return "OOD_British_Isles_English"

    return ""


def find_first_key(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in row and row[key] is not None:
            return key
    return None


def get_nested_value(obj: Any, key: str, default: str = "") -> str:
    if isinstance(obj, dict) and key in obj and obj[key] is not None:
        return str(obj[key])
    return default


def get_value(row: Dict[str, Any], candidates: List[str], default: str = "") -> str:
    for key in candidates:
        if key in row and row[key] is not None:
            value = row[key]

            if isinstance(value, dict):
                # Common nested speaker structure.
                for nested_key in ["speaker_id", "id", "gender", "part1_id", "part2_id"]:
                    nested = get_nested_value(value, nested_key, "")
                    if nested:
                        return nested

            text = str(value).strip()
            if text:
                return text

    return default


def get_audio_value(row: Dict[str, Any], candidates: List[str]):
    key = find_first_key(row, candidates)
    if key is None:
        return None, None
    return key, row.get(key)


def load_audio_from_value(audio_value):
    if audio_value is None:
        raise ValueError("audio_value is None")

    if isinstance(audio_value, dict):
        # Hugging Face Audio feature often has bytes/path.
        if audio_value.get("bytes") is not None:
            audio_bytes = audio_value["bytes"]
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            return audio_array, sample_rate

        if audio_value.get("array") is not None and audio_value.get("sampling_rate") is not None:
            audio_array = audio_value["array"]
            sample_rate = int(audio_value["sampling_rate"])

            if getattr(audio_array, "ndim", 1) > 1:
                audio_array = audio_array.mean(axis=1)

            return audio_array, sample_rate

        if audio_value.get("path") is not None:
            path = str(audio_value["path"])
            waveform, sample_rate = torchaudio.load(path)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            return waveform.squeeze().numpy(), sample_rate

        # Some datasets store audio under nested {"audio": {"path": ...}} style.
        for nested_key in ["audio", "context", "file", "mp3", "wav"]:
            if nested_key in audio_value and audio_value[nested_key] is not None:
                return load_audio_from_value(audio_value[nested_key])

    if isinstance(audio_value, str):
        path = Path(audio_value)

        if path.exists():
            waveform, sample_rate = torchaudio.load(str(path))

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            return waveform.squeeze().numpy(), sample_rate

    raise ValueError(f"Unsupported audio value: {type(audio_value)}")


def resample_to_16k(audio_array, sample_rate):
    if sample_rate == 16000:
        return audio_array, 16000

    waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=16000,
    )
    waveform = resampler(waveform)

    return waveform.squeeze().numpy(), 16000


def save_wav(audio_array, sample_rate: int, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_array, sample_rate)


def safe_get_configs(hf_name: str, requested_configs) -> List[Optional[str]]:
    if requested_configs != "auto":
        return requested_configs

    try:
        configs = get_dataset_config_names(hf_name)
        if not configs:
            return [None]

        # Avoid exploding over too many configs for giant datasets.
        return configs

    except Exception as e:
        print(f"Warning: could not list configs for {hf_name}: {e}")
        return [None]


def safe_get_splits(hf_name: str, config: Optional[str], requested_splits) -> List[str]:
    if requested_splits != "auto":
        return requested_splits

    try:
        if config is None:
            splits = get_dataset_split_names(hf_name)
        else:
            splits = get_dataset_split_names(hf_name, config)

        if not splits:
            return ["train"]

        return splits

    except Exception as e:
        print(f"Warning: could not list splits for {hf_name}/{config or 'default'}: {e}")
        return ["train"]


def load_hf_dataset(hf_name: str, config: Optional[str], split: str, streaming: bool):
    display = hf_name if config is None else f"{hf_name}/{config}"
    print(f"Loading dataset: {display}, split={split}, streaming={streaming}")

    if config is None:
        ds = load_dataset(hf_name, split=split, streaming=streaming)
    else:
        ds = load_dataset(hf_name, config, split=split, streaming=streaming)

    try:
        if hasattr(ds, "features") and ds.features is not None:
            for col, feature in ds.features.items():
                if isinstance(feature, Audio):
                    ds = ds.cast_column(col, Audio(decode=False))
    except Exception as e:
        print(f"Warning: could not cast audio column decode=False: {e}")

    return ds


def iter_limited(ds, max_scan: int):
    i = 0
    iterator = iter(ds)

    while True:
        if max_scan > 0 and i >= max_scan:
            break

        try:
            row = next(iterator)
        except StopIteration:
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"Warning: failed while reading dataset row {i}: {e}")
            break

        yield i, row
        i += 1


def all_target_labels_full(per_label_count: Dict[str, int], max_per_label: int, active_target_labels: List[str]) -> bool:
    if max_per_label <= 0:
        return False

    return all(
        per_label_count.get(label, 0) >= max_per_label
        for label in active_target_labels
    )


def infer_label_from_row_or_preset(preset: Dict[str, Any], row: Dict[str, Any]) -> Tuple[str, str]:
    preset_label = preset.get("target_label_from_preset")

    raw_label = get_value(row, preset["label_candidates"], default="")

    if preset_label:
        return preset_label, raw_label if raw_label else preset_label

    target_label = normalize_target_label(raw_label)
    return target_label, raw_label


def infer_speaker(row: Dict[str, Any], preset: Dict[str, Any], default: str) -> str:
    # Direct candidates.
    for key in preset["speaker_candidates"]:
        if key in row and row[key] is not None:
            value = row[key]

            if isinstance(value, dict):
                speaker_id = value.get("speaker_id") or value.get("id")
                part1_id = value.get("part1_id")
                part2_id = value.get("part2_id")

                if speaker_id:
                    return safe_name(speaker_id)

                if part1_id or part2_id:
                    return safe_name(f"{part1_id}_{part2_id}")

            text = str(value).strip()
            if text:
                return safe_name(text)

    # MNSC often has speaker nested dict.
    if "speaker" in row and isinstance(row["speaker"], dict):
        speaker = row["speaker"]
        speaker_id = speaker.get("speaker_id")
        if speaker_id:
            return safe_name(speaker_id)

    return safe_name(default)


def prepare_one(
    preset_name: str,
    preset: Dict[str, Any],
    config: Optional[str],
    split: str,
    max_total: int,
    max_per_label: int,
    max_scan: int,
    streaming: bool,
    global_index_start: int,
    active_target_labels: List[str],
):
    hf_name = preset["hf_name"]

    ds = load_hf_dataset(
        hf_name=hf_name,
        config=config,
        split=split,
        streaming=streaming,
    )

    rows = []
    per_label_count = {}
    skipped = 0

    config_name = config or "default"
    split_name = safe_name(split)

    out_dir = TARGET_ROOT / preset_name / safe_name(config_name) / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for row_id, row in tqdm(iter_limited(ds, max_scan), desc=f"{preset_name}:{config_name}:{split}"):
        if max_total > 0 and len(rows) >= max_total:
            print(f"Reached max_total={max_total}. Stopping {preset_name}/{config_name}/{split}.")
            break

        if all_target_labels_full(per_label_count, max_per_label, active_target_labels):
            print(f"Reached max_per_label={max_per_label} for all requested target labels. Stopping.")
            break

        target_label, raw_label = infer_label_from_row_or_preset(preset, row)

        if target_label not in active_target_labels:
            continue

        current_count = per_label_count.get(target_label, 0)

        if max_per_label > 0 and current_count >= max_per_label:
            continue

        audio_key, audio_value = get_audio_value(row, preset["audio_candidates"])

        if audio_key is None:
            skipped += 1

            if skipped <= 5:
                print(f"Warning: no audio key. keys={list(row.keys())}")

            if skipped >= 20 and len(rows) == 0:
                print(
                    f"Stopping {preset_name}/{config_name}/{split}: "
                    "many rows have no audio column. This dataset is probably text-only or incompatible."
                )
                break

            continue

        try:
            audio_array, sr = load_audio_from_value(audio_value)
            audio_array, sr = resample_to_16k(audio_array, sr)
        except Exception as e:
            skipped += 1

            if skipped <= 10:
                print(f"Warning: failed audio row={row_id}: {e}")

            continue

        utterance_id = f"{safe_name(preset_name)}_{safe_name(config_name)}_{safe_name(split)}_{row_id:08d}"
        wav_path = out_dir / target_label / f"{utterance_id}.wav"

        try:
            save_wav(audio_array, sr, wav_path)
        except Exception as e:
            skipped += 1

            if skipped <= 10:
                print(f"Warning: failed save row={row_id}: {e}")

            continue

        speaker = infer_speaker(
            row=row,
            preset=preset,
            default=f"{preset_name}_{config_name}_{split}_speaker_unknown",
        )

        gender = get_value(row, preset["gender_candidates"], default="")
        text = get_value(row, preset["text_candidates"], default="")

        rows.append(
            {
                "split": "targeted_external",
                "index": global_index_start + len(rows),
                "utterance_id": utterance_id,
                "audio_path": str(wav_path),
                "speaker": speaker,
                "label": target_label,
                "gender": gender,
                "text": text,
                "dataset": preset_name,
                "hf_name": hf_name,
                "source_config": config_name,
                "source_split": split,
                "source_row_id": row_id,
                "original_label": raw_label if raw_label else target_label,
                "target_label": target_label,
                "is_mapped_to_baseline": False,
            }
        )

        per_label_count[target_label] = per_label_count.get(target_label, 0) + 1

    print(f"Prepared {len(rows)} rows for {preset_name}/{config_name}/{split}. Skipped={skipped}")

    if rows:
        print("Counts by label:")
        print(pd.Series([r["label"] for r in rows]).value_counts())
        print("Per-label saved counter:")
        print(per_label_count)

    return rows


def save_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"No rows to save: {path}")
        return

    keys = sorted(set().union(*[r.keys() for r in rows]))

    standard = [
        "split",
        "index",
        "utterance_id",
        "audio_path",
        "speaker",
        "label",
        "gender",
        "text",
    ]

    fieldnames = standard + [k for k in keys if k not in standard]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "mnsc_v1",
            "mnsc_v1_extend",
            "imda_stt",
            "singlish_speaker2050",
        ],
        choices=list(DATASET_PRESETS.keys()),
    )

    parser.add_argument(
        "--target-labels",
        nargs="+",
        default=[
            "OOD_Singaporean_English",
        ],
        choices=sorted(TARGET_LABELS),
        help="Which target labels to collect. Dedicated Singapore datasets should normally use OOD_Singaporean_English.",
    )

    parser.add_argument("--max-total-per-config", type=int, default=2000)
    parser.add_argument("--max-per-label", type=int, default=1000)
    parser.add_argument("--max-scan", type=int, default=200000)
    parser.add_argument("--max-configs", type=int, default=20)
    parser.add_argument("--max-splits", type=int, default=3)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--output", type=str, default=str(OUTPUT_METADATA))

    args = parser.parse_args()

    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    failures = {}

    print("=" * 80)
    print("Preparing targeted dedicated Singapore/Malaysia/British external datasets")
    print("=" * 80)
    print("Datasets:", args.datasets)
    print("Target labels:", args.target_labels)
    print("Max total per config:", args.max_total_per_config)
    print("Max per label:", args.max_per_label)
    print("Max scan:", args.max_scan)
    print("Max configs:", args.max_configs)
    print("Max splits:", args.max_splits)
    print("Streaming:", not args.no_streaming)

    for dataset_name in args.datasets:
        preset = DATASET_PRESETS[dataset_name]
        hf_name = preset["hf_name"]

        configs = safe_get_configs(hf_name, preset["configs"])

        if args.max_configs > 0:
            configs = configs[:args.max_configs]

        for config in configs:
            splits = safe_get_splits(hf_name, config, preset["splits"])

            if args.max_splits > 0:
                splits = splits[:args.max_splits]

            for split in splits:
                try:
                    rows = prepare_one(
                        preset_name=dataset_name,
                        preset=preset,
                        config=config,
                        split=split,
                        max_total=args.max_total_per_config,
                        max_per_label=args.max_per_label,
                        max_scan=args.max_scan,
                        streaming=not args.no_streaming,
                        global_index_start=len(all_rows),
                        active_target_labels=args.target_labels,
                    )
                    all_rows.extend(rows)

                except Exception as e:
                    key = f"{dataset_name}/{config or 'default'}/{split}"
                    failures[key] = str(e)
                    print("!" * 80)
                    print(f"Failed: {key}")
                    print(e)
                    print("!" * 80)

    save_csv(all_rows, Path(args.output))

    summary = {
        "total_rows": len(all_rows),
        "datasets": pd.Series([r["dataset"] for r in all_rows]).value_counts().to_dict() if all_rows else {},
        "labels": pd.Series([r["label"] for r in all_rows]).value_counts().to_dict() if all_rows else {},
        "failures": failures,
    }

    REPORT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nMetadata saved to: {args.output}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()