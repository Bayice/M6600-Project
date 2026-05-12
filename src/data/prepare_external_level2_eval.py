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
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, get_dataset_config_names, get_dataset_split_names, load_dataset
from tqdm import tqdm


OUTPUT_ROOT = Path("data/external_level2")
OUTPUT_METADATA = Path("data/processed/eval/external_level2_test.csv")
REPORT_PATH = Path("results/logs/external_level2_report.json")


TARGET_LABELS = [
    "Arabic",
    "Hindi",
    "Korean",
    "Mandarin",
    "Spanish",
    "Vietnamese",
    "OOD_British_Isles_English",
    "OOD_Singaporean_English",
]


# This script builds a held-out second-source external evaluation set.
# It must NOT be used for supervised training or pseudo-label generation.


DATASET_PRESETS = {
    # English as a Second Language TTS dataset.
    # Useful for Arabic/Hindi/Korean/Mandarin/Spanish/Vietnamese accented English
    # if the dataset exposes native-language / accent metadata.
    "esltts": {
        "hf_name": "MushanW/ESLTTS",
        "configs": "auto",
        "splits": "auto",
        "forced_label": None,
        "label_candidates": [
            "native_language",
            "l1",
            "L1",
            "first_language",
            "accent",
            "language",
            "speaker_native_language",
            "country",
            "locale",
            "label",
        ],
        "speaker_candidates": [
            "speaker",
            "speaker_id",
            "client_id",
            "id",
            "speaker_name",
            "spk_id",
        ],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": [
            "sentence",
            "text",
            "transcription",
            "transcript",
            "normalized_text",
        ],
        "audio_candidates": ["audio", "wav", "file", "path", "audio_path", "mp3"],
    },

    # Worldwide English accent corpus. Useful if it exposes accent/country metadata.
    "globe": {
        "hf_name": "MushanW/GLOBE",
        "configs": "auto",
        "splits": "auto",
        "forced_label": None,
        "label_candidates": [
            "accent",
            "native_language",
            "l1",
            "country",
            "region",
            "locale",
            "label",
            "language",
        ],
        "speaker_candidates": [
            "speaker",
            "speaker_id",
            "client_id",
            "id",
            "speaker_name",
            "spk_id",
        ],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": [
            "sentence",
            "text",
            "transcription",
            "transcript",
            "normalized_text",
        ],
        "audio_candidates": ["audio", "wav", "file", "path", "audio_path", "mp3"],
    },

    "globe_v2": {
        "hf_name": "MushanW/GLOBE_V2",
        "configs": "auto",
        "splits": "auto",
        "forced_label": None,
        "label_candidates": [
            "accent",
            "native_language",
            "l1",
            "country",
            "region",
            "locale",
            "label",
            "language",
        ],
        "speaker_candidates": [
            "speaker",
            "speaker_id",
            "client_id",
            "id",
            "speaker_name",
            "spk_id",
        ],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": [
            "sentence",
            "text",
            "transcription",
            "transcript",
            "normalized_text",
        ],
        "audio_candidates": ["audio", "wav", "file", "path", "audio_path", "mp3"],
    },

    # Common Accent fallback.
    # Useful for multiple labels, but should be treated as weaker/less controlled.
    "common_accent": {
        "hf_name": "DTU54DL/common-accent",
        "configs": [None],
        "splits": ["train"],
        "forced_label": None,
        "label_candidates": ["accent", "label", "class", "category"],
        "speaker_candidates": ["speaker", "speaker_id", "client_id", "id"],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": ["sentence", "text", "transcription"],
        "audio_candidates": ["audio", "file", "path", "audio_path"],
    },

    # British Isles English dedicated dataset.
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
        "splits": ["train"],
        "forced_label": "OOD_British_Isles_English",
        "label_candidates": ["accent", "dialect", "region", "label", "class"],
        "speaker_candidates": ["speaker", "speaker_id", "client_id", "id"],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": ["sentence", "text", "transcription"],
        "audio_candidates": ["audio", "file", "path", "audio_path"],
    },

    # Dedicated Singaporean English / Singlish source.
    "mnsc_v1": {
        "hf_name": "MERaLiON/Multitask-National-Speech-Corpus-v1",
        "configs": "auto",
        "splits": "auto",
        "forced_label": "OOD_Singaporean_English",
        "label_candidates": ["accent", "label", "language", "locale", "country"],
        "speaker_candidates": [
            "speaker",
            "speaker_id",
            "client_id",
            "id",
            "part1_id",
            "part2_id",
        ],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": [
            "sentence",
            "text",
            "transcription",
            "transcript",
            "answer",
            "instruction",
        ],
        "audio_candidates": ["audio", "context", "file", "path", "audio_path", "mp3", "wav"],
    },

    "mnsc_v1_extend": {
        "hf_name": "AudioLLMs/Multitask-National-Speech-Corpus-v1-extend",
        "configs": "auto",
        "splits": "auto",
        "forced_label": "OOD_Singaporean_English",
        "label_candidates": ["accent", "label", "language", "locale", "country"],
        "speaker_candidates": [
            "speaker",
            "speaker_id",
            "client_id",
            "id",
            "part1_id",
            "part2_id",
        ],
        "gender_candidates": ["gender", "sex"],
        "text_candidates": [
            "sentence",
            "text",
            "transcription",
            "transcript",
            "answer",
            "instruction",
        ],
        "audio_candidates": ["audio", "context", "file", "path", "audio_path", "mp3", "wav"],
    },
}


def safe_name(text: str, max_len: int = 120) -> str:
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    return text.strip("_")[:max_len] or "unknown"


def normalize_label(raw_label: str) -> str:
    x = str(raw_label).strip().lower()

    if not x or x in {"none", "nan", "unknown"}:
        return ""

    if "arabic" in x:
        return "Arabic"
    if "hindi" in x or "india" in x or "indian" in x:
        return "Hindi"
    if "korean" in x or "korea" in x:
        return "Korean"
    if "mandarin" in x or "chinese" in x or "china" in x:
        return "Mandarin"
    if "spanish" in x or "spain" in x or "mexico" in x or "latin" in x:
        return "Spanish"
    if "vietnamese" in x or "vietnam" in x:
        return "Vietnamese"

    if "singapore" in x or "singlish" in x:
        return "OOD_Singaporean_English"

    if any(
        k in x
        for k in [
            "british",
            "irish",
            "scottish",
            "welsh",
            "england",
            "northern",
            "southern",
            "midlands",
        ]
    ):
        return "OOD_British_Isles_English"

    return ""


def english_like_text(text: str, min_words: int = 3) -> bool:
    text = str(text).strip()

    if not text:
        return False

    # Must contain English letters.
    if re.search(r"[A-Za-z]", text) is None:
        return False

    # Count alphabetic words.
    words = re.findall(r"[A-Za-z]+", text)

    if len(words) < min_words:
        return False

    # Avoid text that is mostly non-Latin.
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    total_non_space = len(re.sub(r"\s+", "", text))

    if total_non_space == 0:
        return False

    latin_ratio = latin_chars / total_non_space

    return latin_ratio >= 0.5


def find_first_key(row: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in row and row[key] is not None:
            return key
    return None


def value_to_string(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, dict):
        # Try common nested metadata values.
        for key in [
            "text",
            "sentence",
            "transcription",
            "transcript",
            "label",
            "accent",
            "language",
            "native_language",
            "country",
            "speaker_id",
            "id",
        ]:
            if key in value and value[key] is not None:
                return str(value[key])
        return ""

    return str(value)


def get_value(row: Dict[str, Any], candidates: List[str], default: str = "") -> str:
    for key in candidates:
        if key in row and row[key] is not None:
            text = value_to_string(row[key]).strip()
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


def safe_get_configs(hf_name: str, requested_configs, max_configs: int) -> List[Optional[str]]:
    if requested_configs != "auto":
        configs = requested_configs
    else:
        try:
            configs = get_dataset_config_names(hf_name)
            if not configs:
                configs = [None]
        except Exception as e:
            print(f"Warning: could not list configs for {hf_name}: {e}")
            configs = [None]

    if max_configs > 0:
        configs = configs[:max_configs]

    return configs


def safe_get_splits(hf_name: str, config: Optional[str], requested_splits, max_splits: int) -> List[str]:
    if requested_splits != "auto":
        splits = requested_splits
    else:
        try:
            if config is None:
                splits = get_dataset_split_names(hf_name)
            else:
                splits = get_dataset_split_names(hf_name, config)

            if not splits:
                splits = ["train"]
        except Exception as e:
            print(f"Warning: could not list splits for {hf_name}/{config or 'default'}: {e}")
            splits = ["train"]

    if max_splits > 0:
        splits = splits[:max_splits]

    return splits


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


def infer_speaker(row: Dict[str, Any], preset: Dict[str, Any], default: str) -> str:
    for key in preset["speaker_candidates"]:
        if key in row and row[key] is not None:
            value = row[key]

            if isinstance(value, dict):
                for nested_key in ["speaker_id", "id", "part1_id", "part2_id"]:
                    if nested_key in value and value[nested_key] is not None:
                        return safe_name(value[nested_key])

            text = str(value).strip()
            if text:
                return safe_name(text)

    return safe_name(default)


def all_targets_full(counts: Dict[str, int], target_labels: List[str], max_per_label: int) -> bool:
    if max_per_label <= 0:
        return False

    return all(counts.get(label, 0) >= max_per_label for label in target_labels)


def prepare_one_dataset_config_split(
    preset_name: str,
    preset: Dict[str, Any],
    config: Optional[str],
    split: str,
    target_labels: List[str],
    max_per_label: int,
    max_total_per_dataset: int,
    max_scan: int,
    streaming: bool,
    global_start_index: int,
) -> List[Dict[str, Any]]:
    hf_name = preset["hf_name"]
    config_name = config or "default"

    ds = load_hf_dataset(
        hf_name=hf_name,
        config=config,
        split=split,
        streaming=streaming,
    )

    rows = []
    per_label_count = {}
    skipped = 0
    text_filtered = 0
    label_filtered = 0

    out_dir = OUTPUT_ROOT / preset_name / safe_name(config_name) / safe_name(split)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row_id, row in tqdm(
        iter_limited(ds, max_scan=max_scan),
        desc=f"{preset_name}:{config_name}:{split}",
    ):
        if max_total_per_dataset > 0 and len(rows) >= max_total_per_dataset:
            print(f"Reached max_total_per_dataset={max_total_per_dataset}.")
            break

        if all_targets_full(per_label_count, target_labels, max_per_label):
            print(f"Reached max_per_label={max_per_label} for all target labels.")
            break

        forced_label = preset.get("forced_label")
        raw_label = get_value(row, preset["label_candidates"], default="")

        if forced_label:
            label = forced_label
            raw_label_source = forced_label
        else:
            label = normalize_label(raw_label)
            raw_label_source = raw_label

        if label not in target_labels:
            label_filtered += 1
            continue

        if max_per_label > 0 and per_label_count.get(label, 0) >= max_per_label:
            continue

        text = get_value(row, preset["text_candidates"], default="")

        if not english_like_text(text):
            text_filtered += 1
            continue

        audio_key, audio_value = get_audio_value(row, preset["audio_candidates"])

        if audio_key is None:
            skipped += 1

            if skipped <= 5:
                print(f"Warning: no audio key. Row keys={list(row.keys())}")

            if skipped >= 20 and len(rows) == 0:
                print("Too many rows without audio. Stopping this dataset/config/split.")
                break

            continue

        try:
            audio_array, sr = load_audio_from_value(audio_value)
            audio_array, sr = resample_to_16k(audio_array, sr)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"Warning: failed to load audio row={row_id}: {e}")
            continue

        utterance_id = (
            f"level2_{safe_name(preset_name)}_"
            f"{safe_name(config_name)}_{safe_name(split)}_{row_id:08d}"
        )

        output_wav = out_dir / safe_name(label) / f"{utterance_id}.wav"

        try:
            save_wav(audio_array, sr, output_wav)
        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"Warning: failed to save wav row={row_id}: {e}")
            continue

        speaker = infer_speaker(
            row=row,
            preset=preset,
            default=f"{preset_name}_{config_name}_{split}_speaker_unknown",
        )

        gender = get_value(row, preset["gender_candidates"], default="")

        rows.append(
            {
                "split": "external_level2_test",
                "index": global_start_index + len(rows),
                "utterance_id": utterance_id,
                "audio_path": str(output_wav),
                "speaker": speaker,
                "label": label,
                "gender": gender,
                "text": text,
                "dataset": preset_name,
                "hf_name": hf_name,
                "source_config": config_name,
                "source_split": split,
                "source_row_id": row_id,
                "original_label": raw_label_source,
                "is_level2_eval": True,
                "is_mapped_to_baseline": label in {
                    "Arabic",
                    "Hindi",
                    "Korean",
                    "Mandarin",
                    "Spanish",
                    "Vietnamese",
                },
            }
        )

        per_label_count[label] = per_label_count.get(label, 0) + 1

    print(f"\nPrepared {len(rows)} rows for {preset_name}/{config_name}/{split}")
    print(f"Skipped audio errors/no audio: {skipped}")
    print(f"Filtered by label: {label_filtered}")
    print(f"Filtered by text: {text_filtered}")

    if rows:
        print(pd.Series([r["label"] for r in rows]).value_counts())

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


def cap_global_per_label(df: pd.DataFrame, target_labels: List[str], max_per_label: int, seed: int) -> pd.DataFrame:
    if len(df) == 0:
        return df

    parts = []

    for label in target_labels:
        group = df[df["label"] == label].copy()

        if len(group) == 0:
            continue

        if max_per_label > 0 and len(group) > max_per_label:
            group = group.sample(n=max_per_label, random_state=seed)

        parts.append(group)

    if not parts:
        return pd.DataFrame(columns=df.columns)

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["index"] = range(len(out))

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "esltts",
            "globe",
            "globe_v2",
            "common_accent",
            "english_dialects",
            "mnsc_v1",
            "mnsc_v1_extend",
        ],
        choices=list(DATASET_PRESETS.keys()),
    )

    parser.add_argument(
        "--target-labels",
        nargs="+",
        default=TARGET_LABELS,
        choices=TARGET_LABELS,
    )

    parser.add_argument("--max-per-label", type=int, default=1000)
    parser.add_argument("--max-total-per-dataset", type=int, default=12000)
    parser.add_argument("--max-scan", type=int, default=200000)
    parser.add_argument("--max-configs", type=int, default=20)
    parser.add_argument("--max-splits", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--output", type=str, default=str(OUTPUT_METADATA))

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    failures = {}

    print("=" * 80)
    print("Preparing precise external level-2 evaluation set")
    print("=" * 80)
    print("Datasets:", args.datasets)
    print("Target labels:", args.target_labels)
    print("Max per label:", args.max_per_label)
    print("Max total per dataset:", args.max_total_per_dataset)
    print("Max scan:", args.max_scan)
    print("Streaming:", not args.no_streaming)

    for dataset_name in args.datasets:
        preset = DATASET_PRESETS[dataset_name]
        hf_name = preset["hf_name"]

        configs = safe_get_configs(
            hf_name=hf_name,
            requested_configs=preset["configs"],
            max_configs=args.max_configs,
        )

        for config in configs:
            splits = safe_get_splits(
                hf_name=hf_name,
                config=config,
                requested_splits=preset["splits"],
                max_splits=args.max_splits,
            )

            for split in splits:
                try:
                    rows = prepare_one_dataset_config_split(
                        preset_name=dataset_name,
                        preset=preset,
                        config=config,
                        split=split,
                        target_labels=args.target_labels,
                        max_per_label=args.max_per_label,
                        max_total_per_dataset=args.max_total_per_dataset,
                        max_scan=args.max_scan,
                        streaming=not args.no_streaming,
                        global_start_index=len(all_rows),
                    )
                    all_rows.extend(rows)

                except Exception as e:
                    key = f"{dataset_name}/{config or 'default'}/{split}"
                    failures[key] = str(e)
                    print("!" * 80)
                    print(f"Failed: {key}")
                    print(e)
                    print("!" * 80)

    if all_rows:
        df = pd.DataFrame(all_rows)
        df = df.drop_duplicates(subset=["audio_path"]).reset_index(drop=True)
        df = cap_global_per_label(
            df=df,
            target_labels=args.target_labels,
            max_per_label=args.max_per_label,
            seed=args.seed,
        )
        rows_to_save = df.to_dict("records")
    else:
        df = pd.DataFrame()
        rows_to_save = []

    save_csv(rows_to_save, output_path)

    summary = {
        "total_rows": int(len(df)),
        "label_counts": df["label"].value_counts().to_dict() if len(df) else {},
        "dataset_counts": df["dataset"].value_counts().to_dict() if len(df) else {},
        "source_config_counts": df["source_config"].value_counts().head(50).to_dict()
        if len(df) and "source_config" in df.columns
        else {},
        "target_labels": args.target_labels,
        "max_per_label": args.max_per_label,
        "failures": failures,
    }

    REPORT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nMetadata saved to: {output_path}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()