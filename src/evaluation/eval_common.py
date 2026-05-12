"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import csv
import json
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_from_disk
from transformers import pipeline


BASELINE_LABELS = [
    "Arabic",
    "Hindi",
    "Korean",
    "Mandarin",
    "Spanish",
    "Vietnamese",
]


def normalize_label(label: str) -> str:
    text = str(label).strip().lower()

    mapping = {
        "arabic": "Arabic",
        "hindi": "Hindi",
        "korean": "Korean",
        "mandarin": "Mandarin",
        "chinese": "Mandarin",
        "spanish": "Spanish",
        "vietnamese": "Vietnamese",
        "viet": "Vietnamese",
    }

    if text in mapping:
        return mapping[text]

    if text.startswith("ood_"):
        return str(label).strip()

    return str(label).strip()


def is_baseline_label(label: str) -> bool:
    return normalize_label(label) in BASELINE_LABELS


def safe_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def row_get_first(row: pd.Series, candidates: List[str], default: str = "") -> str:
    for col in candidates:
        if col in row and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value:
                return value
    return default


def get_true_label_from_row(row: pd.Series) -> Tuple[str, bool]:
    if "is_mapped_to_baseline" in row:
        label = row_get_first(row, ["target_label", "label", "true_label"], "")
        mapped = safe_bool(row["is_mapped_to_baseline"])
        return normalize_label(label), mapped

    label = row_get_first(
        row,
        ["true_label", "label", "speaker_native_language", "target_label", "original_label"],
        "",
    )

    if not label or label.upper() == "UNKNOWN":
        return "", False

    return normalize_label(label), True


def load_audio_file_mono_16k(audio_path: str):
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=16000,
        )
        waveform = resampler(waveform)

    return waveform.squeeze().numpy()


def resample_array_to_16k(audio_array, sample_rate: int):
    if sample_rate == 16000:
        return audio_array

    waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=16000,
    )
    waveform = resampler(waveform)

    return waveform.squeeze().numpy()


def load_audio_from_hf_value(audio_value: Any):
    if not isinstance(audio_value, dict):
        raise ValueError(f"Unsupported HF audio value type: {type(audio_value)}")

    if audio_value.get("bytes") is not None:
        audio_bytes = audio_value["bytes"]
        audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        return resample_array_to_16k(audio_array, sample_rate)

    if audio_value.get("path") is not None:
        return load_audio_file_mono_16k(audio_value["path"])

    raise ValueError("HF audio value has neither bytes nor path.")


def get_num_labels_from_pipeline(classifier) -> int:
    model = classifier.model
    if hasattr(model, "config") and hasattr(model.config, "num_labels"):
        return int(model.config.num_labels)
    return 6


def predict_audio_array(classifier, audio_array, top_k: int) -> Dict[str, Any]:
    safe_top_k = min(top_k, get_num_labels_from_pipeline(classifier))
    results = classifier(audio_array, top_k=safe_top_k)
    top = results[0]

    return {
        "pred_label": normalize_label(top["label"]),
        "pred_score": float(top["score"]),
        "all_results": results,
    }


def predict_audio_file(classifier, audio_path: str, top_k: int) -> Dict[str, Any]:
    audio_array = load_audio_file_mono_16k(audio_path)
    return predict_audio_array(classifier, audio_array, top_k=top_k)


def build_classifier(model_path: str, device: int, local_files_only: bool = True):
    return pipeline(
        task="audio-classification",
        model=model_path,
        device=device,
        local_files_only=local_files_only,
    )


def load_metadata_sample(
    metadata_path: Path,
    samples: int,
    seed: int,
    split: str = "all",
) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if split != "all" and "split" in df.columns:
        df = df[df["split"].astype(str) == split].copy()

    if "audio_path" not in df.columns:
        raise ValueError(f"No audio_path column in {metadata_path}")

    df = df[df["audio_path"].notna()].copy()
    df = df[df["audio_path"].astype(str).str.len() > 0].copy()

    exists_mask = df["audio_path"].apply(lambda p: Path(str(p)).exists())
    missing = len(df) - int(exists_mask.sum())

    if missing > 0:
        print(f"Warning: dropping {missing} rows with missing audio files from {metadata_path}")

    df = df[exists_mask].copy()

    if len(df) == 0:
        raise ValueError(f"No valid audio rows found in {metadata_path}")

    if samples > 0:
        n = min(samples, len(df))
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def run_user_audio_source(
    classifier,
    model_name: str,
    audio_paths: List[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    rows = []

    for i, audio_path in enumerate(audio_paths):
        path = Path(audio_path)

        if not path.exists():
            rows.append(
                {
                    "model_name": model_name,
                    "source": "user_audio",
                    "sample_id": i,
                    "audio_path": str(path),
                    "pred_label": "ERROR",
                    "error": "audio file not found",
                }
            )
            continue

        try:
            pred = predict_audio_file(classifier, str(path), top_k=top_k)

            out = {
                "model_name": model_name,
                "source": "user_audio",
                "sample_id": i,
                "audio_path": str(path),
                "true_label": "",
                "has_comparable_label": False,
                "pred_label": pred["pred_label"],
                "pred_score": pred["pred_score"],
                "correct": "",
            }

            for item in pred["all_results"]:
                out[f"score_{normalize_label(item['label'])}"] = float(item["score"])

            rows.append(out)

        except Exception as e:
            rows.append(
                {
                    "model_name": model_name,
                    "source": "user_audio",
                    "sample_id": i,
                    "audio_path": str(path),
                    "pred_label": "ERROR",
                    "error": str(e),
                }
            )

    return rows


def run_metadata_source(
    classifier,
    model_name: str,
    source_name: str,
    df: pd.DataFrame,
    top_k: int,
) -> List[Dict[str, Any]]:
    rows = []

    for i, row in df.iterrows():
        audio_path = str(row["audio_path"])
        true_label, has_label = get_true_label_from_row(row)

        speaker = row_get_first(row, ["speaker", "speaker_code", "client_id"], "")
        utterance_id = row_get_first(row, ["utterance_id", "index", "source_row_id"], str(i))

        try:
            pred = predict_audio_file(classifier, audio_path, top_k=top_k)
            pred_label = pred["pred_label"]

            correct = ""
            if has_label:
                correct = pred_label == true_label

            out = {
                "model_name": model_name,
                "source": source_name,
                "sample_id": i,
                "audio_path": audio_path,
                "speaker": speaker,
                "utterance_id": utterance_id,
                "true_label": true_label,
                "has_comparable_label": has_label,
                "pred_label": pred_label,
                "pred_score": pred["pred_score"],
                "correct": correct,
            }

            for col in [
                "split",
                "index",
                "dataset",
                "hf_name",
                "source_config",
                "original_label",
                "target_label",
                "is_mapped_to_baseline",
                "text",
                "label",
            ]:
                if col in row:
                    out[col] = row[col]

            for item in pred["all_results"]:
                out[f"score_{normalize_label(item['label'])}"] = float(item["score"])

            rows.append(out)

        except Exception as e:
            rows.append(
                {
                    "model_name": model_name,
                    "source": source_name,
                    "sample_id": i,
                    "audio_path": audio_path,
                    "speaker": speaker,
                    "utterance_id": utterance_id,
                    "true_label": true_label,
                    "has_comparable_label": has_label,
                    "pred_label": "ERROR",
                    "pred_score": "",
                    "correct": False if has_label else "",
                    "error": str(e),
                }
            )

    return rows


def summarize_prediction_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(rows)

    summary: Dict[str, Any] = {}

    if len(df) == 0:
        return {"rows": 0}

    summary["rows"] = int(len(df))
    valid = df[df["pred_label"] != "ERROR"].copy()
    summary["valid_rows"] = int(len(valid))

    if len(valid) == 0:
        return summary

    summary["prediction_counts"] = valid["pred_label"].value_counts().to_dict()

    if "has_comparable_label" in valid.columns:
        comparable = valid[
            valid["has_comparable_label"].astype(str).str.lower().isin(["true", "1"])
        ].copy()

        if len(comparable) > 0:
            comparable["correct"] = comparable["correct"].astype(bool)
            summary["comparable_rows"] = int(len(comparable))
            summary["accuracy"] = float(comparable["correct"].mean())
            summary["correct"] = int(comparable["correct"].sum())
            summary["confusion"] = pd.crosstab(
                comparable["true_label"],
                comparable["pred_label"],
            ).to_dict()

    if "dataset" in valid.columns:
        summary["prediction_by_dataset"] = pd.crosstab(
            valid["dataset"],
            valid["pred_label"],
        ).to_dict()

    if "label" in valid.columns:
        summary["prediction_by_label"] = pd.crosstab(
            valid["label"],
            valid["pred_label"],
        ).to_dict()

    return summary


def save_csv(rows: List[Dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"No rows to save: {output_path}")
        return

    keys = sorted(set().union(*[set(r.keys()) for r in rows]))

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


def save_json(data: Dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON: {output_path}")


def external_distribution(metadata_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(metadata_path)

    out: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
    }

    if "dataset" in df.columns:
        out["dataset_counts"] = df["dataset"].value_counts().to_dict()

    if "label" in df.columns:
        out["label_counts"] = df["label"].value_counts().to_dict()

    if "is_mapped_to_baseline" in df.columns:
        mapped = df[df["is_mapped_to_baseline"].astype(str).str.lower().isin(["true", "1"])]
        ood = df[~df["is_mapped_to_baseline"].astype(str).str.lower().isin(["true", "1"])]

        out["mapped_rows"] = int(len(mapped))
        out["ood_rows"] = int(len(ood))
        out["mapped_label_counts"] = mapped["label"].value_counts().to_dict()
        out["ood_label_counts"] = ood["label"].value_counts().to_dict()

    return out