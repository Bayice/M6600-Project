"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import argparse
import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_from_disk
from transformers import pipeline


REMOTE_MODEL_NAME = "kaysrubio/accent-id-distilhubert-finetuned-l2-arctic2"
LOCAL_MODEL_DIR = Path("models/hf/accent-id-distilhubert-finetuned-l2-arctic2")

OFFICIAL_METADATA = Path("data/processed/metadata_l2_arctic_official.csv")
EXTERNAL_METADATA = Path("data/processed/external_metadata.csv")
HF_DATASET_DIR = Path("data/raw/L2Arctic_hf")

DEFAULT_OUTPUT = Path("results/logs/exp0_baseline_predictions.csv")


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
    return mapping.get(text, str(label).strip())


def get_model_path(force_remote: bool = False) -> Tuple[str, bool]:
    if force_remote:
        print("Force remote loading enabled.")
        return REMOTE_MODEL_NAME, False

    config_path = LOCAL_MODEL_DIR / "config.json"
    preprocessor_path = LOCAL_MODEL_DIR / "preprocessor_config.json"

    if config_path.exists() and preprocessor_path.exists():
        print(f"Found local model: {LOCAL_MODEL_DIR}")
        return str(LOCAL_MODEL_DIR), True

    print("Local model not found. Falling back to Hugging Face Hub.")
    print("You can run this first:")
    print("python src\\download_baseline_model.py")
    return REMOTE_MODEL_NAME, False


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


def predict_audio_array(classifier, audio_array, top_k: int) -> Dict[str, Any]:
    results = classifier(audio_array, top_k=top_k)
    top = results[0]

    return {
        "pred_label": normalize_label(top["label"]),
        "pred_score": float(top["score"]),
        "all_results": results,
    }


def predict_audio_file(classifier, audio_path: str, top_k: int) -> Dict[str, Any]:
    audio_array = load_audio_file_mono_16k(audio_path)
    return predict_audio_array(classifier, audio_array, top_k=top_k)


def row_get_first(row: pd.Series, candidates: List[str], default: str = "") -> str:
    for col in candidates:
        if col in row and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value:
                return value
    return default


def get_true_label_from_row(row: pd.Series) -> Tuple[str, bool]:
    if "is_mapped_to_baseline" in row:
        mapped = row["is_mapped_to_baseline"]
        if isinstance(mapped, str):
            has_label = mapped.strip().lower() == "true"
        else:
            has_label = bool(mapped)

        label = row_get_first(row, ["target_label", "label", "true_label"], "")
        return normalize_label(label), has_label

    label = row_get_first(
        row,
        ["true_label", "label", "speaker_native_language", "target_label", "original_label"],
        "",
    )

    if not label or label.upper() == "UNKNOWN":
        return "", False

    return normalize_label(label), True


def load_metadata_rows(metadata_path: Path, samples: int, seed: int, split: str = "all") -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)

    if split != "all" and "split" in df.columns:
        df = df[df["split"].astype(str) == split].copy()

    if "audio_path" not in df.columns:
        raise ValueError(f"Metadata file has no audio_path column: {metadata_path}")

    df = df[df["audio_path"].notna()]
    df = df[df["audio_path"].astype(str).str.len() > 0]

    exists_mask = df["audio_path"].apply(lambda p: Path(str(p)).exists())
    missing_count = len(df) - int(exists_mask.sum())

    if missing_count > 0:
        print(f"Warning: dropping {missing_count} rows with missing audio_path files.")

    df = df[exists_mask].copy()

    if len(df) == 0:
        raise ValueError(f"No valid audio rows found in metadata: {metadata_path}")

    if samples > 0:
        n = min(samples, len(df))
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df


def run_user_audio(classifier, audio_paths: List[str], top_k: int) -> List[Dict[str, Any]]:
    rows = []

    print("\n" + "=" * 80)
    print("User-provided audio inference")
    print("=" * 80)

    for i, audio_path in enumerate(audio_paths):
        path = Path(audio_path)

        if not path.exists():
            print(f"Warning: user audio not found, skipping: {path}")
            continue

        pred = predict_audio_file(classifier, str(path), top_k=top_k)

        print(
            f"[{i+1}/{len(audio_paths)}] audio={path} "
            f"pred={pred['pred_label']} score={pred['pred_score']:.4f}"
        )

        rows.append(
            {
                "source": "user_audio",
                "sample_id": i,
                "audio_path": str(path),
                "speaker": "",
                "true_label": "",
                "has_comparable_label": False,
                "pred_label": pred["pred_label"],
                "pred_score": pred["pred_score"],
                "correct": "",
            }
        )

    return rows


def run_metadata_source(
    classifier,
    metadata_path: Path,
    source_name: str,
    samples: int,
    seed: int,
    split: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    df = load_metadata_rows(
        metadata_path=metadata_path,
        samples=samples,
        seed=seed,
        split=split,
    )

    rows = []

    print("\n" + "=" * 80)
    print(f"Baseline on metadata source: {source_name}")
    print(f"Metadata: {metadata_path}")
    print(f"Samples: {len(df)}")
    print("=" * 80)

    for i, row in df.iterrows():
        audio_path = str(row["audio_path"])
        true_label, has_label = get_true_label_from_row(row)
        speaker = row_get_first(row, ["speaker", "speaker_code", "client_id"], "")
        utterance_id = row_get_first(row, ["utterance_id", "index", "source_row_id"], str(i))

        try:
            pred = predict_audio_file(classifier, audio_path, top_k=top_k)
            pred_label = pred["pred_label"]
            pred_score = pred["pred_score"]

            correct = ""
            if has_label:
                correct = pred_label == true_label

            print(
                f"[{i+1}/{len(df)}] speaker={speaker:8s} "
                f"true={true_label or 'N/A':28s} pred={pred_label:10s} "
                f"score={pred_score:.4f} correct={correct}"
            )

            out = {
                "source": source_name,
                "sample_id": i,
                "audio_path": audio_path,
                "speaker": speaker,
                "utterance_id": utterance_id,
                "true_label": true_label,
                "has_comparable_label": has_label,
                "pred_label": pred_label,
                "pred_score": pred_score,
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
            ]:
                if col in row:
                    out[col] = row[col]

            for item in pred["all_results"]:
                out[f"score_{normalize_label(item['label'])}"] = float(item["score"])

            rows.append(out)

        except Exception as e:
            print(f"[{i+1}/{len(df)}] ERROR audio={audio_path}: {e}")

            rows.append(
                {
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

    print_summary(rows, source_name=source_name)

    return rows


def run_hf_source(classifier, hf_split: str, samples: int, seed: int, top_k: int) -> List[Dict[str, Any]]:
    if not HF_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"HF dataset not found: {HF_DATASET_DIR}\n"
            "Please run: python src\\data\\prepare_l2arctic.py"
        )

    ds = load_from_disk(str(HF_DATASET_DIR))

    if hf_split not in ds:
        raise ValueError(f"Split {hf_split} not found. Available splits: {list(ds.keys())}")

    split_ds = ds[hf_split].cast_column("audio", Audio(decode=False))

    n = min(samples, len(split_ds)) if samples > 0 else len(split_ds)
    sampled = split_ds.shuffle(seed=seed).select(range(n))

    rows = []

    print("\n" + "=" * 80)
    print(f"Baseline on HF L2Arctic dataset: split={hf_split}, samples={n}")
    print("=" * 80)

    for i, ex in enumerate(sampled):
        speaker = str(ex.get("speaker_code", "")).upper()
        true_label = normalize_label(ex.get("speaker_native_language", ""))

        try:
            audio_array = load_audio_from_hf_value(ex["audio"])
            pred = predict_audio_array(classifier, audio_array, top_k=top_k)

            pred_label = pred["pred_label"]
            pred_score = pred["pred_score"]
            correct = pred_label == true_label

            print(
                f"[{i+1}/{n}] speaker={speaker:8s} "
                f"true={true_label:12s} pred={pred_label:10s} "
                f"score={pred_score:.4f} correct={correct}"
            )

            out = {
                "source": "hf_l2arctic",
                "sample_id": i,
                "hf_split": hf_split,
                "speaker": speaker,
                "audio_path": "",
                "true_label": true_label,
                "has_comparable_label": True,
                "pred_label": pred_label,
                "pred_score": pred_score,
                "correct": correct,
                "text": ex.get("text", ""),
            }

            for item in pred["all_results"]:
                out[f"score_{normalize_label(item['label'])}"] = float(item["score"])

            rows.append(out)

        except Exception as e:
            print(f"[{i+1}/{n}] ERROR: {e}")

            rows.append(
                {
                    "source": "hf_l2arctic",
                    "sample_id": i,
                    "hf_split": hf_split,
                    "speaker": speaker,
                    "true_label": true_label,
                    "has_comparable_label": True,
                    "pred_label": "ERROR",
                    "pred_score": "",
                    "correct": False,
                    "error": str(e),
                }
            )

    print_summary(rows, source_name="hf_l2arctic")
    return rows


def print_summary(rows: List[Dict[str, Any]], source_name: str):
    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("No rows to summarize.")
        return

    valid = df[df["pred_label"] != "ERROR"].copy()

    print("\n" + "=" * 80)
    print(f"Summary: {source_name}")
    print("=" * 80)
    print(f"Rows: {len(df)}")
    print(f"Valid predictions: {len(valid)}")

    if len(valid) == 0:
        return

    print("\nPrediction counts:")
    print(valid["pred_label"].value_counts())

    if "has_comparable_label" in valid.columns:
        comparable = valid[
            valid["has_comparable_label"].astype(str).str.lower().isin(["true", "1"])
        ].copy()

        if len(comparable) > 0:
            comparable["correct"] = comparable["correct"].astype(bool)
            acc = comparable["correct"].mean()

            print("\nComparable-label accuracy:")
            print(f"Correct: {comparable['correct'].sum()} / {len(comparable)}")
            print(f"Accuracy: {acc:.4f}")

            print("\nConfusion table:")
            print(pd.crosstab(comparable["true_label"], comparable["pred_label"]))

    if "dataset" in valid.columns:
        print("\nPrediction counts by external dataset:")
        print(pd.crosstab(valid["dataset"], valid["pred_label"]))

    if "target_label" in valid.columns:
        print("\nPrediction counts by target label:")
        print(pd.crosstab(valid["target_label"], valid["pred_label"]))


def save_results(rows: List[Dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print("No rows to save.")
        return

    all_keys = sorted(set().union(*[set(r.keys()) for r in rows]))

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["user_audio", "official", "hf", "external", "metadata"],
    )
    parser.add_argument("--audio", nargs="*", default=[])
    parser.add_argument("--metadata", type=str, default="")
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--hf-split", type=str, default="scripted", choices=["scripted", "spontaneous"])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--force-remote", action="store_true")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))

    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1

    if device == 0:
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    model_path, use_local_only = get_model_path(force_remote=args.force_remote)
    print("Loading model:", model_path)

    classifier = pipeline(
        task="audio-classification",
        model=model_path,
        device=device,
        local_files_only=use_local_only,
    )

    if args.source == "user_audio":
        rows = run_user_audio(
            classifier=classifier,
            audio_paths=args.audio,
            top_k=args.top_k,
        )

    elif args.source == "official":
        rows = run_metadata_source(
            classifier=classifier,
            metadata_path=OFFICIAL_METADATA,
            source_name="official_l2arctic",
            samples=args.samples,
            seed=args.seed,
            split=args.split,
            top_k=args.top_k,
        )

    elif args.source == "external":
        rows = run_metadata_source(
            classifier=classifier,
            metadata_path=EXTERNAL_METADATA,
            source_name="external",
            samples=args.samples,
            seed=args.seed,
            split=args.split,
            top_k=args.top_k,
        )

    elif args.source == "metadata":
        if not args.metadata:
            raise ValueError("--metadata is required when --source metadata")

        rows = run_metadata_source(
            classifier=classifier,
            metadata_path=Path(args.metadata),
            source_name="custom_metadata",
            samples=args.samples,
            seed=args.seed,
            split=args.split,
            top_k=args.top_k,
        )

    elif args.source == "hf":
        rows = run_hf_source(
            classifier=classifier,
            hf_split=args.hf_split,
            samples=args.samples,
            seed=args.seed,
            top_k=args.top_k,
        )

    else:
        raise ValueError(f"Unsupported source: {args.source}")

    save_results(rows, Path(args.output))


if __name__ == "__main__":
    main()

'''
python src\exp0_run_baseline.py --source external --samples 1000 --seed 42 --output results\logs\exp0_external_1000.csv

python src\exp0_run_baseline.py --source official --samples 300 --seed 42 --output results\logs\exp0_official_300.csv

python src\exp0_run_baseline.py --source hf --hf-split scripted --samples 300 --seed 42 --output results\logs\exp0_hf_300.csv


'''