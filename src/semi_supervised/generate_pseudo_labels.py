"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import pipeline


DEFAULT_POOL_METADATA = Path("data/processed/external_metadata_plus_targeted.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/semi_supervised")


def parse_labels(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def load_audio_mono_16k(audio_path: str):
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


def load_excluded_audio_paths(csv_paths: List[str]) -> Set[str]:
    excluded = set()

    for csv_path in csv_paths:
        path = Path(csv_path)

        if not path.exists():
            print(f"Warning: exclude CSV not found, skipping: {path}")
            continue

        df = pd.read_csv(path)

        if "audio_path" not in df.columns:
            print(f"Warning: no audio_path column in exclude CSV, skipping: {path}")
            continue

        for p in df["audio_path"].dropna().astype(str):
            excluded.add(p)

    return excluded


def load_pool(
    pool_metadata: Path,
    excluded_audio_paths: Set[str],
    max_pool: int,
    seed: int,
) -> pd.DataFrame:
    if not pool_metadata.exists():
        raise FileNotFoundError(f"Pool metadata not found: {pool_metadata}")

    df = pd.read_csv(pool_metadata)

    if "audio_path" not in df.columns:
        raise ValueError(f"{pool_metadata} does not contain audio_path column.")

    df = df.copy()
    df["audio_path"] = df["audio_path"].astype(str)

    before = len(df)
    df = df[~df["audio_path"].isin(excluded_audio_paths)].copy()
    print(f"Pool rows before exclude: {before}")
    print(f"Excluded paths: {len(excluded_audio_paths)}")
    print(f"Pool rows after exclude: {len(df)}")

    exists_mask = df["audio_path"].apply(lambda p: Path(p).exists())
    missing = int((~exists_mask).sum())

    if missing > 0:
        print(f"Warning: dropping {missing} pool rows with missing audio files.")

    df = df[exists_mask].copy()

    if max_pool > 0 and len(df) > max_pool:
        df = df.sample(n=max_pool, random_state=seed).reset_index(drop=True)
        print(f"Subsampled pool to max_pool={max_pool}")
    else:
        df = df.reset_index(drop=True)

    return df


def build_classifier(model_path: str):
    device = 0 if torch.cuda.is_available() else -1

    if device == 0:
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    local_only = Path(model_path).exists()

    print(f"Loading teacher model: {model_path}")
    print(f"local_files_only={local_only}")

    return pipeline(
        task="audio-classification",
        model=model_path,
        device=device,
        local_files_only=local_only,
    )


def predict_pool(
    classifier,
    pool_df: pd.DataFrame,
    allowed_labels: List[str],
    confidence_threshold: float,
    top_k: int,
) -> pd.DataFrame:
    allowed = set(allowed_labels)
    rows = []

    for idx, row in tqdm(pool_df.iterrows(), total=len(pool_df), desc="pseudo-labeling"):
        audio_path = str(row["audio_path"])

        try:
            audio = load_audio_mono_16k(audio_path)
            results = classifier(audio, top_k=top_k)

            top = results[0]
            pred_label = str(top["label"])
            pred_score = float(top["score"])

            selected = (
                pred_label in allowed
                and pred_score >= confidence_threshold
            )

            out = row.to_dict()
            out.update(
                {
                    "pseudo_pred_label": pred_label,
                    "pseudo_pred_score": pred_score,
                    "pseudo_selected": selected,
                    "pseudo_error": "",
                }
            )

            for item in results:
                label = str(item["label"])
                out[f"score_{label}"] = float(item["score"])

            rows.append(out)

        except Exception as e:
            out = row.to_dict()
            out.update(
                {
                    "pseudo_pred_label": "ERROR",
                    "pseudo_pred_score": 0.0,
                    "pseudo_selected": False,
                    "pseudo_error": str(e),
                }
            )
            rows.append(out)

    return pd.DataFrame(rows)


def cap_selected_per_label(
    pred_df: pd.DataFrame,
    max_per_label: int,
    seed: int,
) -> pd.DataFrame:
    selected = pred_df[pred_df["pseudo_selected"] == True].copy()

    if len(selected) == 0:
        return selected

    parts = []

    for label, group in selected.groupby("pseudo_pred_label"):
        if max_per_label > 0 and len(group) > max_per_label:
            group = group.sample(n=max_per_label, random_state=seed)
        parts.append(group)

    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return out


def make_pseudo_train_rows(selected_df: pd.DataFrame, pseudo_source_name: str) -> pd.DataFrame:
    out = selected_df.copy()

    out["label"] = out["pseudo_pred_label"]
    out["split"] = pseudo_source_name
    out["is_pseudo_labeled"] = True
    out["pseudo_confidence"] = out["pseudo_pred_score"]
    out["original_label_before_pseudo"] = out.get("label", "")

    # Keep common metadata fields first if they exist.
    if "index" in out.columns:
        out["index"] = range(len(out))

    return out


def summarize(pred_df: pd.DataFrame, selected_df: pd.DataFrame) -> Dict:
    summary = {
        "pool_rows": int(len(pred_df)),
        "selected_rows": int(len(selected_df)),
        "prediction_counts": pred_df["pseudo_pred_label"].value_counts().to_dict()
        if "pseudo_pred_label" in pred_df.columns
        else {},
        "selected_counts": selected_df["pseudo_pred_label"].value_counts().to_dict()
        if len(selected_df) > 0
        else {},
    }

    if "pseudo_pred_score" in selected_df.columns and len(selected_df) > 0:
        summary["selected_score_mean"] = float(selected_df["pseudo_pred_score"].mean())
        summary["selected_score_min"] = float(selected_df["pseudo_pred_score"].min())
        summary["selected_score_max"] = float(selected_df["pseudo_pred_score"].max())

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher-model", type=str, required=True)
    parser.add_argument("--pool-metadata", type=str, default=str(DEFAULT_POOL_METADATA))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument(
        "--allowed-labels",
        type=str,
        required=True,
        help="Comma-separated label list.",
    )

    parser.add_argument(
        "--exclude-csvs",
        nargs="*",
        default=[],
        help="CSV files whose audio_path values should be excluded from pseudo-label pool.",
    )

    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--max-per-label", type=int, default=500)
    parser.add_argument("--max-pool", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, required=True)

    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_labels = parse_labels(args.allowed_labels)

    print("=" * 80)
    print("Generating pseudo labels")
    print("=" * 80)
    print("Teacher model:", args.teacher_model)
    print("Pool metadata:", args.pool_metadata)
    print("Output dir:", output_dir)
    print("Allowed labels:", allowed_labels)
    print("Confidence threshold:", args.confidence_threshold)
    print("Max per label:", args.max_per_label)
    print("Max pool:", args.max_pool)
    print("Exclude CSVs:", args.exclude_csvs)

    excluded = load_excluded_audio_paths(args.exclude_csvs)

    pool_df = load_pool(
        pool_metadata=Path(args.pool_metadata),
        excluded_audio_paths=excluded,
        max_pool=args.max_pool,
        seed=args.seed,
    )

    classifier = build_classifier(args.teacher_model)

    pred_df = predict_pool(
        classifier=classifier,
        pool_df=pool_df,
        allowed_labels=allowed_labels,
        confidence_threshold=args.confidence_threshold,
        top_k=args.top_k,
    )

    selected_df = cap_selected_per_label(
        pred_df=pred_df,
        max_per_label=args.max_per_label,
        seed=args.seed,
    )

    pseudo_train_df = make_pseudo_train_rows(
        selected_df=selected_df,
        pseudo_source_name=f"pseudo_{args.name}",
    )

    all_pred_path = output_dir / "pseudo_all_predictions.csv"
    selected_path = output_dir / "pseudo_selected.csv"
    train_rows_path = output_dir / "pseudo_train_rows.csv"
    summary_path = output_dir / "pseudo_summary.json"

    pred_df.to_csv(all_pred_path, index=False, encoding="utf-8")
    selected_df.to_csv(selected_path, index=False, encoding="utf-8")
    pseudo_train_df.to_csv(train_rows_path, index=False, encoding="utf-8")

    summary = summarize(pred_df, selected_df)
    summary.update(
        {
            "teacher_model": args.teacher_model,
            "pool_metadata": args.pool_metadata,
            "allowed_labels": allowed_labels,
            "confidence_threshold": args.confidence_threshold,
            "max_per_label": args.max_per_label,
            "exclude_csvs": args.exclude_csvs,
        }
    )

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSummary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nSaved:")
    print(all_pred_path)
    print(selected_path)
    print(train_rows_path)
    print(summary_path)


if __name__ == "__main__":
    main()