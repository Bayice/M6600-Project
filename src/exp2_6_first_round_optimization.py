"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.evaluation.eval_common import (
    build_classifier,
    external_distribution,
    load_metadata_sample,
    run_metadata_source,
    run_user_audio_source,
    save_csv,
    save_json,
    summarize_prediction_rows,
)


HF_BASELINE_LOCAL = Path("models/hf/accent-id-distilhubert-finetuned-l2-arctic2")
HF_BASELINE_REMOTE = "kaysrubio/accent-id-distilhubert-finetuned-l2-arctic2"

OFFICIAL_METADATA = Path("data/processed/metadata_l2_arctic_official.csv")
EXTERNAL_METADATA = Path("data/processed/external_metadata.csv")

DEFAULT_USER_AUDIO = [
    "data/samples/XuHe_audio.wav",
    "data/samples/ZhiyuanLu_Audio.wav",
    "data/samples/ZhiyuanLu_audio.wav",
]


def model_entry(name: str, path: str, local_only: bool = True) -> Dict[str, object]:
    return {
        "name": name,
        "path": path,
        "local_only": local_only,
    }


def discover_models(include_smoke: bool = False) -> List[Dict[str, object]]:
    models: List[Dict[str, object]] = []

    if (HF_BASELINE_LOCAL / "config.json").exists():
        models.append(
            model_entry(
                name="hf_kaysrubio_local",
                path=str(HF_BASELINE_LOCAL),
                local_only=True,
            )
        )
    else:
        models.append(
            model_entry(
                name="hf_kaysrubio_remote",
                path=HF_BASELINE_REMOTE,
                local_only=False,
            )
        )

    checkpoint_root = Path("models/checkpoints")
    candidates = sorted(checkpoint_root.glob("*/best_model"))

    for best_model in candidates:
        exp_dir = best_model.parent
        name = exp_dir.name

        if not include_smoke and "smoke" in name.lower():
            continue

        if (best_model / "config.json").exists():
            models.append(
                model_entry(
                    name=name,
                    path=str(best_model),
                    local_only=True,
                )
            )

    return models


def print_models(models: List[Dict[str, object]]):
    print("=" * 80)
    print("Models to evaluate")
    print("=" * 80)

    for i, m in enumerate(models):
        print(f"{i:02d}. {m['name']}")
        print(f"    path={m['path']}")
        print(f"    local_only={m['local_only']}")


def safe_model_name(name: str) -> str:
    return name.replace("\\", "_").replace("/", "_").replace(":", "_")


def plot_bar_series(series: pd.Series, title: str, xlabel: str, ylabel: str, output_path: Path, rotate: bool = True):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate:
        plt.xticks(rotation=45, ha="right")
    else:
        plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure: {output_path}")


def plot_external_distribution(metadata_path: Path, output_dir: Path):
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing external metadata: {metadata_path}")

    df = pd.read_csv(metadata_path)

    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Plotting external dataset distribution")
    print("=" * 80)

    if "label" in df.columns:
        label_counts = df["label"].fillna("UNKNOWN").value_counts()
        label_counts.to_csv(table_dir / "external_label_distribution.csv", header=["count"])

        plot_bar_series(
            series=label_counts,
            title="External Dataset Label Distribution",
            xlabel="Label",
            ylabel="Number of Samples",
            output_path=figure_dir / "external_label_distribution.png",
            rotate=True,
        )

    if "dataset" in df.columns:
        dataset_counts = df["dataset"].fillna("UNKNOWN").value_counts()
        dataset_counts.to_csv(table_dir / "external_dataset_distribution.csv", header=["count"])

        plot_bar_series(
            series=dataset_counts,
            title="External Dataset Source Distribution",
            xlabel="Dataset",
            ylabel="Number of Samples",
            output_path=figure_dir / "external_dataset_distribution.png",
            rotate=True,
        )

    if "is_mapped_to_baseline" in df.columns:
        mapped = df["is_mapped_to_baseline"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        mapped_counts = pd.Series(
            {
                "mapped_to_original_6_labels": int(mapped.sum()),
                "OOD_or_unmapped": int((~mapped).sum()),
            }
        )
        mapped_counts.to_csv(table_dir / "external_mapped_vs_ood.csv", header=["count"])

        plot_bar_series(
            series=mapped_counts,
            title="External Data: Original 6 Labels vs OOD",
            xlabel="Category",
            ylabel="Number of Samples",
            output_path=figure_dir / "external_mapped_vs_ood.png",
            rotate=False,
        )

    if "dataset" in df.columns and "label" in df.columns:
        table = pd.crosstab(df["dataset"].fillna("UNKNOWN"), df["label"].fillna("UNKNOWN"))
        table.to_csv(table_dir / "external_dataset_label_crosstab.csv", encoding="utf-8")

        plt.figure(figsize=(14, 7))
        plt.imshow(table.values, aspect="auto")
        plt.title("External Dataset × Label Distribution")
        plt.xlabel("Label")
        plt.ylabel("Dataset")
        plt.xticks(range(len(table.columns)), table.columns, rotation=90)
        plt.yticks(range(len(table.index)), table.index)
        cbar = plt.colorbar()
        cbar.set_label("Number of Samples")
        plt.tight_layout()

        heatmap_path = figure_dir / "external_dataset_label_heatmap.png"
        plt.savefig(heatmap_path, dpi=200)
        plt.close()

        print(f"Saved figure: {heatmap_path}")


def source_accuracy_rows(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    valid = df[df["pred_label"] != "ERROR"].copy()

    comparable = valid[
        valid["has_comparable_label"].astype(str).str.lower().isin(["true", "1"])
    ].copy()

    if len(comparable) == 0:
        return pd.DataFrame()

    comparable["correct"] = comparable["correct"].astype(bool)

    rows = []

    for (model_name, source), g in comparable.groupby(["model_name", "source"]):
        rows.append(
            {
                "model_name": model_name,
                "source": source,
                "rows": len(g),
                "correct": int(g["correct"].sum()),
                "accuracy": float(g["correct"].mean()),
            }
        )

    return pd.DataFrame(rows)


def combined_official_external_rows(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    valid = df[df["pred_label"] != "ERROR"].copy()

    comparable = valid[
        valid["has_comparable_label"].astype(str).str.lower().isin(["true", "1"])
    ].copy()

    comparable = comparable[
        comparable["source"].astype(str).isin(["official_l2arctic", "external"])
    ].copy()

    if len(comparable) == 0:
        return pd.DataFrame()

    comparable["correct"] = comparable["correct"].astype(bool)

    rows = []

    for model_name, g in comparable.groupby("model_name"):
        official = g[g["source"] == "official_l2arctic"]
        external = g[g["source"] == "external"]

        rows.append(
            {
                "model_name": model_name,
                "combined_rows": len(g),
                "combined_correct": int(g["correct"].sum()),
                "combined_accuracy": float(g["correct"].mean()),
                "official_rows": len(official),
                "official_accuracy": float(official["correct"].mean()) if len(official) else "",
                "external_comparable_rows": len(external),
                "external_comparable_accuracy": float(external["correct"].mean()) if len(external) else "",
            }
        )

    return pd.DataFrame(rows)


def save_prediction_count_tables(df: pd.DataFrame, output_dir: Path):
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    valid = df[df["pred_label"] != "ERROR"].copy()

    if len(valid) == 0:
        return

    pred_counts = pd.crosstab(
        [valid["model_name"], valid["source"]],
        valid["pred_label"],
    )

    pred_counts_path = table_dir / "prediction_counts_by_model_source.csv"
    pred_counts.to_csv(pred_counts_path, encoding="utf-8")
    print(f"Saved: {pred_counts_path}")

    if "label" in valid.columns:
        by_label = pd.crosstab(
            [valid["model_name"], valid["source"], valid["label"]],
            valid["pred_label"],
        )

        by_label_path = table_dir / "prediction_counts_by_model_source_label.csv"
        by_label.to_csv(by_label_path, encoding="utf-8")
        print(f"Saved: {by_label_path}")

    if "dataset" in valid.columns:
        by_dataset = pd.crosstab(
            [valid["model_name"], valid["source"], valid["dataset"]],
            valid["pred_label"],
        )

        by_dataset_path = table_dir / "prediction_counts_by_model_source_dataset.csv"
        by_dataset.to_csv(by_dataset_path, encoding="utf-8")
        print(f"Saved: {by_dataset_path}")


def build_comparison_tables(all_rows: List[Dict[str, object]], output_dir: Path):
    df = pd.DataFrame(all_rows)

    if len(df) == 0:
        print("No rows for comparison tables.")
        return

    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    # 1. Separate accuracy by source.
    source_acc = source_accuracy_rows(df)
    source_acc_path = table_dir / "model_source_accuracy_summary.csv"
    source_acc.to_csv(source_acc_path, index=False, encoding="utf-8")
    print(f"Saved: {source_acc_path}")

    print("\n" + "=" * 80)
    print("Model/source accuracy summary")
    print("=" * 80)
    print(source_acc)

    # 2. Official + external combined comparable accuracy.
    combined = combined_official_external_rows(df)
    combined_path = table_dir / "model_official_external_combined_summary.csv"
    combined.to_csv(combined_path, index=False, encoding="utf-8")
    print(f"Saved: {combined_path}")

    print("\n" + "=" * 80)
    print("Official + external combined comparable accuracy")
    print("=" * 80)
    print(combined)

    # 3. User audio predictions only.
    user_audio_df = df[df["source"] == "user_audio"].copy()
    user_audio_path = table_dir / "user_audio_predictions.csv"
    user_audio_df.to_csv(user_audio_path, index=False, encoding="utf-8")
    print(f"Saved: {user_audio_path}")

    print("\n" + "=" * 80)
    print("User audio predictions")
    print("=" * 80)

    if len(user_audio_df) > 0:
        cols = [
            c for c in [
                "model_name",
                "audio_path",
                "pred_label",
                "pred_score",
                "score_Arabic",
                "score_Hindi",
                "score_Korean",
                "score_Mandarin",
                "score_Spanish",
                "score_Vietnamese",
            ]
            if c in user_audio_df.columns
        ]
        print(user_audio_df[cols])
    else:
        print("No user audio rows.")

    # 4. Prediction count tables.
    save_prediction_count_tables(df, output_dir)


def evaluate_one_model(
    model: Dict[str, object],
    device: int,
    top_k: int,
    user_audio: List[str],
    official_df: pd.DataFrame,
    external_df: pd.DataFrame,
    output_dir: Path,
):
    model_name = str(model["name"])
    model_path = str(model["path"])
    local_only = bool(model["local_only"])

    print("\n" + "=" * 80)
    print(f"Evaluating model: {model_name}")
    print(f"Path: {model_path}")
    print("=" * 80)

    classifier = build_classifier(
        model_path=model_path,
        device=device,
        local_files_only=local_only,
    )

    all_rows = []
    per_source_summaries = {}

    if user_audio:
        rows = run_user_audio_source(
            classifier=classifier,
            model_name=model_name,
            audio_paths=user_audio,
            top_k=top_k,
        )
        all_rows.extend(rows)
        per_source_summaries["user_audio"] = summarize_prediction_rows(rows)

    official_rows = run_metadata_source(
        classifier=classifier,
        model_name=model_name,
        source_name="official_l2arctic",
        df=official_df,
        top_k=top_k,
    )
    all_rows.extend(official_rows)
    per_source_summaries["official_l2arctic"] = summarize_prediction_rows(official_rows)

    external_rows = run_metadata_source(
        classifier=classifier,
        model_name=model_name,
        source_name="external",
        df=external_df,
        top_k=top_k,
    )
    all_rows.extend(external_rows)
    per_source_summaries["external"] = summarize_prediction_rows(external_rows)

    combined_rows = official_rows + external_rows
    per_source_summaries["official_plus_external"] = summarize_prediction_rows(combined_rows)

    model_safe_name = safe_model_name(model_name)

    csv_path = output_dir / "per_model" / f"{model_safe_name}_predictions.csv"
    json_path = output_dir / "per_model" / f"{model_safe_name}_summary.json"

    save_csv(all_rows, csv_path)

    summary = {
        "model_name": model_name,
        "model_path": model_path,
        "all_sources": summarize_prediction_rows(all_rows),
        "per_source": per_source_summaries,
    }

    save_json(summary, json_path)

    return all_rows, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--only-model", type=str, default="")
    parser.add_argument("--skip-user-audio", action="store_true")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/first_round_optimization",
    )

    parser.add_argument(
        "--user-audio",
        nargs="*",
        default=DEFAULT_USER_AUDIO,
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else -1

    if device == 0:
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    print("\n" + "=" * 80)
    print("External data distribution")
    print("=" * 80)

    dist = external_distribution(EXTERNAL_METADATA)
    print(json.dumps(dist, indent=2, ensure_ascii=False))
    save_json(dist, output_dir / "external_distribution.json")

    plot_external_distribution(
        metadata_path=EXTERNAL_METADATA,
        output_dir=output_dir,
    )

    print("\nSampling official and external metadata...")

    official_df = load_metadata_sample(
        metadata_path=OFFICIAL_METADATA,
        samples=args.samples,
        seed=args.seed,
    )

    external_df = load_metadata_sample(
        metadata_path=EXTERNAL_METADATA,
        samples=args.samples,
        seed=args.seed,
    )

    official_sample_path = output_dir / f"official_sample_{args.samples}_seed{args.seed}.csv"
    external_sample_path = output_dir / f"external_sample_{args.samples}_seed{args.seed}.csv"

    official_df.to_csv(official_sample_path, index=False, encoding="utf-8")
    external_df.to_csv(external_sample_path, index=False, encoding="utf-8")

    print(f"Saved official sample: {official_sample_path}")
    print(f"Saved external sample: {external_sample_path}")

    models = discover_models(include_smoke=args.include_smoke)

    if args.only_model:
        models = [
            m for m in models
            if args.only_model.lower() in str(m["name"]).lower()
        ]

    if not models:
        raise ValueError("No models found to evaluate.")

    print_models(models)

    all_rows = []
    all_summaries = []

    user_audio = [] if args.skip_user_audio else args.user_audio

    for model in models:
        rows, summary = evaluate_one_model(
            model=model,
            device=device,
            top_k=args.top_k,
            user_audio=user_audio,
            official_df=official_df,
            external_df=external_df,
            output_dir=output_dir,
        )

        all_rows.extend(rows)
        all_summaries.append(summary)

    save_csv(all_rows, output_dir / "all_model_predictions.csv")

    save_json(
        {"summaries": all_summaries},
        output_dir / "all_model_summaries.json",
    )

    build_comparison_tables(all_rows, output_dir)

    print("\nDone.")
    print(f"All outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()