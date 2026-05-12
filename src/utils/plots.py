"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_bar_chart(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    rotate_xticks: bool = False,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotate_xticks:
        plt.xticks(rotation=45, ha="right")
    else:
        plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure: {output_path}")


def save_label_distribution(df: pd.DataFrame, output_path: Path):
    counts = df["label"].fillna("UNKNOWN").replace("", "UNKNOWN").value_counts().sort_index()
    save_bar_chart(
        series=counts,
        title="Utterance Count by Accent / First Language",
        xlabel="Accent / First Language",
        ylabel="Number of Utterances",
        output_path=output_path,
        rotate_xticks=True,
    )


def save_speaker_distribution(df: pd.DataFrame, output_path: Path):
    counts = df["speaker"].fillna("UNKNOWN").replace("", "UNKNOWN").value_counts().sort_index()
    save_bar_chart(
        series=counts,
        title="Utterance Count by Speaker",
        xlabel="Speaker",
        ylabel="Number of Utterances",
        output_path=output_path,
        rotate_xticks=True,
    )


def save_split_distribution(df: pd.DataFrame, output_path: Path):
    counts = df["split"].fillna("UNKNOWN").replace("", "UNKNOWN").value_counts().sort_index()
    save_bar_chart(
        series=counts,
        title="Utterance Count by Dataset Split",
        xlabel="Split",
        ylabel="Number of Utterances",
        output_path=output_path,
        rotate_xticks=False,
    )


def save_label_speaker_heatmap(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean = df.copy()
    clean["label"] = clean["label"].fillna("UNKNOWN").replace("", "UNKNOWN")
    clean["speaker"] = clean["speaker"].fillna("UNKNOWN").replace("", "UNKNOWN")

    table = pd.crosstab(clean["label"], clean["speaker"])

    plt.figure(figsize=(14, 6))
    plt.imshow(table.values, aspect="auto")
    plt.title("Label-Speaker Utterance Count Heatmap")
    plt.xlabel("Speaker")
    plt.ylabel("Accent / First Language")

    plt.xticks(range(len(table.columns)), table.columns, rotation=90)
    plt.yticks(range(len(table.index)), table.index)

    cbar = plt.colorbar()
    cbar.set_label("Number of Utterances")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure: {output_path}")


def save_duration_histogram(df: pd.DataFrame, output_path: Path):
    if "duration_seconds" not in df.columns:
        print("Skipping duration histogram: duration_seconds column not found.")
        return

    duration = pd.to_numeric(df["duration_seconds"], errors="coerce").dropna()

    if len(duration) == 0:
        print("Skipping duration histogram: no valid duration values found.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(duration, bins=50)
    plt.title("Audio Duration Distribution")
    plt.xlabel("Duration in Seconds")
    plt.ylabel("Number of Utterances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved figure: {output_path}")


def generate_basic_metadata_figures(df: pd.DataFrame, figure_dir: Path, prefix: str):
    figure_dir.mkdir(parents=True, exist_ok=True)

    save_label_distribution(df, figure_dir / f"{prefix}_label_distribution.png")
    save_speaker_distribution(df, figure_dir / f"{prefix}_speaker_distribution.png")
    save_split_distribution(df, figure_dir / f"{prefix}_split_distribution.png")
    save_label_speaker_heatmap(df, figure_dir / f"{prefix}_label_speaker_heatmap.png")
    save_duration_histogram(df, figure_dir / f"{prefix}_duration_histogram.png")