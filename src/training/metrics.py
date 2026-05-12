"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_classification_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
    }


def save_predictions_and_confusion(
    logits,
    labels,
    metadata_df: pd.DataFrame,
    id2label: Dict[int, str],
    output_dir: Path,
    split_name: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = np.argmax(logits, axis=-1)

    rows = []

    for i, pred_id in enumerate(preds):
        true_id = int(labels[i])
        pred_label = id2label[int(pred_id)]
        true_label = id2label[true_id]

        meta = metadata_df.iloc[i].to_dict()

        row = dict(meta)
        row.update(
            {
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": pred_label == true_label,
            }
        )
        rows.append(row)

    pred_df = pd.DataFrame(rows)
    pred_path = output_dir / f"{split_name}_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")
    print(f"Saved predictions -> {pred_path}")

    labels_order = [id2label[i] for i in sorted(id2label.keys())]

    cm = confusion_matrix(
        pred_df["true_label"],
        pred_df["pred_label"],
        labels=labels_order,
    )

    cm_df = pd.DataFrame(cm, index=labels_order, columns=labels_order)
    cm_csv = output_dir / f"{split_name}_confusion_matrix.csv"
    cm_df.to_csv(cm_csv, encoding="utf-8")
    print(f"Saved confusion matrix CSV -> {cm_csv}")

    cm_png = output_dir / f"{split_name}_confusion_matrix.png"
    save_confusion_matrix_png(cm_df, cm_png)

    acc = float(pred_df["correct"].mean())
    macro_f1 = float(f1_score(pred_df["true_label"], pred_df["pred_label"], average="macro"))

    summary = {
        "split": split_name,
        "num_examples": int(len(pred_df)),
        "accuracy": acc,
        "macro_f1": macro_f1,
    }

    summary_path = output_dir / f"{split_name}_summary.json"
    pd.Series(summary).to_json(summary_path, indent=2)
    print(f"Saved summary -> {summary_path}")

    return pred_df, cm_df, summary


def save_confusion_matrix_png(cm_df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_df.values, aspect="auto")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.xticks(range(len(cm_df.columns)), cm_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(cm_df.index)), cm_df.index)

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            plt.text(j, i, str(cm_df.values[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved confusion matrix PNG -> {output_path}")