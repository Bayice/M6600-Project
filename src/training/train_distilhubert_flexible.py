"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import argparse
import inspect
import json
from pathlib import Path
from typing import List

import pandas as pd
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.training.audio_dataset import AccentCsvDataset, AudioClassificationCollator
from src.training.metrics import compute_classification_metrics, save_predictions_and_confusion


BASELINE_LABELS = [
    "Arabic",
    "Hindi",
    "Korean",
    "Mandarin",
    "Spanish",
    "Vietnamese",
]

DEFAULT_BASE_MODEL = "ntu-spml/distilhubert"


def read_labels_from_csvs(paths: List[str]) -> List[str]:
    labels = set()

    for path in paths:
        if not path:
            continue

        p = Path(path)
        if not p.exists():
            continue

        df = pd.read_csv(p)
        if "label" not in df.columns:
            continue

        for x in df["label"].dropna().astype(str):
            x = x.strip()
            if x:
                labels.add(x)

    baseline_first = [x for x in BASELINE_LABELS if x in labels]
    others = sorted([x for x in labels if x not in BASELINE_LABELS])

    return baseline_first + others


def get_label_list(args) -> List[str]:
    if args.label_mode == "baseline":
        return BASELINE_LABELS

    if args.label_mode == "union":
        labels = read_labels_from_csvs([args.train_csv, args.dev_csv, args.test_csv])
        if not labels:
            raise ValueError("No labels found from train/dev/test CSVs.")
        return labels

    raise ValueError(f"Unsupported label mode: {args.label_mode}")


def freeze_feature_encoder_if_requested(model, freeze: bool):
    if not freeze:
        return

    print("Freezing feature encoder / feature extractor if available.")

    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        return

    if hasattr(model, "hubert") and hasattr(model.hubert, "feature_extractor"):
        for param in model.hubert.feature_extractor.parameters():
            param.requires_grad = False
        return

    if hasattr(model, "wav2vec2") and hasattr(model.wav2vec2, "feature_extractor"):
        for param in model.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
        return

    print("Warning: no known feature encoder found to freeze.")


def build_training_args(args):
    kwargs = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "fp16": args.fp16 and torch.cuda.is_available(),
        "remove_unused_columns": False,
        "dataloader_num_workers": args.num_workers,
        "report_to": "none",
        "seed": args.seed,
    }

    sig = inspect.signature(TrainingArguments.__init__)

    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--dev-csv", type=str, required=True)
    parser.add_argument("--test-csv", type=str, default="")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument(
        "--label-mode",
        type=str,
        default="baseline",
        choices=["baseline", "union"],
        help="baseline = original 6 labels; union = all labels from train/dev/test",
    )

    parser.add_argument("--model-name", type=str, default=DEFAULT_BASE_MODEL)

    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)

    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-dev", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)

    parser.add_argument("--freeze-feature-encoder", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default="")
    args = parser.parse_args()

    args.output_dir = str(Path(args.output_dir))
    args.fp16 = not args.no_fp16

    set_seed(args.seed)

    label_list = get_label_list(args)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}

    print("=" * 80)
    print("Training flexible DistilHuBERT accent classifier")
    print("=" * 80)
    print("Train CSV: ", args.train_csv)
    print("Dev CSV:   ", args.dev_csv)
    print("Test CSV:  ", args.test_csv if args.test_csv else "None")
    print("Output:    ", args.output_dir)
    print("Model:     ", args.model_name)
    print("Label mode:", args.label_mode)
    print("Num labels:", len(label_list))
    print("Labels:    ", label_list)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "label_list.json").write_text(
        json.dumps(label_list, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name)

    model = AutoModelForAudioClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    freeze_feature_encoder_if_requested(model, args.freeze_feature_encoder)

    train_dataset = AccentCsvDataset(
        csv_path=args.train_csv,
        label_list=label_list,
        max_seconds=args.max_seconds,
        sample_rate=args.sample_rate,
        limit=args.limit_train,
    )

    dev_dataset = AccentCsvDataset(
        csv_path=args.dev_csv,
        label_list=label_list,
        max_seconds=args.max_seconds,
        sample_rate=args.sample_rate,
        limit=args.limit_dev,
    )

    test_dataset = None
    if args.test_csv:
        test_dataset = AccentCsvDataset(
            csv_path=args.test_csv,
            label_list=label_list,
            max_seconds=args.max_seconds,
            sample_rate=args.sample_rate,
            limit=args.limit_test,
        )

    collator = AudioClassificationCollator(
        feature_extractor=feature_extractor,
        sample_rate=args.sample_rate,
    )

    training_args = build_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        compute_metrics=compute_classification_metrics,
    )

    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    print("\nSaving best model and feature extractor...")
    trainer.save_model(str(output_dir / "best_model"))
    feature_extractor.save_pretrained(str(output_dir / "best_model"))

    print("\nEvaluating on Dev set...")
    dev_output = trainer.predict(dev_dataset)
    save_predictions_and_confusion(
        logits=dev_output.predictions,
        labels=dev_output.label_ids,
        metadata_df=dev_dataset.df,
        id2label=id2label,
        output_dir=output_dir,
        split_name="dev",
    )

    if test_dataset is not None:
        print("\nEvaluating on Test set...")
        test_output = trainer.predict(test_dataset)
        save_predictions_and_confusion(
            logits=test_output.predictions,
            labels=test_output.label_ids,
            metadata_df=test_dataset.df,
            id2label=id2label,
            output_dir=output_dir,
            split_name="test",
        )

    print("\nDone.")
    print(f"Model and results saved to: {output_dir}")


if __name__ == "__main__":
    main()