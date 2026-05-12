"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    print("\n" + "=" * 80)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 80)

    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-feature-encoder", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")

    parser.add_argument(
        "--which",
        type=str,
        default="minimum",
        choices=["file", "speaker0", "minimum", "speaker4fold"],
    )

    args = parser.parse_args()

    common = [
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.eval_batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--learning-rate", str(args.learning_rate),
        "--max-seconds", str(args.max_seconds),
        "--seed", str(args.seed),
    ]

    if args.freeze_feature_encoder:
        common.append("--freeze-feature-encoder")

    if args.no_fp16:
        common.append("--no-fp16")

    experiments = []

    if args.which in ["file", "minimum"]:
        experiments.append(
            {
                "name": "exp1_file_level_distilhubert",
                "train": "data/processed/train_file_level.csv",
                "dev": "data/processed/val_file_level.csv",
                "test": "data/processed/test_file_level.csv",
                "out": "models/checkpoints/exp1_file_level_distilhubert",
            }
        )

    if args.which in ["speaker0", "minimum"]:
        experiments.append(
            {
                "name": "exp2_speaker_disjoint_fold0_distilhubert",
                "train": "data/processed/train_speaker_disjoint.csv",
                "dev": "data/processed/dev_speaker_disjoint.csv",
                "test": "data/processed/test_speaker_disjoint.csv",
                "out": "models/checkpoints/exp2_speaker_disjoint_fold0_distilhubert",
            }
        )

    if args.which == "speaker4fold":
        for fold_id in range(4):
            experiments.append(
                {
                    "name": f"exp2_speaker_disjoint_fold{fold_id}_distilhubert",
                    "train": f"data/processed/splits/split2_speaker_disjoint_fold{fold_id}/train.csv",
                    "dev": f"data/processed/splits/split2_speaker_disjoint_fold{fold_id}/dev.csv",
                    "test": f"data/processed/splits/split2_speaker_disjoint_fold{fold_id}/test.csv",
                    "out": f"models/checkpoints/exp2_speaker_disjoint_fold{fold_id}_distilhubert",
                }
            )

    for exp in experiments:
        for key in ["train", "dev", "test"]:
            if not Path(exp[key]).exists():
                raise FileNotFoundError(f"Missing {key} CSV for {exp['name']}: {exp[key]}")

        cmd = [
            sys.executable,
            "-m",
            "src.training.train_distilhubert",
            "--train-csv", exp["train"],
            "--dev-csv", exp["dev"],
            "--test-csv", exp["test"],
            "--output-dir", exp["out"],
        ] + common

        run_command(cmd)

    print("\nAll requested experiments finished.")


if __name__ == "__main__":
    main()