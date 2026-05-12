"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import argparse
from pathlib import Path

from src.data.splitters import (
    load_and_clean_metadata,
    make_split1_file_level,
    make_split2_speaker_disjoint_folds,
    print_split2_fold_summary,
    print_summary,
    write_split1_outputs,
    write_split2_outputs,
)
from src.utils.constants import L2ARCTIC_OFFICIAL_METADATA, PROCESSED_DIR


DEFAULT_OUTPUT_DIR = PROCESSED_DIR
DEFAULT_SPLIT_DIR = PROCESSED_DIR / "splits"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, default=str(L2ARCTIC_OFFICIAL_METADATA))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split-dir", type=str, default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    split_dir = Path(args.split_dir)

    print("=" * 80)
    print("Making L2-ARCTIC data splits")
    print("=" * 80)
    print("Metadata:  ", metadata_path)
    print("Output dir:", output_dir)
    print("Split dir: ", split_dir)
    print("Seed:      ", args.seed)
    print("Dev ratio: ", args.dev_ratio)

    df = load_and_clean_metadata(metadata_path)

    print_summary("Full metadata", df)

    split1_train, split1_dev, split1_test = make_split1_file_level(
        df=df,
        seed=args.seed,
        train_ratio=0.8,
        dev_ratio=0.1,
        test_ratio=0.1,
    )

    write_split1_outputs(
        train_df=split1_train,
        dev_df=split1_dev,
        test_df=split1_test,
        output_dir=output_dir,
        split_dir=split_dir,
    )

    print_summary("Split-1 file-level train", split1_train)
    print_summary("Split-1 file-level dev", split1_dev)
    print_summary("Split-1 file-level test", split1_test)

    split2_folds = make_split2_speaker_disjoint_folds(
        df=df,
        seed=args.seed,
        dev_ratio_within_train_speakers=args.dev_ratio,
    )

    write_split2_outputs(
        folds=split2_folds,
        output_dir=output_dir,
        split_dir=split_dir,
    )

    print_split2_fold_summary(split2_folds)

    print("\nDone.")
    print("\nMain outputs:")
    print(f"  {output_dir / 'train_file_level.csv'}")
    print(f"  {output_dir / 'val_file_level.csv'}")
    print(f"  {output_dir / 'test_file_level.csv'}")
    print(f"  {output_dir / 'train_speaker_disjoint.csv'}")
    print(f"  {output_dir / 'dev_speaker_disjoint.csv'}")
    print(f"  {output_dir / 'val_speaker_disjoint.csv'}")
    print(f"  {output_dir / 'test_speaker_disjoint.csv'}")

    print("\nDetailed fold outputs:")
    print(f"  {split_dir / 'split1_file_level_train.csv'}")
    print(f"  {split_dir / 'split2_speaker_disjoint_fold0' / 'train.csv'}")
    print(f"  {split_dir / 'split2_speaker_disjoint_fold0' / 'dev.csv'}")
    print(f"  {split_dir / 'split2_speaker_disjoint_fold0' / 'test.csv'}")


if __name__ == "__main__":
    main()