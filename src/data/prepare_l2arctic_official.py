"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import argparse
import csv
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.constants import (
    L2ARCTIC_OFFICIAL_EXTRACT_DIR,
    L2ARCTIC_OFFICIAL_METADATA,
    L2ARCTIC_OFFICIAL_RAW_DIR,
    SPEAKER_TO_GENDER,
    SPEAKER_TO_L1,
)


def unzip_file(zip_path: Path, output_dir: Path, force: bool = False):
    if output_dir.exists() and any(output_dir.rglob("*")) and not force:
        print(f"Already extracted, skipping: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path} -> {output_dir}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)


def extract_all_speaker_zips(raw_dir: Path, extract_dir: Path, force: bool = False):
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Official L2-ARCTIC folder not found: {raw_dir}\n"
            "Expected speaker zip files such as ABA.zip, BWC.zip, etc."
        )

    extract_dir.mkdir(parents=True, exist_ok=True)

    speaker_zips = sorted(raw_dir.glob("*.zip"))

    # Exclude non-speaker zips such as suitcase_corpus.zip.
    speaker_zips = [p for p in speaker_zips if p.stem.upper() in SPEAKER_TO_L1]

    if not speaker_zips:
        raise FileNotFoundError(f"No speaker zip files found in {raw_dir}")

    print(f"Found {len(speaker_zips)} speaker zip files.")

    for zip_path in speaker_zips:
        speaker = zip_path.stem.upper()
        output_dir = extract_dir / speaker
        unzip_file(zip_path, output_dir, force=force)


def find_wav_files_for_speaker(speaker_dir: Path) -> List[Path]:
    return sorted(speaker_dir.rglob("*.wav"))


def find_prompts_file(raw_dir: Path) -> Optional[Path]:
    candidates = [
        raw_dir / "PROMPTS",
        raw_dir / "prompts",
        raw_dir / "PROMPTS.txt",
        raw_dir / "etc" / "txt.done.data",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_prompts(raw_dir: Path) -> Dict[str, str]:
    """
    Best-effort prompt loader.

    Official L2-ARCTIC may include a PROMPTS file.
    This function tries to map utterance IDs such as arctic_a0001 to text.
    If the format is not recognized, text can remain blank.
    """
    prompts_path = find_prompts_file(raw_dir)

    if prompts_path is None:
        print("No PROMPTS file found. Text column will be empty.")
        return {}

    print(f"Loading prompts from: {prompts_path}")

    mapping = {}

    with prompts_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) >= 2:
                utt_id = parts[0].strip().strip("()").strip('"')
                text = parts[1].strip().strip('"').strip()
                text = text.replace('\\"', '"').strip()

                if utt_id:
                    mapping[utt_id] = text

    print(f"Loaded {len(mapping)} prompts.")
    return mapping


def infer_utterance_id(wav_path: Path) -> str:
    return wav_path.stem


def build_metadata(raw_dir: Path, extract_dir: Path, metadata_path: Path):
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(raw_dir)
    rows = []

    for speaker, label in sorted(SPEAKER_TO_L1.items()):
        speaker_dir = extract_dir / speaker

        if not speaker_dir.exists():
            print(f"Warning: speaker dir not found, skipping: {speaker_dir}")
            continue

        wav_files = find_wav_files_for_speaker(speaker_dir)

        if not wav_files:
            print(f"Warning: no wav files found for speaker {speaker}")
            continue

        print(f"Speaker {speaker}: found {len(wav_files)} wav files")

        for wav_path in wav_files:
            utt_id = infer_utterance_id(wav_path)
            text = prompts.get(utt_id, "")

            rows.append(
                {
                    "split": "official",
                    "index": len(rows),
                    "utterance_id": utt_id,
                    "audio_path": str(wav_path),
                    "speaker": speaker,
                    "label": label,
                    "gender": SPEAKER_TO_GENDER.get(speaker, ""),
                    "text": text,
                }
            )

    fieldnames = [
        "split",
        "index",
        "utterance_id",
        "audio_path",
        "speaker",
        "label",
        "gender",
        "text",
    ]

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nMetadata saved to:", metadata_path)
    print("Total rows:", len(rows))

    print("\nLabel counts:")
    label_count = {}
    for row in rows:
        label_count[row["label"]] = label_count.get(row["label"], 0) + 1

    for label, count in sorted(label_count.items()):
        print(f"  {label:12s}: {count}")

    print("\nSpeaker counts:")
    speaker_count = {}
    for row in rows:
        speaker_count[row["speaker"]] = speaker_count.get(row["speaker"], 0) + 1

    for speaker, count in sorted(speaker_count.items()):
        print(f"  {speaker:8s}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default=str(L2ARCTIC_OFFICIAL_RAW_DIR))
    parser.add_argument("--extract-dir", type=str, default=str(L2ARCTIC_OFFICIAL_EXTRACT_DIR))
    parser.add_argument("--metadata-path", type=str, default=str(L2ARCTIC_OFFICIAL_METADATA))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    extract_dir = Path(args.extract_dir)
    metadata_path = Path(args.metadata_path)

    print("=" * 80)
    print("Preparing official L2-ARCTIC release")
    print("=" * 80)
    print("Raw dir:      ", raw_dir)
    print("Extract dir:  ", extract_dir)
    print("Metadata path:", metadata_path)
    print("")

    extract_all_speaker_zips(
        raw_dir=raw_dir,
        extract_dir=extract_dir,
        force=args.force,
    )

    build_metadata(
        raw_dir=raw_dir,
        extract_dir=extract_dir,
        metadata_path=metadata_path,
    )

    print("\nDone.")
    print("Next:")
    print(f"  Check metadata: {metadata_path}")
    print("  Then run baseline on official wav files.")


if __name__ == "__main__":
    main()