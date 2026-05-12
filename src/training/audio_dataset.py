"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.audio_io import load_audio_file_mono, resample_waveform
from src.utils.constants import BASELINE_LABELS


class AccentCsvDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        label_list: Optional[List[str]] = None,
        max_seconds: float = 8.0,
        sample_rate: int = 16000,
        limit: int = 0,
    ):
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.label_list = label_list or BASELINE_LABELS
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        self.max_seconds = max_seconds
        self.sample_rate = sample_rate

        df = pd.read_csv(self.csv_path)

        required = ["audio_path", "label"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column in {self.csv_path}: {col}")

        df = df.copy()
        df["audio_path"] = df["audio_path"].astype(str)
        df["label"] = df["label"].astype(str)

        df = df[df["audio_path"].str.len() > 0]
        df = df[df["label"].isin(self.label_list)]

        exists_mask = df["audio_path"].apply(lambda p: Path(p).exists())
        missing_count = len(df) - int(exists_mask.sum())

        if missing_count > 0:
            print(f"Warning: dropping {missing_count} rows with missing audio files from {self.csv_path}")

        df = df[exists_mask].reset_index(drop=True)

        if limit > 0:
            df = df.head(limit).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f"No valid rows found in {self.csv_path}")

        self.df = df

        print(f"Loaded dataset: {self.csv_path}")
        print(f"Rows: {len(self.df)}")
        print("Label counts:")
        print(self.df["label"].value_counts().sort_index())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        audio_path = str(row["audio_path"])
        label = str(row["label"])

        waveform, sr = load_audio_file_mono(audio_path)
        waveform = resample_waveform(waveform, sr, target_rate=self.sample_rate)

        audio = waveform.squeeze(0)

        if self.max_seconds and self.max_seconds > 0:
            max_len = int(self.max_seconds * self.sample_rate)
            if audio.numel() > max_len:
                audio = audio[:max_len]

        return {
            "audio": audio.numpy(),
            "labels": self.label2id[label],
        }


class AudioClassificationCollator:
    def __init__(self, feature_extractor, sample_rate: int = 16000):
        self.feature_extractor = feature_extractor
        self.sample_rate = sample_rate

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        audios = [f["audio"] for f in features]
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)

        batch = self.feature_extractor(
            audios,
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = labels
        return batch