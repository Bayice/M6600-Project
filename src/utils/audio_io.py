"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import io
from pathlib import Path
from typing import Any, Tuple

import soundfile as sf
import torch
import torchaudio


def load_audio_file_mono(audio_path: str) -> Tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, sample_rate


def resample_waveform(waveform: torch.Tensor, sample_rate: int, target_rate: int = 16000) -> torch.Tensor:
    if sample_rate == target_rate:
        return waveform

    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=target_rate,
    )
    return resampler(waveform)


def load_audio_file_mono_16k(audio_path: str):
    waveform, sample_rate = load_audio_file_mono(audio_path)
    waveform = resample_waveform(waveform, sample_rate, target_rate=16000)
    return waveform.squeeze().numpy()


def load_audio_from_hf_value(audio_value: Any):
    """
    Load audio from Hugging Face Audio(decode=False) value.

    Supports:
    - {"bytes": ...}
    - {"path": ...}
    """
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


def save_wav(audio_array, sample_rate: int, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio_array, sample_rate)