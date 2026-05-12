"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
from transformers import pipeline

from src.utils.audio_io import load_audio_file_mono_16k
from src.utils.constants import LOCAL_BASELINE_MODEL, REMOTE_BASELINE_MODEL
from src.utils.labels import normalize_l2_label


def get_baseline_model_path(force_remote: bool = False) -> Tuple[str, bool]:
    if force_remote:
        print("Force remote loading enabled.")
        return REMOTE_BASELINE_MODEL, False

    config_path = LOCAL_BASELINE_MODEL / "config.json"
    preprocessor_path = LOCAL_BASELINE_MODEL / "preprocessor_config.json"

    if config_path.exists() and preprocessor_path.exists():
        print(f"Found local model: {LOCAL_BASELINE_MODEL}")
        return str(LOCAL_BASELINE_MODEL), True

    print("Local model not found. Falling back to Hugging Face Hub.")
    print("You can run this first:")
    print("python src\\download_baseline_model.py")
    return REMOTE_BASELINE_MODEL, False


def build_baseline_pipeline(force_remote: bool = False):
    device = 0 if torch.cuda.is_available() else -1

    if device == 0:
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using CPU")

    model_path, use_local_only = get_baseline_model_path(force_remote=force_remote)
    print("Loading model:", model_path)

    classifier = pipeline(
        task="audio-classification",
        model=model_path,
        device=device,
        local_files_only=use_local_only,
    )

    return classifier


def predict_audio_array(classifier, audio_array, top_k: int = 6) -> Dict:
    results = classifier(audio_array, top_k=top_k)
    top = results[0]

    return {
        "pred_label": normalize_l2_label(top["label"]),
        "pred_score": float(top["score"]),
        "all_results": results,
    }


def predict_audio_file(classifier, audio_path: str, top_k: int = 6) -> Dict:
    audio_array = load_audio_file_mono_16k(audio_path)
    return predict_audio_array(classifier, audio_array, top_k=top_k)