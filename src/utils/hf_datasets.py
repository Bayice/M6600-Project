"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

from pathlib import Path
from typing import Optional

from datasets import Audio, load_dataset, load_from_disk


def is_valid_saved_dataset(path: Path) -> bool:
    if not path.exists():
        return False

    if (path / "dataset_dict.json").exists():
        return True

    if (path / "state.json").exists() or (path / "dataset_info.json").exists():
        return True

    return False


def get_split_names(ds):
    if hasattr(ds, "keys"):
        return list(ds.keys())
    return ["train"]


def get_split(ds, split_name):
    if hasattr(ds, "keys"):
        return ds[split_name]
    return ds


def load_or_download_dataset(dataset_name: str, save_dir: Path, force_download: bool = False):
    if is_valid_saved_dataset(save_dir) and not force_download:
        print(f"Found local dataset at: {save_dir}")
        print("Loading from disk...")
        return load_from_disk(str(save_dir))

    print("Local dataset not found or force_download=True.")
    print(f"Downloading dataset from Hugging Face: {dataset_name}")

    ds = load_dataset(dataset_name)

    save_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to: {save_dir}")
    ds.save_to_disk(str(save_dir))

    return ds


def get_audio_column(ds_split, audio_candidates=None) -> Optional[str]:
    if audio_candidates is None:
        audio_candidates = ["audio", "wav", "file", "path", "audio_path"]

    for col, feature in ds_split.features.items():
        if isinstance(feature, Audio):
            return col

    for col in audio_candidates:
        if col in ds_split.column_names:
            return col

    return None


def disable_audio_decoding(ds):
    """
    Avoid requiring torchcodec during metadata generation.
    We only need audio path / metadata here, not decoded waveform arrays.
    """
    for split in get_split_names(ds):
        ds_split = get_split(ds, split)
        audio_col = get_audio_column(ds_split)

        if audio_col is not None:
            ds[split] = ds_split.cast_column(audio_col, Audio(decode=False))
            print(f"Disabled audio decoding for split={split}, column={audio_col}")

    return ds


def load_hf_split(hf_name: str, config: Optional[str], split: str, streaming: bool):
    name_display = hf_name if config is None else f"{hf_name}/{config}"
    print(f"Loading dataset: {name_display}, split={split}, streaming={streaming}")

    if config is None:
        ds = load_dataset(hf_name, split=split, streaming=streaming)
    else:
        ds = load_dataset(hf_name, config, split=split, streaming=streaming)

    try:
        if hasattr(ds, "features") and ds.features is not None:
            for col, feature in ds.features.items():
                if isinstance(feature, Audio):
                    ds = ds.cast_column(col, Audio(decode=False))
    except Exception as e:
        print(f"Warning: could not cast audio column to decode=False: {e}")

    return ds


def iter_rows(ds, max_scan: int):
    count = 0
    for row in ds:
        yield row
        count += 1
        if max_scan > 0 and count >= max_scan:
            break