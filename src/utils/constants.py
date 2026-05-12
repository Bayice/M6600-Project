"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(".")
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

RESULTS_DIR = Path("results")
LOG_DIR = RESULTS_DIR / "logs"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"

MODEL_DIR = Path("models")
HF_MODEL_DIR = MODEL_DIR / "hf"


# ---------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------

REMOTE_BASELINE_MODEL = "kaysrubio/accent-id-distilhubert-finetuned-l2-arctic2"
LOCAL_BASELINE_MODEL = HF_MODEL_DIR / "accent-id-distilhubert-finetuned-l2-arctic2"


# ---------------------------------------------------------------------
# L2-ARCTIC paths
# ---------------------------------------------------------------------

L2ARCTIC_HF_NAME = "KoelLabs/L2Arctic"
L2ARCTIC_HF_DIR = RAW_DIR / "L2Arctic_hf"

L2ARCTIC_OFFICIAL_RAW_DIR = RAW_DIR / "l2arctic_release_v5.0"
L2ARCTIC_OFFICIAL_EXTRACT_DIR = RAW_DIR / "l2arctic_release_v5.0_extracted"

L2ARCTIC_HF_METADATA = PROCESSED_DIR / "metadata_l2_arctic.csv"
L2ARCTIC_OFFICIAL_METADATA = PROCESSED_DIR / "metadata_l2_arctic_official.csv"

EXTERNAL_METADATA = PROCESSED_DIR / "external_metadata.csv"


# ---------------------------------------------------------------------
# L2-ARCTIC speaker metadata
# ---------------------------------------------------------------------

SPEAKER_TO_L1 = {
    # Arabic
    "ABA": "Arabic",
    "SKA": "Arabic",
    "YBAA": "Arabic",
    "ZHAA": "Arabic",

    # Hindi
    "ASI": "Hindi",
    "RRBI": "Hindi",
    "SVBI": "Hindi",
    "TNI": "Hindi",

    # Korean
    "HJK": "Korean",
    "HKK": "Korean",
    "YDCK": "Korean",
    "YKWK": "Korean",

    # Mandarin / Chinese
    "BWC": "Mandarin",
    "LXC": "Mandarin",
    "NCC": "Mandarin",
    "TXHC": "Mandarin",

    # Spanish
    "EBVS": "Spanish",
    "ERMS": "Spanish",
    "MBMPS": "Spanish",
    "NJS": "Spanish",

    # Vietnamese
    "HQTV": "Vietnamese",
    "PNV": "Vietnamese",
    "THV": "Vietnamese",
    "TLV": "Vietnamese",
}


SPEAKER_TO_GENDER = {
    "ABA": "m",
    "SKA": "f",
    "YBAA": "m",
    "ZHAA": "f",

    "ASI": "m",
    "RRBI": "m",
    "SVBI": "f",
    "TNI": "f",

    "HJK": "m",
    "HKK": "m",
    "YDCK": "f",
    "YKWK": "f",

    "BWC": "m",
    "LXC": "f",
    "NCC": "f",
    "TXHC": "m",

    "EBVS": "m",
    "ERMS": "m",
    "MBMPS": "f",
    "NJS": "f",

    "HQTV": "m",
    "PNV": "f",
    "THV": "f",
    "TLV": "m",
}


BASELINE_LABELS = [
    "Arabic",
    "Hindi",
    "Korean",
    "Mandarin",
    "Spanish",
    "Vietnamese",
]