"""
Author: He Xu
UNI: xh2707
Course: MEEC E6600 - Mathematics of Machine Learning, Signals, and Control
"""

import re
from typing import Tuple


def safe_name(text: str, max_len: int = 120) -> str:
    text = str(text)
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    return text.strip("_")[:max_len] or "unknown"


def normalize_l2_label(label: str) -> str:
    text = str(label).strip().lower()

    mapping = {
        "arabic": "Arabic",
        "hindi": "Hindi",
        "korean": "Korean",
        "mandarin": "Mandarin",
        "chinese": "Mandarin",
        "china": "Mandarin",
        "spanish": "Spanish",
        "spain": "Spanish",
        "vietnamese": "Vietnamese",
        "vietnam": "Vietnamese",
        "viet": "Vietnamese",
    }

    return mapping.get(text, str(label).strip())


def normalize_external_label_for_baseline(label: str) -> Tuple[str, bool]:
    """
    Return:
        target_label, is_mapped_to_baseline

    If is_mapped_to_baseline=True, target_label is one of the six L2-ARCTIC classes.
    If False, target_label is an OOD label and should not be treated as correct/incorrect
    for the six-class baseline.
    """
    x = str(label).strip().lower()

    if not x or x in {"nan", "none", "unknown"}:
        return "OOD_UNKNOWN", False

    # Strong mappings to L2-ARCTIC six classes.
    if "arabic" in x:
        return "Arabic", True
    if "hindi" in x:
        return "Hindi", True
    if "korean" in x:
        return "Korean", True
    if "mandarin" in x or "chinese" in x or "china" in x:
        return "Mandarin", True
    if "spanish" in x or "spain" in x:
        return "Spanish", True
    if "vietnamese" in x or "vietnam" in x:
        return "Vietnamese", True

    # Weak mapping. Useful for analysis, but be careful using it for training.
    if "india" in x or "south asia" in x or "pakistan" in x or "sri lanka" in x:
        return "Hindi", True

    # OOD categories.
    if "singapore" in x or "singlish" in x:
        return "OOD_Singaporean_English", False
    if "malaysia" in x or "malaysian" in x:
        return "OOD_Malaysian_English", False
    if "hong kong" in x:
        return "OOD_Hong_Kong_English", False
    if "filipino" in x or "philippines" in x:
        return "OOD_Filipino_English", False
    if "german" in x or "austrian" in x or "dutch" in x:
        return "OOD_Germanic_European_English", False
    if "africa" in x or "zimbabwe" in x or "namibia" in x:
        return "OOD_Southern_African_English", False
    if "jamaica" in x or "bermuda" in x or "bahamas" in x or "trinidad" in x or "west indies" in x:
        return "OOD_Caribbean_English", False
    if any(k in x for k in ["irish", "scottish", "welsh", "northern", "southern", "midlands", "england", "british"]):
        return "OOD_British_Isles_English", False
    if "american" in x or "united states" in x or "usa" in x:
        return "OOD_North_American_English", False

    return "OOD_" + safe_name(label), False