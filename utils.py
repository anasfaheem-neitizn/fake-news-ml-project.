"""Shared text helpers. train.py aur app.py dono yahi use karte hain,
taaki training aur prediction mein text bilkul ek jaise clean ho."""
import re

# Source ke naam jo model ko "cheat" karne dete hain (jaise Reuters).
SOURCE_PATTERNS = [
    r"^.{0,80}?\(reuters\)\s*-?\s*",   # "WASHINGTON (Reuters) - " dateline
    r"\breuters\b",
    r"\bfeatured image\b",
    r"\bimage via\b",
    r"\bgetty images?\b",
    r"\bphoto by\b",
    r"\b21st century wire\b",
    r"pic\.twitter\.com\S*",
]


def strip_source(text: str) -> str:
    """Source ke naam hatao taaki model sirf content se seekhe."""
    for pattern in SOURCE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


def clean_text(text: str) -> str:
    """Lowercase, URL aur punctuation hatao, extra spaces ek karo."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
