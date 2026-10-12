"""utils.py - shared helper functions used by train_model.py and app.py."""
import re

# Real articles in this dataset often start with "CITY (Reuters) -".
# If we leave it in, the model just learns "Reuters = real" (data leakage).
SOURCE_PATTERN = re.compile(r"^.{0,80}?\(reuters\)\s*-?\s*", flags=re.IGNORECASE)


def clean_text(text):
    """Lowercase text and remove URLs, source tags, numbers and punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = SOURCE_PATTERN.sub("", text)            # remove "WASHINGTON (Reuters) -"
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove links
    text = re.sub(r"\breuters\b", " ", text)       # remove leftover source word
    text = re.sub(r"[^a-z\s]", " ", text)          # keep letters only
    text = re.sub(r"\s+", " ", text).strip()       # collapse extra spaces
    return text
