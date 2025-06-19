"""
Text normalization utilities for Whisper fine-tuning
"""

import re
import regex
from transformers.models.whisper.english_normalizer import (
    remove_symbols_and_diacritics,
    remove_symbols,
)


class BasicTextNormalizer:
    """Text normalizer for Dutch/Flemish text with parentheses removal to keep speech events like (lacht)"""

    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        self.clean = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters = split_letters

    def __call__(self, s: str):
        s = s.lower()
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(
            r"\s+", " ", s
        )  # replace any successive whitespace characters with a space

        return s
