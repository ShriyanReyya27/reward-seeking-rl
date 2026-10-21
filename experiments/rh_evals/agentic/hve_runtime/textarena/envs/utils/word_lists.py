"""Replacement for TextArena's ``EnglishDictionary`` (envs/utils/word_lists.py).

The original builds Hunspell en_GB/en_US + NLTK word sets at construction. Only Wordle uses it
here, and only ``is_english_word`` on 5-letter guesses, so this reads the 5-letter words that the
original accepts from ``data/wordle5_valid.txt`` (made by ``hve_tools/make_wordlists.py`` from the
original code). Longer or shorter words are never checked by the vendored engines.
"""

from pathlib import Path

_DATA = Path(__file__).resolve().parents[3] / "data" / "wordle5_valid.txt"
_WORDS = None


def _words():
    global _WORDS
    if _WORDS is None:
        _WORDS = set(_DATA.read_text().split())
    return _WORDS


class EnglishDictionary:
    """Dictionary Utils for English words."""

    def __init__(self, keep_proper_nouns=False, include_nltk=True, keep_non_alpha=False):
        self.keep_proper_nouns = keep_proper_nouns
        self.include_nltk = include_nltk
        self.keep_non_alpha = keep_non_alpha

    def is_english_word(self, word: str) -> bool:
        """Check if a word is in the UK and/or US and/or NLTK English dictionary."""
        return word.lower() in _words()
