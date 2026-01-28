"""
Preprocessing utilities for the SciFact vector space IR system.

Transform raw document or query text into normalized tokens:
1) lowercase
2) strip markup
3) remove punctuation/special characters and numeric tokens
4) whitespace tokenization
5) stopword removal
6) optional Porter stemming

No I/O or dataset paths are referenced here; callers provide the stopword set.
"""

from __future__ import annotations

import re
from typing import List, Set

from nltk.stem import PorterStemmer

# Precompile regexes for speed and determinism.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Keep only a–z and whitespace after lowercasing; drops punctuation, digits, symbols.
_NON_ALPHA_RE = re.compile(r"[^a-z\s]+")

_stemmer = PorterStemmer()


def preprocess_text(text: str, stopwords: Set[str], stem: bool = False) -> List[str]:
    """
    Clean and tokenize text for indexing or querying.

    Args:
        text: Raw input text.
        stopwords: Set of stopwords to remove.
        stem: If True, apply Porter stemming.

    Returns:
        List of normalized tokens (order preserved).
    """
    # 1) Normalize case.
    text = text.lower()

    # 2) Remove simple HTML/XML markup.
    text = _HTML_TAG_RE.sub(" ", text)

    # 3) Remove punctuation, digits, and other symbols; keep letters and spaces.
    text = _NON_ALPHA_RE.sub(" ", text)

    # 4) Tokenize on whitespace.
    tokens = text.split()

    # 5) Remove stopwords and empty tokens.
    tokens = [tok for tok in tokens if tok and tok not in stopwords]

    # 6) Optional Porter stemming.
    if stem:
        tokens = [_stemmer.stem(tok) for tok in tokens]

    return tokens
