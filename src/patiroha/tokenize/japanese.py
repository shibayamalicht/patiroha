"""Japanese compound noun extraction and TF-IDF tokenization.

Provides Janome-based tokenization with configurable POS tags, n-gram filter
patterns, and minimum token length.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from janome.tokenizer import Tokenizer

from patiroha.stopwords import get_stopwords
from patiroha.tokenize.filters import apply_ngram_filters
from patiroha.tokenize.normalize import normalize_text, strip_html

# Module-level tokenizer instance (reused across calls)
_DEFAULT_TOKENIZER: Tokenizer | None = None

# Default rejection patterns for compound nouns
DEFAULT_REJECT_PATTERNS: list[str] = [
    r"[\d０-９]+$",  # Pure digits
    r"(図|表|式|第)[\d０-９]+.*",  # Figure/table refs
    r"^(上記|前記|本開示|当該|該)",  # Boilerplate prefixes
    r"[0-9０-９]+[)）]?$",  # Trailing reference numbers
    r"[0-9０-９]+[a-zA-Zａ-ｚＡ-Ｚ]",  # Alphanumeric refs
    r"^[ぁ-ん]$",  # Single hiragana
]


def _get_tokenizer() -> Tokenizer:
    global _DEFAULT_TOKENIZER
    if _DEFAULT_TOKENIZER is None:
        _DEFAULT_TOKENIZER = Tokenizer()
    return _DEFAULT_TOKENIZER


def _is_valid_compound(
    word: str,
    stopwords: frozenset[str],
    min_length: int,
    reject_patterns: list[re.Pattern[str]],
) -> bool:
    """Check if a compound noun passes all validation filters."""
    if len(word) < min_length:
        return False
    if word in stopwords:
        return False
    return not any(p.search(word) for p in reject_patterns)


def extract_keywords(
    text: str,
    stopwords: Iterable[str] | None = None,
    apply_filters: bool = True,
    clean_html: bool = False,
    tokenizer: Tokenizer | None = None,
    pos_tags: Sequence[str] = ("名詞",),
    min_length: int = 2,
    extra_reject_patterns: Sequence[str] | None = None,
    disable_default_reject: bool = False,
) -> list[str]:
    """Extract compound nouns (keywords) from Japanese patent text.

    Args:
        text: Input text to analyze.
        stopwords: Stopword set. If None, uses default patent stopwords.
        apply_filters: If True, applies n-gram filters before extraction.
        clean_html: If True, strips HTML tags first (recommended for NPL).
        tokenizer: Janome Tokenizer instance. If None, uses default.
        pos_tags: POS tags to extract. Default is ("名詞",) for nouns only.
            Add "動詞" for verbs, "形容詞" for adjectives, etc.
        min_length: Minimum character length for extracted keywords.
        extra_reject_patterns: Additional regex patterns to reject compounds.
        disable_default_reject: If True, skips built-in rejection patterns.

    Returns:
        List of extracted keyword strings.
    """
    if not text:
        return []

    sw: frozenset[str] = frozenset(stopwords) if stopwords is not None else get_stopwords()
    tok = tokenizer or _get_tokenizer()
    pos_set = set(pos_tags)

    # Build rejection patterns
    raw_patterns: list[str] = [] if disable_default_reject else list(DEFAULT_REJECT_PATTERNS)
    if extra_reject_patterns:
        raw_patterns.extend(extra_reject_patterns)
    compiled_reject = [re.compile(p) for p in raw_patterns]

    # Pre-processing
    if clean_html:
        text = strip_html(text)
    text = normalize_text(text)
    if apply_filters:
        text = apply_ngram_filters(text)
    text = re.sub(r"【.*?】", "", text)
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', " ", text)

    words: list[str] = []

    # Japanese morphological analysis: compound noun extraction
    tokens = tok.tokenize(text)
    compound_word = ""

    for token in tokens:
        pos = token.part_of_speech.split(",")[0]
        if pos in pos_set:
            compound_word += token.surface
        else:
            if _is_valid_compound(compound_word, sw, min_length, compiled_reject):
                words.append(compound_word)
            compound_word = ""

    # Handle last accumulated compound word
    if _is_valid_compound(compound_word, sw, min_length, compiled_reject):
        words.append(compound_word)

    # English fallback: extract 3+ letter words if no Japanese keywords found
    if not words and re.search(r"[a-zA-Z]", text):
        skips = {w.lower() for w in sw}
        candidates = re.findall(r"\b[a-zA-Z]{3,}\b", text)
        for w in candidates:
            if w.lower() not in skips:
                words.append(w)

    return words


def tokenize_for_tfidf(
    text: str,
    stopwords: Iterable[str] | None = None,
    tokenizer: Tokenizer | None = None,
    pos_tags: Sequence[str] = ("名詞",),
) -> str:
    """Tokenize text for TF-IDF vectorization, returning space-separated tokens.

    Args:
        text: Input text to tokenize.
        stopwords: Stopword set. If None, uses default patent stopwords.
        tokenizer: Janome Tokenizer instance. If None, uses default.
        pos_tags: POS tags to include in output.

    Returns:
        Space-separated string of processed tokens.
    """
    if not isinstance(text, str) or not text:
        return ""

    import unicodedata

    sw: frozenset[str] = frozenset(stopwords) if stopwords is not None else get_stopwords()
    tok = tokenizer or _get_tokenizer()
    pos_set = set(pos_tags)

    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\(（][\w\s]+[\)）]", " ", text)
    text = re.sub(r"\b(図|fig|step|s)\s?\d+\b", " ", text)
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]', " ", text)

    tokens = list(tok.tokenize(text))
    processed_tokens: list[str] = []
    i = 0

    while i < len(tokens):
        token1 = tokens[i]
        base_form = token1.base_form if token1.base_form != "*" else token1.surface

        if base_form in sw or len(base_form) < 2:
            i += 1
            continue

        if (i + 1) < len(tokens):
            token2 = tokens[i + 1]
            base_form2 = token2.base_form if token2.base_form != "*" else token2.surface
            pos1 = token1.part_of_speech.split(",")[0]
            pos2 = token2.part_of_speech.split(",")[0]
            if pos1 in pos_set and pos2 in pos_set and base_form2 not in sw:
                compound_word = base_form + base_form2
                processed_tokens.append(compound_word)
                i += 2
                continue

        pos = token1.part_of_speech.split(",")[0]
        if pos in pos_set:
            processed_tokens.append(base_form)
        i += 1

    return " ".join(processed_tokens)
