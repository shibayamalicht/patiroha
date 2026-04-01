"""N-gram filters for removing patent boilerplate, reference symbols, and formulaic phrases."""

from __future__ import annotations

import re
from re import Pattern
from typing import Union

# Filter definitions: (category, pattern, pattern_type, priority)
_NGRAM_ROWS: list[tuple[str, str, str, int]] = [
    # Reference symbols with parentheses
    (
        "参照符号付き要素",
        r"[一-龥ぁ-んァ-ンA-Za-z0-9／\-＋・]+?(?:部|層|面|体|板|孔|溝|片|部材|要素|機構|装置|手段|電極|端子|領域|基板|回路|材料|工程)\s*[（(]\s*[0-9０-９A-Za-z]+[A-Za-z]?\s*[）)]",
        "regex",
        1,
    ),
    (
        "参照符号付き要素",
        r"(?:上記|前記)?[一-龥ぁ-んァ-ンA-Za-z0-9／\-＋・]+?(?:部|層|面|体|板|孔|溝|片|部材|要素|機構|装置|手段|電極|端子|領域|基板|回路|材料|工程)\s*[0-9０-９A-Za-z]+[A-Za-z]?",
        "regex",
        1,
    ),
    ("参照符号付き要素", r"[A-Z]+[0-9]+", "regex", 1),
    # Section headers
    ("見出し・章句", "一実施形態において", "literal", 1),
    ("見出し・章句", "他の実施形態において", "literal", 1),
    ("見出し・章句", "別の実施形態において", "literal", 1),
    ("見出し・章句", "本明細書において", "literal", 1),
    ("見出し・章句", "本明細書では", "literal", 1),
    ("見出し・章句", "本発明の一側面", "literal", 1),
    ("見出し・章句", "一実施例において", "literal", 1),
    ("見出し・章句", "他の実施例において", "literal", 1),
    ("見出し・章句", "好ましい態様として", "literal", 2),
    ("見出し・章句", "好適には", "literal", 2),
    ("見出し・章句", "用語の定義", "literal", 2),
    ("見出し・章句", "図示しない", "literal", 2),
    # Figure / table references
    ("図表参照", r"図[ 　]*[０-９0-9]+に示す", "regex", 1),
    ("図表参照", r"表[ 　]*[０-９0-9]+に示す", "regex", 1),
    ("図表参照", r"式[ 　]*[０-９0-9]+に示す", "regex", 1),
    ("図表参照", r"請求項[ 　]*[０-９0-9]+", "regex", 1),
    ("図表参照", r"(?:【|\[)\s*[０-９0-9]{4,5}\s*(?:】|\])", "regex", 1),
    ("図表参照", r"[（(][０-９0-9]+[）)]", "regex", 2),
    ("図表参照", r"第\s*[０-９0-9]+の?実施形態", "regex", 2),
    ("図表参照", r"段落\s*[０-９0-9]+", "regex", 2),
    ("図表参照", r"図[ 　]*[０-９0-9]+[A-Za-z]?", "regex", 2),
    # Definition introductions
    ("定義導入", r"以下、[^、。]+を[^、。]+と称する", "regex", 1),
    ("定義導入", r"以下、[^、。]+を[^、。]+という", "regex", 1),
    # Functional phrases
    ("機能句", "してもよい", "literal", 1),
    ("機能句", "であってもよい", "literal", 1),
    ("機能句", "することができる", "literal", 1),
    ("機能句", "行うことができる", "literal", 1),
    ("機能句", "に限定されない", "literal", 1),
    ("機能句", "に限られない", "literal", 1),
    ("機能句", "一例として", "literal", 2),
    ("機能句", "例示的には", "literal", 2),
    # Reference phrases
    ("参照句", "前述のとおり", "literal", 2),
    ("参照句", "前述の通り", "literal", 2),
    ("参照句", "後述するように", "literal", 2),
    ("参照句", "後述のとおり", "literal", 2),
    # Scope expressions
    ("範囲表現", r"少なくとも(?:一|１)つ", "regex", 2),
    ("範囲表現", "少なくとも一部", "literal", 2),
    ("範囲表現", r"複数の(?:実施形態|構成|要素)", "regex", 3),
    # Problem phrases
    ("課題句", r"(?:上記|前記)の?課題", "regex", 1),
    # Logical connectives
    ("接続・論理", "一方で", "literal", 3),
    ("接続・論理", "他方で", "literal", 3),
    ("接続・論理", "すなわち", "literal", 3),
]

# Pre-compile regex patterns at module load time
_CompiledEntry = tuple[str, Union[Pattern[str], str], str, int]
_NGRAM_COMPILED: list[_CompiledEntry] = []
for _cat, _pat, _ptype, _pri in _NGRAM_ROWS:
    if _ptype == "regex":
        _NGRAM_COMPILED.append((_cat, re.compile(_pat), _ptype, _pri))
    else:
        _NGRAM_COMPILED.append((_cat, _pat, _ptype, _pri))


def apply_ngram_filters(text: str) -> str:
    """Remove patent boilerplate patterns from text.

    Applies regex and literal pattern matching to remove reference symbols,
    section headers, figure references, formulaic phrases, etc.

    Args:
        text: Input text to filter.

    Returns:
        Filtered text with boilerplate removed.
    """
    for _cat, pat, ptype, _pri in _NGRAM_COMPILED:
        if ptype == "literal":
            assert isinstance(pat, str)
            if pat in text:
                text = text.replace(pat, "")
        else:
            assert not isinstance(pat, str)
            text = pat.sub("", text)
    return text
