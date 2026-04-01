"""Tests for patiroha.tokenize."""

from patiroha.tokenize import apply_ngram_filters, extract_keywords, normalize_text, strip_html


def test_normalize_text_nfkc():
    assert normalize_text("ＡＢＣ") == "ABC"


def test_normalize_text_whitespace():
    assert normalize_text("a  b   c") == "a b c"


def test_normalize_text_mu():
    assert "μ" in normalize_text("µm")


def test_normalize_text_nan():
    import pandas as pd

    assert normalize_text(pd.NA) == ""
    assert normalize_text(None) == ""


def test_strip_html():
    assert strip_html("<p>hello</p>") == " hello "


def test_ngram_filter_removes_literal():
    text = "一実施形態において、装置が配置される。"
    filtered = apply_ngram_filters(text)
    assert "一実施形態において" not in filtered


def test_ngram_filter_removes_figure_ref():
    text = "図 3に示す構成を有する。"
    filtered = apply_ngram_filters(text)
    assert "図" not in filtered or "3に示す" not in filtered


def test_extract_keywords_basic():
    text = "セルロースナノファイバーを含有する樹脂組成物に関する。"
    keywords = extract_keywords(text)
    assert len(keywords) > 0
    # Should extract compound nouns
    assert any("セルロース" in kw for kw in keywords) or any("樹脂" in kw for kw in keywords)


def test_extract_keywords_empty():
    assert extract_keywords("") == []
    assert extract_keywords(None) == []  # type: ignore[arg-type]


def test_extract_keywords_english_fallback():
    text = "This polymer composite exhibits excellent thermal properties."
    keywords = extract_keywords(text)
    assert len(keywords) > 0
    assert any("polymer" in kw.lower() or "composite" in kw.lower() for kw in keywords)


def test_extract_keywords_with_filters():
    text = "図1に示す部材(101)は、本発明の一実施形態において使用される光学フィルムである。"
    keywords = extract_keywords(text, apply_filters=True)
    # Reference symbols should be removed
    for kw in keywords:
        assert "(101)" not in kw
