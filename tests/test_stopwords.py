"""Tests for patiroha.stopwords."""

import pytest

from patiroha.stopwords import StopwordManager, get_stopwords, list_categories, list_words
from patiroha.stopwords.catalog import CATEGORIES, NPL_CATEGORIES, PATENT_CATEGORIES


def test_categories_defined():
    assert len(CATEGORIES) == 7
    assert set(CATEGORIES.keys()) == {"general", "patent_terms", "structure", "it_control", "chemistry", "misc", "npl"}


def test_patent_categories_exclude_npl():
    assert "npl" not in PATENT_CATEGORIES
    assert "npl" in NPL_CATEGORIES


def test_get_stopwords_patent():
    sw = get_stopwords("patent")
    assert isinstance(sw, frozenset)
    assert "本発明" in sw
    assert len(sw) > 100


def test_get_stopwords_npl():
    sw_patent = get_stopwords("patent")
    sw_npl = get_stopwords("npl")
    assert len(sw_npl) > len(sw_patent)
    assert "論文" in sw_npl


def test_fullwidth_expansion():
    sw = get_stopwords("patent")
    assert "Inc" in sw or "Ｉｎｃ" in sw


def test_stopword_manager_include():
    mgr = StopwordManager(include=["general"])
    sw = mgr.build()
    assert "する" in sw
    assert "本発明" not in sw


def test_stopword_manager_exclude():
    mgr = StopwordManager(exclude=["chemistry"])
    sw = mgr.build()
    assert "する" in sw
    assert "溶液" not in sw


def test_stopword_manager_add_remove():
    mgr = StopwordManager(include=["general"])
    mgr.add(["カスタム用語"])
    sw = mgr.build()
    assert "カスタム用語" in sw

    mgr.remove(["する"])
    sw = mgr.build()
    assert "する" not in sw


def test_stopword_manager_add_overrides_remove():
    mgr = StopwordManager(include=["general"])
    mgr.remove(["テスト"])
    mgr.add(["テスト"])
    sw = mgr.build()
    assert "テスト" in sw


# --- New: list_categories / list_words / summary ---


def test_list_categories():
    cats = list_categories()
    assert isinstance(cats, dict)
    assert len(cats) == 7
    assert "general" in cats
    assert "npl" in cats
    assert all(isinstance(v, int) and v > 0 for v in cats.values())


def test_list_words():
    words = list_words("general")
    assert isinstance(words, list)
    assert len(words) > 0
    assert words == sorted(words)  # Should be sorted
    assert "する" in words


def test_list_words_all_categories():
    for cat in CATEGORIES:
        words = list_words(cat)
        assert len(words) > 0


def test_list_words_unknown_category():
    with pytest.raises(KeyError, match="Unknown category"):
        list_words("nonexistent")


def test_manager_list_active_words():
    mgr = StopwordManager(include=["general", "patent_terms"])
    mgr.add(["自社用語"])
    active = mgr.list_active_words()
    assert "general" in active
    assert "patent_terms" in active
    assert "custom" in active
    assert "自社用語" in active["custom"]
    assert "chemistry" not in active


def test_manager_summary():
    mgr = StopwordManager(include=["general"])
    mgr.add(["テスト語"])
    mgr.remove(["する"])
    s = mgr.summary()
    assert "general" in s
    assert "テスト語" in s
    assert "する" in s
    assert "Total after expansion" in s
