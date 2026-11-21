"""Loanwords count as known: source languages, wasei toggle, exceptions."""

import pytest

from i1sort.config import load_config
from i1sort.filters import TokenFilter
from i1sort.known import KnownSet
from i1sort.loanwords import LoanwordRule, loanword_languages


@pytest.fixture(scope="module")
def parser():
    cfg = load_config()
    if not cfg.resolve(cfg["dictionaries"]["jmdict"]).exists():
        pytest.skip("JMdict not available")
    from i1sort.parse.pipeline import Parser

    return Parser(cfg, use_cache=False)


def filt(parser, sentence, word, **sw):
    """How the word counts: "loanword" when it is known as a loanword, else its filter (None)."""
    cfg = load_config(overrides={"known": {"loanwords": True, "loanwords_wasei": True, "not_loanword_files": [], **sw}})
    tf = TokenFilter(cfg, parser.db, ignore=())
    ks = KnownSet(loanwords=LoanwordRule.from_config(cfg))
    for w in parser.parse(sentence, tf).words:
        if w.surface == word:
            assert w.filter is None  # a loanword stays a word in the sentence
            why = ks.why(w, parser.db)
            return "loanword" if why and "loanword" in why else None
    raise AssertionError(word)


def test_setting_forms():
    assert loanword_languages(True) == {"english"}
    assert loanword_languages(False) == set()
    assert loanword_languages("english, French") == {"english", "french"}
    assert loanword_languages(["English", "German"]) == {"english", "german"}


@pytest.mark.parametrize("sentence,word,langs,expect", [
    ("テレビを見た", "テレビ", ["English"], "loanword"),          # unspecified source = English
    ("アンケートに答えた", "アンケート", ["English"], None),        # French
    ("アンケートに答えた", "アンケート", ["English", "French"], "loanword"),
    ("パンを食べた", "パン", ["English", "French"], None),          # Portuguese
    ("パンを食べた", "パン", ["Portuguese"], "loanword"),
    ("テレビを見た", "テレビ", False, None),
])
def test_languages(parser, sentence, word, langs, expect):
    assert filt(parser, sentence, word, loanwords=langs) == expect


def test_wasei_toggle(parser):
    assert filt(parser, "父はサラリーマンだ", "サラリーマン", loanwords=["English"], loanwords_wasei=True) == "loanword"
    assert filt(parser, "父はサラリーマンだ", "サラリーマン", loanwords=["English"], loanwords_wasei=False) is None


def test_not_loanwords_list(parser):
    assert filt(parser, "彼はオタクだ", "オタク", loanwords=["English"],
                not_loanword_files=["word_lists/not_loanwords.txt"]) is None
