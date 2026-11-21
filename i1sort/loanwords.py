"""Loanwords that count as known: the ``[known] loanwords`` languages.

A word is a loanword when its JMdict entry is spelled only in katakana and its source
language is one of the chosen ones. JMdict leaves plain English loans unmarked, so an
entry with no source counts as English (unless it is slang, like ガチムチ). Wasei words
(サラリーマン) count only with ``loanwords_wasei``; ``not_loanword_files`` lists katakana
words that are Japanese (オタク). A loanword stays a word in the sentence, known like any
other known word; it is never a target.
"""

from __future__ import annotations

from .config import Config
from .filters import _NAME_MISC, load_wordlist


def loanword_languages(setting) -> set[str]:
    """``loanwords`` setting → lowercase language names: true = English, false/[] = off,
    "english, french" or ["English", "French"]."""
    if setting is True:
        return {"english"}
    if not setting:
        return set()
    items = setting.split(",") if isinstance(setting, str) else setting
    return {str(x).strip().lower() for x in items if str(x).strip()}


def _katakana_word(t: str) -> bool:
    return bool(t) and all(("゠" <= c <= "ヿ") or c in "ー・＝=ｰ" for c in t)


def source_labels(db, seq: int | None) -> list[str]:
    """JMdict source labels of a katakana-only loanword entry ("English (wasei)"), with
    "English (unspecified)" for plain English loans JMdict leaves unmarked; [] if the entry
    is not a loanword (kanji or hiragana spellings, onomatopoeia, names, Japanese slang)."""
    if not seq:
        return []
    e = db.entry(seq)
    if e is None or "on-mim" in e.misc or e.misc & _NAME_MISC:
        return []
    terms = [f.term for f in db.forms_of_entry(seq)]
    if not terms or not all(_katakana_word(t) for t in terms):
        return []
    labels = list(e.lsource)
    if not labels and not (e.misc & {"sl", "m-sl"}):  # ガチムチ: Japanese slang
        labels = ["English (unspecified)"]  # JMdict leaves plain English loans unmarked
    return labels


class LoanwordRule:
    def __init__(self, languages: set[str], wasei: bool = True, not_loanwords: set[str] | None = None):
        self.languages = languages
        self.wasei = wasei
        self.not_loanwords = not_loanwords or set()

    @classmethod
    def from_config(cls, cfg: Config) -> "LoanwordRule | None":
        from .config import as_list

        kc = cfg["known"]
        langs = loanword_languages(kc.get("loanwords", True))
        if not langs:
            return None
        nots: set[str] = set()
        for f in as_list(kc.get("not_loanword_files")):
            nots |= load_wordlist(cfg.resolve(f))
        return cls(langs, bool(kc.get("loanwords_wasei", True)), nots)

    def classify(self, w, db) -> str | None:
        """Why the word is a loanword of the chosen languages ("English loanword (…)"), or None."""
        if w.kind == "expression" or w.pos0 in ("助詞", "助動詞") or not w.seq:
            return None
        if tuple(w.pos[:3]) == ("名詞", "固有名詞", "人名"):
            return None  # a person's name, even one JMdict also lists as a loanword
        if any(s and s in self.not_loanwords for s in (w.surface, w.lemma, w.headword)):
            return None  # オタク: a Japanese word JMdict gives no source for
        for label in source_labels(db, w.seq):
            lang, _, flags = label.partition(" (")
            if lang.lower() not in self.languages:
                continue
            if "wasei" in flags and not self.wasei:
                continue
            if label == "English (unspecified)":
                return "English loanword (JMdict: katakana only, no other source)"
            return f"{lang} loanword (JMdict: {label})"
        return None
