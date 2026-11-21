"""Map word units to JMdict entries (word identity).

A unit's candidates are the JMdict entries with a spelling equal to its
dictionary form, its Sudachi normalized form, or (for kana) its reading. Each
candidate is scored on how it matched, whether its reading agrees with the
tokenizer's reading of the lemma, part-of-speech compatibility, and commonness.
The best entry's sequence number becomes the word key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..dictdb import DictDB, Entry, FormRow
from ..textutil import has_kanji, is_kana, is_katakana, kata_to_hira, readings_match
from .base import POS

# Sudachi POS (pos[0], optionally pos[1]) → compatible JMdict POS families.
_COMPAT: dict[str, set[str]] = {
    "名詞": {"noun", "vs", "adj-na", "adv", "suffix", "pn", "exp", "ctr"},
    "代名詞": {"pn", "noun"},
    "動詞": {"verb", "vs", "aux", "exp"},
    "形容詞": {"adj-i", "aux", "exp"},
    "形状詞": {"adj-na", "noun", "adv", "adj-i", "exp"},
    "副詞": {"adv", "noun", "adj-na", "int", "exp", "conj"},
    "連体詞": {"adj-pn", "pn", "exp"},
    "接続詞": {"conj", "exp", "adv"},
    "感動詞": {"int", "exp", "adv", "noun", "adj-na"},
    "接頭辞": {"prefix", "noun"},
    "接尾辞": {"suffix", "noun", "ctr"},
    "助詞": {"prt", "conj", "exp"},
    "助動詞": {"aux", "exp", "verb", "adj-i"},
}

_NAME_MISC = {"organization", "company", "product", "work", "char", "person", "place", "given",
              "surname", "fem", "masc", "serv", "group", "creat", "dei", "ev", "fict", "leg", "myth",
              "obj", "oth", "relig", "ship", "station", "unclass", "doc"}
_ARCHAIC = {"arch", "obs", "dated", "rare"}


@dataclass
class Resolution:
    seq: int
    entry: Entry
    score: float
    how: str


_EXACT_POS = {"代名詞": "pn", "副詞": "adv", "連体詞": "adj-pn", "接続詞": "conj", "感動詞": "int"}


def pos_compatible(pos: POS, entry: Entry) -> bool:
    fams = entry.pos_family()
    want = _COMPAT.get(pos[0] if pos else "", set())
    return bool(fams & want)


class Resolver:
    def __init__(self, db: DictDB):
        self.db = db

    def _score(self, entry: Entry, row: FormRow, how: str, lemma_reading: str, pos: POS | None) -> float:
        s = {"lemma": 6.0, "norm": 5.0, "surface": 4.0, "reading": 3.0, "joined": 6.0}[how]
        if how == "reading":
            if "uk" in entry.misc or is_kana(row.term):
                s += 1.5
        lr = kata_to_hira(lemma_reading or "")
        rr = kata_to_hira(row.reading)
        if lr and is_kana(lr) and is_kana(rr):
            if lr == rr:
                s += 4
            elif readings_match(lr, rr):
                s += 2
            else:
                s -= 3
        if pos:
            s += 3 if pos_compatible(pos, entry) else -3
            exact = _EXACT_POS.get(pos[0])
            if exact and exact in entry.pos:
                s += 2.5  # わし (代名詞) is 儂 "I", not 鷲 "eagle"
        # A kanji spelling reached through its reading: that spelling's commonness says
        # nothing about the kana text (いけない is いける, not the common 生ける).
        if row.common and not (how == "reading" and has_kanji(row.term)):
            s += 1.5
        if entry.common:
            s += 1
        if how != "reading" and is_katakana(row.term) and "uk" in entry.misc:
            s += 0.5  # クズ "scum" is 屑 (usually written in kana), not the plant 葛
        if how != "reading" and row.term == entry.headword:
            s += 0.5  # 人手 is the headword of 人手 "manpower", only a variant of 海星 "starfish"
        if row.rare:
            s -= 2
        if entry.misc & _NAME_MISC and not entry.common:
            s -= 2
        if entry.senses and all(sn.misc & _ARCHAIC for sn in entry.senses):
            s -= 1.5
        s += max(-1.0, min(1.0, entry.score / 2_000_000))  # Yomitan popularity, tie-breaker
        # Frequency tie-breakers. For kana matches, how often *this* word is written in
        # kana (JPDB/Jiten ㋕ ranks) separates homophones like 済む/住む/澄む for すむ.
        if how == "reading" or (is_kana(row.reading) and is_kana(lemma_reading or "x") and not has_kanji(row.term)):
            kr = self.db.entry_kana_rank(row.seq, row.reading)
            if kr:
                s += max(0.0, min(3.0, 3.0 - 0.6 * math.log10(kr)))
        fr = self.db.entry_rank(row.seq)
        if fr:
            s += max(0.0, min(1.5, 1.5 - 0.3 * math.log10(fr)))
        return s

    def candidates(self, forms: list[tuple[str, str]], lemma_reading: str, pos: POS | None,
                   allow_reading: bool = True, strict_reading: bool = False) -> list[Resolution]:
        best: dict[int, Resolution] = {}
        seen: set[tuple[str, str]] = set()
        for text, how in forms:
            if not text or (text, how) in seen:
                continue
            seen.add((text, how))
            rows = list(self.db.forms_by_term(text))
            hows = [how] * len(rows)
            if allow_reading and is_kana(text):
                rr = self.db.forms_by_reading(text)
                rows += rr
                hows += ["reading"] * len(rr)
            for row, h in zip(rows, hows):
                e = self.db.entry(row.seq)
                if e is None:
                    continue
                if pos and pos[0] in ("助詞", "助動詞") and not pos_compatible(pos, e):
                    continue
                if (row.term.startswith(("っ", "ッ")) or e.headword.startswith(("っ", "ッ"))) \
                        and not text.startswith(("っ", "ッ")):
                    continue  # チュウ (a kiss) is not っちゅう
                if h == "reading" and not is_kana(row.term) and "uk" not in e.misc and (strict_reading or is_katakana(text)):
                    continue  # ハハ ≠ 母, カバネ ≠ 姓: katakana/variant text needs a kana-written entry
                sc = self._score(e, row, h, lemma_reading, pos)
                cur = best.get(row.seq)
                if cur is None or sc > cur.score:
                    best[row.seq] = Resolution(row.seq, e, sc, f"{h}:{text}→{row.term}【{row.reading}】")
        return sorted(best.values(), key=lambda r: -r.score)

    def resolve(self, lemma: str, norm: str, surface: str, lemma_reading: str, pos: POS | None,
                strict_reading: bool = False) -> Resolution | None:
        forms = [(lemma, "lemma")]
        if norm and norm != lemma:
            forms.append((norm, "norm"))
        if surface and surface not in (lemma, norm):
            forms.append((surface, "surface"))
        # A kana lemma whose reading is known: also try the reading itself.
        # katakana text (チュウ) never means a kanji word reached only by reading (注), even when
        # Sudachi's normalized form is hiragana
        cands = self.candidates(forms, lemma_reading, pos, strict_reading=strict_reading or (is_katakana(lemma) and is_katakana(surface or lemma)))
        if not cands:
            return None
        if pos and pos[0] == "代名詞":
            # Sudachi's pronoun tag is reliable: あたい is 私 "I", never 値 "price"
            pn = [c for c in cands if "pn" in c.entry.pos]
            cands = pn or cands
        top = cands[0]
        # Require a minimally plausible match: a pure reading hit on an incompatible POS is noise.
        if top.score < 3.0:
            return None
        return top

    def resolve_joined(self, text: str, joined_reading: str, pos_hint: POS | None = None,
                       allow_reading: bool = True) -> list[Resolution]:
        """Candidates for a merged span (compound or expression) spelled exactly ``text``."""
        return self.candidates([(text, "joined")], joined_reading, pos_hint, allow_reading=allow_reading)


def fallback_key(norm: str, pos: POS) -> str:
    return f"~{norm}|{pos[0] if pos else ''}"


def has_kanji_form(s: str) -> bool:
    return has_kanji(s)
