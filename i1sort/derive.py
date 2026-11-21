"""Coverage: when a word the learner never studied still counts as known.

A word in a sentence is covered when its key is known, or when it is a predictable
relative of a known word. The links are built from JMdict on the fly, after Jiten's
"derived forms" (structural rule + gloss check):

  potential  使える ← 使う        a 下一段 verb whose senses say "able to"
  affix      お客さん ← 客, 父さん ← お父さん, 人たち ← 人, 奴ら ← 奴
             お/ご prefixes; さん/ちゃん/様/くん/君/殿 and plural たち/達/ら/ども/がた
             suffixes. Except for plurals, the glosses must overlap: お前 "you" is not 前.
  stem       代わり ← 代わる: a verb's ます stem used as a noun (glosses must overlap);
             やり方 ← やる: stem + 方 "way of doing"
  variant    風【ふう】 ← 風【かぜ】, 後【ご】 ← 後【あと】: the same single kanji read another
             way, when the known word's glosses already cover the first sense
  parts      言う通り, 気をつける: a merged expression whose content parts are all covered;
             捜査官 (捜査 + 官), 見つけ出す: a merged compound read as its parts are read;
             どういう (どう言う): a 連体詞/adverb/conjunction/expression entry spelled as two
             known words with matching readings

  morpheme   語 in 異世界語 ← 日本語, 小学 in 小学生 ← 小学校: a short kanji noun glued to a
             neighbouring noun (a compound JMdict does not list), used the same way, with the
             same reading, at the same end of a known word

Known vocab phrases also teach their parts (先に → 先); that happens when the known set
is built (known.py, "phrase").
"""

from __future__ import annotations

import re
from collections import defaultdict

from .textutil import has_kanji, kata_to_hira, readings_match

DERIVATIONS = ("potential", "affix", "stem", "variant", "parts", "morpheme", "phrase")

PREFIXES = (("お", "お"), ("御", "お"), ("ご", "ご"), ("御", "ご"))
HONORIFIC_SUFFIXES = (("さん", "さん"), ("ちゃん", "ちゃん"), ("様", "さま"), ("さま", "さま"),
                      ("くん", "くん"), ("君", "くん"), ("殿", "どの"))
PLURAL_SUFFIXES = (("たち", "たち"), ("達", "たち"), ("ら", "ら"), ("等", "ら"), ("ども", "ども"),
                   ("共", "ども"), ("がた", "がた"), ("方", "がた"))
GLOSS_EXPECT = ("honorific", "polite", "respectful")

_GODAN_I = {"う": "い", "つ": "ち", "る": "り", "む": "み", "ぶ": "び", "ぬ": "に", "く": "き", "ぐ": "ぎ", "す": "し"}
_GODAN_POS = {"v5u": "う", "v5t": "つ", "v5r": "る", "v5m": "む", "v5b": "ぶ", "v5n": "ぬ", "v5k": "く",
              "v5g": "ぐ", "v5s": "す", "v5k-s": "く", "v5r-i": "る", "v5u-s": "う", "v5aru": "る"}
_ICHIDAN_POS = {"v1", "v1-s"}
_TAILS = ("ような", "みたいな", "って", "か", "よ", "ね", "な", "ぞ", "ぜ", "さ", "わ")
_SPLIT_POS = frozenset({"adj-pn", "adv", "conj", "exp"})
_FUNCTION_POS = ("助詞", "助動詞", "補助記号", "記号", "空白", "接頭辞", "接尾辞")

_STOP = frozenset(
    "the and for with that this from into onto one ones someone something somebody etc esp usu used use "
    "make made become becoming have has having his her their its not but who which when such also very "
    "more most each other another person thing state being about over out off all any may can will would "
    "to of a an in on at by as or be is are was it oneself one's".split()
)
_SUFFIXES = ("iness", "ness", "ality", "ility", "ity", "ance", "ence", "ment", "ion", "ing",
             "ers", "er", "ed", "es", "s", "ly", "al")


def _stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    # refusal/refuse → refus, rejection/reject → reject, declining/decline → declin
    return word[:-1] if word.endswith("e") and len(word) > 4 else word


def _content(gloss: str) -> set[str]:
    gloss = re.sub(r"\([^)]*\)", " ", gloss.lower())
    return {_stem(t) for t in re.findall(r"[a-z]+", gloss) if t not in _STOP and len(t) > 1}


def _vocab(entry) -> set[str]:
    out: set[str] = set()
    for s in entry.senses:
        out |= _content(s.gloss)
    return out


def first_sense_covered(derived, base, expect: tuple[str, ...] = ()) -> bool:
    """Jiten's sense check, on the derived word's main sense: it shares a content word
    with the base's glosses, or reads exactly as the grammar point predicts."""
    if not derived.senses:
        return False
    g = derived.senses[0].gloss
    if expect and any(e in g.lower() for e in expect):
        return True
    return bool(_content(g) & _vocab(base))


def affix_cores(text: str, reading: str) -> list[tuple[str, str, bool]]:
    """Strip one honorific prefix and/or one honorific or plural suffix.

    Returns (core text, core reading, plural_only) for every way of stripping at least one
    affix whose spelling and reading both match: お兄ちゃん/おにいちゃん → 兄/にい."""
    pres = [("", "")] + [(t, r) for t, r in PREFIXES if text.startswith(t) and reading.startswith(r)]
    sufs = [("", "", True)] + [(t, r, False) for t, r in HONORIFIC_SUFFIXES if text.endswith(t) and reading.endswith(r)] \
        + [(t, r, True) for t, r in PLURAL_SUFFIXES if text.endswith(t) and reading.endswith(r)]
    out = []
    for pt, pr in pres:
        for st, sr, plural in sufs:
            if not pt and not st:
                continue
            ct = text[len(pt): len(text) - len(st)]
            cr = reading[len(pr): len(reading) - len(sr)]
            if ct and cr and (has_kanji(ct) or len(ct) >= 2):
                out.append((ct, cr, plural and not pt))
    return out


def masu_stems(text: str, reading: str, pos) -> list[tuple[str, str]]:
    out = []
    for tag in pos:
        end = _GODAN_POS.get(tag)
        if end and text.endswith(end) and reading.endswith(end) and len(reading) >= 2:
            out.append((text[:-1] + _GODAN_I[end], reading[:-1] + _GODAN_I[end]))
        elif tag in _ICHIDAN_POS and text.endswith("る") and reading.endswith("る") and len(reading) >= 2:
            out.append((text[:-1], reading[:-1]))
    return out


class Coverage:
    """Answers "is this word known, and why" for one known set and dictionary."""

    def __init__(self, known, db, derivations=DERIVATIONS):
        self.known = known
        self.db = db
        self.on = set(derivations)
        self._memo: dict[tuple[str, bool], str | None] = {}
        self._built = False

    # ------------------------------------------------------------------ indexes
    def _forms(self, seq: int, primary: bool = False):
        """JMdict spellings with hiragana readings. Only real spellings: a kanji word's bare
        reading would let 柱【はしら】 strip ら and land on 橋【はし】. ``primary``: the headword
        and common spellings only (元【もと】 lists 由 as a rare spelling; 由 is よし)."""
        e = self.db.entry(seq) if primary else None
        rows = [(f.term, kata_to_hira(f.reading)) for f in self.db.forms_of_entry(seq)
                if not primary or f.common or (e is not None and f.term == e.headword)]
        return list(dict.fromkeys(rows))

    def _build(self) -> None:
        # (text, reading) → [(key, seq, stripped, plural_only)]
        self.cores: dict[tuple[str, str], list[tuple[str, int, bool, bool]]] = defaultdict(list)
        self.stems: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
        self.spell: dict[str, list[tuple[str, int]]] = defaultdict(list)
        self.orig: dict[tuple[str, str], list[str]] = defaultdict(list)  # known words' own forms
        self.orig_reading: dict[str, list[str]] = defaultdict(list)
        # first/last character → {(form, reading): keys}, for the morpheme rule
        self.orig_by_char: dict[str, dict[tuple[str, str], list[str]]] = defaultdict(lambda: defaultdict(list))
        for key in self.known.words:
            if not key.startswith("j"):
                continue
            seq = int(key[1:])
            e = self.db.entry(seq)
            if e is None:
                continue
            verb = any(t.startswith("v5") or t in _ICHIDAN_POS for t in e.pos)
            primary = set(self._forms(seq, primary=True))
            for t, r in self._forms(seq):
                self.cores[(t, r)].append((key, seq, False, False))
                if not e.pos & {"prt", "aux", "aux-v", "aux-adj"}:
                    self.orig[(t, r)].append(key)
                    if key not in self.orig_reading[r]:
                        self.orig_reading[r].append(key)
                if has_kanji(t) and len(t) >= 2:
                    self.orig_by_char[t[0]][(t, r)].append(key)
                    if t[-1] != t[0]:
                        self.orig_by_char[t[-1]][(t, r)].append(key)
                if has_kanji(t):
                    for ct, cr, plural in affix_cores(t, r):
                        if has_kanji(ct):
                            self.cores[(ct, cr)].append((key, seq, True, plural))
                    if len(t) == 1 and (t, r) in primary:
                        self.spell[t].append((key, seq))
                if verb:
                    for st in masu_stems(t, r, e.pos):
                        self.stems[st].append((key, seq))
        self._built = True

    # ------------------------------------------------------------------ queries
    def ok(self, key: str, learning: bool) -> bool:
        return self.known.is_known(key, learning)

    def why(self, word, learning: bool = True) -> str | None:
        """None if the word is not covered; otherwise the reason ("known", "via 客 (affix)")."""
        if self.ok(word.key, learning):
            return self.known.level(word.key)
        if self.known.exact_match(word):
            return "known"
        loan = getattr(self.known, "loanwords", None)
        if loan is not None:
            why = loan.classify(word, self.db)
            if why:
                return why
        top = getattr(self.known, "auto_known_rank", 0)
        if top and word.seq is not None:
            r = self.db.entry_rank(word.seq)
            if r is not None and r <= top:
                return f"frequent (rank {r:.0f} <= {top:.0f})"
        if word.key.startswith("j") and word.seq is not None:
            mk = (word.key, learning)
            if mk not in self._memo:
                self._memo[mk] = self._lexical(word, learning)
            if self._memo[mk]:
                return self._memo[mk]
        if "parts" in self.on and word.parts:
            r = self._parts(word, learning)
            if r:
                return r
        if "parts" in self.on and word.seq is None and word.kind == "token":
            r = self._tails(kata_to_hira(word.surface), kata_to_hira(word.surface), learning)
            if r:
                return r
        if "morpheme" in self.on and word.bound:
            return self._morpheme(word, learning)
        return None

    def _tails(self, t: str, r: str, learning: bool) -> str | None:
        """まだか, いいよ, どのような: a known word plus a particle or ような/みたいな."""
        if not self._built:
            self._build()
        for tail in _TAILS:
            if len(t) > len(tail) and t.endswith(tail) and r.endswith(tail):
                head_t, head_r = t[: -len(tail)], r[: -len(tail)]
                keys = list(self.orig.get((head_t, head_r), ()))
                if head_t == head_r:  # kana: まだ is filed under 未だ
                    keys += self.orig_reading.get(head_r, [])
                for key in keys:
                    if self.ok(key, learning):
                        return f"via {self.known.words[key].word} + {tail} (parts)"
        return None

    def _morpheme(self, word, learning: bool) -> str | None:
        if not self._built:
            self._build()
        t, r = word.surface, word.reading
        if not r:
            return None
        for (kt, kr), keys in self.orig_by_char.get(t[-1] if word.bound == "suffix" else t[0], {}).items():
            if len(kt) <= len(t):
                continue
            if word.bound == "suffix" and not (kt.endswith(t) and kr.endswith(r)):
                continue
            if word.bound == "prefix" and not (kt.startswith(t) and (
                    kr.startswith(r) or (r[-1] in "くつちき" and kr.startswith(r[:-1] + "っ")))):  # 小学 → 小学校
                continue
            for key in keys:
                if self.ok(key, learning):
                    return f"via {self.known.words[key].word} (morpheme)"
        return None

    def _parts(self, word, learning: bool) -> str | None:
        content = [p for p in word.parts if p.pos0 not in _FUNCTION_POS]
        if not content:
            return None
        if word.kind == "compound" or word.kind == "token":
            if any(p.pos0 == "接頭辞" and p.surface in ("お", "ご", "御") for p in word.parts):
                return None  # honorific compounds go through the affix rule's gloss check (お前)
            # Only compositional compounds: the reading is the parts' readings joined
            # (rendaku allowed: 馬小屋 うま+ごや). A compound read some other way is its own word.
            *head, last = word.parts
            joined = "".join(kata_to_hira("".join(m.reading for m in p.morphs)) for p in head) \
                + kata_to_hira(last.morphs[-1].lemma_reading if len(last.morphs) == 1 else "".join(m.reading for m in last.morphs))
            if not word.reading or not readings_match(word.reading, joined):
                return None
        names = []
        for p in content:
            if not p.key or self.why(p, learning) is None:
                return None
            names.append(p.headword or p.lemma)
        return f"via {' + '.join(names)} (parts)"

    def _lexical(self, word, learning: bool) -> str | None:
        if not self._built:
            self._build()
        db = self.db
        e = db.entry(word.seq)
        if e is None:
            return None
        if "potential" in self.on:
            from .known import derived_bases

            for b in derived_bases(word, db):
                if self.ok(b, learning):
                    return f"via {self.known.words[b].word} (potential)"
        forms = self._forms(word.seq)
        if word.reading:
            same = [(t, r) for t, r in forms if r == word.reading]
            forms = same or forms

        def name(key: str) -> str:
            kw = self.known.words.get(key)
            return kw.word if kw else key

        if "affix" in self.on:
            for t, r in forms:
                variants = [(t, r, False, False)] + [(ct, cr, True, pl) for ct, cr, pl in affix_cores(t, r)]
                for ct, cr, stripped, plural in variants:
                    for key, seq, k_stripped, k_plural in self.cores.get((ct, cr), ()):
                        if key == word.key or not self.ok(key, learning):
                            continue
                        if not stripped and not k_stripped:
                            continue  # same form, different entry: not a derivation
                        if not has_kanji(ct) and k_stripped:
                            continue  # kana cores only match a known word's own spelling
                        be = db.entry(seq)
                        if be is None:
                            continue
                        if (not stripped or plural) and (not k_stripped or k_plural):
                            return f"via {name(key)} (affix)"  # plurals need no gloss check
                        if first_sense_covered(e, be, GLOSS_EXPECT):
                            return f"via {name(key)} (affix)"
        if "stem" in self.on:
            for t, r in forms:
                if t.endswith("方") and r.endswith("かた") and len(t) > 1:
                    for key, seq in self.stems.get((t[:-1], r[:-2]), ()):
                        if self.ok(key, learning):
                            return f"via {name(key)} + 方 (stem)"
                cands = [(t, r)] + [(ct, cr) for ct, cr, _ in affix_cores(t, r)]
                for ct, cr in cands:
                    for key, seq in self.stems.get((ct, cr), ()):
                        be = db.entry(seq)
                        if be is not None and self.ok(key, learning) and first_sense_covered(e, be):
                            return f"via {name(key)} (stem)"
        if "parts" in self.on and e.pos & _SPLIT_POS:
            # どういう = どう + 言う: a function-type word spelled as two known words
            for t, r in forms:
                tl = self._tails(t, r, learning)
                if tl:
                    return tl
                for k in range(1, len(t)):
                    if t[:k] == t[k:]:
                        continue  # reduplication: こそこそ is not こそ + こそ
                    for j in range(1, len(r)):
                        lk = [x for x in self.orig.get((t[:k], r[:j]), ()) if self.ok(x, learning)]
                        rk = [x for x in self.orig.get((t[k:], r[j:]), ()) if self.ok(x, learning)]
                        if lk and rk:
                            return f"via {name(lk[0])} + {name(rk[0])} (parts)"
        if "variant" in self.on:
            for t, r in forms:
                if len(t) != 1 or not has_kanji(t) or t != e.headword:
                    continue
                for key, seq in self.spell.get(t, ()):
                    be = db.entry(seq)
                    if key != word.key and be is not None and self.ok(key, learning) and first_sense_covered(e, be):
                        return f"via {name(key)}【{kata_to_hira(be.reading)}】 (variant)"
        return None
