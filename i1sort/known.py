"""The known-word set, built from Anki history over AnkiConnect plus manual lists.

Sources ([[known.sources]] in config.toml) name a note type and a field:

  kind = "vocab"     the field holds one word (Kaishi `Word`); it is resolved straight to
                     a JMdict entry, using `reading_field` to pick between homographs
  kind = "sentence"  every counted word in the field (after the token filters)

Levels, per note, from Anki's own search over its cards (the best card wins):

  known     some card has interval >= known_interval_days, or FSRS stability
            >= known_stability_days (0 = don't use stability)
  learning  studied (not new) but not known
  unknown   never seen (not stored)

A word's level is the best level over all notes it appears in. The ``[known] files``
lists (one word per line, or CSV/TSV with the word in the first column and optionally the
reading in the second; Jiten/jpdb exports work) add words at level known, looked up in the
dictionary. ``exact_files`` add words matched by exact text only.
"""

from __future__ import annotations

import csv
import html
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .anki import AnkiConnect, quote
from .config import Config
from .derive import DERIVATIONS
from .textutil import has_japanese, is_kana, kata_to_hira

LEVELS = {"learning": 1, "known": 2}
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class KnownWord:
    key: str
    word: str
    reading: str = ""
    level: str = "learning"
    sources: set[str] = field(default_factory=set)


@dataclass
class KnownSet:
    words: dict[str, KnownWord] = field(default_factory=dict)
    unresolved: list[tuple[str, str]] = field(default_factory=list)  # (source, text) vocab with no JMdict entry
    stats: dict[str, dict[str, int]] = field(default_factory=dict)  # source → counts
    derivations: tuple[str, ...] = DERIVATIONS  # which derive.py rules count as coverage
    exact: set[str] = field(default_factory=set)  # exact_files: known when a token's text is one of these
    ignored: set[str] = field(default_factory=set)  # mark-deck words bound for an ignore_files list, not written yet
    marked: list = field(default_factory=list)  # mark.MarkDeck per configured mark deck (read from Anki)
    loanwords: object = None  # loanwords.LoanwordRule: loanwords of the chosen languages count as known
    auto_known_rank: float = 0  # words this frequent (rank <= N) count as known; 0 = off

    def add(self, key: str, word: str, reading: str, level: str, source: str) -> None:
        kw = self.words.get(key)
        if kw is None:
            self.words[key] = KnownWord(key, word, reading, level, {source})
            return
        kw.sources.add(source)
        if LEVELS[level] > LEVELS[kw.level]:
            kw.level = level

    def level(self, key: str) -> str:
        kw = self.words.get(key)
        return kw.level if kw else "unknown"

    def exact_match(self, word) -> bool:
        """The token's own text (surface, dictionary form, normalized form) is in an exact list;
        the list entry itself is never looked up in the dictionary, and neither is the token:
        the JMdict entry it resolved to does not matter (it may be the wrong one)."""
        if not self.exact:
            return False
        m = word.morphs[0] if len(word.morphs) == 1 else None
        return any(t and t in self.exact for t in (word.surface, word.lemma, m.norm if m else ""))

    def is_known(self, key: str, learning_counts: bool = True) -> bool:
        lv = self.level(key)
        return lv == "known" or (learning_counts and lv == "learning")

    def coverage(self, db):
        """The derivation rules (derive.py) for this set; built on first use."""
        cov = self.__dict__.get("_coverage")
        if cov is None or cov.db is not db:
            from .derive import Coverage

            cov = Coverage(self, db, self.derivations)
            self.__dict__["_coverage"] = cov
        return cov

    def why(self, word, db, learning_counts: bool = True) -> str | None:
        """None if unknown; else "known", "learning" or how it derives from a known word."""
        return self.coverage(db).why(word, learning_counts)

    def covers(self, word, db, learning_counts: bool = True) -> bool:
        """Known directly, or derived from a known word (使える ← 使う, お客さん ← 客)."""
        return self.why(word, db, learning_counts) is not None

    def counts(self) -> dict[str, int]:
        out = {"known": 0, "learning": 0}
        for w in self.words.values():
            out[w.level] += 1
        return out


def strip_field(value: str) -> str:
    s = html.unescape(_TAG_RE.sub("", value or "")).replace(" ", " ")
    s = re.sub(r"\[[^\]]*\]", "", s)  # Anki furigana 漢字[かんじ]
    return s.strip()


class KnownBuilder:
    def __init__(self, cfg: Config, anki: AnkiConnect | None, parser=None, token_filter=None, log=None):
        self.cfg = cfg
        self.kc = cfg["known"]
        self.anki = anki
        self._parser = parser
        self._tf = token_filter
        self.log = log or (lambda m: print(m, file=sys.stderr))

    # lazily create the parser: vocab-only runs still need the resolver
    @property
    def parser(self):
        if self._parser is None:
            from .parse.pipeline import Parser

            self._parser = Parser(self.cfg)
        return self._parser

    @property
    def token_filter(self):
        if self._tf is None:
            from .filters import TokenFilter

            self._tf = TokenFilter(self.cfg, self.parser.db)
        return self._tf

    # ------------------------------------------------------------------ word resolution
    def resolve_vocab(self, text: str, reading: str = "") -> list[tuple[str, str, str]]:
        """Return [(key, display word, reading)] for a vocabulary field."""
        text = strip_field(text)
        reading = strip_field(reading)
        if not text or not has_japanese(text):
            return []
        r = self.parser.resolver.candidates([(text, "lemma")], reading, None)
        if r and r[0].score >= 3.0:
            top = r[0]
            rd = reading or (top.entry.reading if is_kana(top.entry.reading) else "")
            return [(f"j{self.parser.db.canonical_seq(top.seq)}", text, kata_to_hira(rd))]
        # Not a single JMdict spelling: parse it (phrases, odd spellings).
        res = self.parser.parse(text, self.token_filter, clean=False)
        return [(w.key, w.headword or w.lemma, w.reading) for w in res.counted]

    def phrase_parts(self, text: str, reading: str = "") -> list[tuple[str, str, str]]:
        """Content words inside a vocab phrase: 先に → 先, この先 → 先, 代わりに → 代わり.

        Only when the phrase splits into several words whose readings add up to the
        card's reading, so a one-word card (方【かた】) never adds a homograph (方【ほう】)."""
        text = strip_field(text)
        reading = kata_to_hira(strip_field(reading))
        if not text or not has_japanese(text):
            return []
        res = self.parser.parse(text, self.token_filter, clean=False)
        units = [w for w in res.words if w.filter not in ("symbol", "whitespace")]
        content = lambda w: w.counted  # noqa: E731
        if len(units) == 1 and units[0].kind == "expression" and units[0].filter != "grammar":
            units = units[0].parts  # 失礼します → 失礼 + し + ます (parts carry no filters)
            content = lambda w: w.pos0 not in ("助詞", "助動詞", "接頭辞", "接尾辞")  # noqa: E731
        if len(units) < 2:
            return []
        if reading:
            got = "".join(kata_to_hira("".join(m.reading for m in w.morphs)) for w in units)
            if got != reading:
                return []
        return [(w.key, w.headword or w.lemma, w.reading) for w in units
                if content(w) and w.key.startswith("j") and w.lemma not in ("する", "為る")]

    def sentence_words(self, text: str) -> list[tuple[str, str, str]]:
        res = self.parser.parse(text, self.token_filter)
        return [(w.key, w.headword or w.lemma, w.reading) for w in res.counted]

    # ------------------------------------------------------------------ sources
    def _queries(self, src: dict) -> tuple[str, str]:
        base = f"note:{quote(src['note_type'])}"
        if src.get("query"):
            base += f" ({src['query']})"
        days = int(self.kc.get("known_interval_days", 21))
        known = f"prop:ivl>={days}"
        stab = float(self.kc.get("known_stability_days") or 0)
        if stab > 0:
            known = f"({known} OR prop:s>={stab:g})"
        return f"{base} -is:new", f"{base} {known}"

    def from_anki(self, ks: KnownSet) -> None:
        sources = self.kc.get("sources") or []
        if not sources:
            self.log("warning: no [[known.sources]] in config; known words come only from files")
            return
        assert self.anki is not None
        queries = []
        for src in sources:
            queries.extend(self._queries(src))
        results = self.anki.find_notes_many(queries)
        for i, src in enumerate(sources):
            studied, known = results[2 * i], set(results[2 * i + 1])
            name = f"{src['note_type']}:{src['field']}"
            stats = {"notes": len(studied), "known_notes": len(known), "words": 0}
            infos = self.anki.notes_info(studied)
            kind = src.get("kind", "vocab")
            for info in infos:
                fields = info["fields"]
                if src["field"] not in fields:
                    raise ValueError(f"note type {src['note_type']!r} has no field {src['field']!r}")
                level = "known" if info["noteId"] in known else "learning"
                text = fields[src["field"]]["value"]
                if kind == "vocab":
                    rf = src.get("reading_field")
                    reading = fields[rf]["value"] if rf and rf in fields else ""
                    words = self.resolve_vocab(text, reading)
                    if not words:
                        if has_japanese(strip_field(text)):
                            ks.unresolved.append((name, strip_field(text)))
                        continue
                    if words[0][0].startswith("~"):
                        ks.unresolved.append((name, strip_field(text)))
                    if "phrase" in ks.derivations and len(words) == 1:
                        own = {words[0][0]}
                        for key, word, rd in self.phrase_parts(text, reading):
                            if key not in own:
                                ks.add(key, word, rd, level, f"{name} (in {strip_field(text)})")
                else:
                    words = self.sentence_words(text)
                for key, word, rd in words:
                    ks.add(key, word, rd, level, name)
                    stats["words"] += 1
            ks.stats[name] = stats
            self.log(f"  {name}: {stats['notes']} studied notes ({stats['known_notes']} known) → {stats['words']} words")
        self.parser.flush()

    def from_file(self, ks: KnownSet, path: Path, source: str, extra: Iterable[str] = ()) -> None:
        """A known-word list, each line looked up in the dictionary. ``extra``: words to treat
        as if they were in the file (mark-deck words not written yet)."""
        if not path.exists() and not extra:
            return
        n = 0
        text = path.read_text(encoding="utf-8-sig") if path.exists() else ""
        rows = list(_rows(text)) + [[w] for w in extra]
        for row in rows:
            word = row[0].strip()
            if not word or word.startswith("#") or not has_japanese(word):
                continue
            if len(word) == 1 and is_kana(word):
                continue  # か, て: particles, and as words ambiguous (蚊, 手); use an exact file
            reading = row[1].strip() if len(row) > 1 and is_kana(row[1].strip()) else ""
            words = self.resolve_vocab(word, reading)
            for key, w, rd in words:
                ks.add(key, w, rd, "known", source)
                n += 1
            if "phrase" in ks.derivations and len(words) == 1:
                for key, w, rd in self.phrase_parts(word, reading):
                    if key != words[0][0]:
                        ks.add(key, w, rd, "known", f"{source} (in {word})")
        ks.stats[source] = {"words": n}
        self.log(f"  {source}: {n} words")

    def build(self, use_anki: bool = True) -> KnownSet:
        from .loanwords import LoanwordRule

        ks = KnownSet(derivations=tuple(self.kc.get("derivations", DERIVATIONS)),
                      loanwords=LoanwordRule.from_config(self.cfg),
                      auto_known_rank=float(self.kc.get("auto_known_rank") or 0))
        unknown = set(ks.derivations) - set(DERIVATIONS)
        if unknown:
            raise ValueError(f"[known] derivations: unknown {sorted(unknown)}; choose from {list(DERIVATIONS)}")
        from . import mark
        from .config import as_list
        from .filters import load_wordlist

        if use_anki:
            self.from_anki(ks)
            if mark.deck_names(self.cfg):
                ks.marked = mark.read(self.cfg, self.anki, log=self.log)
        pend = mark.pending(ks.marked)
        for f in as_list(self.kc.get("files")):
            path = self.cfg.resolve(f)
            self.from_file(ks, path, f"file:{f}", extra=sorted(mark.pending_for(pend, path)))
        for f in as_list(self.kc.get("exact_files")):
            path = self.cfg.resolve(f)
            words = load_wordlist(path) | mark.pending_for(pend, path)
            ks.exact |= words
            ks.stats[f"exact:{f}"] = {"words": len(words)}
            self.log(f"  exact:{f}: {len(words)} words (exact text match)")
        for f in as_list(self.cfg["filters"].get("ignore_files")):
            ks.ignored |= mark.pending_for(pend, self.cfg.resolve(f))
        return ks


def _rows(text: str) -> Iterable[list[str]]:
    sample = text[:2000]
    delim = "\t" if "\t" in sample else ("," if "," in sample else None)
    if delim is None:
        for line in text.splitlines():
            yield [line]
        return
    yield from csv.reader(io.StringIO(text), delimiter=delim)


def export(ks: KnownSet, path: Path, levels: Iterable[str] = ("known", "learning")) -> int:
    levels = set(levels)
    rows = sorted((w for w in ks.words.values() if w.level in levels), key=lambda w: (-LEVELS[w.level], w.key))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("# word\treading\tlevel\tkey\tsources\n")
        for w in rows:
            fh.write(f"{w.word}\t{w.reading}\t{w.level}\t{w.key}\t{','.join(sorted(w.sources))}\n")
    return len(rows)


# ---------------------------------------------------------------------------- derived forms

_ABLE = re.compile(r"\bable to\b|^to be \w+able\b|\bcan\b")
_E_TO_U = {"え": "う", "け": "く", "げ": "ぐ", "せ": "す", "て": "つ", "ね": "ぬ", "べ": "ぶ", "め": "む", "れ": "る"}


def derived_bases(word, db) -> list[str]:
    """Keys of words this one is a predictable derivation of.

    Potential verbs: a 下一段 verb whose godan base exists and whose JMdict senses say
    "able to" (使える ← 使う, 読める ← 読む), or "to seem" for the spontaneous form
    (思える ← 思う), or whose entry is rare while the base is common (聞ける ← 聞く).
    分かれる is common and has no "able to" sense, so it is not a form of 分かる.
    """
    if word.pos0 != "動詞" or not word.morphs or word.kind != "token":
        return []
    m = word.morphs[0]
    lemma = m.lemma
    if len(m.pos) < 5 or not m.pos[4].startswith("下一段") or len(lemma) < 2 or not lemma.endswith("る"):
        return []
    u = _E_TO_U.get(lemma[-2])
    if not u or word.seq is None:
        return []
    e = db.entry(word.seq)
    if e is None:
        return []
    able = any(_ABLE.search(s.gloss) for s in e.senses) or e.gloss.startswith("to seem")
    base = lemma[:-2] + u
    out = []
    rows = db.forms_by_term(base)
    if not rows and is_kana(base):
        rows = db.forms_by_reading(base)  # もらえる → もらう, written 貰う in JMdict
    for f in rows:
        be = db.entry(f.seq)
        if be is None or not any(t.startswith("v5") for t in be.pos):
            continue
        # Sudachi lemmatizes 聞ければ as 聞ける, whose only JMdict entry is a rare word
        # ("to tell"): a rare 下一段 entry next to a common godan base is the potential.
        if able or (not e.common and be.common):
            out.append(f"j{db.canonical_seq(f.seq)}")
    return list(dict.fromkeys(out))
