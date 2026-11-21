"""Plan a deck's new cards: decide, pick one card per new word, order by word priority.

For every new card (decisions are per note; all its new cards share them):

  junk         sentence filters (empty, music, foreign, no_audio, no_words,
               too_short, too_long) or a duplicate of a better card
  known        U = 0 unknown words
  i1           U = 1, the best card for its target word
  same-target  U = 1, but another card teaches the same word better
  not-ready    U >= 2

U counts distinct counted words whose level is unknown (learning counts as known
unless learning_counts_as_known = false).

Order of new positions: i1 by target priority (freq.py), then same-target, then
not-ready by U, then known, then junk. Positions start at the deck's current
lowest new-card position, so other decks' cards keep their places, and advance one
per note (Anki gives a note's new cards the same position).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .config import Config
from .filters import TokenFilter, detect_deck_names, load_ignore_set, sentence_junk
from .freq import WordPriority, deck_counts
from .known import KnownSet
from .parse.base import ParseResult, Word
from .furigana import word_readings
from .textutil import has_kanji, is_kana, kata_to_hira

DECISIONS = ("i1", "same-target", "not-ready", "known", "junk")


@dataclass
class CardPlan:
    note_id: int
    card_ids: list[int]
    deck: str
    model: str
    old_due: int
    suspended: bool
    tags: list[str]
    res: ParseResult
    has_audio: bool = True
    cards: list[dict] = field(default_factory=list)  # this note's new cards in the deck: cardId, due, queue
    note_cards: list[int] = field(default_factory=list)  # all of the note's cards (any deck, any state)
    fields: dict[str, str] = field(default_factory=dict)  # current values of the note's i1sort_* (and definition) fields
    definition_field: str = ""  # [sort.fields] definition: the field i1sort fills with the target's definition
    frequency_field: str = ""  # [sort.fields] frequency: the field for the target's harmonic frequency rank
    target_frequency: str = ""  # that rank (floor of the harmonic mean over [frequency] dictionaries)
    unknown: list[Word] = field(default_factory=list)
    derived: list[tuple[Word, str]] = field(default_factory=list)  # covered via a known relative
    decision: str = ""
    reason: str = ""
    target: Word | None = None
    target_reading: str = ""  # kana reading of the target's dictionary form (i1sort_target)
    readings: dict[int, str | None] | None = None  # id(word) → reading in the line (furigana fields)
    priority: float = 0.0
    quality: float = 0.0
    length: int = 0
    new_due: int | None = None

    @property
    def u(self) -> int:
        return len(self.unknown)


@dataclass
class DeckPlan:
    deck: str
    cards: list[CardPlan]
    names: dict[str, str]
    priority: WordPriority

    def counts(self) -> Counter:
        return Counter(c.decision for c in self.cards)

    def junk_counts(self) -> Counter:
        return Counter(c.reason.split(":")[0] for c in self.cards if c.decision == "junk")


def _dup_key(res: ParseResult) -> str:
    return re.sub(r"[\s、。！？!?…・―—\-~〜ーっッ「」『』\"'“”.,]", "", res.cleaned)


def _word_tokens(res: ParseResult) -> int:
    return sum(1 for w in res.words if w.filter not in ("symbol", "whitespace"))


class Planner:
    def __init__(self, cfg: Config, parser, known: KnownSet):
        self.cfg = cfg
        self.sc = cfg["sort"]
        self.parser = parser
        self.known = known
        self.learning_ok = bool(cfg["known"].get("learning_counts_as_known", True))

    def fields_for(self, model: str) -> dict:
        f = (self.sc.get("fields") or {}).get(model)
        if not f:
            raise ValueError(f"no [sort.fields] entry for note type {model!r}: add its sentence/audio fields to config.toml")
        return f

    def plan_deck(self, deck: str, cards: list[dict], notes: dict[int, dict]) -> DeckPlan:
        """``cards``: cardsInfo results for the deck's new cards; ``notes``: notesInfo by note id."""
        by_note: dict[int, list[dict]] = defaultdict(list)
        for c in cards:
            by_note[c["note"]].append(c)

        # 1. parse every note's sentence (no filters yet: deck names come first)
        plans: list[CardPlan] = []
        for nid, cs in by_note.items():
            note = notes[nid]
            f = self.fields_for(note["modelName"])
            text = note["fields"][f["sentence"]]["value"]
            audio = note["fields"].get(f.get("audio", ""), {}).get("value", "") if f.get("audio") else "x"
            res = self.parser.parse(text, None)
            cs.sort(key=lambda c: c["due"])
            plans.append(CardPlan(
                note_id=nid, card_ids=[c["cardId"] for c in cs], deck=deck, model=note["modelName"],
                old_due=min(c["due"] for c in cs), suspended=all(c["queue"] == -1 for c in cs),
                tags=note.get("tags", []), res=res, has_audio=bool(audio.strip()),
                cards=[{"cardId": c["cardId"], "due": c["due"], "queue": c["queue"]} for c in cs],
                note_cards=list(note.get("cards") or [c["cardId"] for c in cs]),
                fields={k: v["value"] for k, v in note["fields"].items()
                        if k.startswith("i1sort_") or k in (f.get("definition"), f.get("frequency"))},
                definition_field=f.get("definition") or "",
                frequency_field=f.get("frequency") or "",
            ))
        self.parser.flush()
        plans.sort(key=lambda p: p.old_due)

        # 2. per-deck names, then token filters
        names = detect_deck_names([p.res for p in plans], self.parser.db)
        ignore = (load_ignore_set(self.cfg) | self.known.ignored) if self.known.ignored else None
        tf = TokenFilter(self.cfg, self.parser.db, deck_names=names, ignore=ignore)
        for p in plans:
            tf.apply(p.res.words, annotated=p.res.annotated)

        # 3. word priority from global rank + in-deck counts
        counts = deck_counts([w.key for w in p.res.counted] for p in plans)
        w_deck = float(self.sc.get("weight_deck_freq", 1.0)) if self.sc.get("use_deck_freq", True) else 0.0
        prio = WordPriority(self.parser.db, counts, float(self.sc.get("weight_global_freq", 1.0)),
                            w_deck, float(self.sc.get("rank_floor", 100000)))

        # 4. sentence filters, duplicates, unknown words
        lo, hi = self.sc.get("ideal_length", [6, 14])
        seen_dup: dict[str, CardPlan] = {}
        for p in plans:
            p.length = _word_tokens(p.res)
            junk = sentence_junk(p.res, self.cfg, p.has_audio)
            max_tokens = int(self.sc.get("max_tokens", 0) or 0)
            if not junk and max_tokens and p.length > max_tokens:
                junk = ("too_long", f"{p.length} tokens > {max_tokens}")
            if junk:
                p.decision, p.reason = "junk", f"{junk[0]}: {junk[1]}"
                continue
            dk = _dup_key(p.res)
            first = seen_dup.get(dk)
            if first is not None:
                p.decision, p.reason = "junk", f"duplicate: same line as card at position {first.old_due}"
                continue
            seen_dup[dk] = p
            uniq: dict[str, Word] = {}
            for w in p.res.counted:
                why = self.known.why(w, self.parser.db, self.learning_ok)
                if why is None:
                    uniq.setdefault(w.key, w)
                elif why not in ("known", "learning"):
                    p.derived.append((w, why))
            p.unknown = list(uniq.values())
            n_known = sum(1 for w in p.res.counted if self.known.level(w.key) == "known")
            # known loanwords and very frequent words count as known here too
            n_known += sum(1 for _, why in p.derived if " loanword (" in why or why.startswith("frequent (rank"))
            n_counted = len(p.res.counted)
            dist = max(0, lo - p.length, p.length - hi)
            p.quality = round(-dist + 2.0 * n_known / max(1, n_counted), 3)
            if p.u == 0:
                p.decision = "known"
            elif p.u == 1:
                p.target = p.unknown[0]
                p.priority = round(prio.priority(p.target.key), 4)
                p.decision = "i1"
            else:
                p.decision = "not-ready"
                p.priority = round(max(prio.priority(w.key) for w in p.unknown), 4)
                p.reason = f"{p.u} unknown"

        # 5. one card per target word: best quality, then earliest
        by_target: dict[str, list[CardPlan]] = defaultdict(list)
        for p in plans:
            if p.decision == "i1":
                by_target[p.target.key].append(p)
        for key, group in by_target.items():
            group.sort(key=lambda p: (-p.quality, p.old_due))
            for other in group[1:]:
                other.decision = "same-target"
                other.reason = f"target taught by card at position {group[0].old_due}"

        for p in plans:
            if p.target is not None:
                p.target_reading = self.target_reading(p.target)
                p.target_frequency = self.target_frequency(p.target, p.target_reading, prio)
            p.readings = word_readings(p.res.words, fix=self._common_reading)

        # 6. new positions
        return DeckPlan(deck, order_positions(plans), names, prio)

    def _common_reading(self, w: Word, reading: str) -> str:
        """Sudachi's reading of a word written as its JMdict spelling, unless JMdict marks
        another reading of that spelling common and this one not (転生: てんせい)."""
        if w.seq is None or not has_kanji(w.surface):
            return reading
        rows = [r for r in self.parser.db.forms_of_entry(w.seq) if r.term == w.surface]
        same = [r for r in rows if kata_to_hira(r.reading) == kata_to_hira(reading)]
        if not rows or (same and same[0].common) or not any(r.common for r in rows):
            return reading
        return max(rows, key=lambda r: (r.common, not r.rare, r.score)).reading

    def target_frequency(self, w: Word, reading: str, prio: WordPriority) -> str:
        """Harmonic frequency rank of the target as i1sort_target shows it (headword + reading),
        like Yomitan's {frequency-harmonic-rank}; else the entry's best rank; "" if unranked."""
        import math

        r = self.parser.db.freq_rank(w.headword or w.lemma, reading) if reading else None
        if r is None:
            r = prio.global_rank(w.key)
        return str(math.floor(r)) if r else ""

    def target_reading(self, w: Word) -> str:
        """The reading of the word as i1sort_target shows it (JMdict headword, else lemma): the
        headword's JMdict reading that agrees with the tokenizer's when JMdict marks it common
        (今日: きょう or こんにち), else the headword's most common reading (転生 is てんせい,
        not Sudachi's てんしょう)."""
        word = w.headword or w.lemma
        if is_kana(word):
            return word
        if w.seq is not None:
            rows = [r for r in self.parser.db.forms_of_entry(w.seq) if r.term == word]
            same = [r for r in rows if kata_to_hira(r.reading) == kata_to_hira(w.reading)]
            if same and (same[0].common or not any(r.common for r in rows)):
                return same[0].reading
            if rows:
                return max(rows, key=lambda r: (r.common, not r.rare, r.score)).reading
        return w.reading

    def across_decks(self, decks: list[DeckPlan]) -> None:
        """One i+1 card per word across all decks (``one_card_per_word = "all"``): the best
        sentence keeps it, the other decks' cards for that word become same-target."""
        if str(self.sc.get("one_card_per_word", "all")) != "all" or len(decks) < 2:
            return
        by_target: dict[str, list[tuple[int, CardPlan]]] = defaultdict(list)
        for k, dp in enumerate(decks):
            for p in dp.cards:
                if p.decision == "i1":
                    by_target[p.target.key].append((k, p))
        changed = set()
        for group in by_target.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda kp: (-kp[1].quality, kp[0], kp[1].old_due))
            best_deck = decks[group[0][0]].deck
            for k, p in group[1:]:
                p.decision = "same-target"
                p.reason = f"target taught in {best_deck}"
                changed.add(k)
        for k in changed:
            decks[k].cards = order_positions(decks[k].cards)


def order_positions(plans: list[CardPlan]) -> list[CardPlan]:
    """Sort a deck's cards into plan order and number them from its lowest position."""
    rank = {d: i for i, d in enumerate(DECISIONS)}
    order = sorted(plans, key=lambda p: (
        rank[p.decision],
        -p.priority if p.decision in ("i1", "same-target") else (p.u if p.decision == "not-ready" else 0),
        -p.priority if p.decision == "not-ready" else 0,
        p.old_due,
    ))
    # Anki repositions per note (siblings share a position), so one step per note
    pos = min((p.old_due for p in plans), default=0)
    for p in order:
        p.new_due = pos
        pos += 1
    return order
