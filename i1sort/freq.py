"""Word frequency for prioritising new words.

* Global: the average rank (harmonic mean over the configured rank-based Yomitan
  frequency dictionaries, see dictdb.py) of the word's JMdict entry, best over its
  spellings. Lower rank = more common.
* In-deck: in how many of the deck's sentences the word is counted.

priority(word) = w_global * g + w_deck * d, both in [0, 1]:
  g = 1 - log10(rank) / log10(rank_floor)   (rank 1 → 1, rank >= rank_floor → 0, no data → 0)
  d = log(1 + count) / log(1 + max_count_in_deck)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from .dictdb import DictDB


class WordPriority:
    def __init__(self, db: DictDB, deck_counts: Counter, w_global: float = 1.0, w_deck: float = 1.0,
                 rank_floor: float = 100_000):
        self.db = db
        self.deck_counts = deck_counts
        self.w_global = w_global
        self.w_deck = w_deck
        self.rank_floor = rank_floor
        self.max_count = max(deck_counts.values(), default=1)

    def global_rank(self, key: str) -> float | None:
        if not key.startswith("j"):
            return None
        seq = int(key[1:])
        r = self.db.entry_rank(seq)
        if r is None:
            # some lists rank a spelling without its reading (モテる): use the best of those
            ranks = [x for f in self.db.forms_of_entry(seq) if (x := self.db.freq_rank(f.term)) is not None]
            r = min(ranks) if ranks else None
        return r

    def g(self, key: str) -> float:
        r = self.global_rank(key)
        if r is None or r <= 0:
            return 0.0
        return max(0.0, 1.0 - math.log10(max(r, 1.0)) / math.log10(self.rank_floor))

    def d(self, key: str) -> float:
        c = self.deck_counts.get(key, 0)
        return math.log1p(c) / math.log1p(self.max_count) if self.max_count > 0 else 0.0

    def priority(self, key: str) -> float:
        return self.w_global * self.g(key) + self.w_deck * self.d(key)

    def formula(self) -> str:
        if self.w_deck == 0:
            return f"priority = {self.w_global:g} × (1 − log10(rank) / log10({self.rank_floor:g}))"
        return (f"priority = {self.w_global:g} × (1 − log10(rank) / log10({self.rank_floor:g}))"
                f" + {self.w_deck:g} × ln(1 + deck count) / ln(1 + {self.max_count})")


def deck_counts(word_keys_per_sentence: Iterable[Iterable[str]]) -> Counter:
    """Number of sentences each word is counted in (duplicates within a sentence count once)."""
    c: Counter = Counter()
    for keys in word_keys_per_sentence:
        c.update(set(keys))
    return c
