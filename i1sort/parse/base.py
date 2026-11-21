"""Core data types shared by tokenizer backends and the parsing pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

POS = tuple[str, ...]


@dataclass
class Morph:
    """One tokenizer unit (a Sudachi morpheme)."""

    surface: str
    lemma: str  # dictionary form
    norm: str  # normalized form (Sudachi: 可愛い for かわいい/カワイイ)
    reading: str  # reading of the surface, katakana
    lemma_reading: str  # reading of the dictionary form, katakana
    pos: POS  # full part-of-speech tuple
    oov: bool = False
    begin: int = 0
    end: int = 0
    # Finer units this morph splits into (Sudachi A/B units), used by compound re-segmentation.
    a_units: list["Morph"] | None = None
    b_units: list["Morph"] | None = None

    @property
    def inflected(self) -> bool:
        return self.surface != self.lemma


@dataclass
class Word:
    """A word unit after compound/expression merging: what gets a word key."""

    surface: str
    lemma: str  # dictionary form of the whole unit
    reading: str  # hiragana reading of the lemma
    pos: POS
    morphs: list[Morph]
    kind: str = "token"  # token | compound | expression
    seq: int | None = None  # JMdict entry ID
    key: str = ""  # j<seq> or ~<normalized>|<pos>
    headword: str = ""  # JMdict headword of the resolved entry
    gloss: str = ""
    filter: str | None = None  # reason code if this word never counts as unknown
    filter_detail: str = ""
    note: str = ""  # how the unit was formed / resolved
    # The words a merged expression was built from (言う通り = 言う + 通り). A learner who
    # knows every content part knows the expression (known.py, "parts" derivation).
    parts: list["Word"] = field(default_factory=list)
    # "prefix"/"suffix": a short kanji noun glued to a neighbouring noun with no particle
    # (語 in 異世界語, 初 in 初事件): a piece of a compound JMdict does not list. Set by filters.
    bound: str = ""

    @property
    def pos0(self) -> str:
        return self.pos[0] if self.pos else ""

    @property
    def counted(self) -> bool:
        return self.filter is None

    def to_json(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("morphs", "parts")}
        d["pos"] = list(self.pos)
        d["morphs"] = [
            {k: v for k, v in asdict(m).items() if k not in ("a_units", "b_units")} for m in self.morphs
        ]
        d["parts"] = [p.to_json() for p in self.parts]
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Word":
        d = dict(d)
        d["pos"] = tuple(d["pos"])
        d["morphs"] = [Morph(**{**m, "pos": tuple(m["pos"])}) for m in d["morphs"]]
        d["parts"] = [cls.from_json(p) for p in d.get("parts") or []]
        return cls(**d)


@dataclass
class ParseResult:
    text: str  # original
    cleaned: str  # after subtitle cleanup
    words: list[Word] = field(default_factory=list)
    has_music: bool = False  # ♪ in the line (song lyrics)
    removed: list[str] = field(default_factory=list)  # speaker labels, annotations, readings
    annotated: list[str] = field(default_factory=list)  # kanji given an inline reading (names)

    @property
    def counted(self) -> list[Word]:
        return [w for w in self.words if w.filter is None]


class Backend(Protocol):
    name: str

    def version(self) -> str: ...

    def tokenize(self, text: str) -> list[Morph]: ...
