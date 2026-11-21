"""SudachiPy tokenizer backend (checked against sudachipy 0.7.0)."""

from __future__ import annotations

import sys
from importlib import metadata

from .base import Morph

try:
    from sudachipy import Dictionary, SplitMode
except ImportError:  # pragma: no cover
    Dictionary = None  # type: ignore
    SplitMode = None  # type: ignore


def _pkg_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "?"


class SudachiBackend:
    name = "sudachi"

    def __init__(self, dict_name: str = "full", split_mode: str = "C"):
        if Dictionary is None:
            raise RuntimeError("sudachipy is not installed")
        self.dict_name = dict_name
        try:
            self.dictionary = Dictionary(dict=dict_name)
        except Exception as e:  # sudachidict_full missing → core
            if dict_name == "core":
                raise
            print(f"warning: sudachidict_{dict_name} unavailable ({e}); using core", file=sys.stderr)
            self.dict_name = "core"
            self.dictionary = Dictionary(dict="core")
        self.mode_name = split_mode.upper()
        self.mode = getattr(SplitMode, self.mode_name)
        self.tokenizer = self.dictionary.tokenizer(mode=self.mode)
        self._lemma_reading_cache: dict[tuple[int, int], str] = {}

    def version(self) -> str:
        return (
            f"sudachipy={_pkg_version('sudachipy')};"
            f"dict={self.dict_name}@{_pkg_version('sudachidict_' + self.dict_name)};mode={self.mode_name}"
        )

    def _morph(self, m, with_units: bool) -> Morph:
        surface = m.surface()
        lemma = m.dictionary_form()
        if lemma == surface:
            lemma_reading = m.reading_form()
        else:
            key = (m.dictionary_id(), m.word_id())
            lemma_reading = self._lemma_reading_cache.get(key)
            if lemma_reading is None:
                try:
                    lemma_reading = m.dictionary_form_morpheme().reading_form()
                except Exception:
                    lemma_reading = m.reading_form()
                self._lemma_reading_cache[key] = lemma_reading
        out = Morph(
            surface=surface,
            lemma=lemma,
            norm=m.normalized_form(),
            reading=m.reading_form(),
            lemma_reading=lemma_reading,
            pos=tuple(m.part_of_speech()),
            oov=m.is_oov(),
            begin=m.begin(),
            end=m.end(),
        )
        if with_units and not out.oov:
            a = m.split(SplitMode.A)
            if len(a) > 1:
                out.a_units = [self._morph(x, False) for x in a]
                b = m.split(SplitMode.B)
                if len(b) > 1:
                    out.b_units = [self._morph(x, False) for x in b]
        return out

    def tokenize(self, text: str) -> list[Morph]:
        return [self._morph(m, True) for m in self.tokenizer.tokenize(text)]
