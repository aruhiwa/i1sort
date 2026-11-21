"""Configuration: defaults deep-merged with the user's config.toml."""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "data_dir": "data",
    "dictionaries": {
        # Yomitan-format JMdict (sequence numbers = JMdict entry IDs). Jitendex also works.
        "jmdict": "dict/JMdict_english.zip",
        "jmnedict": "dict/JMnedict.zip",
    },
    "frequency": {
        # Yomitan frequency dictionaries. Only rank-based ones (1 = most common) are
        # used; occurrence-based ones are skipped. `mode` overrides the declared mode.
        "dictionaries": [
            {"path": "dict/frequency/jiten_freq_global.zip"},
            {"path": "dict/frequency/JPDB_v2.2_Frequency_Kana_2024-10-13.zip"},
            {"path": "dict/frequency/[JA Freq] YoutubeFreqV3.zip", "mode": "rank-based"},
            {"path": "dict/frequency/Corpus of Everyday Japanese Conversation.zip"},
            {"path": "dict/frequency/Corpus of Spontaneous Japanese - CSJ.zip"},
            {"path": "dict/frequency/NINJAL Web Japanese Corpus - NWJC.zip"},
            {"path": "dict/frequency/BCCWJ_SUW_LUW_combined.zip", "mode": "rank-based"},
        ],
    },
    "parser": {
        "sudachi_dict": "full",  # full, core or small; falls back to core
        "split_mode": "C",
        # Re-segment noun compounds with the rules from japanese-compound-merging.md.
        "compound_resegment": True,
        # Merge consecutive tokens into JMdict expressions (気がする, しょうがない).
        "merge_expressions": True,
        "max_expression_tokens": 6,
        # An expression of several content words (気がする) merges only if its frequency
        # rank is at most this many times its rarest content word's rank. 0 = always merge.
        "expression_max_ratio": 6.0,
    },
    "anki": {
        "url": "http://127.0.0.1:8765",
        "timeout": 120,
        "profile": None,  # refuse to run if Anki's active profile differs
    },
    "known": {
        "sources": [],  # [{note_type, field, kind = "vocab"|"sentence", reading_field?, query?}]
        "known_interval_days": 21,
        "known_stability_days": 0,  # FSRS stability threshold; 0 = interval only
        "learning_counts_as_known": True,
        "files": [],  # known-word lists looked up in the dictionary: word per line or CSV/TSV, word first
        "exact_files": [],  # known words matched only by exact token text (no dictionary lookup)
        # Loanwords from these languages count as known (true = English, false = off): JMdict
        # katakana-only entries whose source is one of them (unspecified = English);
        # not_loanword_files lists katakana words that are Japanese (one word per line).
        "loanwords": True,
        "loanwords_wasei": True,  # also wasei words built from those languages (サラリーマン)
        "not_loanword_files": [],
        # Words whose frequency rank (harmonic mean over [frequency] dictionaries) is <= this
        # count as known. 0 = off.
        "auto_known_rank": 0,
        # derive.py: which relatives of a known word also count as known
        "derivations": ["potential", "affix", "stem", "variant", "parts", "morpheme", "phrase"],
    },
    # Cards you move to these decks have their word appended to a file (mark.py). What the
    # words then mean depends on where the file is listed: [known] files / exact_files,
    # [filters] ignore_files.
    "mark_decks": {
        "known": "",  # deck name; empty = off
        "exact": "",
        "ignored": "",
        # the field each deck's word is read from: the dictionary form for the known list (looked
        # up, so all conjugations count), the form in the sentence for exact and ignored
        "known_field": "i1sort_target",
        "exact_field": "i1sort_cloze_body",
        "ignored_field": "i1sort_cloze_body",
        "known_file": "",  # the file each deck's words are appended to
        "exact_file": "",
        "ignored_file": "",
    },
    # Definitions for the target word, from a Yomitan-format dictionary with JMdict sequence
    # numbers (Jitendex), into the [sort.fields] `definition` field (definitions.py).
    "definitions": {
        "dictionary": "",  # path to the dictionary zip; empty = off
        "compact": False,  # add Yomitan's compact glossary styles (layout "compact-popup-anki")
    },
    "sort": {
        "decks": [],  # decks to sort (analyze/apply default)
        "fields": {},  # note type -> {sentence = field, audio = field}
        "min_words": 2,  # fewer counted words → junk
        "max_tokens": 30,  # more word tokens (particles included) → junk; 0 = no limit
        "one_card_per_word": "all",  # "all": one i+1 card per word across sorted decks; "deck": per deck
        "ideal_length": [6, 14],  # preferred sentence length (word tokens) among same-target candidates
        "require_audio": True,  # empty audio field → junk
        "weight_global_freq": 1.0,
        "use_deck_freq": False,  # true: also weigh how often the word appears in the deck
        "weight_deck_freq": 1.0,
        "rank_floor": 100000,  # global rank at or beyond this scores 0
        "suspend_junk": True,
        "suspend_known": False,
        "tag_prefix": "i1sort",
        "write_fields": True,
    },
    "filters": {
        "particles": True,
        "aux_verbs": True,
        "symbols": True,
        "whitespace": True,
        "grammar": True,  # subsidiary verbs after て, する after a noun, そう/よう/みたい stems
        "interjections": True,
        "names": True,
        "honorifics": True,  # さん, くん, ちゃん, 様 ... after a name or noun
        "numbers": True,
        "affixes": True,  # lone prefixes/suffixes that did not merge into a word
        "non_japanese": True,  # Latin letters, digits, emoji
        "onomatopoeia": False,
        "ignore_list": True,
        "ignore_files": ["ignore.txt"],  # words that never count: exact text, one per line
    },
}


# Settings that were renamed: a config still using the old name stops with this hint.
_RENAMED = {
    ("known", "manual_file"): 'merged into [known] files = ["known.txt", ...]',
    ("known", "import_files"): "merged into [known] files",
    ("filters", "ignore_file"): 'now [filters] ignore_files = ["ignore.txt", ...] (a list)',
    ("mark_decks", "field"): "now one per deck: known_field, exact_field, ignored_field",
    ("filters", "loanwords"): "moved to [known] loanwords (loanwords now count as known, not filtered)",
    ("filters", "loanwords_wasei"): "moved to [known] loanwords_wasei",
    ("filters", "not_loanword_files"): "moved to [known] not_loanword_files",
    ("filters", "auto_known_rank"): "moved to [known] auto_known_rank (those words now count as known, not filtered)",
}


def as_list(v) -> list:
    """A setting that takes one or more values: "a.txt" or ["a.txt", "b.txt"]."""
    if not v:
        return []
    return [v] if isinstance(v, str) else list(v)


def _check_renamed(user: dict, path) -> None:
    old = [f"[{sec}] {key}: {hint}" for (sec, key), hint in _RENAMED.items() if key in (user.get(sec) or {})]
    if old:
        raise ValueError(f"{path}: settings were renamed:\n  " + "\n  ".join(old))


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class Config:
    raw: dict[str, Any]
    base_dir: Path
    path: Path | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def resolve(self, p: str | os.PathLike) -> Path:
        p = Path(os.path.expanduser(str(p)))
        return p if p.is_absolute() else (self.base_dir / p)

    @property
    def data_dir(self) -> Path:
        d = self.resolve(self.raw["data_dir"])
        d.mkdir(parents=True, exist_ok=True)
        return d


def find_config(explicit: str | None = None) -> Path | None:
    if explicit:
        return Path(explicit)
    env = os.environ.get("I1SORT_CONFIG")
    if env:
        return Path(env)
    for cand in (Path.cwd() / "config.toml", Path(__file__).resolve().parent.parent / "config.toml"):
        if cand.exists():
            return cand
    return None


def load_config(path: str | None = None, overrides: dict | None = None) -> Config:
    cfg_path = find_config(path)
    raw = copy.deepcopy(DEFAULTS)
    base_dir = Path(__file__).resolve().parent.parent
    if cfg_path is not None and cfg_path.exists():
        with open(cfg_path, "rb") as fh:
            user = tomllib.load(fh)
        _check_renamed(user, cfg_path)
        raw = _deep_merge(raw, user)
        base_dir = cfg_path.resolve().parent
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config(raw=raw, base_dir=base_dir, path=cfg_path)
