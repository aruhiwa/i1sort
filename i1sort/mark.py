"""Mark decks: words you mark while studying, collected into word-list files.

When a new card's target turns out to be a word you already know, or one you never want to
learn, move the card to one of the decks in ``[mark_decks]`` (``known``, ``exact``,
``ignored``). Every run reads those decks and takes the word from each deck's field:
``known_field`` (default ``i1sort_target``, the dictionary form: the known list looks it up,
so every conjugation counts), ``exact_field`` and ``ignored_field`` (default
``i1sort_cloze_body``, the word as written in the sentence: these decks are for words i1sort
matched to the wrong dictionary entry, names and made-up words, so the dictionary form would
be the wrong word). A reading written into the field (炙(あぶ)る) is dropped. ``apply`` appends new words to the deck's file
(``known_file``, ``exact_file``, ``ignored_file``), so they stay after you delete the cards,
and suspends the cards (tagged ``<prefix>::suspended``, so ``restore`` can put them back).

The files only collect words. What the words mean comes from where the file is listed, like
any other list: ``[known] files`` (looked up in the dictionary), ``[known] exact_files`` or
``[filters] ignore_files``. Until ``apply`` writes them, a deck's words are treated as if
they were already in its file, so ``analyze`` plans with them too. A card whose field is
empty (a not-ready card has no target) is left alone and listed: type the word into the
field, or move the card back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .anki import cards_info, find_cards, quote
from .config import Config

KINDS = ("known", "exact", "ignored")


@dataclass
class MarkDeck:
    kind: str  # "known" | "exact" | "ignored"
    deck: str
    file: Path
    word_field: str = "i1sort_target"  # the field the word is read from
    words: dict[str, list[int]] = field(default_factory=dict)  # word → card ids
    cards: list[dict] = field(default_factory=list)  # cardId, note, due, queue, type
    tags: dict[int, list[str]] = field(default_factory=dict)  # note id → tags
    no_word: list[int] = field(default_factory=list)  # card ids with an empty field
    no_field: list[int] = field(default_factory=list)  # card ids whose note type lacks the field
    in_file: set[str] = field(default_factory=set)

    @property
    def new_words(self) -> list[str]:
        return [w for w in self.words if w not in self.in_file]

    @property
    def marked_cards(self) -> list[dict]:
        bad = set(self.no_word) | set(self.no_field)
        return [c for c in self.cards if c["cardId"] not in bad]


def settings(cfg: Config) -> dict:
    return cfg.raw.get("mark_decks") or {}


def file_for(cfg: Config, kind: str) -> Path | None:
    f = settings(cfg).get(f"{kind}_file")
    return cfg.resolve(f) if f else None


def _key(p: Path) -> Path:
    return p.resolve()


def pending(marks: list[MarkDeck]) -> dict[Path, set[str]]:
    """Words in the mark decks that are not in their file yet, by (resolved) file."""
    out: dict[Path, set[str]] = {}
    for m in marks:
        out.setdefault(_key(m.file), set()).update(m.new_words)
    return out


def pending_for(pend: dict[Path, set[str]], path: Path) -> set[str]:
    return pend.get(_key(path), set()) if pend else set()


def listed_as(cfg: Config, path: Path) -> list[str]:
    """The settings that list this file (and so decide what its words mean)."""
    from .config import as_list

    where = []
    for label, files in (("[known] files", cfg["known"].get("files")),
                         ("[known] exact_files", cfg["known"].get("exact_files")),
                         ("[filters] ignore_files", cfg["filters"].get("ignore_files"))):
        if any(_key(cfg.resolve(f)) == _key(path) for f in as_list(files)):
            where.append(label)
    return where


def deck_names(cfg: Config) -> list[str]:
    md = settings(cfg)
    return [md[k] for k in KINDS if md.get(k)]


def exclude_query(cfg: Config) -> str:
    """Search terms that leave the mark decks out of a sorted deck (if they are subdecks)."""
    return "".join(f" -deck:{quote(d)}" for d in deck_names(cfg))


DEFAULT_FIELDS = {"known": "i1sort_target", "exact": "i1sort_cloze_body", "ignored": "i1sort_cloze_body"}
_INLINE_READING = re.compile(r"[(（][ぁ-ゖァ-ヺー・]+[)）]")


def field_for(cfg: Config, kind: str) -> str:
    return settings(cfg).get(f"{kind}_field") or DEFAULT_FIELDS[kind]


def _word(value: str) -> str:
    """The word in a field: no HTML, furigana, inline readings (炙(あぶ)る) or spaces."""
    from .known import strip_field

    return re.sub(r"\s+", "", _INLINE_READING.sub("", strip_field(value)))


def read(cfg: Config, anki, log=print) -> list[MarkDeck]:
    """The configured mark decks with their cards and words (read-only)."""
    from .filters import load_wordlist

    md = settings(cfg)
    out = []
    for kind in KINDS:
        deck = md.get(kind)
        if not deck:
            continue
        path = file_for(cfg, kind)
        if path is None:
            raise ValueError(f"[mark_decks] {kind} = {deck!r} needs a file: set [mark_decks] {kind}_file")
        m = MarkDeck(kind, deck, path, field_for(cfg, kind))
        fld = m.word_field
        m.in_file = load_wordlist(m.file)
        cids = find_cards(anki, f"deck:{quote(deck)}")
        if cids:
            m.cards = [{k: c[k] for k in ("cardId", "note", "due", "queue", "type")} for c in cards_info(anki, cids)]
            notes = {n["noteId"]: n for n in anki.notes_info(sorted({c["note"] for c in m.cards}))}
            for c in m.cards:
                n = notes.get(c["note"], {})
                m.tags[c["note"]] = list(n.get("tags", []))
                f = n.get("fields", {})
                if fld not in f:
                    m.no_field.append(c["cardId"])
                    continue
                w = _word(f[fld]["value"])
                if not w:
                    m.no_word.append(c["cardId"])
                    continue
                m.words.setdefault(w, []).append(c["cardId"])
        log(f"  mark deck {deck} ({kind}): {len(m.cards)} cards, {len(m.words)} words, "
            f"{len(m.new_words)} not yet in {m.file.name}")
        out.append(m)
    return out


def changes(m: MarkDeck, prefix: str):
    """Suspend the deck's marked cards (tracked), whatever their state: you put them there."""
    from .apply import DeckChanges, _tag_set

    track = f"{prefix}::suspended"
    ch = DeckChanges(m.deck)
    for c in m.marked_cards:
        if c["queue"] == -1:
            continue  # already suspended (by you or an earlier run)
        ch.suspend.append(c["cardId"])
        if track.lower() not in _tag_set(m.tags.get(c["note"], []), prefix) and c["note"] not in ch.tag_add[track]:
            ch.tag_add[track].append(c["note"])
    return ch


def summary(marks: list[MarkDeck], cfg: Config) -> str:
    lines = []
    for m in marks:
        fld = m.word_field
        new = m.new_words
        shown = "、".join(new[:12]) + ("…" if len(new) > 12 else "")
        lines.append(f"  {m.deck}: {len(m.words)} words, {len(new)} new for {m.file.name}"
                     + (f": {shown}" if new else ""))
        where = listed_as(cfg, m.file)
        if where:
            lines.append(f"    {m.file.name} is in {', '.join(where)}")
        else:
            lines.append(f"    {m.file.name} is not in [known] files, [known] exact_files or "
                         f"[filters] ignore_files, so its words have no effect")
        if m.no_word:
            lines.append(f"    {len(m.no_word)} card(s) with an empty {fld} field are left alone "
                         f"(type the word into the field, or move the card back)")
        if m.no_field:
            lines.append(f"    {len(m.no_field)} card(s) whose note type has no {fld} field are left alone")
    return "\n".join(lines)


def append_words(m: MarkDeck) -> list[str]:
    """Add the deck's new words to its file, one per line. Returns the words added."""
    new = m.new_words
    if not new:
        return []
    m.file.parent.mkdir(parents=True, exist_ok=True)
    text = m.file.read_text(encoding="utf-8") if m.file.exists() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    m.file.write_text(text + "".join(w + "\n" for w in new), encoding="utf-8")
    m.in_file |= set(new)
    return new


def remove_words(path: Path, words: list[str]) -> int:
    """Undo append_words: drop lines that are exactly one of these words."""
    if not path.exists() or not words:
        return 0
    drop = set(words)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [ln for ln in lines if ln.strip() not in drop]
    path.write_text("".join(kept), encoding="utf-8")
    return len(lines) - len(kept)
