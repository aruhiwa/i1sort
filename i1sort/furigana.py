"""Furigana for the sentence fields: a reading for every parsed word, put on the raw field.

Readings are Sudachi's reading of each piece of the word as written, with i1sort's own
fixes (表 is おもて) and corrections for Sudachi's systematic misreadings in speech:

* 私 is わたし (Sudachi: わたくし)
* 何 is なん before だ/で/です/の/と/ど/な/て/じゃ or a counter, otherwise なに (Sudachi: always なん)
* 言う is いう (Sudachi writes the pronunciation ゆう)
* いつの間に is いつのまに (Sudachi: あいだ)

Words with digits get no furigana (30万 would be read digit by digit).

The field keeps its HTML and spacing: each word with kanji is found in the raw text in
order and replaced by Anki furigana syntax (``私[わたし]は 毎日[まいにち]``), with a space
before it so Anki's ``furigana:`` filter knows where the base text starts.
"""

from __future__ import annotations

import re

from .textutil import furigana, has_kanji, is_kana_char, kata_to_hira

_NAN_NEXT = ("だ", "で", "と", "ど", "の", "な", "て", "じゃ", "ん")


def _nan(nxt) -> bool:
    """何 is なん before these (何だ, 何で, 何の, 何と, 何なの, 何て, 何人)."""
    if nxt is None:
        return False
    if len(nxt.pos) > 2 and nxt.pos[2] == "助数詞" and nxt.pos[0] == "接尾辞":
        return True
    return kata_to_hira(nxt.surface).startswith(_NAN_NEXT)


def _morph_reading(w, m, prev2, prev, nxt, fix) -> str | None:
    if any(c.isdigit() for c in m.surface):
        return None
    r = m.reading or m.surface
    if len(w.morphs) == 1 and m.surface == m.lemma and getattr(w, "reading", ""):
        r = w.reading  # i1sort's reading fixes (表 おもて, 金 かね)
        if fix is not None:
            r = fix(w, r)  # JMdict's common reading (転生 てんせい, not てんしょう)
    r = kata_to_hira(r)
    if m.surface == "私":
        return "わたし"
    if m.surface == "何":
        return "なん" if _nan(nxt) else "なに"
    if m.surface.startswith("言") and r.startswith("ゆ") and (m.lemma == "言う" or m.norm == "言う"):
        return "い" + r[1:]
    if m.surface == "間" and prev is not None and prev.surface == "の" and prev2 is not None and prev2.surface == "いつ":
        return "ま"
    return r


def word_readings(words, fix=None) -> dict[int, str | None]:
    """id(word) → kana reading of the word as written in the line (None: no furigana).
    ``fix(word, reading)`` may correct the reading of an uninflected single-piece word."""
    flat = [(i, m) for i, w in enumerate(words) for m in (getattr(w, "morphs", None) or [])]
    per: dict[int, list[str | None]] = {}
    for k, (i, m) in enumerate(flat):
        prev2 = flat[k - 2][1] if k >= 2 else None
        prev = flat[k - 1][1] if k >= 1 else None
        nxt = flat[k + 1][1] if k + 1 < len(flat) else None
        per.setdefault(i, []).append(_morph_reading(words[i], m, prev2, prev, nxt, fix))
    out: dict[int, str | None] = {}
    for i, w in enumerate(words):
        rs = per.get(i)
        if rs is None:  # no morphs (tests): the word's own reading if it is uninflected
            out[id(w)] = w.reading if getattr(w, "reading", "") and w.surface == w.lemma else None
        else:
            out[id(w)] = None if any(r is None for r in rs) else "".join(rs)
    return out


def surface_pattern(surface: str) -> re.Pattern:
    """The word as written in the field; a kanji may carry an inline reading: 炙(あぶ)る."""
    return re.compile("".join(
        re.escape(c) + (r"(?:[(（][ぁ-ゖァ-ヺー・]+[)）])?" if "㐀" <= c <= "鿿" else "") for c in surface))


def ruby_spans(raw: str, words, readings: dict[int, str | None]) -> list[tuple[int, int, str]]:
    """(start, end, furigana text) for each word with kanji, found in order in the raw field
    (HTML tags and [sound:] references are never matched)."""
    masked = re.sub(r"<[^>]*>|\[sound:[^\]]*\]", lambda m: "\0" * len(m.group(0)), raw)
    cur, out = 0, []
    for w in words:
        if not has_kanji(w.surface):
            continue
        m = surface_pattern(w.surface).search(masked, cur)
        if not m:
            continue  # spelled differently in the field (the cleaner changed it): no furigana
        cur = m.end()
        r = readings.get(id(w))
        if r:
            ruby = furigana(raw[m.start():m.end()], r)
            if "[" in ruby:
                out.append((m.start(), m.end(), ruby))
    return out


def render(raw: str, spans: list[tuple[int, int, str]], start: int = 0, end: int | None = None) -> str:
    """raw[start:end] with the spans inside it replaced by their furigana."""
    end = len(raw) if end is None else end
    parts, pos = [], start
    for s, e, ruby in spans:
        if s < start or e > end:
            continue
        parts.append(raw[pos:s])
        # a space marks where the base text starts (Anki's furigana filter removes it); a word
        # that starts with kana (お 前[まえ]) is its own separator
        parts.append((" " if s > start and not is_kana_char(ruby[0]) else "") + ruby)
        pos = e
    parts.append(raw[pos:end])
    return "".join(parts)


def strip(text: str) -> str:
    """What Anki's ``kanji:`` filter shows: the text without furigana (inverse of render)."""
    return re.sub(r" ?([^ >]+?)\[(.+?)\]", r"\1", text)
