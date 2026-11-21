"""Small Japanese text helpers: kana conversion, script tests, reading comparison."""

from __future__ import annotations

import re
import unicodedata

_KATA_START, _KATA_END = 0x30A1, 0x30F6  # ァ..ヶ
_HIRA_OFFSET = 0x60


def kata_to_hira(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if _KATA_START <= o <= _KATA_END:
            out.append(chr(o - _HIRA_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


def hira_to_kata(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if _KATA_START - _HIRA_OFFSET <= o <= _KATA_END - _HIRA_OFFSET:
            out.append(chr(o + _HIRA_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


def is_hiragana_char(ch: str) -> bool:
    return "ぁ" <= ch <= "ゟ"


def is_katakana_char(ch: str) -> bool:
    return "゠" <= ch <= "ヿ" or "ㇰ" <= ch <= "ㇿ" or "ｦ" <= ch <= "ﾟ"


def is_kana_char(ch: str) -> bool:
    return is_hiragana_char(ch) or is_katakana_char(ch) or ch == "ー"


def is_kanji_char(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF
        or 0x3400 <= o <= 0x4DBF
        or 0x20000 <= o <= 0x2FFFF
        or 0xF900 <= o <= 0xFAFF
        or ch in "々〆ヶ"
    )


def is_kana(s: str) -> bool:
    return bool(s) and all(is_kana_char(c) for c in s)


def is_katakana(s: str) -> bool:
    return bool(s) and all(is_katakana_char(c) or c == "ー" for c in s)


def is_hiragana(s: str) -> bool:
    return bool(s) and all(is_hiragana_char(c) or c == "ー" for c in s)


def has_kanji(s: str) -> bool:
    return any(is_kanji_char(c) for c in s)


def is_single_kanji(s: str) -> bool:
    return len(s) == 1 and is_kanji_char(s)


def is_japanese_char(ch: str) -> bool:
    return is_kana_char(ch) or is_kanji_char(ch)


def has_japanese(s: str) -> bool:
    return any(is_japanese_char(c) for c in s)


_LATIN_RE = re.compile(r"^[A-Za-zＡ-Ｚａ-ｚ0-9０-９'’.\-]+$")


def is_latin(s: str) -> bool:
    return bool(_LATIN_RE.match(s))


_VOWEL_OF = {}
for _row, _v in (
    ("あかさたなはまやらわがざだばぱぁゃゎ", "あ"),
    ("いきしちにひみりぎじぢびぴぃ", "い"),
    ("うくすつぬふむゆるぐずづぶぷぅゅっ", "う"),
    ("えけせてねへめれげぜでべぺぇ", "え"),
    ("おこそとのほもよろをごぞどぼぽぉょ", "お"),
):
    for _c in _row:
        _VOWEL_OF[_c] = _v


def vowel_of(ch: str) -> str | None:
    return _VOWEL_OF.get(kata_to_hira(ch))


def strip_voicing(s: str) -> str:
    d = unicodedata.normalize("NFD", s)
    d = d.replace("゙", "").replace("゚", "")
    return unicodedata.normalize("NFC", d)


def readings_match(a: str, b: str, *, ignore_voicing: bool = True) -> bool:
    """Compare two readings as described in japanese-compound-merging.md §6.3.

    Both are converted to hiragana, づ→ず, ぢ→じ, を→お, voicing marks are
    stripped (rendaku is not a sign of a single word), and ー matches any vowel.
    """

    def prep(s: str) -> str:
        s = kata_to_hira(s).replace("づ", "ず").replace("ぢ", "じ").replace("を", "お")
        return strip_voicing(s) if ignore_voicing else s

    a, b = prep(a), prep(b)
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        if x == "ー" and y in "あいうえお":
            continue
        if y == "ー" and x in "あいうえお":
            continue
        return False
    return True


def reading_key(s: str) -> str:
    """Normalize a reading for index lookups: hiragana, no long-vowel/voicing changes."""
    return kata_to_hira(s)


_FULLWIDTH_TRANS = {i: i - 0xFEE0 for i in range(0xFF01, 0xFF5F)}


def to_halfwidth_ascii(s: str) -> str:
    return s.translate(_FULLWIDTH_TRANS)


# ---------------------------------------------------------------------- furigana
_INLINE_READING = re.compile(r"([^\s\[\]()（）]+?)[(（]([ぁ-ゖァ-ヺー]+)[)）]")


def furigana(text: str, reading: str) -> str:
    """Anki furigana for ``text`` read as ``reading``: 食べ物 + たべもの → 食[た]べ 物[もの].

    Kana in the text is matched against the reading so each kanji run gets only its own
    part; when that fails (irregular readings like 今日 are fine, mismatches are not) the
    whole text gets the whole reading. Text without kanji is returned as is. An inline
    reading already in the text (炙(あぶ)る) is used as given."""
    if not text:
        return ""
    if _INLINE_READING.search(text):
        out = _INLINE_READING.sub(lambda m: f" {m.group(1)}[{m.group(2)}]", text).lstrip()
        if not has_kanji(_strip_ruby(out)):
            return out
        text = _INLINE_READING.sub(r"\1", text)
    if not has_kanji(text) or not reading:
        return text
    reading = kata_to_hira(reading)
    # runs: kanji (or other non-kana: digits, letters) vs kana
    runs: list[tuple[bool, str]] = []
    for ch in text:
        k = not is_kana_char(ch)
        if runs and runs[-1][0] == k:
            runs[-1] = (k, runs[-1][1] + ch)
        else:
            runs.append((k, ch))
    pat = "".join("(.+?)" if k else f"({re.escape(kata_to_hira(s))})" for k, s in runs)
    m = re.fullmatch(pat, reading)
    if not m:
        return f"{text}[{reading}]"
    out = []
    for i, (k, s) in enumerate(runs):
        if k and m.group(i + 1) != kata_to_hira(s):
            out.append(f"{' ' if out else ''}{s}[{m.group(i + 1)}]")
        else:
            out.append(s)
    return "".join(out)


def _strip_ruby(s: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", s)
