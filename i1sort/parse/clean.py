"""Subtitle-line cleanup applied before parsing.

Removes HTML, Anki furigana, inline readings like 禰豆子(ねずこ), speaker labels
like （カズマ） and bracketed annotations like （笑）, and records what was removed
so the report can show it.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field

_TAG_RE = re.compile(r"<[^>]+>")
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_SOUND_RE = re.compile(r"\[sound:[^\]]*\]")
# Anki furigana: " 漢字[かんじ]" — the space before the kanji is a furigana separator.
_ANKI_FURI_RE = re.compile(r" ?([^\s\[\]]+?)\[([ぁ-んァ-ヶー・]+)\]")
_KANJI = r"[㐀-䶿一-鿿豈-﫿々〆ヶ]"
# Inline reading after kanji: 禰豆子(ねずこ), ダメ神(ダメがみ), 佐藤和真(さとうカズマ)
_INLINE_READING_RE = re.compile(r"(?<=" + _KANJI + r")[(（]([ぁ-んァ-ヶー・ 　]+)[)）]")
# Bracketed annotations / speaker labels. Innermost first, repeated until stable.
_BRACKET_RE = re.compile(r"[（(]([^（()）]*)[)）]|【([^【】]*)】|〔([^〔〕]*)〕|［([^［］]*)］|≪([^≪≫]*)≫|＜([^＜＞]*)＞|\[([^\[\]]*)\]")
# Speaker label written as "名前：" at the start of the line.
_LABEL_COLON_RE = re.compile(r"^\s*([^\s：:「」]{1,12})[：:](?=\S)")
_JAPANESE_RE = re.compile(r"[ぁ-ゖァ-ヺ" + _KANJI[1:-1] + r"]")
_BRACKET_CHARS_RE = re.compile(r"[（()）【】〔〕［］≪≫＜＞]")
_MUSIC_RE =re.compile(r"[♪♫♬♩〜～]+")
_SPACE_RE = re.compile(r"[ 　\t]+")


@dataclass
class Cleaned:
    text: str
    removed: list[str] = field(default_factory=list)
    has_music: bool = False
    annotated: list[str] = field(default_factory=list)  # kanji given an inline reading (names)


def clean_text(raw: str) -> Cleaned:
    removed: list[str] = []
    s = html.unescape(raw or "")
    # halfwidth katakana and punctuation (ﾘﾋﾞｭｰﾏﾝ, ｢｣｡) from broadcast subtitles
    s = re.sub(r"[\uff61-\uff9f]+", lambda m: unicodedata.normalize("NFKC", m.group(0)), s)
    s = _SOUND_RE.sub("", s)
    s = _BR_RE.sub(" ", s)
    s = _TAG_RE.sub("", s)
    s = s.replace(" ", " ")
    if "[" in s and _ANKI_FURI_RE.search(s):
        s = _ANKI_FURI_RE.sub(lambda m: m.group(1), s)

    annotated: list[str] = []

    def drop_reading(m: re.Match) -> str:
        removed.append(m.group(0))
        run = re.search(_KANJI + r"+$", s[: m.start()])
        if run:
            annotated.append(run.group(0))
        return ""

    s = _INLINE_READING_RE.sub(drop_reading, s)

    def drop_bracket(m: re.Match) -> str:
        removed.append(m.group(0))
        return " "

    before, n_removed = s, len(removed)
    for _ in range(4):
        s2 = _BRACKET_RE.sub(drop_bracket, s)
        if s2 == s:
            break
        s = s2
    # Unbalanced opening bracket at the start, e.g. "（佐藤和真 text" after nested removal.
    s = re.sub(r"^\s*[（(][^）)]*$", "", s)
    if not _JAPANESE_RE.search(s) and re.search(r"[ぁ-ゖ]", before):
        # The whole line is bracketed speech: off-screen or background voices (（大丈夫ですか）).
        # Keep it. Labels and captions without hiragana (（キョン）, （銃声）) stay removed.
        del removed[n_removed:]
        s = _BRACKET_CHARS_RE.sub(" ", before)

    # brackets that span several subtitles leave an unmatched half: 《テル：あっ … だ…》
    # 《…》 marks thoughts: keep the words, drop the marks
    s = re.sub(r"[《》]", " ", s)
    s = re.sub(r"^\s*[≪＜]|[≫＞]\s*$", "", s).strip()
    m = _LABEL_COLON_RE.match(s)
    if m and not re.search(r"\d", m.group(1)):
        removed.append(m.group(0))
        s = s[m.end():]

    has_music = bool(re.search(r"[♪♫♬♩]", s))
    s = re.sub(r"[♪♫♬♩]+", " ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return Cleaned(s, removed, has_music, annotated)


_ELONG_SMALL = {"ぁ": "あ", "ぃ": "い", "ぅ": "う", "ぇ": "え", "ぉ": "お"}


_KATA_PARTICLE_RE = re.compile(r"[\u3400-\u9fff々][ヲガハヘデニモト](?![ァ-ヺー])")
_KATA_OKURI_RE = re.compile(r"[\u3400-\u9fff々][ァ-ヺ]{1,3}(?![ァ-ヺー])")


def katakana_speech(s: str) -> str | None:
    """指令ヲ伝エル, 朝ダ！ 起キロ！: speech written with katakana grammar (a robot, a crow).
    Returns the line in hiragana when it shows katakana particles or endings after kanji;
    the caller keeps it only if the tokenizer reads it better (日本ハム, 赤ペン stay)."""
    from ..textutil import kata_to_hira

    if _KATA_PARTICLE_RE.search(s) or len(_KATA_OKURI_RE.findall(s)) >= 2:
        return kata_to_hira(s)
    return None


def normalize_for_tokenizer(s: str) -> str:
    """Light normalization that helps the tokenizer with casual subtitle spelling.

    * small vowels used for elongation after a hiragana with the same vowel
      become ー (すごぉい → すごーい, すげぇ → すげー); the pipeline then picks a spelling
    * repeated ー/〜 runs collapse to one ー
    """
    from ..textutil import is_hiragana_char, vowel_of

    out: list[str] = []
    for ch in s:
        # small vowels after a same-vowel hiragana mark elongation (すごぉい, ねぇ, すげぇ);
        # turn them into ー and let the pipeline pick the spelling the tokenizer likes best
        if ch in _ELONG_SMALL and out and is_hiragana_char(out[-1]) and vowel_of(out[-1]) == _ELONG_SMALL[ch]:
            if out[-1] != "ー":
                out.append("ー")
            continue
        if ch in "ー〜～" and out and out[-1] == "ー":
            continue
        if ch in "〜～" and out and (is_hiragana_char(out[-1]) or "\u3400" <= out[-1] <= "\u9fff"):  # 広～い
            out.append("ー")
            continue
        out.append(ch)
    # のですう, ますぅ: a stretched polite ending (ぅ became ー above)
    s = re.sub(r"(?<=[でま])す[うー]+(?=[\s　、。！？!?…‥」』]|$)", "す", "".join(out))
    # どーしよう, どーした: どう (dropping the ー gives the rare verb 度する)
    s = re.sub(r"(?<![ぁ-ゖ])どー(?=[しすせさだでかもぞ])", "どう", s)
    # ち～っす, ちわっす: a greeting the tokenizer shreds (ち+ー+っす → 地位)
    s = re.sub(r"(?<![ぁ-ゖァ-ヺ\u3400-\u9fff々])(?:ち[ーぃい]*っす|ちわっす)", " ", s)  # not 気持ちっす
    # かっくいー: かっこいい
    s = re.sub(r"(?<![ぁ-ゖ])かっくい(?:ー|い)?", "かっこいい", s)
    # いっつも: いつも
    s = re.sub(r"いっつも", "いつも", s)
    # いっくよー, いっくぞ: an emphatic っ inside いく (otherwise read as 一句)
    return re.sub(r"(?<![ぁ-ゖ])いっく(?=[よぞわね])", "いく", s)
