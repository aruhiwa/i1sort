"""Compound gate rules (japanese-compound-merging.md §4) against a mock dictionary."""

from i1sort.parse.base import Morph
from i1sort.parse.compounds import CompoundJudge, Unit, resegment

# spelling → (readings, rank). Ranks loosely follow the doc's examples.
DICT = {
    "顕微鏡": ({"けんびきょう"}, 19855), "顕微": ({"けんび"}, 20606), "鏡": ({"かがみ"}, 2420),
    "電子": ({"でんし"}, 3000), "電子顕微鏡": ({"でんしけんびきょう"}, 60000),
    "感染": ({"かんせん"}, 1828), "対策": ({"たいさく"}, 1015), "感染対策": ({"かんせんたいさく"}, 154948),
    "経済": ({"けいざい"}, 900), "政策": ({"せいさく"}, 2600), "経済政策": ({"けいざいせいさく"}, 65000),
    "生活": ({"せいかつ"}, 500), "習慣": ({"しゅうかん"}, 3000), "生活習慣": ({"せいかつしゅうかん"}, 20000),
    "生活習慣病": ({"せいかつしゅうかんびょう"}, 25000), "病": ({"やまい"}, 8000),
    "二人": ({"ふたり"}, 300), "三脚": ({"さんきゃく"}, 30000), "二人三脚": ({"ににんさんきゃく"}, 25000),
    "授業": ({"じゅぎょう"}, 2000), "中": ({"なか"}, 50), "授業中": ({"じゅぎょうちゅう"}, 20000),
    "日本": ({"にほん"}, 100), "政府": ({"せいふ"}, 800), "日本政府": ({"にほんせいふ"}, 30000),
    "販売": ({"はんばい"}, 3000), "会社": ({"かいしゃ"}, 400), "販売会社": ({"はんばいがいしゃ"}, 90000),
    "苛め": ({"いじめ"}, 6000), "自殺": ({"じさつ"}, 3000), "いじめ自殺": ({"いじめじさつ"}, 240000),
    "一目": ({"ひとめ"}, 5000), "一目惚れ": ({"ひとめぼれ"}, 9000),
}


def make_judge() -> CompoundJudge:
    return CompoundJudge(
        has_spelling=lambda s: s in DICT,
        readings_of=lambda s: DICT[s][0] if s in DICT else set(),
        rank=lambda s: DICT[s][1] if s in DICT else None,
        rank_by_reading=lambda k: 6000 if k == "いじめ" else None,
    )


def m(surface: str, reading: str, pos=("名詞", "普通名詞", "一般", "*", "*", "*")) -> Morph:
    return Morph(surface, surface, surface, reading, reading, pos)


SUFFIX = ("接尾辞", "名詞的", "一般", "*", "*", "*")
PROPER = ("名詞", "固有名詞", "地名", "国", "*", "*")


def segs(morphs) -> list[str]:
    out = []
    for it in resegment(morphs, make_judge()):
        out.append(it.surface if isinstance(it, Unit) else it.surface)
    return out


def test_single_kanji_merges():
    assert segs([m("顕微", "ケンビ"), m("鏡", "キョウ", SUFFIX)]) == ["顕微鏡"]


def test_transparent_compound_splits():
    assert segs([m("感染", "カンセン"), m("対策", "タイサク")]) == ["感染", "対策"]
    assert segs([m("経済", "ケイザイ"), m("政策", "セイサク")]) == ["経済", "政策"]


def test_three_parts_judged_as_two_words():
    # 電子 + 顕微鏡 is rejected, then 顕微 + 鏡 merges
    assert segs([m("電子", "デンシ"), m("顕微", "ケンビ"), m("鏡", "キョウ", SUFFIX)]) == ["電子", "顕微鏡"]
    # 生活習慣 + 病: single kanji
    assert segs([m("生活", "セイカツ"), m("習慣", "シュウカン"), m("病", "ビョウ", SUFFIX)]) == ["生活習慣病"]


def test_different_reading_merges():
    assert segs([m("二人", "フタリ"), m("三脚", "サンキャク")]) == ["二人三脚"]


def test_rendaku_is_not_a_different_reading():
    assert segs([m("販売", "ハンバイ"), m("会社", "カイシャ")]) == ["販売", "会社"]


def test_productive_suffix_and_proper_noun():
    assert segs([m("授業", "ジュギョウ"), m("中", "チュウ", SUFFIX)]) == ["授業", "中"]
    assert segs([m("日本", "ニホン", PROPER), m("政府", "セイフ")]) == ["日本", "政府"]


def test_kana_part_takes_headword_frequency():
    assert segs([m("いじめ", "イジメ"), m("自殺", "ジサツ")]) == ["いじめ", "自殺"]


def test_bound_part_merges():
    # 惚れ is not a dictionary word on its own
    assert segs([m("一目", "ヒトメ"), m("惚れ", "ホレ")]) == ["一目惚れ"]


def test_never_merge_honorific():
    assert segs([m("田中", "タナカ", ("名詞", "固有名詞", "人名", "姓", "*", "*")), m("さん", "サン", SUFFIX)]) == ["田中", "さん"]
