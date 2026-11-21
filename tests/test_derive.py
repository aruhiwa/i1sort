"""Derived-form coverage (derive.py): which relatives of a known word count as known."""

import pytest

from i1sort.config import load_config
from i1sort.filters import TokenFilter
from i1sort.known import KnownBuilder, KnownSet


@pytest.fixture(scope="module")
def env():
    cfg = load_config(overrides={"mark_decks": {"known": "", "exact": "", "ignored": ""}, "known": {"sources": [], "files": [], "exact_files": []},
                                 "filters": {"ignore_list": False}})
    if not cfg.resolve(cfg["dictionaries"]["jmdict"]).exists():
        pytest.skip("JMdict not available")
    from i1sort.parse.pipeline import Parser

    p = Parser(cfg, use_cache=False)
    return cfg, p, TokenFilter(cfg, p.db, ignore=())


def word(env, sentence, lemma):
    _, p, tf = env
    for w in p.parse(sentence, tf).words:
        if (w.headword or w.lemma) == lemma or w.lemma == lemma:
            return w
    raise AssertionError(f"{lemma} not in {[w.headword or w.lemma for w in p.parse(sentence, tf).words]}")


def known(env, *pairs, derivations=None):
    """pairs: (text, reading) vocab entries, all at level known."""
    cfg, p, _ = env
    kb = KnownBuilder(cfg, None, parser=p, log=lambda m: None)
    ks = KnownSet() if derivations is None else KnownSet(derivations=tuple(derivations))
    for text, reading in pairs:
        for key, w, rd in kb.resolve_vocab(text, reading):
            ks.add(key, w, rd, "known", "test")
    return ks


def why(env, ks, w):
    return ks.why(w, env[1].db)


@pytest.mark.parametrize("known_words,sentence,target,rule", [
    ([("客", "きゃく")], "お客さん こんにちは", "お客さん", "affix"),
    ([("お父さん", "おとうさん")], "父さんは見た", "父さん", "affix"),
    ([("お兄さん", "おにいさん")], "お兄ちゃん何してるの", "お兄ちゃん", "affix"),
    ([("人", "ひと")], "あの人たちが来た", "人たち", "affix"),
    ([("奴", "やつ")], "派手な奴らだ", "奴ら", "affix"),
    ([("代わる", "かわる")], "その代わり、やるなら今だ", "代わり", "stem"),
    ([("やる", "やる")], "やり方が気に入らない", "やり方", "stem"),
    ([("風", "かぜ")], "あんなふうになりたくない", "風", "variant"),
    ([("言う", "いう"), ("通り", "とおり")], "言う通りにする", "言う通り", "parts"),
    ([("気", "き"), ("付ける", "つける")], "気をつけろ", "気をつける", "parts"),
    ([("どう", "どう"), ("言う", "いう")], "どういう事だ", "どう言う", "parts"),
    ([("見つける", "みつける"), ("出す", "だす")], "犯人を見つけ出した", "見つけ出す", "parts"),
    ([("人間", "にんげん")], "人間界の地に", "人間界", "parts"),
    ([("日本語", "にほんご")], "異世界語とか", "語", "morpheme"),
    ([("小学校", "しょうがっこう")], "小学生って誰だ", "小学", "morpheme"),
    ([("未だ", "まだ")], "まだか 早くしろ！", "まだか", "parts"),
    ([("思う", "おもう")], "とても思えない", "思える", "potential"),
    ([("聞く", "きく")], "何か聞ければいい", "聞ける", "potential"),
])
def test_covered(env, known_words, sentence, target, rule):
    ks = known(env, *known_words)
    w = word(env, sentence, target)
    got = why(env, ks, w)
    assert got is not None and got.endswith(f"({rule})"), got


@pytest.mark.parametrize("known_words,sentence,target", [
    ([("前", "まえ")], "お前は誰だ", "お前"),          # お前 "you" is not 前 "front"
    ([("橋", "はし")], "柱が話をしている", "柱"),      # はしら is not はし + plural ら
    ([("元", "もと")], "その由を伝えた", "由"),         # 由 is only a rare spelling of もと
    ([("こそ", "こそ")], "こそこそ逃げる", "こそこそ"),  # reduplication, and こそ is a particle
    ([("分かる", "わかる")], "道が分かれる", "分かれる"),  # not a potential
    ([("言う", "いう")], "言う通りにする", "言う通り"),   # 通り unknown: parts not all covered
    ([("屋根", "やね")], "木の根がこんなところまで", "根"),  # 根 stands alone here: not a morpheme
])
def test_not_covered(env, known_words, sentence, target):
    ks = known(env, *known_words)
    assert why(env, ks, word(env, sentence, target)) is None


def test_derivations_can_be_turned_off(env):
    ks = known(env, ("客", "きゃく"), derivations=["potential"])
    assert why(env, ks, word(env, "お客さん こんにちは", "お客さん")) is None


def test_vocab_phrase_teaches_parts(env):
    cfg, p, _ = env
    kb = KnownBuilder(cfg, None, parser=p, log=lambda m: None)
    assert [k[1] for k in kb.phrase_parts("先に", "さきに")] == ["先"]
    assert [k[1] for k in kb.phrase_parts("失礼します", "しつれいします")] == ["失礼"]
    assert kb.phrase_parts("方", "かた") == []          # one word: never adds a homograph
    assert kb.phrase_parts("先に", "まえに") == []      # readings must add up


def test_phrase_parts_from_file(env, tmp_path):
    cfg, p, _ = env
    f = tmp_path / "known.txt"
    f.write_text("先に\tさきに\n", encoding="utf-8")
    cfg2 = load_config(overrides={"mark_decks": {"known": "", "exact": "", "ignored": ""}, "known": {"sources": [], "files": [str(f)], "exact_files": []}})
    ks = KnownBuilder(cfg2, None, parser=p, log=lambda m: None).build(use_anki=False)
    assert why(env, ks, word(env, "先に飛ぶよ", "先")) == "known"


@pytest.mark.parametrize("sentence,lemma,reading", [
    ("表が出たら行く", "表", "おもて"),     # Sudachi alone says ひょう
    ("お金がない", "お金", "おかね"),
    ("金がない", "金", "かね"),
])
def test_reading_fixes(env, sentence, lemma, reading):
    assert word(env, sentence, lemma).reading == reading


@pytest.mark.parametrize("sentence,lemma,reading", [
    ("会話から入んだよ", "入る", "はいる"),
    ("米も欲しい", "米", "こめ"),
    ("それじゃあ あたいは行くね", "私", "あたい"),
])
def test_more_reading_fixes(env, sentence, lemma, reading):
    assert word(env, sentence, lemma).reading == reading


@pytest.mark.parametrize("known_words,sentence,target", [
    ([("魅力", "みりょく")], "魅力的な人だ", "魅力的"),
    ([("高い", "たかい")], "この高さは無理", "高さ"),
    ([("憎い", "にくい")], "憎しみ以外の感情", "憎しみ"),
])
def test_whole_token_covered_by_pieces(env, known_words, sentence, target):
    ks = known(env, *known_words)
    got = why(env, ks, word(env, sentence, target))
    assert got is not None and got.endswith("(parts)"), got


def test_whole_token_is_its_own_target(env):
    ks = known(env, ("世界", "せかい"))
    assert why(env, ks, word(env, "憎しみ以外の感情", "憎しみ")) is None


def test_honorific_stem_noun_covered(env):
    ks = known(env, ("断る", "ことわる"))
    got = why(env, ks, word(env, "それはお断りだ", "お断り"))
    assert got is not None and "断る" in got, got
