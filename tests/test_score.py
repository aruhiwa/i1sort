"""Planner decisions and ordering on fake cards (no Anki needed)."""

import pytest

from i1sort.config import load_config
from i1sort.known import KnownSet
from i1sort.score import DECISIONS, Planner

MODEL = "M"


@pytest.fixture(scope="module")
def env():
    cfg = load_config(overrides={"sort": {"fields": {MODEL: {"sentence": "Expression", "audio": "Audio"}},
                                          "min_words": 3, "one_card_per_word": "all"},
                                 "filters": {"ignore_list": False}})
    if not cfg.resolve(cfg["dictionaries"]["jmdict"]).exists():
        pytest.skip("JMdict not available")
    from i1sort.parse.pipeline import Parser

    return cfg, Parser(cfg, use_cache=False)


def known_from(parser, sentences):
    ks = KnownSet()
    from i1sort.filters import TokenFilter

    tf = TokenFilter(parser.cfg, parser.db, ignore=())
    for s in sentences:
        for w in parser.parse(s, tf).counted:
            ks.add(w.key, w.lemma, w.reading, "known", "test")
    return ks


def deck(texts, start=100, audio=None):
    cards, notes = [], {}
    for i, t in enumerate(texts):
        nid, cid = 1000 + i, 5000 + i
        a = "[sound:x.mp3]" if audio is None or audio[i] else ""
        notes[nid] = {"noteId": nid, "modelName": MODEL, "tags": [],
                      "fields": {"Expression": {"value": t, "order": 0}, "Audio": {"value": a, "order": 1}}}
        cards.append({"cardId": cid, "note": nid, "due": start + i, "queue": 0})
    return cards, notes


def plan(env, known_sentences, texts, **kw):
    cfg, parser = env
    ks = known_from(parser, known_sentences)
    cards, notes = deck(texts, **kw)
    return Planner(cfg, parser, ks).plan_deck("D", cards, notes)


BASE = ["私は毎日学校で日本語を勉強している", "友達と一緒に映画を見に行った", "この本はとても面白い"]


def by_text(dp):
    return {p.res.cleaned: p for p in dp.cards}


def test_decisions(env):
    texts = [
        "私は毎日学校で映画を見る",          # all known → known
        "私は毎日学校で猫を見る",            # 猫 unknown → i1
        "友達は猫と犬が大好きだ",            # 猫, 犬, 大好き → not-ready
        "♪ 毎日学校で日本語を勉強する",      # music → junk
    ]
    dp = plan(env, BASE + ["大好きな友達"], texts)
    d = {t: p.decision for t, p in by_text(dp).items()}
    assert d["私は毎日学校で映画を見る"] == "known"
    assert d["私は毎日学校で猫を見る"] == "i1"
    assert by_text(dp)["私は毎日学校で猫を見る"].target.lemma == "猫"
    assert d["友達は猫と犬が大好きだ"] == "not-ready"
    assert d["毎日学校で日本語を勉強する"] == "junk"


def test_one_card_per_target_prefers_ideal_length(env):
    texts = ["猫を見る友達", "私は毎日学校で友達と一緒に猫を見る"]
    dp = plan(env, BASE + ["見る"], texts)
    d = by_text(dp)
    assert d["私は毎日学校で友達と一緒に猫を見る"].decision == "i1"
    assert d["猫を見る友達"].decision == "same-target"


def test_junk_reasons(env):
    texts = ["はい！", "本を見る", "友達と一緒に本を読んだ", "友達と一緒に本を読んだ", "Hello there my friend 本"]
    dp = plan(env, BASE + ["読む"], texts, audio=[1, 1, 0, 1, 1])
    reasons = {p.res.cleaned: p.reason.split(":")[0] for p in dp.cards if p.decision == "junk"}
    assert reasons["はい！"] == "no_words"
    assert reasons["本を見る"] == "too_short"
    assert reasons["友達と一緒に本を読んだ"] in ("no_audio", "duplicate")
    assert reasons["Hello there my friend 本"] == "foreign"
    # the copy with audio is kept, the one without is junk
    assert sum(1 for p in dp.cards if p.res.cleaned == "友達と一緒に本を読んだ" and p.decision == "junk") == 1


def test_order_and_positions(env):
    texts = ["私は毎日学校で映画を見る", "私は毎日学校で猫を見る", "友達は猫と犬が大好きだ", "はい！"]
    dp = plan(env, BASE, texts, start=40)
    rank = {d: i for i, d in enumerate(DECISIONS)}
    seq = [rank[p.decision] for p in dp.cards]
    assert seq == sorted(seq)
    assert [p.new_due for p in dp.cards] == list(range(40, 44))


def test_deterministic(env):
    texts = ["私は毎日学校で猫を見る", "私は毎日学校で犬を見る", "友達は猫と犬が大好きだ"]
    a = plan(env, BASE, texts)
    b = plan(env, BASE, texts)
    assert [(p.note_id, p.decision, p.new_due) for p in a.cards] == [(p.note_id, p.decision, p.new_due) for p in b.cards]


def test_deck_frequency_switch(env):
    cfg, parser = env
    texts = ["私は毎日学校で猫を見る", "猫と友達は毎日学校で遊ぶ", "私は毎日学校で鷲を見る"]
    ks = known_from(parser, BASE + ["遊ぶ"])
    cards, notes = deck(texts)
    cfg_on = load_config(overrides={**cfg.raw, "sort": {**cfg.raw["sort"], "use_deck_freq": True}})
    on = Planner(cfg_on, parser, ks).plan_deck("D", cards, notes)
    cfg_off = load_config(overrides={**cfg.raw, "sort": {**cfg.raw["sort"], "use_deck_freq": False}})
    off = Planner(cfg_off, parser, ks).plan_deck("D", cards, notes)
    p_on = {p.target.lemma: p.priority for p in on.cards if p.target}
    p_off = {p.target.lemma: p.priority for p in off.cards if p.target}
    assert p_off["猫"] == round(off.priority.g(next(p.target.key for p in off.cards if p.target and p.target.lemma == "猫")), 4)
    assert p_on["猫"] > p_off["猫"]  # 猫 appears twice in the deck: only the deck term adds to it
    assert "deck count" not in off.priority.formula()


def test_max_tokens_junks_long_lines(env):
    cfg, parser = env
    long = "私は毎日学校で日本語を勉強して、友達と一緒に映画を見に行って、この本はとても面白いと思ったけど、猫を見た"
    ks = known_from(parser, BASE)
    cards, notes = deck([long, "私は毎日学校で猫を見る"])
    cfg2 = load_config(overrides={**cfg.raw, "sort": {**cfg.raw["sort"], "max_tokens": 10}})
    dp = Planner(cfg2, parser, ks).plan_deck("D", cards, notes)
    p = by_text(dp)[long]
    assert p.decision == "junk" and p.reason.startswith("too_long")


def test_one_card_per_word_across_decks(env):
    cfg, parser = env
    ks = known_from(parser, BASE)
    planner = Planner(cfg, parser, ks)
    a = planner.plan_deck("A", *deck(["私は毎日学校で猫を見る"], start=10))
    b = planner.plan_deck("B", *deck(["猫だ", "友達と一緒に猫を見に行った"], start=50))
    assert [p.decision for p in a.cards if p.target] == ["i1"]
    planner.across_decks([a, b])
    i1 = [(dp.deck, p.res.cleaned) for dp in (a, b) for p in dp.cards if p.decision == "i1"]
    assert len(i1) == 1  # 猫 is taught once, by the better sentence
    assert all(p.reason.startswith("target taught in") for dp in (a, b) for p in dp.cards
               if p.decision == "same-target" and p.target and p.target.lemma == "猫")
    assert [p.new_due for p in b.cards] == list(range(50, 50 + len(b.cards)))


def test_one_card_per_word_per_deck(env):
    cfg, parser = env
    ks = known_from(parser, BASE)
    cfg2 = load_config(overrides={**cfg.raw, "sort": {**cfg.raw["sort"], "one_card_per_word": "deck"}})
    planner = Planner(cfg2, parser, ks)
    a = planner.plan_deck("A", *deck(["私は毎日学校で猫を見る"], start=10))
    b = planner.plan_deck("B", *deck(["友達と一緒に猫を見に行った"], start=50))
    planner.across_decks([a, b])
    assert sum(p.decision == "i1" for dp in (a, b) for p in dp.cards) == 2


def test_cloze_and_segment_fields(env):
    from i1sort.apply import field_values

    cfg, parser = env
    ks = known_from(parser, BASE)
    texts = ["私は<b>毎日</b>学校で猫を見る", "猫だ、猫を見たのは私だ", "はい！"]
    dp = Planner(cfg, parser, ks).plan_deck("D", *deck(texts))
    for p in dp.cards:
        f = field_values(p)
        assert f["i1sort_cloze_prefix"] + f["i1sort_cloze_body"] + f["i1sort_cloze_suffix"] == p.res.text
        if p.target is not None:
            assert f["i1sort_cloze_body"] == p.target.surface
    first = by_text(dp)["私は毎日学校で猫を見る"]
    f = field_values(first)
    assert (f["i1sort_cloze_prefix"], f["i1sort_cloze_body"]) == ("私は<b>毎日</b>学校で", "猫")
    assert f["i1sort_segments"] == "私｜は｜毎日｜学校｜で｜猫｜を｜見る"
    assert "は: particle" in f["i1sort_filtered"] and "を: particle" in f["i1sort_filtered"]
    junk = by_text(dp)["はい！"]
    assert field_values(junk)["i1sort_cloze_prefix"] == "はい！"


def test_furigana_fields(env):
    from i1sort.apply import field_values

    cfg, parser = env
    ks = known_from(parser, BASE)
    texts = ["私は毎日学校で本を読んだ", "私は毎日表を見る", "私は毎日学校で猫を見る", "私は毎日転生した"]
    dp = Planner(cfg, parser, ks).plan_deck("D", *deck(texts))
    f = field_values(by_text(dp)["私は毎日学校で本を読んだ"])
    assert (f["i1sort_target"], f["i1sort_target_furigana"], f["i1sort_target_reading_only"]) == ("読む", "読[よ]む", "よむ")
    assert (f["i1sort_cloze_body"], f["i1sort_cloze_body_furigana"]) == ("読ん", "読[よ]ん")
    f = field_values(by_text(dp)["私は毎日表を見る"])  # i1sort's reading fix: おもて, not ひょう
    assert (f["i1sort_target_furigana"], f["i1sort_cloze_body_furigana"]) == ("表[おもて]", "表[おもて]")
    f = field_values(by_text(dp)["私は毎日学校で猫を見る"])
    assert f["i1sort_target_reading_only"] == "ねこ"
    f = field_values(by_text(dp)["私は毎日転生した"])  # JMdict's common reading, not Sudachi's てんしょう
    assert (f["i1sort_target_furigana"], f["i1sort_cloze_body_furigana"]) == ("転生[てんせい]", "転生[てんせい]")


def test_furigana_alignment():
    from i1sort.textutil import furigana

    assert furigana("食べ物", "たべもの") == "食[た]べ 物[もの]"
    assert furigana("お断り", "おことわり") == "お 断[ことわ]り"
    assert furigana("今日", "きょう") == "今日[きょう]"
    assert furigana("テレビ", "テレビ") == "テレビ"
    assert furigana("炙(あぶ)る", "あぶる") == "炙[あぶ]る"  # the field's own reading
    assert furigana("ヤバい", "やばい") == "ヤバい"
    assert furigana("見る", "みた") == "見る[みた]"  # kana does not line up: whole word


def test_sentence_furigana(env):
    from i1sort.apply import field_values
    from i1sort.furigana import strip

    cfg, parser = env
    ks = known_from(parser, BASE)
    texts = ["私は<b>毎日</b>学校で猫を見る", "何を言うの？何だよ、私の猫だ", "いつの間に30万も猫を炙(あぶ)った", "猫 日本語"]
    dp = Planner(cfg, parser, ks).plan_deck("D", *deck(texts))
    for p in dp.cards:
        f = field_values(p)
        # Anki's kanji: filter gives back the plain fields
        for part in ("prefix", "body", "suffix"):
            if "(" not in f[f"i1sort_cloze_{part}"]:
                assert strip(f[f"i1sort_cloze_{part}_furigana"]) == f[f"i1sort_cloze_{part}"], part
    f = field_values(by_text(dp)["私は毎日学校で猫を見る"])
    assert f["i1sort_cloze_prefix_furigana"] == "私[わたし]は<b> 毎日[まいにち]</b> 学校[がっこう]で"
    assert f["i1sort_cloze_body_furigana"] == "猫[ねこ]"
    assert f["i1sort_cloze_suffix_furigana"] == "を 見[み]る"
    p = next(p for p in dp.cards if p.res.text.startswith("何を"))
    f = field_values(p)
    text = f["i1sort_cloze_prefix_furigana"] + f["i1sort_cloze_body_furigana"] + f["i1sort_cloze_suffix_furigana"]
    assert "何[なに]を 言[い]う" in text and "何[なん]だよ" in text and "私[わたし]の" in text
    p = next(p for p in dp.cards if p.res.text.startswith("いつの"))
    f = field_values(p)
    text = f["i1sort_cloze_prefix_furigana"] + f["i1sort_cloze_body_furigana"] + f["i1sort_cloze_suffix_furigana"]
    assert "間[ま]に" in text and "30万も" in text and "炙[あぶ]った" in text
    p = next(p for p in dp.cards if p.res.text == "猫 日本語")
    f = field_values(p)
    assert strip(f["i1sort_cloze_prefix_furigana"] + f["i1sort_cloze_body_furigana"]
                 + f["i1sort_cloze_suffix_furigana"]) == "猫 日本語"  # the space survives
