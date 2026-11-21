"""Known-word extraction against a mocked AnkiConnect (no Anki collection needed)."""

from pathlib import Path

import pytest

from i1sort.anki import AnkiConnect, AnkiConnectError
from i1sort.config import load_config
from i1sort.known import KnownBuilder, export, strip_field


class FakeAnki(AnkiConnect):
    """Answers the AnkiConnect actions known.py uses from in-memory data."""

    def __init__(self, notes, studied, known, profile="dev", active="dev"):
        super().__init__(profile=profile)
        self.notes = notes  # nid -> (model, {field: value})
        self.studied = studied  # nids with a non-new card
        self.known = known  # nids with a mature card
        self.active = active
        self.calls = []

    def invoke(self, action, **params):
        self.calls.append(action)
        if action == "version":
            return 6
        if action == "getActiveProfile":
            return self.active
        if action == "apiReflect":
            return {"actions": ["findNotes", "notesInfo", "multi"]}
        if action == "multi":
            return [self.invoke(a["action"], **a.get("params", {})) for a in params["actions"]]
        if action == "findNotes":
            q = params["query"]
            model = q.split('"')[1]
            pool = self.known if "prop:ivl" in q else self.studied
            return [n for n in pool if self.notes[n][0] == model]
        if action == "notesInfo":
            return [{"noteId": n, "modelName": self.notes[n][0],
                     "fields": {k: {"value": v, "order": i} for i, (k, v) in enumerate(self.notes[n][1].items())}}
                    for n in params["notes"]]
        raise AssertionError(f"unexpected action {action}")


def cfg_with(sources):
    cfg = load_config(overrides={"mark_decks": {"known": "", "exact": "", "ignored": ""}, "known": {"sources": sources, "files": [], "exact_files": []}})
    if not cfg.resolve(cfg["dictionaries"]["jmdict"]).exists():
        pytest.skip("JMdict not available")
    return cfg


VOCAB = [{"note_type": "Vocab", "field": "Word", "reading_field": "Reading", "kind": "vocab"}]
SENT = [{"note_type": "Sent", "field": "Expression", "kind": "sentence"}]


def build(cfg, anki):
    return KnownBuilder(cfg, anki, log=lambda m: None).build()


def test_profile_guard():
    anki = FakeAnki({}, [], [], profile="dev", active="User 1")
    with pytest.raises(AnkiConnectError, match="profile"):
        anki.check()


def test_levels_and_best_level_wins():
    cfg = cfg_with(VOCAB + SENT)
    notes = {
        1: ("Vocab", {"Word": "食べる", "Reading": "たべる"}),
        2: ("Vocab", {"Word": "<b>猫</b>", "Reading": "ねこ"}),
        3: ("Vocab", {"Word": "犬", "Reading": "いぬ"}),  # new: not studied
        4: ("Sent", {"Expression": "猫が魚を食べた"}),
    }
    ks = build(cfg, FakeAnki(notes, studied=[1, 2, 4], known=[1]))
    by_word = {w.word: w for w in ks.words.values()}
    assert by_word["食べる"].level == "known"  # vocab mature, sentence only learning → best wins
    assert by_word["猫"].level == "learning"
    assert by_word["魚"].level == "learning"
    assert "犬" not in by_word
    assert len(by_word["食べる"].sources) == 2
    assert ks.counts() == {"known": 1, "learning": 2}


def test_reading_field_picks_homograph():
    cfg = cfg_with(VOCAB)
    notes = {1: ("Vocab", {"Word": "方", "Reading": "ほう"}), 2: ("Vocab", {"Word": "方", "Reading": "かた"})}
    ks = build(cfg, FakeAnki(notes, studied=[1, 2], known=[1, 2]))
    assert len(ks.words) == 2  # 方【ほう】 and 方【かた】 are different words


def test_vocab_key_matches_sentence_key():
    """私 in a vocab deck and 私 in a sentence (Sudachi may read it わたくし) share one key."""
    cfg = cfg_with(VOCAB + SENT)
    notes = {1: ("Vocab", {"Word": "私", "Reading": "わたし"}), 2: ("Sent", {"Expression": "私です"})}
    ks = build(cfg, FakeAnki(notes, studied=[1, 2], known=[1]))
    assert len(ks.words) == 1 and next(iter(ks.words.values())).level == "known"


def test_export_round_trip(tmp_path):
    cfg = cfg_with(VOCAB)
    notes = {1: ("Vocab", {"Word": "食べる", "Reading": "たべる"}), 2: ("Vocab", {"Word": "方", "Reading": "かた"})}
    ks = build(cfg, FakeAnki(notes, studied=[1, 2], known=[1, 2]))
    out = tmp_path / "known.tsv"
    assert export(ks, out) == 2
    cfg2 = load_config(overrides={"mark_decks": {"known": "", "exact": "", "ignored": ""}, "known": {"sources": [], "files": [str(out)], "exact_files": []}})
    ks2 = KnownBuilder(cfg2, None, log=lambda m: None).build(use_anki=False)
    assert set(ks2.words) == set(ks.words)


def test_batched_requests():
    cfg = cfg_with(VOCAB)
    notes = {i: ("Vocab", {"Word": "猫", "Reading": "ねこ"}) for i in range(1, 2501)}
    anki = FakeAnki(notes, studied=list(notes), known=[])
    build(cfg, anki)
    assert anki.calls.count("notesInfo") == 3  # 2500 notes in chunks of 1000
    assert anki.calls.count("multi") == 1  # both searches for every source in one request


def test_strip_field():
    assert strip_field("<a>稍</a>") == "稍"
    assert strip_field("漢字[かんじ]&nbsp;") == "漢字"


def test_potential_covered_by_base():
    cfg = cfg_with(VOCAB)
    from i1sort.parse.pipeline import Parser

    p = Parser(cfg, use_cache=False)
    notes = {1: ("Vocab", {"Word": "使う", "Reading": "つかう"})}
    ks = KnownBuilder(cfg, FakeAnki(notes, studied=[1], known=[1]), parser=p, log=lambda m: None).build()
    words = {w.lemma: w for w in p.parse("魔法が使える。道が分かれる").counted}
    assert ks.covers(words["使える"], p.db)
    assert not ks.covers(words["分かれる"], p.db)  # not a potential form of 分かる


def test_asbplayer_tool(tmp_path):
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import asbplayer_words as aw

    entries = [
        {"profile": "Default", "token": "あたし", "status": 5, "lemmas": ["私", "わたし"]},
        {"profile": "Default", "token": "グリッチ", "status": 2, "lemmas": ["グリッチ"]},
        {"profile": "Default", "token": "猫", "status": 1, "lemmas": ["猫"]},
        {"profile": "Default", "token": "うわっ", "status": 0, "lemmas": ["うわ"], "states": [0]},
    ]
    f = tmp_path / "asb.json"
    f.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "known.txt"
    aw.main([str(f), "--status", "learning,mature", "--form", "both", "-o", str(out)])
    assert out.read_text(encoding="utf-8").splitlines() == ["あたし", "私\tわたし", "グリッチ"]
    aw.main([str(f), "--state", "ignored", "-o", str(out)])
    assert out.read_text(encoding="utf-8").splitlines() == ["うわっ"]
    aw.main([str(f), "--status", "unknown", "--state", "not-ignored", "-o", str(out)])
    assert out.read_text(encoding="utf-8").splitlines() == ["猫"]


def test_exact_files_match_token_text_only(tmp_path):
    from i1sort.parse.pipeline import Parser

    ex = tmp_path / "exact.txt"
    ex.write_text("シーヌ\nパーティー\n", encoding="utf-8")
    cfg = cfg_with([])
    cfg.raw["known"]["exact_files"] = [str(ex)]
    p = Parser(cfg, use_cache=False)
    ks = KnownBuilder(cfg, None, parser=p, log=lambda m: None).build(use_anki=False)
    words = {w.surface: w for w in p.parse("パーティーで死ぬほど踊った").words}
    assert ks.covers(words["パーティー"], p.db)
    assert not ks.covers(words["死ぬ"], p.db)  # シーヌ in an exact list never means 死ぬ
    assert not ks.words  # nothing is looked up in the dictionary


def test_single_kana_lines_are_skipped(tmp_path):
    f = tmp_path / "k.txt"
    f.write_text("か\nて\n猫\n", encoding="utf-8")
    cfg = cfg_with([])
    cfg.raw["known"]["files"] = [str(f)]
    ks = KnownBuilder(cfg, None, log=lambda m: None).build(use_anki=False)
    assert [w.word for w in ks.words.values()] == ["猫"]


def test_auto_known_rank_counts_as_known():
    from i1sort.filters import TokenFilter
    from i1sort.parse.pipeline import Parser

    cfg = cfg_with([])
    cfg.raw["known"]["auto_known_rank"] = 2000
    p = Parser(cfg, use_cache=False)
    ks = KnownBuilder(cfg, None, parser=p, log=lambda m: None).build(use_anki=False)
    words = {w.surface: w for w in p.parse("友達と電話した", TokenFilter(cfg, p.db, ignore=())).words}
    assert words["友達"].filter is None  # still a word in the sentence
    assert ks.why(words["友達"], p.db).startswith("frequent (rank")
    assert p.db.entry_rank(words["友達"].seq) <= 2000
    cfg.raw["known"]["auto_known_rank"] = 0
    assert KnownBuilder(cfg, None, parser=p, log=lambda m: None).build(use_anki=False).why(words["友達"], p.db) is None
