"""Mark decks: words marked by moving their cards, appended to files, cards suspended."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from i1sort import mark
from i1sort.apply import restore_changes, snapshot, write, write_restore
from i1sort.config import load_config
from i1sort.known import KnownBuilder, KnownSet

KNOWN, EXACT, IGNORED = "棚::i1sort_known", "棚::i1sort_exact", "棚::i1sort_ignored"
BODY = "i1sort_cloze_body"  # exact and ignored decks read the word as written in the sentence


class FakeAnki:
    """Legacy reads (findCards by deck, cardsInfo, notesInfo) and v1 suspend/tag writes."""

    def __init__(self):
        self.cards: dict[int, dict] = {}
        self.notes: dict[int, dict] = {}

    def add(self, nid, cid, deck, word, queue=0, type_=0, tags=(), field="i1sort_target"):
        self.notes[nid] = {"noteId": nid, "tags": list(tags), "cards": [cid],
                           "fields": {field: {"value": word, "order": 0}}}
        self.cards[cid] = {"cardId": cid, "note": nid, "deck": deck, "due": 100 + cid, "queue": queue, "type": type_}

    def _ensure_checked(self):
        pass

    def invoke(self, action, **p):
        if action == "findCards":
            deck = p["query"].split('"')[1]
            return [c for c, v in self.cards.items() if v["deck"] == deck]
        if action == "cardsInfo":
            return [dict(self.cards[c]) for c in p["cards"] if c in self.cards]
        raise AssertionError(action)

    def notes_info(self, nids):
        return [json.loads(json.dumps(self.notes[n])) for n in nids]

    item_errors = staticmethod(lambda r: [])

    def call(self, method, **p):
        if method == "cards.setSuspended":
            for i in p["ids"]:
                c = self.cards[int(i)]
                c["queue"] = -1 if p["suspended"] else (0 if c["type"] == 0 else c["type"])
            return {"items": []}
        if method in ("tags.add", "tags.remove"):
            for n in p["noteIds"]:
                tags = self.notes[int(n)]["tags"]
                for t in p["tags"]:
                    if method == "tags.add" and t not in tags:
                        tags.append(t)
                    if method == "tags.remove" and t in tags:
                        tags.remove(t)
            return {"items": []}
        if method == "notes.update":
            return {}
        raise AssertionError(method)

    def call_batch(self, calls):
        return [self.call(m, **p) for m, p in calls]


def make_cfg(tmp_path, listed=True):
    """Each deck appends to its own file; ``listed``: the files are in the usual settings."""
    files = {k: tmp_path / f"marked_{k}.txt" for k in ("known", "exact", "ignored")}
    files["known"].write_text("見張る", encoding="utf-8")  # no final newline: append must add one
    cfg = load_config(overrides={
        "mark_decks": {"known": KNOWN, "exact": EXACT, "ignored": IGNORED,
                       **{f"{k}_file": str(f) for k, f in files.items()}},
        "known": {"sources": [], "files": [str(files["known"])] if listed else [],
                  "exact_files": [str(files["exact"])] if listed else []},
        "filters": {"ignore_files": [str(files["ignored"])] if listed else []},
    })
    return cfg, files


@pytest.fixture
def setup(tmp_path):
    cfg, files = make_cfg(tmp_path)
    a = FakeAnki()
    a.add(1, 11, KNOWN, "程")
    a.add(2, 12, KNOWN, "<b>見張る</b>", queue=1, type_=1)  # answered once before it was moved
    a.add(3, 13, KNOWN, "")  # a not-ready card: no target
    a.add(4, 14, IGNORED, "カズマ", queue=-1, field=BODY)  # already suspended (by you)
    a.add(6, 16, EXACT, "シーヌ", field=BODY)
    a.add(5, 15, "KonoSuba", "猫")  # not in a mark deck
    return cfg, a, files


def test_read_and_changes(setup):
    cfg, a, _ = setup
    known, exact, ignored = mark.read(cfg, a, log=lambda m: None)
    assert (known.kind, exact.kind, ignored.kind) == ("known", "exact", "ignored")
    assert known.words == {"程": [11], "見張る": [12]}
    assert known.new_words == ["程"] and known.no_word == [13]
    assert exact.new_words == ["シーヌ"] and ignored.new_words == ["カズマ"]
    ch = mark.changes(known, "i1sort")
    assert sorted(ch.suspend) == [11, 12]  # the empty card is left alone; studied cards too are yours
    assert sorted(ch.tag_add["i1sort::suspended"]) == [1, 2]
    assert mark.changes(ignored, "i1sort").empty()  # already suspended: not tagged, not touched
    text = mark.summary([known, exact, ignored], cfg)
    assert "1 card(s) with an empty i1sort_target field" in text
    assert "marked_known.txt is in [known] files" in text and "marked_ignored.txt is in [filters] ignore_files" in text


def test_words_mean_what_their_file_is_listed_as(setup):
    """Pending deck words count as if already in their file; the file's listing gives them meaning."""
    cfg, a, files = setup
    if not cfg.resolve(cfg["dictionaries"]["jmdict"]).exists():
        pytest.skip("JMdict not available")
    ks = KnownBuilder(cfg, a, log=lambda m: None).build()
    assert {w.word for w in ks.words.values()} >= {"程", "見張る"}  # looked up in the dictionary
    assert ks.exact == {"シーヌ"} and ks.ignored == {"カズマ"}
    assert not ({"程", "見張る"} & ks.exact)
    files_only = KnownBuilder(cfg, None, log=lambda m: None).build(use_anki=False)
    assert {w.word for w in files_only.words.values()} == {"見張る"} and not files_only.exact


def test_unlisted_files_have_no_effect(tmp_path):
    cfg, files = make_cfg(tmp_path, listed=False)
    a = FakeAnki()
    a.add(1, 11, KNOWN, "程")
    a.add(6, 16, EXACT, "シーヌ", field=BODY)
    ks = KnownBuilder(cfg, a, log=lambda m: None).build()
    assert not ks.words and not ks.exact and not ks.ignored
    assert "so its words have no effect" in mark.summary(ks.marked, cfg)


def test_exact_match_ignores_the_dictionary_entry():
    # 屈し misread as 屈指: an exact line 屈し matches the text, and 屈指 is never written
    ks = KnownSet(exact={"屈し"})
    assert ks.exact_match(SimpleNamespace(surface="屈し", lemma="屈し", headword="屈指", morphs=[]))
    ks = KnownSet(exact={"屈指"})
    assert not ks.exact_match(SimpleNamespace(surface="屈し", lemma="屈し", headword="屈指", morphs=[]))


def test_fields_per_deck(tmp_path):
    cfg, _ = make_cfg(tmp_path)
    a = FakeAnki()
    # the exact deck reads i1sort_cloze_body; a reading written into the field is dropped
    a.notes[7] = {"noteId": 7, "tags": [], "cards": [17],
                  "fields": {"i1sort_target": {"value": "屈指"}, BODY: {"value": "<b>屈(くっ)し</b>"}}}
    a.cards[17] = {"cardId": 17, "note": 7, "deck": EXACT, "due": 1, "queue": 0, "type": 0}
    exact = next(m for m in mark.read(cfg, a, log=lambda m: None) if m.kind == "exact")
    assert exact.word_field == BODY and exact.words == {"屈し": [17]}


def test_apply_and_restore(setup, tmp_path):
    cfg, a, files = setup
    known_file, exact_file, ignore_file = files["known"], files["exact"], files["ignored"]
    marks = mark.read(cfg, a, log=lambda m: None)
    chs = [mark.changes(m, "i1sort") for m in marks]
    snap = snapshot([], [], tmp_path / "s.json", marks=list(zip(marks, chs)))
    for m in marks:
        mark.append_words(m)
    for ch in chs:
        write(a, ch, log=lambda m: None)
    assert known_file.read_text(encoding="utf-8") == "見張る\n程\n"
    assert exact_file.read_text(encoding="utf-8") == "シーヌ\n"
    assert ignore_file.read_text(encoding="utf-8") == "カズマ\n"
    assert a.cards[11]["queue"] == -1 and a.cards[12]["queue"] == -1 and a.cards[13]["queue"] == 0
    assert a.cards[15]["queue"] == 0
    # second run: nothing left to do
    again = mark.read(cfg, a, log=lambda m: None)
    assert not any(m.new_words for m in again) and all(mark.changes(m, "i1sort").empty() for m in again)
    # restore: cards back as they were (the learning card too), words out of the files
    data = json.loads(snap.read_text(encoding="utf-8"))
    assert data["mark_files"] == [{"file": str(known_file), "words": ["程"]},
                                  {"file": str(exact_file), "words": ["シーヌ"]},
                                  {"file": str(ignore_file), "words": ["カズマ"]}]
    for ch in restore_changes(a, data, "i1sort"):
        assert not ch.reposition
        write_restore(a, ch, log=lambda m: None)
    for f in data["mark_files"]:
        mark.remove_words(Path(f["file"]), f["words"])
    assert a.cards[11]["queue"] == 0 and a.cards[12]["queue"] == 1
    assert "i1sort::suspended" not in a.notes[1]["tags"]
    assert known_file.read_text(encoding="utf-8") == "見張る\n"
    assert exact_file.read_text(encoding="utf-8") == "" and ignore_file.read_text(encoding="utf-8") == ""


def test_marked_words_change_the_plan(setup):
    cfg, _, _ = setup
    if not cfg.resolve(cfg["dictionaries"]["jmdict"]).exists():
        pytest.skip("JMdict not available")
    from i1sort.filters import TokenFilter
    from i1sort.parse.pipeline import Parser
    from i1sort.score import Planner

    cfg2 = load_config(overrides={**cfg.raw, "sort": {**cfg.raw["sort"], "fields": {"M": {"sentence": "S", "audio": ""}},
                                                      "min_words": 1}})
    parser = Parser(cfg2, use_cache=False)
    base = KnownSet()
    tf = TokenFilter(cfg2, parser.db, ignore=())
    for w in parser.parse("私は毎日学校で本を見る", tf).counted:
        base.add(w.key, w.lemma, w.reading, "known", "test")
    notes = {1: {"noteId": 1, "modelName": "M", "tags": [], "fields": {"S": {"value": "私は毎日学校で猫を見る"}}},
             2: {"noteId": 2, "modelName": "M", "tags": [], "fields": {"S": {"value": "私は毎日学校で犬を見る"}}}}
    cards = [{"cardId": 10 + n, "note": n, "due": n, "queue": 0} for n in notes]
    base.exact = {"猫"}  # in the known mark deck
    base.ignored = {"犬"}  # in the ignored mark deck
    dp = Planner(cfg2, parser, base).plan_deck("D", cards, notes)
    by = {p.note_id: p for p in dp.cards}
    assert by[1].decision == "known" and by[2].decision == "known"
    assert [w.filter for w in by[2].res.words if w.surface == "犬"] == ["ignored"]


def test_ignore_files_takes_several_lists(tmp_path):
    from i1sort.filters import load_ignore_set

    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("カズマ\n", encoding="utf-8")
    b.write_text("アクア\n", encoding="utf-8")
    assert load_ignore_set(load_config(overrides={"filters": {"ignore_files": [str(a), str(b)]}})) == {"カズマ", "アクア"}
    assert load_ignore_set(load_config(overrides={"filters": {"ignore_files": str(a)}})) == {"カズマ"}


def test_renamed_settings_stop_with_a_hint(tmp_path):
    c = tmp_path / "config.toml"
    c.write_text('[known]\nmanual_file = "known.txt"\n[filters]\nignore_file = "ignore.txt"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"(?s)manual_file.*\[known\] files.*ignore_files"):
        load_config(str(c))
