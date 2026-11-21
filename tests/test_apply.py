"""apply.py against an in-memory AnkiConnect: order, tags, suspension, idempotency, restore."""

from types import SimpleNamespace

import pytest

from i1sort.anki import AnkiConnectError, AnkiV1
from i1sort.apply import compute_changes, refresh, restore_changes, snapshot, write, write_restore
from i1sort.score import CardPlan, DeckPlan


class FakeAnki:
    """Legacy reads (cardsInfo/notesInfo) and v1 writes over one in-memory collection."""

    def __init__(self):
        self.cards: dict[int, dict] = {}
        self.notes: dict[int, dict] = {}
        self.calls: list[str] = []

    def add(self, nid, cid, due, queue=0, type_=0, tags=(), fields=None):
        self.notes.setdefault(nid, {"noteId": nid, "tags": list(tags), "cards": [], "fields": fields or {}})
        self.notes[nid]["cards"].append(cid)
        self.cards[cid] = {"cardId": cid, "note": nid, "due": due, "queue": queue, "type": type_}

    # legacy read side (what cards_info / AnkiConnect.notes_info use)
    def _ensure_checked(self):
        pass

    def invoke(self, action, **p):
        assert action == "cardsInfo"
        return [dict(self.cards[c]) for c in p["cards"] if c in self.cards]

    def notes_info(self, nids):
        return [{**self.notes[n], "tags": list(self.notes[n]["tags"]),
                 "fields": {k: {"value": v} for k, v in self.notes[n]["fields"].items()}} for n in nids]

    # v1 write side
    item_errors = staticmethod(AnkiV1.item_errors)

    def call(self, method, **p):
        self.calls.append(method)
        if method == "scheduling.repositionNewCards":
            ids = [int(i) for i in p["cardIds"]]
            if any(self.cards[i]["queue"] != 0 or self.cards[i]["type"] != 0 for i in ids):
                raise AnkiConnectError("invalidState")
            pos, seen = p["startingFrom"], {}
            for i in ids:
                n = self.cards[i]["note"]
                if n not in seen:
                    seen[n] = pos
                    pos += p["stepSize"]
                self.cards[i]["due"] = seen[n]
            return {"changedCount": len(ids)}
        if method == "cards.setSuspended":
            for i in p["ids"]:
                c = self.cards[int(i)]
                c["queue"] = -1 if p["suspended"] else (0 if c["type"] == 0 else 2)
            return {"items": [{"index": k, "result": {}} for k in range(len(p["ids"]))]}
        if method in ("tags.add", "tags.remove"):
            for n in p["noteIds"]:
                tags = self.notes[int(n)]["tags"]
                for t in p["tags"]:
                    low = [x.lower() for x in tags]
                    if method == "tags.add" and t.lower() not in low:
                        tags.append(t)
                    if method == "tags.remove" and t.lower() in low:
                        del tags[low.index(t.lower())]
            return {"items": [{"index": k, "result": {}} for k in range(len(p["noteIds"]))]}
        if method == "notes.update":
            self.notes[int(p["id"])]["fields"].update(p["patch"]["fields"])
            return {}
        raise AssertionError(method)

    def call_batch(self, calls):
        return [self.call(m, **p) for m, p in calls]


SC = {"tag_prefix": "i1sort", "suspend_junk": False, "suspend_known": False, "write_fields": True}


def card_plan(anki, nid, decision, reason="", target=None, u=0, priority=0.0):
    n = anki.notes[nid]
    return CardPlan(
        note_id=nid, card_ids=list(n["cards"]), deck="D", model="M", old_due=0, suspended=False,
        tags=list(n["tags"]), res=None, cards=[], note_cards=list(n["cards"]),
        decision=decision, reason=reason,
        target=SimpleNamespace(headword=target, lemma=target) if target else None,
        unknown=[object()] * u, priority=priority,
    )


def deck_plan(anki, spec):
    """spec: [(note id, decision, kwargs)] in plan order; cards are the note's new cards."""
    dp = DeckPlan("D", [card_plan(anki, nid, d, **kw) for nid, d, kw in spec], {}, None)
    for p in dp.cards:
        p.cards = [{"cardId": c, "due": 0, "queue": 0} for c in anki.notes[p.note_id]["cards"]
                   if anki.cards[c]["type"] == 0]
    refresh(anki, [dp])
    return dp


def run(anki, spec, sc=SC):
    dp = deck_plan(anki, spec)
    ch = compute_changes(dp, sc)
    write(anki, ch, log=lambda m: None)
    return ch


def again(anki, spec, sc=SC):
    return compute_changes(deck_plan(anki, spec), sc)


@pytest.fixture
def anki():
    a = FakeAnki()
    for k in range(6):  # notes 1..6, one card each, positions 10..15
        a.add(k + 1, 100 + k, 10 + k)
    return a


SPEC = [(5, "i1", {"target": "猫", "u": 1, "priority": 0.9}), (2, "i1", {"target": "犬", "u": 1, "priority": 0.5}),
        (3, "not-ready", {"u": 2}), (1, "known", {}), (6, "junk", {"reason": "too_short: 1 word(s)"}),
        (4, "junk", {"reason": "duplicate: same line"})]


def test_first_run_orders_and_tags(anki):
    ch = run(anki, SPEC)
    assert ch.reposition and ch.start == 10
    assert [anki.cards[100 + n - 1]["due"] for n, _, _ in SPEC] == [10, 11, 12, 13, 14, 15]
    assert anki.notes[5]["tags"] == ["i1sort::i1"]
    assert anki.notes[6]["tags"] == ["i1sort::junk::too_short"]
    assert anki.notes[4]["tags"] == ["i1sort::junk::duplicate"]


def test_second_run_changes_nothing(anki):
    run(anki, SPEC)
    ch = again(anki, SPEC)
    assert ch.empty(), ch
    anki.calls.clear()
    run(anki, SPEC)
    assert anki.calls == []


def test_stale_tags_are_replaced(anki):
    run(anki, SPEC)
    spec2 = [(2, "i1", {"target": "犬", "u": 1}), (5, "known", {})] + SPEC[2:]
    run(anki, spec2)
    assert anki.notes[5]["tags"] == ["i1sort::known"]
    assert anki.cards[101]["due"] == 10 and anki.cards[104]["due"] == 11
    assert again(anki, spec2).empty()


def test_suspension_is_tracked_and_released(anki):
    anki.cards[103]["queue"] = -1  # note 4 suspended by the user
    sc = {**SC, "suspend_junk": True}
    ch = run(anki, SPEC, sc)
    assert ch.frozen_cards == 1  # note 4 keeps its place
    assert anki.cards[105]["queue"] == -1 and "i1sort::suspended" in anki.notes[6]["tags"]
    assert anki.cards[103]["queue"] == -1 and "i1sort::suspended" not in anki.notes[4]["tags"]
    assert anki.cards[103]["due"] == 13  # never repositioned
    assert again(anki, SPEC, sc).empty()
    run(anki, SPEC, SC)  # suspend_junk off again
    assert anki.cards[105]["queue"] == 0 and "i1sort::suspended" not in anki.notes[6]["tags"]
    assert anki.cards[103]["queue"] == -1  # still the user's
    assert again(anki, SPEC, SC).empty()


def test_notes_with_studied_cards_get_no_tags(anki):
    anki.add(5, 200, 3, queue=2, type_=2)  # note 5 also has a review card
    run(anki, SPEC)
    assert anki.notes[5]["tags"] == []
    assert anki.cards[200]["due"] == 3  # the studied card is untouched
    assert anki.cards[104]["due"] == 10  # its new card is still ordered


def test_fields_only_when_present(anki):
    anki.notes[5]["fields"] = {"i1sort_target": "", "Expression": "x"}
    run(anki, SPEC)
    assert anki.notes[5]["fields"] == {"i1sort_target": "猫", "Expression": "x"}
    assert anki.notes[2]["fields"] == {}
    assert again(anki, SPEC).empty()


def test_restore_puts_everything_back(anki, tmp_path):
    import json

    anki.notes[3]["tags"] = ["mine", "i1sort::old"]
    before = {c: dict(v) for c, v in anki.cards.items()}, {n: list(v["tags"]) for n, v in anki.notes.items()}
    sc = {**SC, "suspend_junk": True}
    dp = deck_plan(anki, SPEC)
    ch = compute_changes(dp, sc)
    path = snapshot([dp], [ch], tmp_path / "s.json")
    write(anki, ch, log=lambda m: None)
    assert anki.cards[105]["queue"] == -1
    snap = json.loads(path.read_text(encoding="utf-8"))
    for c in restore_changes(anki, snap, "i1sort"):
        write_restore(anki, c, log=lambda m: None)
    after = {c: dict(v) for c, v in anki.cards.items()}, {n: list(v["tags"]) for n, v in anki.notes.items()}
    assert after[0] == before[0]
    assert {n: sorted(t) for n, t in after[1].items()} == {n: sorted(t) for n, t in before[1].items()}
    assert all(c.empty() for c in restore_changes(anki, snap, "i1sort"))


def test_cloze_cuts_at_the_targets_own_occurrence():
    from i1sort.apply import _cloze

    w = lambda s: SimpleNamespace(surface=s, filter=None)  # noqa: E731
    first, second = w("猫"), w("猫")
    res = SimpleNamespace(text="猫と猫", words=[first, w("と"), second])
    assert _cloze(SimpleNamespace(res=res, target=second)) == ("猫と", "猫", "")
    assert _cloze(SimpleNamespace(res=res, target=first)) == ("", "猫", "と猫")
    # 行クノヨ was read as 行くのよ: the body is cut from the katakana original
    crow = w("ゆくのよ")
    res = SimpleNamespace(text="ユクノヨ！", words=[crow, w("！")])
    assert _cloze(SimpleNamespace(res=res, target=crow)) == ("", "ユクノヨ", "！")
    # target not in the field at all: the whole line is the prefix
    res = SimpleNamespace(text="こんにちは", words=[w("ちわ")])
    assert _cloze(SimpleNamespace(res=res, target=res.words[0])) == ("こんにちは", "", "")


def test_cloze_keeps_inline_readings():
    from i1sort.apply import _cloze

    t = SimpleNamespace(surface="炙る", filter=None)
    res = SimpleNamespace(text="少し炙(あぶ)る", words=[SimpleNamespace(surface="少し", filter=None), t])
    assert _cloze(SimpleNamespace(res=res, target=t)) == ("少し", "炙(あぶ)る", "")


def test_definition_field_written_and_kept(anki):
    class Defs:
        def html(self, seq, term, reading):
            return f'<div style="text-align: left;" class="yomitan-glossary">{term}</div>'

        def is_ours(self, html):
            return html.startswith('<div style="text-align: left;" class="yomitan-glossary">')

    anki.notes[5]["fields"] = {"definition": "", "i1sort_target": ""}
    anki.notes[2]["fields"] = {"definition": "<b>mine</b>", "i1sort_target": "犬"}
    dp = deck_plan(anki, SPEC)
    for p in dp.cards:
        p.definition_field = "definition"
    refresh(anki, [dp])  # re-read with the definition field
    ch = compute_changes(dp, SC, Defs())
    assert ch.fields[5]["definition"].endswith(">猫</div>")
    assert "definition" not in ch.fields.get(2, {}) and ch.kept_definitions == 1  # your text stays
    write(anki, ch, log=lambda m: None)
    dp = deck_plan(anki, SPEC)
    for p in dp.cards:
        p.definition_field = "definition"
    refresh(anki, [dp])  # re-read with the definition field
    assert compute_changes(dp, SC, Defs()).fields == {}  # second run: nothing to change
