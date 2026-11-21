"""Phase 4: write deck plans to Anki (new-card order, tags, suspension, optional fields).

What is written, per deck, in this order:

1. unsuspend  new cards i1sort suspended earlier (tag ``<prefix>::suspended``) that the
              plan no longer wants suspended; cards you suspended yourself are never touched
2. reposition the deck's active new cards in plan order, one native ``reposition_new_cards``
              call starting at the deck's lowest new-card position (one undo entry); skipped
              when the cards are already in that order
3. suspend    junk / known cards when ``suspend_junk`` / ``suspend_known`` is on
4. tags       ``<prefix>::i1``, ``::same-target``, ``::not-ready``, ``::known``,
              ``::junk::<reason>``; stale ``<prefix>::`` tags from earlier runs are removed
5. fields     ``i1sort_target`` (+ ``_furigana``, ``_reading_only``), ``i1sort_unknown_count``,
              ``i1sort_score``, ``i1sort_segments``, ``i1sort_filtered`` and
              ``i1sort_cloze_prefix`` / ``_body`` / ``_suffix`` (each + ``_furigana``), only on
              note types that already have them (fields are never created)

Only new cards are touched. A note that also has a studied (or otherwise non-new) card is
repositioned but gets no tags, suspension or field changes. Cards that are suspended
without the i1sort tag, or buried, keep their place. Every change is computed against the
current state, so a second run with no new reviews changes nothing.

Before writing, the previous state of everything the run touches is saved to
``data/apply_snapshots/<time>.json``; ``i1sort restore <file>`` puts it back.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .anki import AnkiConnectError, AnkiV1, cards_info

CHUNK = 1000
BATCH = 100  # JSON-RPC batch limit of AnkiConnect v1
FIELD_NAMES = ("i1sort_target", "i1sort_target_furigana", "i1sort_target_reading_only",
               "i1sort_unknown_count", "i1sort_score", "i1sort_segments", "i1sort_filtered",
               "i1sort_cloze_prefix", "i1sort_cloze_prefix_furigana", "i1sort_cloze_body",
               "i1sort_cloze_body_furigana", "i1sort_cloze_suffix", "i1sort_cloze_suffix_furigana")
SEGMENT_SEP = "｜"
WRITE_METHODS = ("scheduling.repositionNewCards", "cards.setSuspended", "tags.add", "tags.remove", "notes.update")


@dataclass
class DeckChanges:
    deck: str
    start: int = 0
    order: list[int] = field(default_factory=list)  # card ids to reposition, in plan order
    reposition: bool = False
    moved: int = 0  # notes whose position changes
    unsuspend: list[int] = field(default_factory=list)
    suspend: list[int] = field(default_factory=list)
    tag_add: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))  # tag -> note ids
    tag_remove: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    fields: dict[int, dict[str, str]] = field(default_factory=dict)  # note id -> new field values
    mixed_notes: int = 0  # notes with studied cards: positions only
    frozen_cards: int = 0  # user-suspended or buried new cards, left in place
    kept_definitions: int = 0  # definition fields holding other text, left alone
    kept_frequencies: int = 0  # frequency fields holding other text, left alone
    runs: list[tuple[int, list[int]]] = field(default_factory=list)  # restore: (start, card ids)

    def empty(self) -> bool:
        return not (self.reposition or self.unsuspend or self.suspend or self.fields
                    or any(self.tag_add.values()) or any(self.tag_remove.values()))

    def tag_notes(self) -> int:
        return len({n for ns in list(self.tag_add.values()) + list(self.tag_remove.values()) for n in ns})


def _tag_set(tags, prefix: str) -> dict[str, str]:
    """i1sort tags on a note, keyed by lowercase (Anki tags are case-insensitive)."""
    pre = prefix.lower() + "::"
    return {t.lower(): t for t in tags if t.lower().startswith(pre)}


def decision_tag(p, prefix: str) -> str:
    if p.decision == "junk":
        code = (p.reason.split(":")[0].strip() or "other").replace(" ", "_")
        return f"{prefix}::junk::{code}"
    return f"{prefix}::{p.decision}"


def _segments(res) -> str:
    """The sentence as parsed: one unit per word (merged expressions and compounds stay whole)."""
    return SEGMENT_SEP.join(w.surface for w in res.words if w.filter != "whitespace" and w.surface.strip())


def _filtered(res) -> str:
    """Filtered words and why: "ね: particle; 田中: name" (punctuation and spaces left out)."""
    seen: dict[str, str] = {}
    for w in res.words:
        if w.filter and w.filter not in ("symbol", "whitespace") and w.surface not in seen:
            seen[w.surface] = w.filter
    return "; ".join(f"{s}: {r}" for s, r in seen.items())


def _cloze_span(p) -> tuple[int, int] | None:
    """Where the target word is in the original sentence field (start, end), or None."""
    from .furigana import surface_pattern
    from .textutil import kata_to_hira

    raw = p.res.text or ""
    t = p.target
    if t is None or not t.surface:
        return None
    # the target is the n-th word with this surface in the line: cut at the n-th occurrence
    n = 0
    for w in p.res.words:
        if w is t:
            break
        if w.surface == t.surface:
            n += 1
    candidates = [t.surface]
    hira = kata_to_hira(t.surface)
    if hira != t.surface:
        candidates.append(hira)
    kata = "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in t.surface)
    if kata != t.surface:
        candidates.append(kata)  # 行クノヨ: the line was read in hiragana but is written in katakana
    for surf in candidates:
        hits = list(surface_pattern(surf).finditer(raw))  # inline readings kept: 炙(あぶ)る
        if hits:
            m = hits[n] if n < len(hits) else hits[0]  # fewer occurrences than in the parse: the first
            return m.start(), m.end()
    return None


def _cloze(p) -> tuple[str, str, str]:
    """The original sentence field cut around the target word, so prefix + body + suffix is
    exactly the field. Without a target (or if it cannot be found) everything is the prefix."""
    raw = p.res.text or ""
    span = _cloze_span(p)
    if span is None:
        return raw, "", ""
    return raw[:span[0]], raw[span[0]:span[1]], raw[span[1]:]


def field_values(p) -> dict[str, str]:
    from . import furigana as fg
    from .textutil import furigana

    raw = (p.res.text or "") if p.res is not None else ""
    span = _cloze_span(p) if p.res is not None else None
    a, b = span or (len(raw), len(raw))
    t = p.target
    word = (t.headword or t.lemma) if t is not None else ""
    reading = (getattr(p, "target_reading", "") or getattr(t, "reading", "")) if t is not None else ""
    readings = (getattr(p, "readings", None) or fg.word_readings(p.res.words)) if p.res is not None else {}
    spans = fg.ruby_spans(raw, p.res.words, readings) if p.res is not None else []
    prefix, body, suffix = raw[:a], raw[a:b], raw[b:]
    body_reading = (reading if getattr(t, "surface", "") == word else readings.get(id(t))) if t is not None else None
    return {
        "i1sort_target": word,
        "i1sort_target_furigana": furigana(word, reading) if word else "",
        "i1sort_target_reading_only": reading if word else "",
        "i1sort_unknown_count": str(p.u) if p.decision != "junk" else "",
        "i1sort_score": f"{p.priority:.4f}" if p.decision in ("i1", "same-target", "not-ready") else "",
        "i1sort_segments": _segments(p.res) if p.res is not None else "",
        "i1sort_filtered": _filtered(p.res) if p.res is not None else "",
        "i1sort_cloze_prefix": prefix,
        "i1sort_cloze_prefix_furigana": fg.render(raw, spans, 0, a),
        "i1sort_cloze_body": body,
        "i1sort_cloze_body_furigana": furigana(body, body_reading) if body and body_reading else body,
        "i1sort_cloze_suffix": suffix,
        "i1sort_cloze_suffix_furigana": fg.render(raw, spans, b),
    }


_KEEP = object()


def definition_change(p, definitions):
    """The new value of the note's definition field, None if it is right, or _KEEP when it
    holds text i1sort did not write. Cards with a target (i1, same-target) get the target's
    definition; others get none. The field is written when empty, and replaced or cleared
    only when it holds a definition from the same dictionary on a note i1sort gave a target
    before (its i1sort_target field is filled): anything else is yours."""
    fld = p.definition_field
    cur = p.fields.get(fld, "")
    t = p.target
    want = ""
    if t is not None and p.decision in ("i1", "same-target"):
        want = definitions.html(getattr(t, "seq", None), t.headword or t.lemma, getattr(p, "target_reading", ""))
    if cur == want:
        return None
    if not cur.strip():
        return want
    if p.fields.get("i1sort_target") and definitions.is_ours(cur):
        return want
    return _KEEP


def frequency_change(p):
    """Like definition_change for the frequency field: the target's rank on cards with a
    target, nothing on others; a field holding a number is i1sort's only on a note i1sort
    gave a target before."""
    fld = p.frequency_field
    cur = p.fields.get(fld, "")
    want = getattr(p, "target_frequency", "") if p.target is not None and p.decision in ("i1", "same-target") else ""
    if cur == want:
        return None
    if not cur.strip():
        return want
    if p.fields.get("i1sort_target") and cur.strip().isdigit():
        return want
    return _KEEP


def compute_changes(dp, sc: dict, definitions=None) -> DeckChanges:
    """Diff a DeckPlan (whose cards carry the current due/queue/tags/fields) against Anki.
    ``definitions``: a definitions.DefinitionDict to fill [sort.fields] definition fields."""
    prefix = sc.get("tag_prefix", "i1sort")
    track = f"{prefix}::suspended"
    susp_junk = bool(sc.get("suspend_junk", False))
    susp_known = bool(sc.get("suspend_known", False))
    write_fields = bool(sc.get("write_fields", True))
    ch = DeckChanges(dp.deck)
    if not dp.cards:
        return ch
    ch.start = min(c["due"] for p in dp.cards for c in p.cards)

    order_notes: list[list[dict]] = []  # repositionable cards, grouped per note, in plan order
    for p in dp.cards:  # already in plan order
        new_ids = {c["cardId"] for c in p.cards}
        mixed = any(cid not in new_ids for cid in p.note_cards)
        ch.mixed_notes += mixed
        existing = _tag_set(p.tags, prefix)
        has_track = track.lower() in existing
        want = (p.decision == "junk" and susp_junk) or (p.decision == "known" and susp_known)
        hold = False
        active = []
        for c in p.cards:
            q = c["queue"]
            if not mixed and q == -1 and has_track and not want:
                ch.unsuspend.append(c["cardId"])
                q = 0
            elif not mixed and q == 0 and want:
                ch.suspend.append(c["cardId"])  # after repositioning
                hold = True
            elif not mixed and q == -1 and has_track and want:
                hold = True
            if q == 0:
                active.append(c)
            else:
                ch.frozen_cards += not (q == -1 and has_track)
        if active:
            order_notes.append(active)
        if mixed:
            continue
        desired = {decision_tag(p, prefix)} | ({track} if hold else set())
        desired_l = {t.lower(): t for t in desired}
        for k, t in desired_l.items():
            if k not in existing:
                ch.tag_add[t].append(p.note_id)
        for k, t in existing.items():
            if k not in desired_l:
                ch.tag_remove[t].append(p.note_id)
        if write_fields and p.fields:
            want_f = {k: v for k, v in field_values(p).items() if k in p.fields}
            diff = {k: v for k, v in want_f.items() if p.fields.get(k, "") != v}
            if definitions is not None and p.definition_field and p.definition_field in p.fields:
                new = definition_change(p, definitions)
                if new is _KEEP:
                    ch.kept_definitions += 1
                elif new is not None:
                    diff[p.definition_field] = new
            if getattr(p, "frequency_field", "") and p.frequency_field in p.fields:
                new = frequency_change(p)
                if new is _KEEP:
                    ch.kept_frequencies += 1
                elif new is not None:
                    diff[p.frequency_field] = new
            if diff:
                ch.fields[p.note_id] = diff

    # Positions: already right when every note's cards share one position, those positions
    # rise along the plan order and the first is the deck's start (gaps are fine: they are
    # cards suspended or buried since). Otherwise one call puts the whole deck in order.
    ch.order = [c["cardId"] for cs in order_notes for c in cs]
    dues = [{c["due"] for c in cs} for cs in order_notes]
    in_place = bool(dues) and all(len(d) == 1 for d in dues) and min(dues[0]) == ch.start and all(
        min(a) < min(b) for a, b in zip(dues, dues[1:]))
    if order_notes and not in_place:
        ch.reposition = True
        ch.moved = sum(1 for i, d in enumerate(dues) if d != {ch.start + i})
    return ch


def refresh(anki, plans) -> None:
    """Re-read due/queue/tags/fields of every planned card right before writing."""
    ids = [c["cardId"] for dp in plans for p in dp.cards for c in p.cards]
    info = {c["cardId"]: c for c in cards_info(anki, ids)} if ids else {}
    nids = sorted({p.note_id for dp in plans for p in dp.cards})
    notes = {n["noteId"]: n for n in anki.notes_info(nids)} if nids else {}
    for dp in plans:
        for p in dp.cards:
            fresh = []
            for c in p.cards:
                ci = info.get(c["cardId"])
                if ci is None or ci.get("type") != 0:  # deleted, or no longer new (studied meanwhile)
                    continue
                fresh.append({"cardId": ci["cardId"], "due": ci["due"], "queue": ci["queue"]})
            p.cards = fresh
            n = notes.get(p.note_id)
            if n is not None:
                p.tags = n.get("tags", [])
                p.note_cards = list(n.get("cards") or p.note_cards)
                p.fields = {k: v["value"] for k, v in n["fields"].items()
                            if k.startswith("i1sort_")
                            or k in (getattr(p, "definition_field", ""), getattr(p, "frequency_field", ""))}
        dp.cards = [p for p in dp.cards if p.cards]


def summary(changes: list[DeckChanges]) -> str:
    head = f"{'deck':42} {'reposition':>10} {'tags':>6} {'suspend':>8} {'unsuspend':>9} {'fields':>6}"
    lines = [head]
    for ch in changes:
        rep = f"{ch.moved}" if ch.reposition else "-"
        lines.append(f"{ch.deck[:42]:42} {rep:>10} {ch.tag_notes():>6} {len(ch.suspend):>8} "
                     f"{len(ch.unsuspend):>9} {len(ch.fields):>6}")
    notes = [f"  {ch.deck}: {ch.mixed_notes} notes with studied cards get no tags/suspension"
             for ch in changes if ch.mixed_notes]
    notes += [f"  {ch.deck}: {ch.frozen_cards} suspended/buried new cards keep their place"
              for ch in changes if ch.frozen_cards]
    notes += [f"  {ch.deck}: {ch.kept_definitions} definition fields hold other text and are left alone"
              for ch in changes if ch.kept_definitions]
    notes += [f"  {ch.deck}: {ch.kept_frequencies} frequency fields hold other text and are left alone"
              for ch in changes if ch.kept_frequencies]
    lines.append("reposition = notes that move; tags = notes whose i1sort tags change")
    return "\n".join(lines + notes)


# ---------------------------------------------------------------------------- snapshot

def snapshot(plans, changes: list[DeckChanges], path: Path, marks=()) -> Path:
    """``marks``: (mark.MarkDeck, DeckChanges) pairs; their cards are saved in any state, and the
    words the run adds to the mark files are listed so ``restore`` can take them out again."""
    touched_notes = set()
    for ch in list(changes) + [c for _, c in marks]:
        touched_notes |= {n for ns in list(ch.tag_add.values()) + list(ch.tag_remove.values()) for n in ns}
        touched_notes |= set(ch.fields)
    data = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "decks": []}
    for dp, ch in zip(plans, changes):
        if ch.empty():
            continue
        data["decks"].append({
            "deck": dp.deck,
            "cards": [{"id": c["cardId"], "note": p.note_id, "due": c["due"], "queue": c["queue"]}
                      for p in dp.cards for c in p.cards],
            "notes": {str(p.note_id): {"tags": list(p.tags), "fields": p.fields}
                      for p in dp.cards if p.note_id in touched_notes},
        })
    for m, ch in marks:
        if not ch.empty():
            data["decks"].append({
                "deck": m.deck, "kind": "mark",
                "cards": [{"id": c["cardId"], "note": c["note"], "due": c["due"], "queue": c["queue"]} for c in m.cards],
                "notes": {str(n): {"tags": list(m.tags.get(n, [])), "fields": {}}
                          for n in {c["note"] for c in m.cards} if n in touched_notes},
            })
    files = [{"file": str(m.file), "words": m.new_words} for m, _ in marks if m.new_words]
    if files:
        data["mark_files"] = files
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------- writing

def _chunks(xs, n=CHUNK):
    for k in range(0, len(xs), n):
        yield xs[k:k + n]


def _check(v1: AnkiV1, method: str, result) -> None:
    errs = v1.item_errors(result)
    if errs:
        raise AnkiConnectError(f"{method}: {len(errs)} item(s) failed, first: {errs[0]}")


def write(v1: AnkiV1, ch: DeckChanges, log=print) -> None:
    s = lambda ids: [str(i) for i in ids]  # noqa: E731  (v1 entity ids are decimal strings)
    for part in _chunks(ch.unsuspend):
        _check(v1, "cards.setSuspended", v1.call("cards.setSuspended", ids=s(part), suspended=False))
    if ch.reposition:
        r = v1.call("scheduling.repositionNewCards", cardIds=s(ch.order), startingFrom=ch.start,
                    stepSize=1, randomize=False, shiftExisting=False)
        log(f"  {ch.deck}: repositioned {r.get('changedCount', '?')} cards from position {ch.start}")
    for part in _chunks(ch.suspend):
        _check(v1, "cards.setSuspended", v1.call("cards.setSuspended", ids=s(part), suspended=True))
    for tag, nids in ch.tag_remove.items():
        for part in _chunks(nids):
            _check(v1, "tags.remove", v1.call("tags.remove", noteIds=s(part), tags=[tag]))
    for tag, nids in ch.tag_add.items():
        for part in _chunks(nids):
            _check(v1, "tags.add", v1.call("tags.add", noteIds=s(part), tags=[tag]))
    _write_fields(v1, ch.fields)


def _write_fields(v1: AnkiV1, fields: dict[int, dict[str, str]]) -> None:
    items = list(fields.items())
    for part in _chunks(items, BATCH):
        v1.call_batch([("notes.update", {"id": str(nid), "patch": {"fields": vals}}) for nid, vals in part])


# ---------------------------------------------------------------------------- restore

def restore_changes(anki, snap: dict, prefix: str) -> list[DeckChanges]:
    """Changes that put the snapshot's positions, i1sort tags, suspension and fields back."""
    track = f"{prefix}::suspended".lower()
    out = []
    for d in snap["decks"]:
        ch = DeckChanges(d["deck"])
        ids = [c["id"] for c in d["cards"]]
        now = {c["cardId"]: c for c in cards_info(anki, ids)}
        notes_now = {n["noteId"]: n for n in anki.notes_info(sorted({c["note"] for c in d["cards"]}))}
        snap_notes = {int(k): v for k, v in d["notes"].items()}
        is_mark = d.get("kind") == "mark"  # a mark deck: cards in any state, never repositioned
        cards = [c for c in d["cards"] if c["id"] in now and (is_mark or now[c["id"]].get("type") == 0)]
        will_be_active = set()
        for c in cards:
            q_now = now[c["id"]]["queue"]
            tags_now = _tag_set(notes_now.get(c["note"], {}).get("tags", []), prefix)
            tags_then = _tag_set(snap_notes.get(c["note"], {}).get("tags", []), prefix)
            if q_now == -1 and c["queue"] != -1 and track in tags_now:
                ch.unsuspend.append(c["id"])  # i1sort suspended it
                q_now = 0
            elif q_now == 0 and c["queue"] == -1 and track in tags_then:
                ch.suspend.append(c["id"])  # i1sort had released it (after repositioning)
            if q_now == 0 and not is_mark:
                will_be_active.add(c["id"])
        # positions: runs of notes whose old positions were consecutive
        cs = sorted((c for c in cards if c["id"] in will_be_active), key=lambda c: (c["due"], c["id"]))
        runs: list[list[dict]] = []
        for c in cs:
            if runs and c["due"] - runs[-1][-1]["due"] in (0, 1):
                runs[-1].append(c)
            else:
                runs.append([c])
        ch.runs = [(r[0]["due"], [c["id"] for c in r]) for r in runs
                   if any(now[c["id"]]["due"] != c["due"] for c in r)]
        ch.reposition = bool(ch.runs)
        ch.moved = sum(len(ids) for _, ids in ch.runs)
        for nid, n in notes_now.items():
            if nid not in snap_notes:
                continue
            old = _tag_set(snap_notes[nid]["tags"], prefix)
            cur = _tag_set(n.get("tags", []), prefix)
            for k, t in cur.items():
                if k not in old:
                    ch.tag_remove[t].append(nid)
            for k, t in old.items():
                if k not in cur:
                    ch.tag_add[t].append(nid)
            old_f = snap_notes[nid].get("fields") or {}
            diff = {k: v for k, v in old_f.items() if n["fields"].get(k, {}).get("value") != v}
            if diff:
                ch.fields[nid] = diff
        out.append(ch)
    return out


def write_restore(v1: AnkiV1, ch: DeckChanges, log=print) -> None:
    s = lambda ids: [str(i) for i in ids]  # noqa: E731
    for part in _chunks(ch.unsuspend):
        _check(v1, "cards.setSuspended", v1.call("cards.setSuspended", ids=s(part), suspended=False))
    for start, ids in ch.runs:
        v1.call("scheduling.repositionNewCards", cardIds=s(ids), startingFrom=start, stepSize=1,
                randomize=False, shiftExisting=False)
    for part in _chunks(ch.suspend):
        _check(v1, "cards.setSuspended", v1.call("cards.setSuspended", ids=s(part), suspended=True))
    for tag, nids in ch.tag_remove.items():
        for part in _chunks(nids):
            _check(v1, "tags.remove", v1.call("tags.remove", noteIds=s(part), tags=[tag]))
    for tag, nids in ch.tag_add.items():
        for part in _chunks(nids):
            _check(v1, "tags.add", v1.call("tags.add", noteIds=s(part), tags=[tag]))
    _write_fields(v1, ch.fields)
    log(f"  {ch.deck}: restored")
