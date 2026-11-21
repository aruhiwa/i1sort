"""Dry-run reports: report.csv and a single self-contained report.html."""

from __future__ import annotations

import csv
import html
from collections import Counter
from pathlib import Path

from .score import DECISIONS, CardPlan, DeckPlan


def _word(w) -> str:
    return w.headword or w.lemma


def _row(p: CardPlan) -> dict:
    return {
        "new_pos": p.new_due,
        "old_pos": p.old_due,
        "deck": p.deck,
        "decision": p.decision,
        "U": p.u if p.decision not in ("junk",) else "",
        "target": _word(p.target) if p.target else "",
        "target_reading": p.target.reading if p.target and any("\u3400" <= c <= "\u9fff" for c in _word(p.target)) else "",
        "priority": p.priority if p.decision in ("i1", "same-target", "not-ready") else "",
        "quality": p.quality if p.decision != "junk" else "",
        "unknown": " ".join(_word(w) for w in p.unknown),
        "derived": " ".join(f"{_word(w)}←{why[4:]}" if why.startswith("via ") else f"{_word(w)}:{why}"
                            for w, why in p.derived),
        "sentence": p.res.cleaned,
        "original": p.res.text if p.res.text != p.res.cleaned else "",
        "filtered": " ".join(f"{w.surface}:{w.filter}" for w in p.res.words
                             if w.filter not in (None, "symbol", "whitespace", "particle", "aux")),
        "reason": p.reason,
        "suspended": "yes" if p.suspended else "",
        "note_id": p.note_id,
        "card_ids": " ".join(map(str, p.card_ids)),
    }


def write_csv(plans: list[DeckPlan], path: Path) -> None:
    rows = [_row(p) for dp in plans for p in dp.cards]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["new_pos"])
        w.writeheader()
        w.writerows(rows)


def _sentence_html(p: CardPlan) -> str:
    unknown = {w.key for w in p.unknown}
    derived = {id(w): why for w, why in p.derived}
    out = []
    for w in p.res.words:
        s = html.escape(w.surface)
        if p.target is not None and w.key == p.target.key and w.filter is None:
            out.append(f'<b class="t">{s}</b>')
        elif w.key in unknown and w.filter is None:
            out.append(f'<span class="u">{s}</span>')
        elif id(w) in derived:
            out.append(f'<span class="dv" title="known {html.escape(derived[id(w)])}">{s}</span>')
        elif w.filter in ("name", "interjection", "honorific", "grammar", "number", "affix", "non_japanese"):
            out.append(f'<span class="f" title="{html.escape(w.filter + (": " + w.filter_detail if w.filter_detail else ""))}">{s}</span>')
        else:
            out.append(s)
    return "".join(out)


_CSS = """
body{font:14px/1.4 system-ui,sans-serif;margin:1.5em;color:#222}
h1{font-size:1.4em} h2{font-size:1.1em;margin-top:1.4em}
table{border-collapse:collapse} td,th{padding:3px 7px;border-bottom:1px solid #eee;vertical-align:top}
th{background:#f4f4f4;position:sticky;top:0;cursor:pointer;user-select:none;text-align:left}
.sum td{border:none;padding:1px 10px 1px 0}
.t{color:#b00;background:#fee} .u{color:#a60;text-decoration:underline dotted} .f{color:#888}
.dv{color:#276;border-bottom:1px dashed #6a9}
.d-i1{color:#070;font-weight:600} .d-junk{color:#999} .d-known{color:#36c} .d-not-ready{color:#a60} .d-same-target{color:#686}
.small{color:#777;font-size:12px} .controls{margin:1em 0} input,select{font-size:14px;margin-right:1em}
code{background:#f4f4f4;padding:1px 4px}
"""

_JS = """
const rows=[...document.querySelectorAll('#cards tbody tr')];
function apply(){const d=document.getElementById('dec').value,k=document.getElementById('deck').value,q=document.getElementById('q').value.toLowerCase();let n=0;
 for(const r of rows){const ok=(!d||r.dataset.d===d)&&(!k||r.dataset.k===k)&&(!q||r.textContent.toLowerCase().includes(q));r.style.display=ok?'':'none';if(ok)n++}
 document.getElementById('n').textContent=n+' cards shown'}
document.querySelectorAll('#cards th').forEach((th,i)=>th.onclick=()=>{const tb=document.querySelector('#cards tbody');const asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
 const num=th.dataset.num==='1';rows.sort((a,b)=>{let x=a.children[i].dataset.v??a.children[i].textContent,y=b.children[i].dataset.v??b.children[i].textContent;
 if(num){x=parseFloat(x);y=parseFloat(y);x=isNaN(x)?Infinity:x;y=isNaN(y)?Infinity:y;return asc?x-y:y-x}return asc?x.localeCompare(y):y.localeCompare(x)});rows.forEach(r=>tb.appendChild(r))});
['dec','deck','q'].forEach(id=>document.getElementById(id).oninput=apply);apply();
"""


def write_html(plans: list[DeckPlan], path: Path, known_counts: dict, title: str = "i1sort dry run") -> None:
    total = Counter()
    for dp in plans:
        total.update(dp.counts())
    parts = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
             f"<style>{_CSS}</style></head><body><h1>{html.escape(title)}</h1>"]
    parts.append(f"<p class='small'>Known words: {known_counts.get('known', 0)} known, "
                 f"{known_counts.get('learning', 0)} learning. Nothing was changed in Anki.</p>")
    # summary table
    parts.append("<table class='sum'><tr><th>deck</th><th>new cards</th>" + "".join(f"<th>{d}</th>" for d in DECISIONS) + "</tr>")
    for dp in plans + [None]:
        c = dp.counts() if dp else total
        name = html.escape(dp.deck) if dp else "<b>total</b>"
        parts.append(f"<tr><td>{name}</td><td>{sum(c.values())}</td>" + "".join(f"<td class='d-{d}'>{c.get(d, 0)}</td>" for d in DECISIONS) + "</tr>")
    parts.append("</table>")
    junk = Counter()
    for dp in plans:
        junk.update(dp.junk_counts())
    parts.append("<p>Junk by reason: " + ", ".join(f"{k} {v}" for k, v in junk.most_common()) + "</p>")
    if plans:
        parts.append(f"<p>Target order: <code>{html.escape(plans[0].priority.formula())}</code> "
                     "(rank = average frequency rank of the word's JMdict entry; max deck count differs per deck).</p>")
    parts.append("<p class='small'>In sentences: <b class='t'>target</b>, <span class='u'>other unknown</span>, "
                 "<span class='f'>filtered (hover for reason)</span>, "
                 "<span class='dv'>known through a related word (hover)</span>.</p>")
    links: dict[str, Counter] = {}
    for dp in plans:
        for p in dp.cards:
            for w, why in p.derived:
                rule = why.rsplit("(", 1)[-1].rstrip(")")
                links.setdefault(rule, Counter())[f"{_word(w)} ← {why[4:].rsplit(' (', 1)[0]}"] += 1
    if links:
        parts.append("<details><summary>Words counted as known through a related known word: "
                     + ", ".join(f"{k} {sum(c.values())}" for k, c in sorted(links.items())) + "</summary>")
        for rule, c in sorted(links.items()):
            parts.append(f"<p class='small'><b>{rule}</b> ({len(c)} words): "
                         + ", ".join(f"{html.escape(k)} ×{n}" for k, n in c.most_common(80)) + "</p>")
        parts.append("<p class='small'>Turn a rule off with <code>[known] derivations</code> in config.toml.</p></details>")
    for dp in plans:
        if dp.names:
            parts.append(f"<details><summary>{html.escape(dp.deck)}: {len(dp.names)} detected names</summary><p class='small'>"
                         + ", ".join(f"{html.escape(k)} <span class='f'>({html.escape(v)})</span>" for k, v in sorted(dp.names.items()))
                         + "</p></details>")
    # controls + table
    decks = [dp.deck for dp in plans]
    parts.append("<div class='controls'><select id='dec'><option value=''>all decisions</option>"
                 + "".join(f"<option>{d}</option>" for d in DECISIONS) + "</select><select id='deck'><option value=''>all decks</option>"
                 + "".join(f"<option>{html.escape(d)}</option>" for d in decks)
                 + "</select><input id='q' placeholder='search'><span id='n' class='small'></span></div>")
    cols = [("new", 1), ("old", 1), ("deck", 0), ("decision", 0), ("U", 1), ("target", 0), ("priority", 1),
            ("quality", 1), ("sentence", 0), ("unknown", 0), ("filtered / reason", 0)]
    parts.append("<table id='cards'><thead><tr>" + "".join(f"<th data-num='{n}'>{c}</th>" for c, n in cols) + "</tr></thead><tbody>")
    for dp in plans:
        for p in dp.cards:
            r = _row(p)
            filt = html.escape(r["filtered"])
            reason = html.escape(r["reason"])
            target = html.escape(r["target"]) + (f" <span class='small'>{html.escape(r['target_reading'])}</span>" if r["target_reading"] and r["target_reading"] != r["target"] else "")
            parts.append(
                f"<tr data-d='{p.decision}' data-k='{html.escape(p.deck)}'><td>{r['new_pos']}</td><td>{r['old_pos']}</td>"
                f"<td class='small'>{html.escape(p.deck)}</td><td class='d-{p.decision}'>{p.decision}{' (susp.)' if p.suspended else ''}</td>"
                f"<td>{r['U']}</td><td>{target}</td><td>{r['priority']}</td><td>{r['quality']}</td>"
                f"<td>{_sentence_html(p)}</td><td>{html.escape(r['unknown'])}</td>"
                f"<td class='small'>{reason}{'<br>' if reason and filt else ''}{filt}</td></tr>")
    parts.append("</tbody></table>")
    parts.append(f"<script>{_JS}</script></body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def summary_text(plans: list[DeckPlan]) -> str:
    lines = []
    w = max([len(dp.deck) for dp in plans] + [5])
    lines.append(f"{'deck':{w}}  {'new':>6}  " + "  ".join(f"{d:>11}" for d in DECISIONS))
    total = Counter()
    for dp in plans:
        c = dp.counts()
        total.update(c)
        lines.append(f"{dp.deck:{w}}  {sum(c.values()):>6}  " + "  ".join(f"{c.get(d, 0):>11}" for d in DECISIONS))
    if len(plans) > 1:
        lines.append(f"{'total':{w}}  {sum(total.values()):>6}  " + "  ".join(f"{total.get(d, 0):>11}" for d in DECISIONS))
    return "\n".join(lines)
