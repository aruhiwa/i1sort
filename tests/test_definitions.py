"""Definitions rendered like Yomitan's {glossary-no-dictionary}, and when apply writes them."""

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from i1sort.apply import _KEEP, definition_change
from i1sort.definitions import DefinitionDict, render_structured_content, scope_css

HERE = Path(__file__).parent
TITLE = "Test.dict [2026-01-01]"


def make_dict(tmp_path, entries, styles="span[title] { cursor: help; }", tags=None) -> Path:
    z = tmp_path / "dict.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("index.json", json.dumps({"title": TITLE, "format": 3, "sequenced": True}))
        f.writestr("tag_bank_1.json", json.dumps(tags or [["★", "popular", 2, "high priority entry", 2],
                                                          ["priority\xa0form", "frequent", 1, "", 1]]))
        f.writestr("styles.css", styles)
        f.writestr("term_bank_1.json", json.dumps(entries, ensure_ascii=False))
    return z


def sc(*content):
    return [{"type": "structured-content", "content": list(content)}]


def test_structured_content_like_the_browser():
    html = render_structured_content([
        {"tag": "span", "title": "noun", "data": {"class": "tag", "code": "n", "content": "part-of-speech-info"},
         "content": "noun"},
        {"tag": "ol", "content": {"tag": "li", "style": {"listStyleType": "\"①\""}, "data": {"content": "sense"},
                                  "content": {"tag": "ul", "data": {"content": "glossary"},
                                              "content": [{"tag": "li", "content": "cat & dog"}]}}},
        # sentence-key / source-type are not valid dataset names: the browser drops them
        {"tag": "div", "data": {"class": "extra-box", "content": "example-sentence", "sentence-key": "猫", "source": 7,
                                "source-type": "tat"},
         "content": [{"tag": "span", "lang": "ja", "content": [{"tag": "ruby", "content": ["猫", {"tag": "rt", "content": "ねこ"}]}]},
                     {"tag": "span", "content": "a\nb"}]},
        {"tag": "table", "content": {"tag": "tr", "content": [{"tag": "th", "content": "x"},
                                                              {"tag": "td", "rowSpan": 2, "data": {"class": "form-pri"},
                                                               "content": "y"}]}},
        {"tag": "a", "href": "?query=%E7%8C%AB", "content": "猫"},
        {"tag": "a", "href": "https://example.org/?a=1&b=2", "content": "src"},
        {"tag": "div", "content": "日本語"},
        {"tag": "img", "path": "x.svg"},
    ])
    th = "font-weight:bold;border-style:solid;padding:0.25em;vertical-align:top;border-width:1px;border-color:currentColor;"
    td = "border-style:solid;padding:0.25em;vertical-align:top;border-width:1px;border-color:currentColor;"
    assert html == (
        '<span><span data-sc-class="tag" data-sc-code="n" data-sc-content="part-of-speech-info" title="noun">noun</span>'
        '<ol><li data-sc-content="sense" style="list-style-type: &quot;①&quot;;"><ul data-sc-content="glossary">'
        '<li>cat &amp; dog</li></ul></li></ol>'
        '<div data-sc-class="extra-box" data-sc-content="example-sentence" data-sc-source="7">'
        '<span lang="ja"><ruby>猫<rt>ねこ</rt></ruby></span><span>a<br>b</span></div>'
        f'<div style="display:block;"><table style="table-layout:auto;border-collapse:collapse;"><tr>'
        f'<th style="{th}">x</th><td data-sc-class="form-pri" rowspan="2" style="{td}">y</td></tr></table></div>'
        '<a href="#"><span lang="ja">猫</span></a>'
        '<a href="https://example.org/?a=1&amp;b=2"><span>src</span><span style="display:none;"></span></a>'
        '<div lang="ja">日本語</div></span>')


def test_entry_with_tags_and_scoped_styles(tmp_path):
    entries = [["猫", "ねこ", "★ priority\xa0form", "", 1, sc({"tag": "span", "content": "cat"}), 101, ""],
               ["猫", "ねこ", "", "", 1, ["猫", ["redirected from 貓"]], -5, ""]]
    d = DefinitionDict(make_dict(tmp_path, entries), tmp_path, compact=False)
    assert d.html(101, "猫", "ねこ") == (
        '<div style="text-align: left;" class="yomitan-glossary"><ol><li data-dictionary="Test.dict [2026-01-01]">'
        '<i>(priority\xa0form, ★)</i> <span><span>cat</span></span></li>'
        '<style>.yomitan-glossary [data-dictionary="Test.dict [2026-01-01]"] span[title] { cursor: help; }</style>'
        '</ol></div>')
    assert d.html(999) == "" and d.html(-5) == ""  # missing entry; redirects are not indexed
    assert d.is_ours(d.html(101, "猫", "ねこ"))
    assert d.is_ours(d.html(101).replace("2026-01-01", "2025-06-01"))  # another version of the dictionary
    assert not d.is_ours("<b>my note</b>")


def test_css_like_chromium():
    css = """
/* comment */
span[data-sc-content="pos"] { background-color: #565656; margin-right: 0; padding: 0 0.5em; }
div[data-sc-content="xref"], li {
    border-color: #1A73E8; background-color: color-mix(in srgb, #1A73E8 5%, transparent);
    color: var(--fg, #333);
    & span { content: '古'; }
    & td {
        text-align: center;
        & span { display: block; }
    }
}
@media (max-width: 10px) { span { color: red; } }
"""
    s = scope_css(css, ".g")
    assert s == (
        '.g span[data-sc-content="pos"] { background-color: rgb(86, 86, 86); margin-right: 0px; padding: 0px 0.5em; }\n'
        '.g div[data-sc-content="xref"], .g li {\n'
        '  border-color: rgb(26, 115, 232); background-color: color-mix(in srgb, rgb(26, 115, 232) 5%, transparent 95%); '
        'color: var(--fg, #333);\n'
        '  & span { content: "古"; }\n'
        '  & td {\n'
        '  text-align: center;\n'
        '  & span { display: block; }\n'
        '}\n'
        '}')


def test_real_jitendex_matches_yomitan_cards():
    """Definitions Yomitan (browser extension) wrote to Anki cards, byte for byte."""
    from i1sort.config import load_config

    cfg = load_config()
    path = cfg.resolve((cfg.get("definitions") or {}).get("dictionary") or "dict/jitendex-yomitan.zip")
    if not path.exists():
        pytest.skip("Jitendex not available")
    d = DefinitionDict(path, cfg.data_dir, compact=True)
    cards = json.loads((HERE / "data" / "yomitan_jitendex_cards.json").read_text(encoding="utf-8"))
    for c in cards:
        if f'data-dictionary="{d.title}"' not in c["html"]:
            pytest.skip(f"cards were made with another Jitendex version than {d.title}")
        assert d.html(c["seq"], c["word"], c["reading"]) == c["html"], c["word"]


class FakeDefs:
    def html(self, seq, term, reading):
        return f'<div style="text-align: left;" class="yomitan-glossary">{term}</div>' if seq else ""

    def is_ours(self, html):
        return html.startswith('<div style="text-align: left;" class="yomitan-glossary">')


def plan(decision, cur, prev_target="猫", target="犬"):
    t = SimpleNamespace(seq=1, headword=target, lemma=target) if decision in ("i1", "same-target") else None
    fields = {"definition": cur}
    if prev_target is not None:
        fields["i1sort_target"] = prev_target
    return SimpleNamespace(definition_field="definition", fields=fields, target=t, decision=decision, target_reading="")


def test_definition_overwrite_policy():
    d = FakeDefs()
    ours = d.html(1, "猫", "")
    assert definition_change(plan("i1", ""), d) == d.html(1, "犬", "")  # empty: filled
    assert definition_change(plan("i1", d.html(1, "犬", "")), d) is None  # already right
    assert definition_change(plan("i1", ours), d) == d.html(1, "犬", "")  # i1sort's old target: replaced
    assert definition_change(plan("not-ready", ours), d) == ""  # no target any more: cleared
    assert definition_change(plan("i1", "<b>my notes</b>"), d) is _KEEP  # your text stays
    assert definition_change(plan("known", "<b>my notes</b>"), d) is _KEEP
    assert definition_change(plan("i1", ours, prev_target=""), d) is _KEEP  # i1sort never gave it a target
    assert definition_change(plan("known", ""), d) is None


def test_frequency_overwrite_policy():
    from i1sort.apply import frequency_change

    def fp(decision, cur, prev="猫"):
        t = SimpleNamespace(seq=1) if decision in ("i1", "same-target") else None
        return SimpleNamespace(frequency_field="word frequency", fields={"word frequency": cur, "i1sort_target": prev},
                               target=t, decision=decision, target_frequency="2365")

    assert frequency_change(fp("i1", "")) == "2365"
    assert frequency_change(fp("i1", "2365")) is None
    assert frequency_change(fp("same-target", "4000")) == "2365"  # i1sort's number for an older target
    assert frequency_change(fp("known", "4000")) == ""  # no target any more
    assert frequency_change(fp("i1", "rare")) is _KEEP  # your text
    assert frequency_change(fp("i1", "4000", prev="")) is _KEEP  # i1sort never gave the note a target


def test_target_frequency_is_the_harmonic_rank():
    from i1sort.config import load_config

    cfg = load_config(overrides={"sort": {"fields": {"M": {"sentence": "S", "audio": "", "frequency": "freq"}},
                                          "min_words": 1},
                                 "known": {"files": [], "exact_files": []}})
    if not cfg.resolve(cfg["dictionaries"]["jmdict"]).exists():
        pytest.skip("JMdict not available")
    import math

    from i1sort.filters import TokenFilter
    from i1sort.known import KnownSet
    from i1sort.parse.pipeline import Parser
    from i1sort.score import Planner

    parser = Parser(cfg, use_cache=False)
    ks = KnownSet()
    for w in parser.parse("私は", TokenFilter(cfg, parser.db, ignore=())).counted:
        ks.add(w.key, w.lemma, w.reading, "known", "t")
    notes = {1: {"noteId": 1, "modelName": "M", "tags": [], "fields": {"S": {"value": "私は極めて"}, "freq": {"value": ""}}}}
    dp = Planner(cfg, parser, ks).plan_deck("D", [{"cardId": 10, "note": 1, "due": 1, "queue": 0}], notes)
    p = dp.cards[0]
    assert p.frequency_field == "freq" and "freq" in p.fields
    assert p.target.headword == "極めて"
    assert p.target_frequency == str(math.floor(parser.db.freq_rank("極めて", "きわめて")))


def test_kana_counts_only_for_words_usually_in_kana():
    from i1sort.dictdb import _pick_freq

    uk = {("揶揄う", "からかう")}
    assert _pick_freq("逝く", "いく", 9328, 65, uk) == (9328, False)  # いく in kana is 行く
    assert _pick_freq("逝く", "いく", None, 65, uk) is None
    assert _pick_freq("揶揄う", "からかう", 35016, 2175, uk) == (2175, True)  # からかう: written in kana
    assert _pick_freq("てっきり", "てっきり", None, 3644, uk) == (3644, True)  # a kana spelling
    assert _pick_freq("極めて", "きわめて", 2878, None, uk) == (2878, False)


def test_real_frequency_ranks():
    from i1sort.config import load_config
    from i1sort.dictdb import open_dictdb

    cfg = load_config()
    if not cfg.resolve(cfg["dictionaries"]["jmdict"]).exists():
        pytest.skip("JMdict not available")
    db = open_dictdb(cfg)
    assert db.freq_rank("逝く", "いく") > 5000  # not 行く's kana count (65)
    assert db.freq_rank("揶揄う", "からかう") < 3000  # usually written からかう: the kana count is its own
    assert db.freq_rank("神", "み") > 100_000  # a value without a reading goes to 神【かみ】 only
    assert db.freq_rank("神", "かみ") < 1000
