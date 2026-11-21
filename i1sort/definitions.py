"""Definitions from a Yomitan-format dictionary (Jitendex), as Yomitan writes them to Anki.

The output matches Yomitan's ``{glossary-no-dictionary}`` Anki field as the browser
extension renders it with the "group" result mode: the dictionary entry's tags in
``<i>(…)</i>``, its structured content as HTML, and the dictionary's stylesheet scoped to
the entry. The entry is picked by JMdict sequence number (Jitendex is "sequenced"), so the
definition is the one of the exact JMdict entry i1sort resolved the word to.

The rendering follows Yomitan's source (ext/js/display/structured-content-generator.js,
ext/js/templates/anki-template-renderer.js, ext/data/templates/default-anki-field-templates.handlebars):

* structured content becomes elements with ``data-sc-*`` attributes (attributes in the order
  Yomitan sets them; dataset names the DOM rejects, like ``data-sc-sentence-key``, are
  dropped as in the browser), ``lang``, ``title`` and inline styles;
* Yomitan's own class styles (tables, link icons) become inline styles, then the classes
  are removed;
* the dictionary's styles.css is scoped to ``.yomitan-glossary [data-dictionary="…"]`` the
  way Chromium serializes it, plus Yomitan's compact-glossary styles when ``compact`` is on
  (glossary layout "compact-popup-anki").

Images (0.1% of Jitendex entries: plants, rare kanji glyphs) are left out: Yomitan would
copy the image file into Anki's media folder. The rows are read from the dictionary zip
once into ``data/definitions.sqlite``.
"""

from __future__ import annotations

import json
import re
import sqlite3
import zipfile
import zlib
from pathlib import Path

# --------------------------------------------------------------------------- index

SCHEMA = 1


class DefinitionDict:
    """A Yomitan term dictionary zip, indexed by JMdict sequence number."""

    def __init__(self, zip_path: Path, data_dir: Path, compact: bool = True, log=None):
        self.zip_path = Path(zip_path)
        self.compact = compact
        self.log = log or (lambda m: None)
        self.db_path = Path(data_dir) / "definitions.sqlite"
        self.con = self._open()
        meta = dict(self.con.execute("SELECT key, value FROM meta"))
        self.title: str = meta["title"]
        self.tags: dict[str, dict] = json.loads(meta["tags"])
        self.styles: str = meta["styles"]
        self._css: str | None = None

    def _stamp(self) -> str:
        st = self.zip_path.stat()
        return json.dumps([SCHEMA, str(self.zip_path.resolve()), st.st_size, int(st.st_mtime)])

    def _open(self) -> sqlite3.Connection:
        if not self.zip_path.exists():
            raise FileNotFoundError(f"definition dictionary not found: {self.zip_path}")
        stamp = self._stamp()
        if self.db_path.exists():
            con = sqlite3.connect(self.db_path)
            try:
                row = con.execute("SELECT value FROM meta WHERE key='stamp'").fetchone()
                if row and row[0] == stamp:
                    return con
            except sqlite3.DatabaseError:
                pass
            con.close()
            self.db_path.unlink()
        self.log(f"indexing {self.zip_path.name} (once) ...")
        tmp = self.db_path.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
        con = sqlite3.connect(tmp)
        con.executescript("""
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE entry(seq INTEGER, term TEXT, reading TEXT, tags TEXT, glossary BLOB);
        """)
        n = 0
        with zipfile.ZipFile(self.zip_path) as z:
            names = set(z.namelist())
            index = json.loads(z.read("index.json"))
            tags: dict[str, dict] = {}
            for name in sorted(x for x in names if re.fullmatch(r"tag_bank_\d+\.json", x)):
                for t in json.loads(z.read(name)):
                    tags[t[0]] = {"category": t[1], "order": t[2], "notes": t[3], "score": t[4]}
            styles = z.read("styles.css").decode("utf-8") if "styles.css" in names else ""
            for name in sorted(x for x in names if re.fullmatch(r"term_bank_\d+\.json", x)):
                rows = []
                for r in json.loads(z.read(name)):
                    seq = r[6] if len(r) > 6 else 0
                    if not isinstance(seq, int) or seq <= 0:
                        continue  # redirects (old kanji forms) have no JMdict entry
                    rows.append((seq, r[0], r[1], r[2] or "",
                                 zlib.compress(json.dumps(r[5], ensure_ascii=False).encode("utf-8"))))
                con.executemany("INSERT INTO entry VALUES (?,?,?,?,?)", rows)
                n += len(rows)
        con.executemany("INSERT INTO meta VALUES (?,?)", [
            ("stamp", stamp), ("title", index.get("title", "")), ("tags", json.dumps(tags, ensure_ascii=False)),
            ("styles", styles),
        ])
        con.execute("CREATE INDEX entry_seq ON entry(seq)")
        con.execute("CREATE INDEX entry_term ON entry(term)")
        con.commit()
        con.close()
        tmp.replace(self.db_path)
        self.log(f"  {n} entries from {index.get('title')}")
        return sqlite3.connect(self.db_path)

    # ----------------------------------------------------------------------- lookup
    def _rows(self, sql: str, args) -> list[tuple[int, str, str, str, list]]:
        return [(s, t, r, tg, json.loads(zlib.decompress(g))) for s, t, r, tg, g in self.con.execute(sql, args)]

    def entry(self, seq: int, term: str = "", reading: str = ""):
        """The row for this JMdict entry, preferring the given spelling and reading."""
        rows = self._rows("SELECT seq, term, reading, tags, glossary FROM entry WHERE seq=?", (seq,))
        if not rows:
            return None
        from .textutil import kata_to_hira

        def rank(r):
            return (r[1] != term, kata_to_hira(r[2] or r[1]) != kata_to_hira(reading or ""))
        return min(rows, key=rank)

    def html(self, seq: int | None, term: str = "", reading: str = "") -> str:
        """The definition as Yomitan's {glossary-no-dictionary}, or "" if the entry is missing."""
        if not seq:
            return ""
        row = self.entry(seq, term, reading)
        if row is None:
            return ""
        return self.render(row[3], row[4])

    def is_ours(self, html: str) -> bool:
        """A definition from this dictionary (any version) in the format i1sort writes."""
        base = _hb_escape(self.title.split(" [")[0])
        return html.startswith('<div style="text-align: left;" class="yomitan-glossary">') \
            and f'<li data-dictionary="{base}' in html

    # ----------------------------------------------------------------------- rendering
    def render(self, tag_text: str, glossary: list) -> str:
        title = _hb_escape(self.title)
        tags = self._tags(tag_text)
        head = f"<i>({', '.join(_hb_escape(t) for t in tags)})</i> " if tags else ""
        items = [_format_glossary(g) for g in glossary]
        body = "".join(items) if len(items) <= 1 else "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"
        css = self.css()
        style = f"<style>{css}</style>" if css else ""
        return (f'<div style="text-align: left;" class="yomitan-glossary"><ol><li data-dictionary="{title}">'
                f"{head}{body}</li>{style}</ol></div>")

    def _tags(self, tag_text: str) -> list[str]:
        out: dict[str, int] = {}
        for name in tag_text.split(" "):  # names may contain no-break spaces (priority\xa0form)
            if not name:
                continue
            if name not in out:
                out[name] = int((self.tags.get(name) or {}).get("order", 0) or 0)
        return [n for n, _ in sorted(out.items(), key=lambda x: (x[1], _collate_key(x[0])))]

    def css(self) -> str:
        if self._css is None:
            parts = []
            if self.styles:
                parts.append(scope_css(scope_css(self.styles, f'[data-dictionary="{_css_attr_escape(self.title)}"]'),
                                       ".yomitan-glossary"))
            if self.compact:
                parts.append(scope_css(ANKI_COMPACT_GLOSS_STYLES, ".yomitan-glossary"))
            self._css = "".join(parts)
        return self._css


def open_definitions(cfg, log=None) -> DefinitionDict | None:
    dc = cfg.get("definitions") or {}
    path = dc.get("dictionary")
    if not path:
        return None
    return DefinitionDict(cfg.resolve(path), cfg.data_dir, bool(dc.get("compact", True)), log=log)


def _hb_escape(s: str) -> str:
    """Handlebars' escapeExpression."""
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
            .replace("'", "&#x27;").replace("`", "&#x60;").replace("=", "&#x3D;"))


def _collate_key(s: str):
    return s.casefold()


def _css_attr_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


# --------------------------------------------------------------------------- structured content

_VOID = {"br", "img", "hr", "wbr"}
# Yomitan's structured-content-style.json rules that apply without images, in file order.
_CLASS_RULES = [
    ({"gloss-link-external-icon"}, "display:none;"),
    ({"gloss-sc-table-container"}, "display:block;"),
    ({"gloss-sc-table"}, "table-layout:auto;border-collapse:collapse;"),
    ({"gloss-sc-thead", "gloss-sc-tfoot", "gloss-sc-th"}, "font-weight:bold;"),
    ({"gloss-sc-th", "gloss-sc-td"},
     "border-style:solid;padding:0.25em;vertical-align:top;border-width:1px;border-color:currentColor;"),
]
_STYLE_PROPS = [  # structured-content style keys in the order Yomitan applies them → CSS property
    ("fontStyle", "font-style"), ("fontWeight", "font-weight"), ("fontSize", "font-size"), ("color", "color"),
    ("background", "background"), ("backgroundColor", "background-color"), ("verticalAlign", "vertical-align"),
    ("textAlign", "text-align"), ("textEmphasis", "text-emphasis"), ("textShadow", "text-shadow"),
    ("textDecorationLine", "text-decoration"), ("textDecorationStyle", "text-decoration-style"),
    ("textDecorationColor", "text-decoration-color"), ("borderColor", "border-color"),
    ("borderStyle", "border-style"), ("borderRadius", "border-radius"), ("borderWidth", "border-width"),
    ("clipPath", "clip-path"), ("margin", "margin"), ("marginTop", "margin-top"), ("marginLeft", "margin-left"),
    ("marginRight", "margin-right"), ("marginBottom", "margin-bottom"), ("padding", "padding"),
    ("paddingTop", "padding-top"), ("paddingLeft", "padding-left"), ("paddingRight", "padding-right"),
    ("paddingBottom", "padding-bottom"), ("wordBreak", "word-break"), ("whiteSpace", "white-space"),
    ("cursor", "cursor"), ("listStyleType", "list-style-type"),
]
_EM_NUMBER = {"marginTop", "marginLeft", "marginRight", "marginBottom"}


class _El:
    __slots__ = ("tag", "attrs", "style", "children")

    def __init__(self, tag: str, cls: str):
        self.tag = tag
        self.attrs: dict[str, str] = {"class": cls}
        self.style: list[tuple[str, str]] = []  # inline style set from the content
        self.children: list = []

    def set(self, name: str, value: str) -> None:
        self.attrs[name] = value  # a new attribute goes last, an existing one keeps its place

    def set_style(self, prop: str, value: str) -> None:
        if "style" not in self.attrs:
            self.attrs["style"] = ""
        self.style = [(p, v) for p, v in self.style if p != prop] + [(prop, value)]


def _is_japanese(text: str) -> bool:
    for ch in text:
        o = ord(ch)
        if (0x3040 <= o <= 0x30ff or 0x4e00 <= o <= 0x9fff or 0x3400 <= o <= 0x4dbf or 0x20000 <= o <= 0x3347f
                or 0xf900 <= o <= 0xfaff or 0x2f800 <= o <= 0x2fa1f or 0xff66 <= o <= 0xff9f or 0xff61 <= o <= 0xff65
                or 0x3000 <= o <= 0x303f or 0xff10 <= o <= 0xff19 or 0xff21 <= o <= 0xff3a or 0xff41 <= o <= 0xff5a
                or 0xff01 <= o <= 0xff0f or 0xff1a <= o <= 0xff1f or 0xff3b <= o <= 0xff3f or 0xff5b <= o <= 0xff60
                or 0xffe0 <= o <= 0xffee):
            return True
    return False


def _set_dataset(el: _El, data: dict) -> None:
    for key, value in data.items():
        key = str(key)
        if key:
            key = key[0].upper() + key[1:]
        key = "sc" + key
        if re.search(r"-[a-z]", key):
            continue  # the DOM rejects it (data-sc-sentence-key), as in the browser
        name = "data-" + re.sub(r"[A-Z]", lambda m: "-" + m.group(0).lower(), key)
        el.set(name, _js_string(value))


def _js_string(v) -> str:
    if v is True:
        return "true"
    if v is False:
        return "false"
    if v is None:
        return "null"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def _append(container: _El, content, language: str | None) -> None:
    if isinstance(content, str):
        if content:
            container.children.append(content)
            if language is None and _is_japanese(content):
                container.set("lang", "ja")
        return
    if isinstance(content, list):
        for item in content:
            _append(container, item, language)
        return
    if isinstance(content, dict):
        node = _element(content, language)
        if node is not None:
            container.children.append(node)


def _element(c: dict, language: str | None):
    tag = c.get("tag")
    if tag == "br":
        return _simple(tag, c, language, children=False, style=False)
    if tag in ("ruby", "rt", "rp"):
        return _simple(tag, c, language, children=True, style=False)
    if tag == "table":
        box = _El("div", "gloss-sc-table-container")
        box.children.append(_simple(tag, c, language, children=True, style=False))
        return box
    if tag in ("thead", "tbody", "tfoot", "tr"):
        return _simple(tag, c, language, children=True, style=False)
    if tag in ("th", "td"):
        return _simple(tag, c, language, children=True, style=True, cell=True)
    if tag in ("div", "span", "ol", "ul", "li", "details", "summary"):
        return _simple(tag, c, language, children=True, style=True)
    if tag == "a":
        return _link(c, language)
    return None  # img (needs the media file in Anki) and unknown tags


def _simple(tag: str, c: dict, language, children: bool, style: bool, cell: bool = False) -> _El:
    el = _El(tag, f"gloss-sc-{tag}")
    data, lang = c.get("data"), c.get("lang")
    if isinstance(data, dict):
        _set_dataset(el, data)
    if isinstance(lang, str):
        el.set("lang", lang)
        language = lang
    if cell:
        if isinstance(c.get("colSpan"), (int, float)):
            el.set("colspan", _js_string(c["colSpan"]))
        if isinstance(c.get("rowSpan"), (int, float)):
            el.set("rowspan", _js_string(c["rowSpan"]))
    if style:
        st, title, opened = c.get("style"), c.get("title"), c.get("open")
        if isinstance(st, dict):
            for key, prop in _STYLE_PROPS:
                v = st.get(key)
                if key == "textDecorationLine" and isinstance(v, list):
                    v = " ".join(v)
                if key in _EM_NUMBER and isinstance(v, (int, float)) and not isinstance(v, bool):
                    v = f"{_js_string(v)}em"
                if isinstance(v, str):
                    el.set_style(prop, v)
        if isinstance(title, str):
            el.set("title", title)
        if opened is True:
            el.set("open", "")
    if children:
        _append(el, c.get("content"), language)
    return el


def _link(c: dict, language) -> _El:
    href = c.get("href") or ""
    internal = href.startswith("?")
    node = _El("a", "gloss-link")
    node.set("data-external", "false" if internal else "true")
    text = _El("span", "gloss-link-text")
    node.children.append(text)
    if isinstance(c.get("lang"), str):
        node.set("lang", c["lang"])
        language = c["lang"]
    _append(text, c.get("content"), language)
    if not internal:
        icon = _El("span", "gloss-link-external-icon icon")
        icon.set("data-icon", "external-link")
        node.children.append(icon)
    node.set("href", "#" if internal else href)
    return node


def _normalize(el: _El) -> None:
    """Class styles → inline, classes and non-sc dataset entries removed (Yomitan _normalizeHtml)."""
    cls = el.attrs.get("class") or ""
    classes = set(cls.split())
    text = "".join(css for names, css in _CLASS_RULES if classes & names)
    text += "".join(f"{p}: {v}; " for p, v in el.style).rstrip(" ")
    el.attrs.pop("class", None)
    if text:
        el.attrs["style"] = text
    else:
        el.attrs.pop("style", None)
    for name in [n for n in el.attrs if n.startswith("data-") and not re.match(r"data-sc(-|$)", n)]:
        del el.attrs[name]
    for ch in el.children:
        if isinstance(ch, _El):
            _normalize(ch)


def _esc_text(s: str) -> str:
    return s.replace("&", "&amp;").replace("\xa0", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;")


def _esc_attr(s: str) -> str:
    return (s.replace("&", "&amp;").replace("\xa0", "&nbsp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _html(node) -> str:
    if isinstance(node, str):
        return "<br>".join(_esc_text(p) for p in node.split("\n"))
    attrs = "".join(f' {k}="{_esc_attr(v)}"' for k, v in node.attrs.items())
    if node.tag in _VOID:
        return f"<{node.tag}{attrs}>"
    return f"<{node.tag}{attrs}>{''.join(_html(c) for c in node.children)}</{node.tag}>"


def render_structured_content(content) -> str:
    root = _El("span", "structured-content")
    _append(root, content, None)
    _normalize(root)
    return _html(root)


def _format_glossary(item) -> str:
    if isinstance(item, str):
        return "<br>".join(_hb_escape(p) for p in item.split("\n"))
    if isinstance(item, dict):
        if item.get("type") == "structured-content":
            return render_structured_content(item.get("content"))
        if item.get("type") == "text":
            return "<br>".join(_hb_escape(p) for p in str(item.get("text", "")).split("\n"))
    return ""


# --------------------------------------------------------------------------- CSS

# Yomitan ext/data/anki-compact-gloss-style.js
ANKI_COMPACT_GLOSS_STYLES = """ul[data-sc-content="glossary"] > li:not(:first-child)::before {
  white-space: pre-wrap;
  content: ' | ';
  display: inline;
  color: #777777;
}

ul[data-sc-content="glossary"] > li {
  display: inline;
}

ul[data-sc-content="glossary"] {
  display: inline;
  list-style: none;
  padding-left: 0;
}"""


def scope_css(css: str, scope: str) -> str:
    """Yomitan's addScopeToCssLegacy as Chromium runs it: every top-level style rule's
    selectors get ``scope`` in front, other rules (@media, @font-face) are dropped, and each
    rule is written back the way Chromium serializes it (one rule per line)."""
    out = []
    for sel, body in _rules(_strip_comments(css)):
        if sel.startswith("@"):
            continue
        sels = ", ".join(f"{scope} {s}" for s in _split_top(sel, ","))
        out.append(_serialize_rule(sels, body))
    return "\n".join(out)


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _rules(css: str) -> list[tuple[str, str]]:
    """Top-level (selector, body) pairs."""
    out, i, n = [], 0, len(css)
    while i < n:
        j = _find_top(css, "{", i)
        if j < 0:
            break
        sel = _ws(css[i:j])
        k = _match_brace(css, j)
        out.append((sel, css[j + 1:k]))
        i = k + 1
    return out


def _find_top(s: str, ch: str, i: int) -> int:
    quote = None
    while i < len(s):
        c = s[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == ch:
            return i
        i += 1
    return -1


def _match_brace(s: str, j: int) -> int:
    depth, i, quote = 0, j, None
    while i < len(s):
        c = s[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(s)


def _split_top(s: str, sep: str) -> list[str]:
    parts, depth, cur, quote = [], 0, "", None
    for c in s:
        if quote:
            cur += c
            if c == quote:
                quote = None
            continue
        if c in "\"'":
            quote = c
        elif c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == sep and depth == 0:
            parts.append(cur.strip())
            cur = ""
            continue
        cur += c
    if cur.strip():
        parts.append(cur.strip())
    return parts


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _body(body: str) -> tuple[list[str], list[str]]:
    """Declarations ("prop: value;") and nested rules (serialized) of a rule body."""
    decls, nested, i, n = [], [], 0, len(body)
    while i < n:
        semi, brace = _find_top(body, ";", i), _find_top(body, "{", i)
        if brace >= 0 and (semi < 0 or brace < semi):
            sel = _ws(body[i:brace])
            k = _match_brace(body, brace)
            nested.append(_serialize_rule(sel, body[brace + 1:k]))
            i = k + 1
            continue
        end = semi if semi >= 0 else n
        decl = body[i:end].strip()
        if decl and ":" in decl:
            prop, _, value = decl.partition(":")
            prop = prop.strip().lower()
            decls.append(f"{prop}: {_value(prop, _ws(value))};")
        i = end + 1
    return decls, nested


def _serialize_rule(sel: str, body: str) -> str:
    decls, nested = _body(body)
    if not nested:
        return f"{sel} {{ {' '.join(decls)} }}" if decls else f"{sel} {{ }}"
    # Chromium indents a nested rule's first line only: its own lines keep their indentation
    lines = ([" ".join(decls)] if decls else []) + nested
    return f"{sel} {{\n" + "".join(f"  {x}\n" for x in lines) + "}"


_LENGTH_PROPS = re.compile(r"^(margin|padding|border(-[a-z]+)?-width|border-width|width|height|min-|max-|top|left|right|bottom"
                           r"|font-size|line-height|gap|row-gap|column-gap|letter-spacing|word-spacing|text-indent|outline-width)")


def _value(prop: str, v: str) -> str:
    """Chromium's computed serialization for the values Jitendex uses."""
    v = re.sub(r"'([^'\"]*)'", r'"\1"', v)  # strings take double quotes
    if "var(" not in v:
        v = _color_mix(v)
        v = re.sub(r"#([0-9a-fA-F]{3,8})\b", lambda m: _hex(m.group(1)), v)
    if _LENGTH_PROPS.match(prop) and "var(" not in v and "calc(" not in v:
        v = " ".join("0px" if t == "0" else t for t in v.split(" "))
    return v


def _hex(h: str) -> str:
    if len(h) in (3, 4):
        h = "".join(c * 2 for c in h)
    if len(h) not in (6, 8):
        return "#" + h
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    if len(h) == 8:
        a = int(h[6:8], 16) / 255
        return f"rgba({r}, {g}, {b}, {round(a, 2):g})"
    return f"rgb({r}, {g}, {b})"


def _color_mix(v: str) -> str:
    """color-mix(in srgb, A 5%, B) → color-mix(in srgb, A 5%, B 95%)."""
    def fix(m):
        args = _split_top(m.group(1), ",")
        if len(args) != 3:
            return m.group(0)
        space, a, b = args
        pa = re.search(r"\s(\d+(?:\.\d+)?)%$", a)
        pb = re.search(r"\s(\d+(?:\.\d+)?)%$", b)
        if pa and not pb:
            b = f"{b} {_js_string(100 - float(pa.group(1)))}%"
        elif pb and not pa:
            a = f"{a} {_js_string(100 - float(pb.group(1)))}%"
        return f"color-mix({space}, {a}, {b})"
    return re.sub(r"color-mix\(((?:[^()]|\([^()]*\))*)\)", fix, v)
