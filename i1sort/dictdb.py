"""JMdict / JMnedict / frequency data, built once into SQLite from Yomitan zips.

Sources (paths in config):
  * JMdict in Yomitan format (``JMdict_english.zip`` from jmdict-yomitan, or
    Jitendex). Each term row is one (spelling, reading) pair of a JMdict entry; its
    sequence number is the JMdict entry ID used as the word key everywhere.
  * JMnedict in Yomitan format (names).
  * Yomitan frequency dictionaries (``term_meta_bank_*.json``).

``i1sort build-data`` (or the first use) writes ``<data_dir>/dict.sqlite``. The
build is keyed by the source files' size+mtime, so replacing a zip rebuilds it.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
import time
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import Config
from .textutil import reading_key

SCHEMA_VERSION = 6

POS_EXTRA = {"exp"}  # Yomitan files 'exp' under the 'expression' category; we treat it as POS.


# --------------------------------------------------------------------------- types


@dataclass(frozen=True)
class Sense:
    pos: frozenset[str]
    misc: frozenset[str]
    gloss: str


@dataclass(frozen=True)
class Entry:
    seq: int
    headword: str
    reading: str
    pos: frozenset[str]
    misc: frozenset[str]
    common: bool
    score: int
    senses: tuple[Sense, ...]
    lsource: tuple[str, ...] = ()  # JMdict source languages ("German", "English (wasei)"); empty = unspecified

    @property
    def gloss(self) -> str:
        return self.senses[0].gloss if self.senses else ""

    def has_pos(self, *tags: str) -> bool:
        return any(t in self.pos for t in tags)

    def pos_family(self) -> set[str]:
        return {pos_family(p) for p in self.pos}


@dataclass(frozen=True)
class FormRow:
    term: str
    reading: str
    seq: int
    common: bool
    rare: bool
    score: int


@dataclass(frozen=True)
class NameRow:
    term: str
    reading: str
    seq: int
    types: frozenset[str]


def pos_family(tag: str) -> str:
    """Collapse JMdict POS tags into coarse families comparable with Sudachi POS."""
    if tag.startswith("v") and tag not in ("vt", "vi"):
        if tag.startswith("vs"):
            return "vs" if tag == "vs" else "verb"
        return "verb"
    if tag in ("adj-i", "adj-ix", "adj-ku", "adj-shiku"):
        return "adj-i"
    if tag in ("adj-na", "adj-nari", "adj-t", "adj-tari", "adj-f"):
        return "adj-na"
    if tag == "adj-no":
        return "noun"
    if tag in ("n", "n-adv", "n-t", "n-pr", "num"):
        return "noun"
    if tag in ("adv", "adv-to"):
        return "adv"
    if tag in ("aux", "aux-v", "aux-adj", "cop"):
        return "aux"
    if tag in ("n-suf", "suf", "ctr"):
        return "suffix"
    if tag in ("n-pref", "pref"):
        return "prefix"
    return tag  # pn, adj-pn, conj, int, prt, exp, unc ...


# --------------------------------------------------------------------------- build helpers


def _iter_bank_rows(zpath: Path, prefix: str) -> Iterator[list]:
    with zipfile.ZipFile(zpath) as z:
        for name in sorted(z.namelist()):
            base = name.rsplit("/", 1)[-1]
            if base.startswith(prefix) and base.endswith(".json"):
                yield from json.loads(z.read(name))


def _read_index(zpath: Path) -> dict:
    with zipfile.ZipFile(zpath) as z:
        for name in z.namelist():
            if name.rsplit("/", 1)[-1] == "index.json":
                return json.loads(z.read(name))
    return {}


def _tag_categories(zpath: Path) -> dict[str, str]:
    cats: dict[str, str] = {}
    for row in _iter_bank_rows(zpath, "tag_bank"):
        cats[row[0]] = row[1]
    return cats


def _gloss_text(content: Any, limit: int = 3) -> str:
    """Pull the first few gloss strings out of Yomitan structured content."""
    out: list[str] = []

    def walk(node: Any, in_li: bool) -> None:
        if len(out) >= limit:
            return
        if isinstance(node, str):
            if in_li and node.strip():
                out.append(node.strip())
            return
        if isinstance(node, list):
            for n in node:
                walk(n, in_li)
            return
        if isinstance(node, dict):
            if node.get("type") == "image":
                return
            data = node.get("data") or {}
            if isinstance(data, dict) and data.get("content") in (
                "notes", "examples", "example-sentence", "references", "antonyms",
                "xref", "attribution", "infoGlossary", "formsTable", "sense-note", "sourceLanguages",
            ):
                return
            if node.get("tag") == "li":
                parts: list[str] = []

                def flat(n: Any) -> None:
                    if isinstance(n, str):
                        parts.append(n)
                    elif isinstance(n, list):
                        for x in n:
                            flat(x)
                    elif isinstance(n, dict):
                        flat(n.get("content"))

                flat(node.get("content"))
                t = "".join(parts).strip()
                if t:
                    out.append(t)
                return
            walk(node.get("content"), in_li)

    for item in content if isinstance(content, list) else [content]:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            out.append(item.get("text", ""))
        else:
            walk(item, False)
        if len(out) >= limit:
            break
    return "; ".join(out[:limit])


def _source_languages(glossary: Any) -> set[str]:
    """Language labels from JMdict lsource (Yomitan "sourceLanguages" list): "German", "English (wasei)".

    JMdict omits lsource for plain English loanwords whose gloss is the English word
    (コンピューター), so an empty set on a katakana word usually means English.
    """
    out: set[str] = set()

    def text(n: Any) -> str:
        if isinstance(n, str):
            return n
        if isinstance(n, list):
            return "".join(text(x) for x in n)
        if isinstance(n, dict):
            return text(n.get("content"))
        return ""

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            data = n.get("data") or {}
            if isinstance(data, dict) and data.get("content") == "sourceLanguages":
                items = n.get("content")
                for li in items if isinstance(items, list) else [items]:
                    t = text(li)
                    out.add(t.split(":", 1)[0].strip() if ":" in t else "English")
                return
            walk(n.get("content"))
        elif isinstance(n, list):
            for x in n:
                walk(x)

    walk(glossary)
    return out


def _redirect_target(glossary: Any) -> str | None:
    """Target of a yomitan-import redirect row: structured content "⟶ <a href=?query=X>"."""
    found: list[str] = []

    def walk(n: Any) -> None:
        if isinstance(n, dict):
            if n.get("tag") == "a" and isinstance(n.get("content"), str):
                found.append(n["content"])
            walk(n.get("content"))
        elif isinstance(n, list):
            for x in n:
                walk(x)

    walk(glossary)
    return found[0] if found else None


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _pick_freq(term: str, reading: str, plain: float | None, kana: float | None,
               uk_forms: set[tuple[str, str]]) -> tuple[float, bool] | None:
    """One value per dictionary for a (term, reading). A ㋕ value counts how often the word
    is written in kana: it is the frequency of a kana spelling, and of a kanji spelling only
    when the word is usually written in kana (揶揄う → からかう); for any other kanji spelling
    it would be the frequency of a different word read the same (逝く ← いく = 行く)."""
    if kana is None:
        return (plain, False) if plain is not None else None
    kanji = any("\u3400" <= c <= "\u9fff" or "\uf900" <= c <= "\ufaff" or c == "々" for c in term)
    if kanji and (term, reading) not in uk_forms and not (reading == "" and any(
            t == term for t, _ in uk_forms)):
        return (plain, False) if plain is not None else None
    if plain is None:
        return kana, True
    return (kana, True) if kana < plain else (plain, False)


def _freq_rows(i: int, vals: dict, uk_forms: set, term_only_rows: list):
    """freq rows of dictionary ``i``: one value per (term, reading), and per (term, "") the
    value given without a reading (YoutubeFreq) or else the best of the term's readings (for
    lookups by spelling alone). Values without a reading are also collected in
    ``term_only_rows``: they go to the term's main reading after all dictionaries are read."""
    readings: dict[str, list[str]] = {}
    for (t, r) in vals:
        if r:
            readings.setdefault(t, []).append(r)
    for (t, r), (plain, kana) in vals.items():
        if not r:
            continue
        got = _pick_freq(t, r, plain, kana, uk_forms)
        if got is not None:
            yield (i, t, r, got[0], int(got[1]))
    for t in {t for (t, r) in vals}:
        term_only = vals.get((t, ""))
        own = [_pick_freq(t, r, *vals[(t, r)], uk_forms) for r in readings.get(t, [])]
        own = [x for x in own if x is not None]
        if term_only is not None:
            got = _pick_freq(t, "", *term_only, uk_forms)
            if got is None:
                continue
            yield (i, t, "", got[0], int(got[1]))
            if not readings.get(t):
                term_only_rows.append((i, t, got[0], int(got[1])))
        elif own:
            best = min(own)
            yield (i, t, "", best[0], int(best[1]))


def parse_freq_value(data: Any) -> tuple[str, float | None, bool]:
    """Return (reading, value, kana_marked) from a Yomitan frequency datum."""
    reading = ""
    if isinstance(data, dict) and "frequency" in data:
        reading = data.get("reading") or ""
        data = data["frequency"]
    kana_marked = False
    if isinstance(data, (int, float)):
        return reading, float(data), False
    if isinstance(data, str):
        m = _NUM_RE.search(data)
        return reading, (float(m.group()) if m else None), "㋕" in data
    if isinstance(data, dict):
        disp = data.get("displayValue")
        if isinstance(disp, str) and "㋕" in disp:
            kana_marked = True
        val = data.get("value")
        if isinstance(val, (int, float)):
            return reading, float(val), kana_marked
        if isinstance(disp, str):
            m = _NUM_RE.search(disp)
            return reading, (float(m.group()) if m else None), kana_marked
    return reading, None, False


def _fingerprint(paths: Iterable[Path]) -> str:
    h = hashlib.sha1(str(SCHEMA_VERSION).encode())
    for p in paths:
        st = p.stat()
        h.update(f"{p}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- build


def _source_paths(cfg: Config) -> tuple[Path, Path | None, list[dict]]:
    jmdict = cfg.resolve(cfg["dictionaries"]["jmdict"])
    jmn = cfg["dictionaries"].get("jmnedict")
    jmnedict = cfg.resolve(jmn) if jmn else None
    freqs = []
    for item in cfg["frequency"]["dictionaries"]:
        if isinstance(item, str):
            item = {"path": item}
        p = cfg.resolve(item["path"])
        if p.exists():
            freqs.append({**item, "path": p})
        else:
            print(f"warning: frequency dictionary not found: {p}", file=sys.stderr)
    return jmdict, (jmnedict if jmnedict and jmnedict.exists() else None), freqs


def build(cfg: Config, db_path: Path, log=print) -> None:
    jmdict, jmnedict, freqs = _source_paths(cfg)
    tmp = db_path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    con.executescript(
        """
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE entry(seq INTEGER PRIMARY KEY, headword TEXT, reading TEXT, pos TEXT,
                           misc TEXT, common INT, score INT, senses TEXT, lsource TEXT);
        CREATE TABLE form(term TEXT, reading TEXT, rkey TEXT, seq INT, common INT, rare INT, score INT);
        CREATE TABLE name(term TEXT, reading TEXT, rkey TEXT, seq INT, types TEXT);
        CREATE TABLE freq_dict(id INTEGER PRIMARY KEY, title TEXT, path TEXT, mode TEXT);
        CREATE TABLE freq(dict INT, term TEXT, reading TEXT, rank REAL, kana INT);
        CREATE TABLE freq_hm(term TEXT, reading TEXT, rank REAL, ndicts INT);
        CREATE TABLE freq_kana(term TEXT, reading TEXT, rank REAL);
        """
    )

    # ---- JMdict
    t0 = time.time()
    cats = _tag_categories(jmdict)
    entries: dict[int, dict] = {}
    forms: dict[tuple[str, str, int], list] = {}
    redirects: list[tuple[str, str, str]] = []  # (term, target, rules)
    for row in _iter_bank_rows(jmdict, "term_bank"):
        term, reading, deftags, rules, score, glossary, seq, termtags = row[:8]
        if seq is not None and seq < 0:
            # yomitan-import redirect rows for search-only spellings: "行ける ⟶ いける"
            target = _redirect_target(glossary)
            if target:
                redirects.append((term, target, rules or ""))
            continue
        reading = reading or term
        tags = deftags.split()
        if "forms" in tags:
            continue
        ttags = termtags.split()
        key = (term, reading, seq)
        f = forms.get(key)
        common = "⭐" in ttags
        rare = any(t in ("R", "r", "⚠️", "⛬") for t in ttags)
        if f is None:
            forms[key] = [common, rare, score]
        else:
            f[0] |= common
            f[1] |= rare
            f[2] = max(f[2], score)
        e = entries.get(seq)
        if e is None:
            e = entries[seq] = {"head": None, "senses": {}, "common": False, "score": -10**9, "lsource": set()}
        e["lsource"].update(_source_languages(glossary))
        if e["head"] is None or score > e["score"]:
            e["head"] = (term, reading)
            e["score"] = score
        e["common"] |= common
        sense_no = tags[0] if tags and tags[0].isdigit() else "1"
        if sense_no not in e["senses"]:
            pos = [t for t in tags if cats.get(t) == "partOfSpeech" or t in POS_EXTRA]
            misc = [t for t in tags if not t.isdigit() and t not in pos]
            e["senses"][sense_no] = (pos, misc, _gloss_text(glossary))
    # Redirects become rare spellings of the entries they point to (same POS family).
    by_term: dict[str, set[int]] = {}
    by_reading: dict[str, set[int]] = {}
    for (t, r, sq) in forms:
        by_term.setdefault(t, set()).add(sq)
        by_reading.setdefault(r, set()).add(sq)
    rule_fam = {"v1": "v1", "v5": "v5", "vs": "vs", "vk": "vk", "vz": "vz", "adj-i": "adj-i"}
    n_redirect = 0
    for term, target, rules in redirects:
        targets = by_term.get(target) or by_reading.get(target) or set()
        want = {rule_fam[r] for r in rules.split() if r in rule_fam}
        for sq in targets:
            e = entries[sq]
            epos = {p for sn in e["senses"].values() for p in sn[0]}
            if want and not any(p == w or p.startswith(w) for p in epos for w in want):
                continue
            rd = target if all("\u3041" <= c <= "\u30ff" or c == "ー" for c in target) else e["head"][1]
            key = (term, rd, sq)
            if key not in forms:
                forms[key] = [False, True, -10**6]
                n_redirect += 1
    erows = []
    for seq, e in entries.items():
        senses = list(e["senses"].values())
        pos = sorted({p for s in senses for p in s[0]})
        misc = sorted({m for s in senses for m in s[1]})
        erows.append((seq, e["head"][0], e["head"][1], " ".join(pos), " ".join(misc),
                      int(e["common"]), e["score"], json.dumps(senses, ensure_ascii=False),
                      ";".join(sorted(e["lsource"]))))
    con.executemany("INSERT INTO entry VALUES (?,?,?,?,?,?,?,?,?)", erows)
    con.executemany(
        "INSERT INTO form VALUES (?,?,?,?,?,?,?)",
        [(t, r, reading_key(r), s, int(v[0]), int(v[1]), v[2]) for (t, r, s), v in forms.items()],
    )
    log(f"  JMdict: {len(entries)} entries, {len(forms)} forms incl. {n_redirect} from {len(redirects)} "
        f"search-only redirects ({time.time() - t0:.1f}s)")

    # ---- JMnedict
    if jmnedict is not None:
        t0 = time.time()
        names: dict[tuple[str, str, int], set] = {}
        for row in _iter_bank_rows(jmnedict, "term_bank"):
            term, reading, deftags, _r, _s, _g, seq = row[:7]
            names.setdefault((term, reading or term, seq), set()).update(deftags.split())
        con.executemany(
            "INSERT INTO name VALUES (?,?,?,?,?)",
            [(t, r, reading_key(r), s, " ".join(sorted(v))) for (t, r, s), v in names.items()],
        )
        log(f"  JMnedict: {len(names)} name forms ({time.time() - t0:.1f}s)")

    # ---- frequency (aggregated in SQLite to keep the build's memory low)
    # Spellings whose word is usually written in kana (JMdict "uk" on the first sense): for
    # them a ㋕ value (how often the word appears written in kana) is the word's frequency.
    uk_forms: set[tuple[str, str]] = set()
    jm_readings: dict[str, set[str]] = {}  # JMdict readings per spelling
    for (t, r, sq) in forms:
        jm_readings.setdefault(t, set()).add(r)
        senses = entries[sq]["senses"]
        if senses and "uk" in next(iter(senses.values()))[1]:
            uk_forms.add((t, r))
    del entries, forms
    con.execute("CREATE TEMP TABLE kana_raw(term TEXT, reading TEXT, rank REAL)")
    term_only: list[tuple[int, str, float, int]] = []  # (dict, term, value, kana) given without a reading
    for i, item in enumerate(freqs):
        t0 = time.time()
        p = item["path"]
        idx = _read_index(p)
        mode = item.get("mode") or idx.get("frequencyMode") or "rank-based"
        title = idx.get("title", p.stem)
        con.execute("INSERT INTO freq_dict VALUES (?,?,?,?)", (i, title, str(p), mode))
        if mode != "rank-based":
            log(f"  frequency: skipping {title} ({mode})")
            continue
        # first plain and first ㋕ value per (term, reading); reading "" = the dictionary
        # gives no reading (YoutubeFreq), which then applies to every reading of the term
        vals: dict[tuple[str, str], list] = {}
        kana_rows: list[tuple[str, str, float]] = []
        for row in _iter_bank_rows(p, "term_meta_bank"):
            if len(row) < 3 or row[1] != "freq":
                continue
            reading, val, kana = parse_freq_value(row[2])
            if val is None or val <= 0:
                continue
            term = row[0]
            if kana and reading:
                # ㋕ value: how often this (term, reading) word is written in kana (JPDB, Jiten)
                kana_rows.append((term, reading, val))
            v = vals.setdefault((term, reading), [None, None])
            if v[int(kana)] is None:
                v[int(kana)] = val
        con.executemany("INSERT INTO freq VALUES (?,?,?,?,?)", list(_freq_rows(i, vals, uk_forms, term_only)))
        first = vals
        con.executemany("INSERT INTO kana_raw VALUES (?,?,?)", kana_rows)
        n = len(first)
        del first, kana_rows
        log(f"  frequency: {title}: {n} values ({time.time() - t0:.1f}s)")
    con.execute("INSERT INTO freq_kana SELECT term, reading, MIN(rank) FROM kana_raw GROUP BY term, reading")
    # A value without a reading counts for the term's main reading only: the most frequent one
    # in the dictionaries that give readings, else the term's only JMdict reading. Yomitan
    # applies it to every reading, which would give rare readings (神【み】) the term's count.
    best: dict[str, tuple[float, str]] = {}
    for t, r, hm in con.execute("SELECT term, reading, COUNT(*) / SUM(1.0 / rank) FROM freq "
                                "WHERE reading != '' GROUP BY term, reading"):
        if t not in best or hm < best[t][0]:
            best[t] = (hm, r)
    extra = []
    for i, t, v, k in term_only:
        r = best[t][1] if t in best else (next(iter(jm_readings[t])) if len(jm_readings.get(t, ())) == 1 else None)
        if r is not None and (k == 0 or _pick_freq(t, r, None, v, uk_forms) is not None):
            extra.append((i, t, r, v, k))
    con.executemany("INSERT INTO freq VALUES (?,?,?,?,?)", extra)
    del term_only, extra, best
    con.execute("INSERT INTO freq_hm SELECT term, reading, COUNT(*) / SUM(1.0 / rank), COUNT(*) "
                "FROM freq GROUP BY term, reading")

    con.executescript(
        """
        CREATE INDEX form_term ON form(term);
        CREATE INDEX form_rkey ON form(rkey);
        CREATE INDEX form_seq ON form(seq);
        CREATE INDEX name_term ON name(term);
        CREATE INDEX name_rkey ON name(rkey);
        CREATE INDEX freq_term ON freq(term, reading);
        CREATE INDEX freq_hm_term ON freq_hm(term, reading);
        CREATE INDEX freq_kana_term ON freq_kana(term, reading);
        """
    )
    srcs = [jmdict] + ([jmnedict] if jmnedict else []) + [f["path"] for f in freqs]
    con.execute("INSERT INTO meta VALUES ('fingerprint', ?)", (_fingerprint(srcs),))
    con.execute("INSERT INTO meta VALUES ('jmdict_title', ?)", (_read_index(jmdict).get("title", ""),))
    con.commit()
    con.close()
    tmp.replace(db_path)


# --------------------------------------------------------------------------- lookup


class DictDB:
    """Read-only lookups over the built SQLite file. All lookups are cached."""

    def __init__(self, db_path: Path):
        self.path = db_path
        self.con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        self.jmdict_title = self._meta("jmdict_title") or ""

    def _meta(self, key: str) -> str | None:
        r = self.con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    # -- JMdict
    @lru_cache(maxsize=200_000)
    def forms_by_term(self, term: str) -> tuple[FormRow, ...]:
        rows = self.con.execute(
            "SELECT term, reading, seq, common, rare, score FROM form WHERE term=?", (term,)
        ).fetchall()
        return tuple(FormRow(t, r, s, bool(c), bool(ra), sc) for t, r, s, c, ra, sc in rows)

    @lru_cache(maxsize=200_000)
    def forms_by_reading(self, kana: str) -> tuple[FormRow, ...]:
        rows = self.con.execute(
            "SELECT term, reading, seq, common, rare, score FROM form WHERE rkey=?", (reading_key(kana),)
        ).fetchall()
        return tuple(FormRow(t, r, s, bool(c), bool(ra), sc) for t, r, s, c, ra, sc in rows)

    @lru_cache(maxsize=100_000)
    def forms_of_entry(self, seq: int) -> tuple[FormRow, ...]:
        rows = self.con.execute(
            "SELECT term, reading, seq, common, rare, score FROM form WHERE seq=?", (seq,)
        ).fetchall()
        return tuple(FormRow(t, r, s, bool(c), bool(ra), sc) for t, r, s, c, ra, sc in rows)

    @lru_cache(maxsize=100_000)
    def entry(self, seq: int) -> Entry | None:
        r = self.con.execute(
            "SELECT seq, headword, reading, pos, misc, common, score, senses, lsource FROM entry WHERE seq=?", (seq,)
        ).fetchone()
        if not r:
            return None
        senses = tuple(Sense(frozenset(p), frozenset(m), g) for p, m, g in json.loads(r[7]))
        return Entry(r[0], r[1], r[2], frozenset(r[3].split()), frozenset(r[4].split()),
                     bool(r[5]), r[6], senses, tuple(x for x in (r[8] or "").split(";") if x))

    def has_spelling(self, s: str) -> bool:
        return bool(self.forms_by_term(s))

    def readings_of_spelling(self, s: str) -> set[str]:
        return {f.reading for f in self.forms_by_term(s)}

    # -- JMnedict
    @lru_cache(maxsize=100_000)
    def names_by_term(self, term: str) -> tuple[NameRow, ...]:
        rows = self.con.execute("SELECT term, reading, seq, types FROM name WHERE term=?", (term,)).fetchall()
        return tuple(NameRow(t, r, s, frozenset(ty.split())) for t, r, s, ty in rows)

    @lru_cache(maxsize=100_000)
    def names_by_reading(self, kana: str) -> tuple[NameRow, ...]:
        rows = self.con.execute(
            "SELECT term, reading, seq, types FROM name WHERE rkey=?", (reading_key(kana),)
        ).fetchall()
        return tuple(NameRow(t, r, s, frozenset(ty.split())) for t, r, s, ty in rows)

    # -- frequency (average rank = harmonic mean over rank-based dictionaries)
    @lru_cache(maxsize=200_000)
    def freq_rank(self, term: str, reading: str = "") -> float | None:
        r = self.con.execute("SELECT rank FROM freq_hm WHERE term=? AND reading=?", (term, reading)).fetchone()
        return r[0] if r else None

    @lru_cache(maxsize=200_000)
    def kana_rank(self, term: str, reading: str) -> float | None:
        """Rank of word (term, reading) when written in kana (㋕ values), if known."""
        r = self.con.execute("SELECT rank FROM freq_kana WHERE term=? AND reading=?", (term, reading)).fetchone()
        return r[0] if r else None

    @lru_cache(maxsize=100_000)
    def freq_rank_by_reading(self, kana: str) -> float | None:
        """Best rank among headwords read as ``kana`` (for kana parts, merging doc §6.4)."""
        best = None
        for f in self.forms_by_reading(kana):
            r = self.freq_rank(f.term, f.reading) or self.freq_rank(f.term)
            if r is not None and (best is None or r < best):
                best = r
        return best

    @lru_cache(maxsize=200_000)
    def freq_values(self, term: str, reading: str = "") -> tuple[tuple[str, float], ...]:
        rows = self.con.execute(
            "SELECT d.title, f.rank FROM freq f JOIN freq_dict d ON d.id=f.dict WHERE f.term=? AND f.reading=?",
            (term, reading),
        ).fetchall()
        return tuple(rows)

    @lru_cache(maxsize=200_000)
    def canonical_seq(self, seq: int) -> int:
        """One key per word for entries JMdict splits by reading only.

        Entries that share a kanji spelling and have the same first gloss are one word
        to a learner whose text shows only the kanji: 私【わたし】/私【わたくし】 ("I; me"),
        何【なに】/何【なん】 ("what"), 貴方【あなた】/【あんた】. Entries with different
        meanings (金 かね "money" / きん "gold") stay apart. The most common entry
        (best frequency rank, then lowest seq) is the canonical one.
        """
        e = self.entry(seq)
        if e is None or not e.senses:
            return seq
        gloss = e.senses[0].gloss.strip().lower()
        if not gloss:
            return seq
        group = {seq}
        for f in self.forms_of_entry(seq):
            if not any("\u3400" <= c <= "\u9fff" for c in f.term):
                continue
            for g in self.forms_by_term(f.term):
                if g.seq in group:
                    continue
                o = self.entry(g.seq)
                if o is not None and o.senses and o.senses[0].gloss.strip().lower() == gloss:
                    group.add(g.seq)
        if len(group) == 1:
            return seq
        return min(group, key=lambda s: (self.entry_rank(s) or float("inf"), s))

    @lru_cache(maxsize=100_000)
    def entry_kana_rank(self, seq: int, reading: str) -> float | None:
        """Best ㋕ rank over an entry's spellings read as ``reading``."""
        best = None
        rk = reading_key(reading)
        for f in self.forms_of_entry(seq):
            if reading_key(f.reading) != rk:
                continue
            r = self.kana_rank(f.term, f.reading)
            if r is not None and (best is None or r < best):
                best = r
        return best

    @lru_cache(maxsize=100_000)
    def entry_rank(self, seq: int) -> float | None:
        """Best average rank over an entry's (spelling, reading) pairs."""
        best = None
        for f in self.forms_of_entry(seq):
            r = self.freq_rank(f.term, f.reading)
            if r is not None and (best is None or r < best):
                best = r
        return best


def open_dictdb(cfg: Config, rebuild: bool = False, log=None) -> DictDB:
    key = ("dictdb",)
    if not rebuild and key in cfg._cache:
        return cfg._cache[key]
    db_path = cfg.data_dir / "dict.sqlite"
    jmdict, jmnedict, freqs = _source_paths(cfg)
    if not jmdict.exists():
        raise FileNotFoundError(f"JMdict not found at {jmdict} (set dictionaries.jmdict in config.toml)")
    srcs = [jmdict] + ([jmnedict] if jmnedict else []) + [f["path"] for f in freqs]
    want = _fingerprint(srcs)
    have = None
    if db_path.exists() and not rebuild:
        try:
            c = sqlite3.connect(db_path)
            r = c.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
            have = r[0] if r else None
            c.close()
        except sqlite3.Error:
            have = None
    if have != want:
        log = log or (lambda m: print(m, file=sys.stderr))
        log(f"Building dictionary data in {db_path} (one-time, ~1 min) ...")
        build(cfg, db_path, log=log)
    db = DictDB(db_path)
    cfg._cache[key] = db
    return db
