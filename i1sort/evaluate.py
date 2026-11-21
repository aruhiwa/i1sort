"""Parser accuracy on the hand-checked test set (tests/parser_cases.yaml).

Case format (YAML list):

  - text: えっ、ちょっと待って！田中くんが食べさせられなかったの？
    words: [ちょっと, 待つ, 食べる]      # the counted words, in order
    filtered: {えっ: interjection, 田中: name, くん: honorific}
    tags: [conjugation, names]

  - same: [可愛い, かわいい, カワイイ]    # each text must yield one counted word, all the same key
    tags: [variants]

An expected word matches a counted word when it is the word's JMdict sequence
number, or one of the entry's spellings (so 可愛い matches a かわいい token), or
its lemma; ``食べる【たべる】`` also checks the reading. ``filtered`` maps a
surface to the expected filter reason (``counted`` = must not be filtered).

Scores: word precision/recall/F1 over all cases, filter-reason accuracy, and
the share of cases that match exactly.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .config import Config
from .filters import TokenFilter
from .parse.base import Word
from .textutil import kata_to_hira


@dataclass
class CaseResult:
    case: dict
    ok: bool
    tp: int = 0
    fp: int = 0
    fn: int = 0
    filt_ok: int = 0
    filt_total: int = 0
    problems: list[str] = field(default_factory=list)
    got: str = ""


def load_cases(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or []
    return data


def _spec_matches(spec, w: Word, db) -> bool:
    if isinstance(spec, int):
        return w.seq == spec
    spec = str(spec)
    reading = None
    if "【" in spec:
        spec, reading = spec.rstrip("】").split("【", 1)
    if reading is not None and kata_to_hira(w.reading or "") != kata_to_hira(reading):
        return False
    if spec in (w.lemma, w.surface, w.headword):
        return True
    if w.seq is not None:
        return any(f.term == spec or f.reading == spec for f in db.forms_of_entry(w.seq))
    return False


def _describe(words: list[Word]) -> str:
    out = []
    for w in words:
        if w.filter in ("whitespace", "symbol"):
            continue
        tag = (w.headword or w.lemma) if w.filter is None else f"({w.surface}:{w.filter})"
        out.append(tag)
    return " ".join(out)


def check_case(case: dict, parser, tf: TokenFilter) -> CaseResult:
    db = parser.db
    if "same" in case:
        keys = []
        problems = []
        for t in case["same"]:
            res = parser.parse(str(t), tf)
            counted = res.counted
            if len(counted) != 1:
                problems.append(f"{t}: expected 1 counted word, got [{_describe(res.words)}]")
                keys.append(None)
            else:
                keys.append(counted[0].key)
        ok = not problems and len(set(keys)) == 1
        if not ok and not problems:
            problems.append("different keys: " + ", ".join(f"{t}→{k}" for t, k in zip(case["same"], keys)))
        n = len(case["same"])
        return CaseResult(case, ok, tp=n if ok else 0, fn=0 if ok else n, problems=problems,
                          got=" ".join(str(k) for k in keys))

    res = parser.parse(case["text"], tf)
    counted = res.counted
    expected = list(case.get("words") or [])
    used = [False] * len(counted)
    tp = 0
    missing = []
    for spec in expected:
        hit = None
        for j, w in enumerate(counted):
            if not used[j] and _spec_matches(spec, w, db):
                hit = j
                break
        if hit is None:
            missing.append(str(spec))
        else:
            used[hit] = True
            tp += 1
    extra = [w.headword or w.lemma for j, w in enumerate(counted) if not used[j]]
    problems = []
    if missing:
        problems.append("missing: " + " ".join(missing))
    if extra:
        problems.append("extra: " + " ".join(extra))
    filt_ok = filt_total = 0
    for surface, reason in (case.get("filtered") or {}).items():
        filt_total += 1
        cands = [w for w in res.words if w.surface == str(surface)]
        if not cands:
            problems.append(f"no token {surface!r} (want {reason})")
            continue
        got = cands[0].filter or "counted"
        if got == reason:
            filt_ok += 1
        else:
            problems.append(f"{surface}: want {reason}, got {got}")
    if "cleaned" in case and res.cleaned != case["cleaned"]:
        problems.append(f"cleaned: want {case['cleaned']!r}, got {res.cleaned!r}")
    ok = not problems
    return CaseResult(case, ok, tp, len(extra), len(missing), filt_ok, filt_total, problems, _describe(res.words))


def run_eval(cfg: Config, cases_path: str, verbose: bool = False, out=sys.stdout) -> int:
    from .parse.pipeline import Parser

    parser = Parser(cfg, use_cache=False)
    tf = TokenFilter(cfg, parser.db, ignore=())
    cases = load_cases(cases_path)
    results = [check_case(c, parser, tf) for c in cases]
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    fo = sum(r.filt_ok for r in results)
    ft = sum(r.filt_total for r in results)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    exact = sum(r.ok for r in results)

    by_tag: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        for t in r.case.get("tags") or ["untagged"]:
            by_tag[t].append(r)

    print(f"parser: {parser.version_key.split('|')[1]}", file=out)
    print(f"cases: {len(results)}   exact: {exact}/{len(results)} ({exact / max(1, len(results)):.0%})", file=out)
    print(f"words: precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}   (tp {tp}, fp {fp}, fn {fn})", file=out)
    print(f"filter reasons: {fo}/{ft} ({fo / max(1, ft):.0%})", file=out)
    print("by tag:", file=out)
    for t, rs in sorted(by_tag.items()):
        print(f"  {t:14s} {sum(r.ok for r in rs)}/{len(rs)}", file=out)
    failing = [r for r in results if not r.ok]
    if failing:
        print(f"\nfailing cases ({len(failing)}):" if verbose else f"\n{len(failing)} failing (use -v to list)", file=out)
        if verbose:
            for r in failing:
                label = r.case.get("text") or " / ".join(map(str, r.case.get("same", [])))
                print(f"  ✗ {label}", file=out)
                for p in r.problems:
                    print(f"      {p}", file=out)
                if r.got:
                    print(f"      got: {r.got}", file=out)
    return 0 if not failing else 1
