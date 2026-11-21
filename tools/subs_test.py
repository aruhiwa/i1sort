"""Test the parser on raw subtitle files (.srt) without Anki decks.

Picks random series from a subtitle tree (by default ones not tested before), parses every
line the way a dry run does (per-series name detection, filters, known words from Anki or
the known-word files), and writes files to review:

  summary.json              per series: lines, counted words, % known, unknown/unresolved counts
  <series>_unknown.tsv      unknown words by frequency: count, headword, reading, surface, POS, kind, example
  <series>_unresolved.tsv   counted tokens with no JMdict entry
  <series>_byreading.tsv    kana tokens resolved to a kanji word by reading alone (homograph risk)
  names.tsv                 names detected per series, with the evidence
  sample.tsv                the files that were parsed (season, series, path)

Every run adds its series to subs_test/tested.tsv so the next pick avoids them.

Run it under a memory cap (a pathological line once used 5+ GB and took the desktop down):

  systemd-run --user --scope -q -p MemoryMax=1500M -p MemorySwapMax=0 \\
      .venv/bin/python tools/subs_test.py pick --series 16 --episodes 3 --out subs_test/run05
  systemd-run --user --scope -q -p MemoryMax=1500M -p MemorySwapMax=0 \\
      .venv/bin/python tools/subs_test.py pick --series 1 --episodes 0 --out subs_test/run06   # 1 whole series
  .venv/bin/python tools/subs_test.py run subs_test/run05/sample.tsv --out subs_test/run05b       # rerun a sample

Layout expected under --root: <season or group>/<series>/*.srt, or <series>/*.srt.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TESTED = ROOT / "subs_test" / "tested.tsv"


def srt_lines(p: Path):
    """Subtitle text of each SRT block, its lines joined with a space."""
    for b in re.split(r"\n\s*\n", p.read_text(encoding="utf-8-sig", errors="replace")):
        ls = [l for l in b.strip().splitlines() if l.strip()]
        ls = [l for l in ls if not re.fullmatch(r"\d+", l.strip()) and "-->" not in l]
        if ls:
            yield " ".join(ls)


def find_series(root: Path) -> list[tuple[str, Path]]:
    """(group, series dir) for every directory that directly holds .srt files."""
    out = []
    for d in sorted({p.parent for p in root.rglob("*.srt")}):
        rel = d.relative_to(root).parts
        group = rel[0] if len(rel) > 1 else ""
        out.append((group, d))
    return out


def tested_series() -> set[str]:
    if not TESTED.exists():
        return set()
    return {l.split("\t")[1] for l in TESTED.read_text(encoding="utf-8").splitlines() if "\t" in l}


def pick(args) -> list[tuple[str, str, Path]]:
    root = Path(args.root)
    rng = random.Random(args.seed)
    done = set() if args.include_tested else tested_series()
    series = [(g, d) for g, d in find_series(root) if d.name not in done]
    if not series:
        sys.exit("no untested series left (use --include-tested)")
    if args.per_group:
        by_group: dict[str, list] = collections.defaultdict(list)
        for g, d in series:
            by_group[g].append((g, d))
        chosen = [x for g in sorted(by_group) for x in rng.sample(by_group[g], min(args.series, len(by_group[g])))]
    else:
        chosen = rng.sample(series, min(args.series, len(series)))
    sample = []
    for g, d in chosen:
        eps = sorted(d.glob("*.srt"))
        if args.episodes > 0:
            eps = sorted(rng.sample(eps, min(args.episodes, len(eps))))
        sample += [(g, d.name, e) for e in eps]
    return sample


def load_sample(path: Path) -> list[tuple[str, str, Path]]:
    out = []
    for l in path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            g, s, f = l.split("\t")
            out.append((g, s, Path(f)))
    return out


def run(sample: list[tuple[str, str, Path]], out: Path, use_anki: bool = True) -> dict:
    from i1sort.anki import AnkiConnectError, from_config
    from i1sort.cli import _make_parser
    from i1sort.config import load_config
    from i1sort.filters import TokenFilter, detect_deck_names
    from i1sort.known import KnownBuilder

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sample.tsv", "w", encoding="utf-8") as fo:
        for g, s, f in sample:
            fo.write(f"{g}\t{s}\t{f}\n")
    cfg = load_config()
    parser = _make_parser(cfg, use_cache=True)
    log = lambda m: print(m, file=sys.stderr)  # noqa: E731
    known = None
    if use_anki:
        try:
            known = KnownBuilder(cfg, from_config(cfg), parser=parser, log=log).build()
        except AnkiConnectError as e:
            print(f"warning: {e}\n  known words come from files only, so % known is low", file=sys.stderr)
    if known is None:
        known = KnownBuilder(cfg, None, parser=parser, log=log).build(use_anki=False)

    groups: dict[str, list[Path]] = collections.OrderedDict()
    seasons: dict[str, str] = {}
    for g, s, f in sample:
        groups.setdefault(s, []).append(f)
        seasons[s] = g
    names_f = open(out / "names.tsv", "w", encoding="utf-8")
    summary = {}
    for series, files in groups.items():
        tag = re.sub(r"[/\\]", "_", series)
        results = [parser.parse(line) for f in files for line in srt_lines(f)]
        names = detect_deck_names(results, parser.db)  # one series = one deck
        for k, v in names.items():
            names_f.write(f"{series}\t{k}\t{v}\n")
        tf = TokenFilter(cfg, parser.db, deck_names=names)
        unk, info, ex = collections.Counter(), {}, {}
        unres, unres_ex = collections.Counter(), {}
        byread, br_ex = collections.Counter(), {}
        filt = collections.Counter()
        n_tok = n_known = 0
        for res in results:
            tf.apply(res.words, annotated=res.annotated)
            for w in res.words:
                if w.filter:
                    filt[w.filter] += 1
                    continue
                n_tok += 1
                if (w.note or "").startswith("reading:") and w.headword and w.headword != w.surface:
                    kk = (w.surface, w.headword, w.reading, "/".join(w.pos[:2]))
                    byread[kk] += 1
                    br_ex.setdefault(kk, res.cleaned)
                if w.seq is None:
                    unres[w.surface] += 1
                    unres_ex.setdefault(w.surface, (res.cleaned, w.pos))
                if known.why(w, parser.db, True):
                    n_known += 1
                    continue
                unk[w.key] += 1
                ex.setdefault(w.key, res.cleaned)
                info[w.key] = (w.headword or w.lemma, w.reading, w.surface, "/".join(w.pos[:2]), w.kind)
        summary[series] = dict(group=seasons[series], files=len(files), lines=len(results), counted=n_tok,
                               known_pct=round(100 * n_known / max(1, n_tok), 1), unknown_types=len(unk),
                               unresolved_tokens=sum(unres.values()), filtered=dict(filt.most_common()))
        with open(out / f"{tag}_unknown.tsv", "w", encoding="utf-8") as fo:
            for k, c in unk.most_common():
                h, r, s, p, kind = info[k]
                fo.write(f"{c}\t{h}\t{r}\t{s}\t{p}\t{kind}\t{ex[k][:80]}\n")
        with open(out / f"{tag}_unresolved.tsv", "w", encoding="utf-8") as fo:
            for s, c in unres.most_common():
                fo.write(f"{c}\t{s}\t{'/'.join(unres_ex[s][1][:3])}\t{unres_ex[s][0][:80]}\n")
        with open(out / f"{tag}_byreading.tsv", "w", encoding="utf-8") as fo:
            for kk, c in byread.most_common():
                fo.write(f"{c}\t" + "\t".join(kk) + f"\t{br_ex[kk][:80]}\n")
        print(f"{series[:40]:40} {len(results):5} lines {n_tok:6} counted {summary[series]['known_pct']:5}% known "
              f"{len(unk):5} unknown {sum(unres.values()):4} unresolved", file=sys.stderr)
    names_f.close()
    parser.flush()
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    TESTED.parent.mkdir(parents=True, exist_ok=True)
    with open(TESTED, "a", encoding="utf-8") as fo:
        for series, v in summary.items():
            fo.write(f"{v['group']}\t{series}\t{out.name}\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick", help="pick random series and parse them")
    p.add_argument("--root", default=str(ROOT / "subs"), help="subtitle tree (default: ./subs)")
    p.add_argument("--series", type=int, default=4, help="series to pick (per group with --per-group)")
    p.add_argument("--per-group", action="store_true", help="pick --series from each top-level folder (season)")
    p.add_argument("--episodes", type=int, default=3, help="random episodes per series; 0 = all")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--include-tested", action="store_true", help="also pick series tested before")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--no-anki", action="store_true", help="known words from files only")
    r = sub.add_parser("run", help="parse the files of an existing sample.tsv")
    r.add_argument("sample", help="sample.tsv from an earlier run")
    r.add_argument("--out", required=True)
    r.add_argument("--no-anki", action="store_true")
    args = ap.parse_args()
    sample = pick(args) if args.cmd == "pick" else load_sample(Path(args.sample))
    run(sample, Path(args.out), use_anki=not args.no_anki)


if __name__ == "__main__":
    main()
