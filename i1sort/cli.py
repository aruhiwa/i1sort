"""i1sort command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import unicodedata

from .config import load_config


def _width(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s: str, w: int) -> str:
    return s + " " * max(0, w - _width(s))


def _make_parser(cfg, use_cache=True):
    from .parse.pipeline import Parser

    return Parser(cfg, use_cache=use_cache)


def cmd_explain(args, cfg) -> int:
    from .filters import TokenFilter

    parser = _make_parser(cfg, use_cache=not args.no_cache)
    tf = TokenFilter(cfg, parser.db, deck_names=args.name or [])
    texts = args.text or [line.rstrip("\n") for line in sys.stdin]
    color = sys.stdout.isatty() and not args.no_color
    known = None
    if args.known:
        from .anki import AnkiConnectError, from_config
        from .known import KnownBuilder

        log = lambda m: print(m, file=sys.stderr)  # noqa: E731
        try:
            known = KnownBuilder(cfg, from_config(cfg), parser=parser, log=log).build()
        except AnkiConnectError as e:
            print(f"warning: {e}\n         known words come from files only ([known] files, exact_files)", file=sys.stderr)
            known = KnownBuilder(cfg, None, parser=parser, log=log).build(use_anki=False)
        learning_ok = bool(cfg["known"].get("learning_counts_as_known", True))
        if known.ignored:  # words in the ignored mark deck
            from .filters import load_ignore_set

            tf = TokenFilter(cfg, parser.db, deck_names=args.name or [], ignore=load_ignore_set(cfg) | known.ignored)

    def c(code: str, s: str) -> str:
        return f"\x1b[{code}m{s}\x1b[0m" if color else s

    for text in texts:
        if not text.strip():
            continue
        res = parser.parse(text, tf)
        print(c("1", res.text))
        if res.cleaned != res.text:
            print(c("2", f"  cleaned: {res.cleaned}"))
        rows = []
        for w in res.words:
            if w.filter in ("whitespace",) and not args.all:
                continue
            if w.filter == "symbol" and not args.all:
                continue
            status = "counted" if w.filter is None else f"filtered: {w.filter}" + (f" ({w.filter_detail})" if w.filter_detail else "")
            if known is not None and w.filter is None:
                why = known.why(w, parser.db, learning_ok)
                status += f" · {why}" if why else " · UNKNOWN"
            pos = "/".join(x for x in w.pos[:3] if x and x != "*")
            jm = f"{w.seq}" if w.seq else "-"
            head = w.headword or ""
            if head and w.reading and head != w.reading:
                head = f"{head}【{w.reading}】"
            rows.append((w.surface, w.lemma, jm, head, pos, status, w))
        ws = [max([_width(r[i]) for r in rows] + [0]) for i in range(5)]
        for r in rows:
            w = r[6]
            line = "  " + "  ".join(_pad(r[i], ws[i]) for i in range(5)) + "  "
            st = r[5]
            line += c("32", st) if w.filter is None else c("2", st)
            print(line)
            if args.verbose:
                if w.kind != "token" or w.note:
                    print(c("2", f"      {w.kind}: {w.note}"))
                if w.gloss:
                    print(c("2", f"      {w.gloss[:100]}"))
                if w.kind == "token" and w.morphs:
                    m = w.morphs[0]
                    print(c("2", f"      sudachi: norm={m.norm} reading={m.reading} lemma_reading={m.lemma_reading} pos={','.join(m.pos)}{' OOV' if m.oov else ''}"))
        counted = [w for w in res.words if w.filter is None]
        print(c("2", f"  → {len(counted)} counted: " + " ".join(w.headword or w.lemma for w in counted)))
        if known is not None:
            unk = list(dict.fromkeys(w.headword or w.lemma for w in counted if known.why(w, parser.db, learning_ok) is None))
            print(c("2", f"  → {len(unk)} unknown: " + " ".join(unk)))
        print()
    parser.flush()
    return 0


def cmd_anki_check(args, cfg) -> int:
    from .anki import from_config

    info = from_config(cfg).check(("findNotes", "notesInfo", "multi"))
    print(f"AnkiConnect API v{info['version']}, profile {info['profile']!r}, {len(info['actions'])} actions")
    for wanted in ("findNotes", "notesInfo", "findCards", "multi", "addTags", "removeTags", "suspend",
                   "unsuspend", "setSpecificValueOfCard", "updateNoteFields", "repositionNewCards"):
        print(f"  {'✓' if wanted in info['actions'] else '✗'} {wanted}")
    return 0


def _build_known(args, cfg):
    from .anki import from_config
    from .known import KnownBuilder

    anki = None if args.no_anki else from_config(cfg)
    return KnownBuilder(cfg, anki).build(use_anki=not args.no_anki)


def cmd_known_export(args, cfg) -> int:
    from .known import export

    ks = _build_known(args, cfg)
    levels = ("known", "learning") if args.level == "all" else (args.level,)
    n = export(ks, Path(args.file), levels)
    c = ks.counts()
    print(f"wrote {n} words to {args.file}  (known {c['known']}, learning {c['learning']})")
    if ks.unresolved:
        print(f"{len(ks.unresolved)} vocab fields had no JMdict entry: "
              + ", ".join(t for _, t in ks.unresolved[:15]) + (" …" if len(ks.unresolved) > 15 else ""))
    return 0


def cmd_known_stats(args, cfg) -> int:
    ks = _build_known(args, cfg)
    c = ks.counts()
    print(f"known {c['known']}, learning {c['learning']}, total {len(ks.words)}")
    for name, st in ks.stats.items():
        print(f"  {name}: " + ", ".join(f"{k} {v}" for k, v in st.items()))
    if ks.unresolved:
        print(f"unresolved vocab ({len(ks.unresolved)}): " + ", ".join(t for _, t in ks.unresolved))
    return 0


def run_analysis(cfg, decks: list[str], log=print):
    """Read the decks' new cards from Anki and plan them. Returns (plans, known set). Read-only."""
    from .anki import cards_info, find_cards, from_config, quote
    from .known import KnownBuilder
    from .mark import exclude_query
    from .parse.pipeline import Parser
    from .score import Planner

    anki = from_config(cfg)
    parser = Parser(cfg)
    log("building known words ...")
    known = KnownBuilder(cfg, anki, parser=parser, log=log).build()
    c = known.counts()
    log(f"  {c['known']} known, {c['learning']} learning")
    planner = Planner(cfg, parser, known)
    wanted = list(decks)
    if str(cfg["sort"].get("one_card_per_word", "all")) == "all":
        # one i+1 card per word across all sorted decks: plan them all, return the asked-for ones
        decks = wanted + [d for d in (cfg["sort"].get("decks") or []) if d not in wanted]
    plans = []
    for deck in decks:
        t0 = time.time()
        cids = find_cards(anki, f"deck:{quote(deck)} is:new{exclude_query(cfg)}")
        if not cids:
            log(f"{deck}: no new cards")
            continue
        cards = cards_info(anki, cids)
        notes = {n["noteId"]: n for n in anki.notes_info(sorted({c["note"] for c in cards}))}
        plans.append(planner.plan_deck(deck, cards, notes))
        log(f"{deck}: {len(cids)} new cards planned in {time.time() - t0:.1f}s")
    planner.across_decks(plans)
    return [dp for dp in plans if dp.deck in wanted], known


def cmd_analyze(args, cfg) -> int:
    from .report import summary_text, write_csv, write_html

    decks = args.deck or cfg["sort"].get("decks") or []
    if not decks:
        print("no decks: pass --deck or set sort.decks in config.toml", file=sys.stderr)
        return 2
    plans, known = run_analysis(cfg, decks, log=lambda m: print(m, file=sys.stderr))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv_path, html_path = out.with_suffix(".csv"), out.with_suffix(".html")
    write_csv(plans, csv_path)
    write_html(plans, html_path, known.counts())
    print(summary_text(plans))
    if known.marked:
        from .mark import summary as mark_summary

        print("\nMark decks (apply adds the new words to the files and suspends the cards):")
        print(mark_summary(known.marked, cfg))
    print(f"\nDry run: nothing changed in Anki. Wrote {csv_path} and {html_path}")
    return 0


def _confirm(args, question: str) -> bool:
    if getattr(args, "yes", False):
        return True
    if not sys.stdin.isatty():
        print("Not applied: no terminal to ask in. Pass --yes to apply without a prompt.")
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def cmd_apply(args, cfg) -> int:
    from . import mark
    from .anki import AnkiConnectError, from_config, v1_from_config
    from .apply import WRITE_METHODS, compute_changes, refresh, snapshot, summary, write
    from .report import summary_text, write_csv, write_html

    decks = args.deck or cfg["sort"].get("decks") or []
    if not decks:
        print("no decks: pass --deck or set sort.decks in config.toml", file=sys.stderr)
        return 2
    log = lambda m: print(m, file=sys.stderr)  # noqa: E731
    v1 = v1_from_config(cfg)
    try:
        info = v1.open(WRITE_METHODS)  # fail before the long analysis, not after
    except AnkiConnectError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    plans, known = run_analysis(cfg, decks, log=log)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(plans, out.with_suffix(".csv"))
    write_html(plans, out.with_suffix(".html"), known.counts())
    print(summary_text(plans))
    anki = from_config(cfg)
    refresh(anki, plans)
    definitions = None
    if any((f or {}).get("definition") for f in (cfg["sort"].get("fields") or {}).values()):
        from .definitions import open_definitions

        definitions = open_definitions(cfg, log=log)
    changes = [compute_changes(dp, cfg["sort"], definitions) for dp in plans]
    prefix = cfg["sort"].get("tag_prefix", "i1sort")
    marks = known.marked
    mark_changes = [mark.changes(m, prefix) for m in marks]
    print(f"\nChanges to write (Anki {info['anki']}, profile {info['profile']!r}):")
    print(summary(changes + mark_changes))
    if marks:
        print("mark decks (new words are added to the files, the cards suspended):")
        print(mark.summary(marks, cfg))
    new_words = [(m, m.new_words) for m in marks if m.new_words]
    if all(ch.empty() for ch in changes + mark_changes) and not new_words:
        print("\nNothing to change: Anki already matches the plan.")
        return 0
    if not _confirm(args, "\nApply these changes?"):
        return 1
    snap = snapshot(plans, changes, cfg.data_dir / "apply_snapshots" / f"{time.strftime('%Y%m%d-%H%M%S')}.json",
                    marks=list(zip(marks, mark_changes)))
    print(f"Saved the previous state to {snap}\n  (undo: i1sort restore {snap})")
    t0 = time.time()
    for m, _ in new_words:
        added = mark.append_words(m)
        print(f"  {m.file}: added {len(added)} {m.kind} word(s)")
    for ch in changes + mark_changes:
        if not ch.empty():
            write(v1, ch, log=print)
    refresh(anki, plans)
    left = [c for c in (compute_changes(dp, cfg["sort"], definitions) for dp in plans) if not c.empty()]
    if marks:
        again = mark.read(cfg, anki, log=lambda m: None)
        left += [c for c in (mark.changes(m, prefix) for m in again) if not c.empty()]
    if left:
        print(f"Applied in {time.time() - t0:.1f}s, but these changes did not take:\n" + summary(left))
        return 1
    print(f"Applied in {time.time() - t0:.1f}s. Checked: Anki now matches the plan.")
    return 0


def cmd_restore(args, cfg) -> int:
    import json as _json

    from .anki import AnkiConnectError, from_config, v1_from_config
    from .apply import WRITE_METHODS, restore_changes, summary, write_restore

    snap = _json.loads(Path(args.snapshot).read_text(encoding="utf-8"))
    v1 = v1_from_config(cfg)
    try:
        v1.open(WRITE_METHODS)
    except AnkiConnectError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    anki = from_config(cfg)
    anki.check(("findNotes", "notesInfo", "cardsInfo"))
    from .filters import load_wordlist
    from .mark import remove_words

    changes = restore_changes(anki, snap, cfg["sort"].get("tag_prefix", "i1sort"))
    files = [(Path(f["file"]), [w for w in f["words"] if w in load_wordlist(Path(f["file"]))])
             for f in snap.get("mark_files") or []]
    files = [(p, ws) for p, ws in files if ws]
    print(f"Restore the state saved {snap.get('time')}:")
    print(summary(changes))
    for p, ws in files:
        print(f"  {p}: remove {len(ws)} word(s) the run added: {'、'.join(ws[:12])}")
    if all(ch.empty() for ch in changes) and not files:
        print("Nothing to restore: Anki already matches the snapshot.")
        return 0
    if not _confirm(args, "Restore?"):
        return 1
    for ch in changes:
        if not ch.empty():
            write_restore(v1, ch, log=print)
    for p, ws in files:
        print(f"  {p}: removed {remove_words(p, ws)} word(s)")
    left = [c for c in restore_changes(anki, snap, cfg["sort"].get("tag_prefix", "i1sort")) if not c.empty()]
    print("Restored." if not left else "Restored, but some changes did not take:\n" + summary(left))
    return 0 if not left else 1


def cmd_build_data(args, cfg) -> int:
    from .dictdb import open_dictdb

    t0 = time.time()
    open_dictdb(cfg, rebuild=True, log=print)
    print(f"done in {time.time() - t0:.1f}s")
    return 0


def cmd_eval_parser(args, cfg) -> int:
    from .evaluate import run_eval

    return run_eval(cfg, args.cases, verbose=args.verbose)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="i1sort", description="i+1 sorter for Japanese sentence decks in Anki")
    ap.add_argument("--config", help="path to config.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("explain", help="show how a sentence is parsed and why each token counts or not")
    p.add_argument("text", nargs="*", help="sentences (default: read lines from stdin)")
    p.add_argument("-v", "--verbose", action="store_true", help="show merge/resolution notes and Sudachi fields")
    p.add_argument("-a", "--all", action="store_true", help="also show punctuation and whitespace")
    p.add_argument("--name", action="append", help="treat this word as a deck character name (repeatable)")
    p.add_argument("-k", "--known", action="store_true",
                   help="also show whether each word is known, and through which known word (needs Anki; "
                        "falls back to the known-word files)")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("analyze", help="dry run: plan the new cards of the decks and write report.csv/report.html")
    p.add_argument("--deck", action="append", help="deck to sort (repeatable; default: sort.decks)")
    p.add_argument("--out", default="report", help="report path without extension (default: report)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("apply", help="plan like analyze, then write the order, tags and suspensions to Anki")
    p.add_argument("--deck", action="append", help="deck to sort (repeatable; default: sort.decks)")
    p.add_argument("--out", default="report", help="report path without extension (default: report)")
    p.add_argument("--yes", action="store_true", help="apply without asking")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("restore", help="undo an apply run from its snapshot file")
    p.add_argument("snapshot", help="data/apply_snapshots/<time>.json")
    p.add_argument("--yes", action="store_true", help="restore without asking")
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser("anki-check", help="check AnkiConnect version, active profile and actions")
    p.set_defaults(func=cmd_anki_check)

    kp = sub.add_parser("known", help="the known-word set").add_subparsers(dest="known_cmd", required=True)
    p = kp.add_parser("export", help="write the known-word set to a file (TSV: word, reading, level, key, sources)")
    p.add_argument("file")
    p.add_argument("--level", choices=["all", "known", "learning"], default="all")
    p.add_argument("--no-anki", action="store_true", help="only the [known] word lists, no Anki")
    p.set_defaults(func=cmd_known_export)
    p = kp.add_parser("stats", help="counts per source")
    p.add_argument("--no-anki", action="store_true")
    p.set_defaults(func=cmd_known_stats)

    p = sub.add_parser("build-data", help="(re)build the dictionary/frequency database")
    p.set_defaults(func=cmd_build_data)

    p = sub.add_parser("eval-parser", help="report parser accuracy on the hand-checked test set")
    p.add_argument("--cases", default="tests/parser_cases.yaml")
    p.add_argument("-v", "--verbose", action="store_true", help="print every failing case")
    p.set_defaults(func=cmd_eval_parser)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
