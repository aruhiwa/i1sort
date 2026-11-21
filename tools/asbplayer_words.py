"""Convert an asbplayer word export into a plain word list for i1sort.

asbplayer (Settings → Dictionary → export) writes a JSON list of entries:
``{"profile", "token", "status", "lemmas", "states"}``. Codes, from asbplayer's
common/settings/settings-dictionary.ts:

  status  0 uncollected, 1 unknown, 2 learning, 3 graduated, 4 young, 5 mature
  states  [0] = ignored

Pick entries by status and state and write one word per line. Use the result as:

  * a regular known-words file (``[known] files``): each line is looked up in the
    dictionary, so conjugated or variant forms count too. Best with ``--form lemma``.
  * an exact-match known-words file (``[known] exact_files``): a token counts as known only
    when its text is exactly a line. Best for ignored entries, which are often non-words.

Examples:

  python tools/asbplayer_words.py EXPORT.json --status learning,mature --form lemma -o word_lists/asbplayer_known.txt
  python tools/asbplayer_words.py EXPORT.json --state ignored -o word_lists/asbplayer_ignored.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STATUS = {"uncollected": 0, "unknown": 1, "learning": 2, "graduated": 3, "young": 4, "mature": 5}
STATE = {"ignored": 0}


def _kana(s: str) -> bool:
    return bool(s) and all("぀" <= c <= "ヿ" or c == "ー" for c in s)


def parse_status(text: str | None) -> set[int] | None:
    if not text:
        return None
    out = set()
    for part in text.split(","):
        part = part.strip().lower()
        if part.isdigit():
            out.add(int(part))
        elif part in STATUS:
            out.add(STATUS[part])
        else:
            raise SystemExit(f"unknown status {part!r}; use {', '.join(STATUS)} or 0-5")
    return out


def select(entries: list[dict], statuses: set[int] | None, state: str, profile: str | None) -> list[dict]:
    out = []
    for d in entries:
        if not isinstance(d, dict) or not d.get("token"):
            continue
        if profile and d.get("profile") != profile:
            continue
        if statuses is not None and d.get("status") not in statuses:
            continue
        ignored = STATE["ignored"] in (d.get("states") or [])
        if state == "ignored" and not ignored or state == "not-ignored" and ignored:
            continue
        out.append(d)
    return out


def lines(entries: list[dict], form: str) -> list[str]:
    """token: the text as seen; lemma: the dictionary form, with its reading after a tab
    when asbplayer has a kana one (私<TAB>わたし); both: one line of each."""
    out: list[str] = []
    for d in entries:
        if form in ("token", "both"):
            out.append(d["token"])
        if form in ("lemma", "both"):
            lemmas = [x for x in (d.get("lemmas") or []) if isinstance(x, str) and x]
            if not lemmas:
                if form == "lemma":
                    out.append(d["token"])
                continue
            reading = lemmas[1] if len(lemmas) > 1 and _kana(lemmas[1]) and lemmas[1] != lemmas[0] else ""
            out.append(f"{lemmas[0]}\t{reading}" if reading else lemmas[0])
    return list(dict.fromkeys(out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("export", help="asbplayer-local-words-*.json")
    ap.add_argument("--status", help="comma-separated: " + ", ".join(STATUS) + " (or 0-5); default: any")
    ap.add_argument("--state", choices=["any", "ignored", "not-ignored"], default="any")
    ap.add_argument("--form", choices=["token", "lemma", "both"], default="token",
                    help="token as seen, dictionary form (+ reading), or both (default: token)")
    ap.add_argument("--profile", help="only entries of this asbplayer profile")
    ap.add_argument("-o", "--out", help="output file (default: stdout)")
    args = ap.parse_args(argv)

    data = json.loads(Path(args.export).read_text(encoding="utf-8"))
    chosen = select(data if isinstance(data, list) else [], parse_status(args.status), args.state, args.profile)
    text = "\n".join(lines(chosen, args.form))
    if args.out:
        Path(args.out).write_text(text + "\n" if text else "", encoding="utf-8")
        print(f"{len(chosen)} entries → {text.count(chr(10)) + 1 if text else 0} lines in {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
