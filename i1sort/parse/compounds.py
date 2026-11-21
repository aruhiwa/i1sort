"""Noun-compound re-segmentation (see japanese-compound-merging.md).

Sudachi's split mode C keeps long compounds whole, including transparent ones
(感染対策, 電子顕微鏡, 重要性). Mode A splits too far (顕微+鏡). We expand noun
C-units into their A-units and merge them back bottom-up, keeping a merge only
when the result is a JMdict spelling *and* behaves like one word:

  Gate 1  the joined text is exactly a JMdict spelling (not just a reading)
  Gate 2  (pairs; runs of 3+ are judged as two words at every split point)
    1. productive prefix/suffix → only the frequency rule may merge;
       proper noun + productive affix never merges
    2. either side a single kanji → merge
    3. proper noun on the left → don't merge
    4. reading differs from the joined part readings → merge (idioms, special readings)
    5. frequency: merge if rank(compound) <= max(rank(left), rank(right))

The exception lists below record taste, not data. Edit them freely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..textutil import is_kana, is_single_kanji, kata_to_hira, readings_match
from .base import Morph

# §5.1 Never merge across these (honorifics and plurals)
NEVER_MERGE_SUFFIXES = frozenset(
    "さん 様 さま 君 くん ちゃん 氏 殿 どの 達 たち ら 共 ども 方 がた".split()
)
# §5.2 Productive suffixes: merge only through the frequency rule
PRODUCTIVE_SUFFIXES = frozenset(
    "中 的 化 性 用 等 製 風 上 内 外 後 前 間 際 毎 別 側 頃 ごろ 末 初 通り どおり 以上 以下 以外 以内 "
    "全体 自体 同士 向け 付き 込み 済み 当たり 派 屋 差 者 人 生 家 員 感 力".split()
)
# §5.3 Productive prefixes (+ お/ご/御, the §5.5 option a): merge only through the frequency rule
PRODUCTIVE_PREFIXES = frozenset(
    "前 元 旧 新 現 故 各 全 両 諸 約 毎 計 同 本 当 超 第 某 他 大 小 不 お ご 御".split()
)
# Dependent/formal nouns never take part in a compound.
FORMAL_NOUNS = frozenset("こと 事 もの 物 ところ 所 ため 為 わけ 訳 はず 筈 とき 時 ほう の ん よう そう".split())

MAX_RUN = 4


@dataclass
class Atom:
    morph: Morph
    role: str  # noun | proper | suffix | prefix | naadj | none
    group: int  # index of the C-unit it came from

    @property
    def surface(self) -> str:
        return self.morph.surface

    @property
    def reading(self) -> str:
        return kata_to_hira(self.morph.reading)


@dataclass
class Unit:
    """Output of re-segmentation: one or more atoms forming a single word."""

    atoms: list[Atom]
    reason: str = ""  # why it merged (empty for single atoms)

    @property
    def surface(self) -> str:
        return "".join(a.surface for a in self.atoms)


def role_of(m: Morph) -> str:
    p = m.pos
    if not p:
        return "none"
    if p[0] == "名詞":
        if p[1] == "数詞":
            return "none"
        if m.lemma in FORMAL_NOUNS or m.surface in FORMAL_NOUNS:
            return "none"
        if p[1] == "固有名詞":
            return "proper"
        return "noun"
    if p[0] == "接尾辞" and p[1] == "名詞的":
        return "suffix"
    if p[0] == "接頭辞":
        return "prefix"
    if p[0] == "形状詞" and p[1] in ("一般", "タリ"):  # タリ: 瞭然 in 一目瞭然
        return "naadj"
    return "none"


def _expandable(m: Morph) -> bool:
    """Noun-like C-units whose finer split we re-judge. Person/place names stay whole."""
    p = m.pos
    if p[0] == "名詞" and p[1] == "固有名詞" and p[2] in ("人名", "地名"):
        return False
    return p[0] in ("名詞", "接頭辞", "接尾辞", "形状詞") and bool(m.a_units)


def _atoms_for(m: Morph, group: int, judge: "CompoundJudge | None" = None) -> list[Atom]:
    units = m.a_units or [m]
    # A verb A-unit inside a noun C-unit is a nominalized stem (立ち+入り+禁止):
    # use the B-units, and treat verb stems there as nouns.
    if any(u.pos[0] in ("動詞", "形容詞") for u in units) and m.b_units:
        units = m.b_units
    out = []
    for u in units:
        r = role_of(u)
        if r == "none" and u.pos[0] in ("動詞",) and len(units) > 1:
            r = "noun"
        out.append(Atom(u, r, group))
    if (len(out) > 1 and any(a.role == "none" for a in out) and m.pos[1] != "固有名詞"
            and (judge is None or judge.has_spelling(m.surface) or judge.has_spelling(m.lemma))):
        # a piece that can't join a compound (憎し+み) would leave a dictionary word split for
        # good; a phrase JMdict lacks (危機的状況, 次の日) still splits into its words
        return [Atom(m, role_of(m), group)]
    return out


class CompoundJudge:
    """Gate 1 and gate 2 over injected lookups, so it can be unit-tested with mocks."""

    def __init__(
        self,
        has_spelling: Callable[[str], bool],
        readings_of: Callable[[str], set[str]],
        rank: Callable[[str], float | None],
        rank_by_reading: Callable[[str], float | None],
        is_common: Callable[[str], bool] = lambda s: False,
    ):
        self.is_common = is_common
        self.has_spelling = has_spelling
        self.readings_of = readings_of
        self.rank = rank
        self.rank_by_reading = rank_by_reading

    def part_rank(self, s: str) -> float:
        r = self.rank(s)
        if r is None and is_kana(s) and not self.has_spelling(s):
            r = self.rank_by_reading(s)  # §6.4 kana parts take their headword's frequency
        return float("inf") if r is None else r

    def freq_rule(self, compound: str, left: str, right: str) -> tuple[bool, str]:
        rc = self.rank(compound)
        if rc is None:
            return False, "no frequency data for compound"
        rl, rr = self.part_rank(left), self.part_rank(right)
        if rc <= max(rl, rr):
            return True, f"frequency ({rc:.0f} <= max({_fmt(rl)}, {_fmt(rr)}))"
        return False, f"compound rarer ({rc:.0f} > max({_fmt(rl)}, {_fmt(rr)}))"

    def pair(self, left: Sequence[Atom], right: Sequence[Atom]) -> tuple[bool, str]:
        ls = "".join(a.surface for a in left)
        rs = "".join(a.surface for a in right)
        comp = ls + rs
        left_proper = any(a.role == "proper" for a in left)
        if ls in PRODUCTIVE_PREFIXES or rs in PRODUCTIVE_SUFFIXES:
            if left_proper:
                return False, "proper noun + productive affix"
            ok, why = self.freq_rule(comp, ls, rs)
            return ok, "productive affix; " + why
        if is_single_kanji(ls) or is_single_kanji(rs):
            return True, "single kanji"
        # A multi-character part that is not a dictionary word on its own is bound
        # (一目+惚れ): same reasoning as the single-kanji rule.
        for side, s in ((left, ls), (right, rs)):
            if len(side) == 1 and side[0].role != "proper" and not self.has_spelling(s) \
                    and not self.has_spelling(side[0].morph.lemma) \
                    and not (is_kana(s) and self.rank_by_reading(s) is not None):  # いじめ = 苛め
                return True, f"bound part ({s} is not a word)"
        if left_proper:
            return False, "proper noun on the left"
        lr = "".join(a.reading for a in left)
        rr = "".join(a.reading for a in right)
        dict_readings = [r for r in self.readings_of(comp) if is_kana(r)]
        if lr and rr and is_kana(lr + rr) and dict_readings:
            if not any(readings_match(r, lr + rr) for r in dict_readings):
                return True, "different reading"
        return self.freq_rule(comp, ls, rs)

    def judge(self, span: Sequence[Atom]) -> tuple[bool, str]:
        comp = "".join(a.surface for a in span)
        if not self.has_spelling(comp):
            return False, "not a JMdict spelling"
        if len(span) == 2:
            return self.pair(span[:1], span[1:])
        reasons = []
        for k in range(1, len(span)):
            left, right = span[:k], span[k:]
            if len(left) > 1 and not self.has_spelling("".join(a.surface for a in left)):
                continue
            if len(right) > 1 and not self.has_spelling("".join(a.surface for a in right)):
                continue
            ok, why = self.pair(left, right)
            ls = "".join(a.surface for a in left)
            rs = "".join(a.surface for a in right)
            if ok:
                return True, f"{ls}+{rs}: {why}"
            reasons.append(f"{ls}+{rs}: {why}")
        return False, "; ".join(reasons) or "no valid two-word split"


def _fmt(x: float) -> str:
    return "∞" if x == float("inf") else f"{x:.0f}"


def _can_continue(a: Atom) -> bool:
    if a.role == "naadj" and a.morph.pos[1] == "タリ":
        return True  # 一目+瞭然
    return a.role in ("noun", "suffix", "proper") and a.surface not in NEVER_MERGE_SUFFIXES


def _can_continue_lexical(a: Atom) -> bool:
    """Honorific suffixes may still close a common lexical word (おば+さん, 兄+ちゃん)."""
    return a.role in ("noun", "suffix") and a.surface in NEVER_MERGE_SUFFIXES


def _can_start(a: Atom) -> bool:
    return a.role in ("noun", "proper", "prefix", "naadj")


def self_common(judge: CompoundJudge, s: str) -> bool:
    return judge.is_common(s)


def resegment(morphs: list[Morph], judge: CompoundJudge) -> list[Unit | Morph]:
    """Return a list whose items are either untouched Morphs or compound Units."""
    # 1. expand noun C-units into atoms
    seq: list[Atom | Morph] = []
    for gi, m in enumerate(morphs):
        if _expandable(m):
            seq.extend(_atoms_for(m, gi, judge))
        else:
            r = role_of(m)
            seq.append(Atom(m, r, gi) if r != "none" else m)

    # 2. bottom-up longest-valid-compound merge
    out: list[Unit | Morph] = []
    i = 0
    n = len(seq)
    while i < n:
        a = seq[i]
        if not isinstance(a, Atom):
            out.append(a)
            i += 1
            continue
        run_end = i + 1
        if _can_start(a):
            while run_end < n and run_end - i < MAX_RUN and isinstance(seq[run_end], Atom) and _can_continue(seq[run_end]):  # type: ignore[arg-type]
                run_end += 1
        merged = False
        # kinship/address words: the run right before さん/ちゃん/... forms a common JMdict
        # word with it (おば+さん, 兄+ちゃん); names never do (田中+さん is not in JMdict)
        if (run_end < n and isinstance(seq[run_end], Atom) and _can_continue_lexical(seq[run_end])  # type: ignore[arg-type]
                and all(x.role != "proper" for x in seq[i:run_end])):  # type: ignore[union-attr]
            span = seq[i:run_end + 1]
            if self_common(judge, "".join(x.surface for x in span)):  # type: ignore[union-attr]
                out.append(Unit(list(span), "common word with honorific suffix"))  # type: ignore[arg-type]
                i = run_end + 1
                continue
        for L in range(run_end - i, 1, -1):
            span = seq[i:i + L]
            ok, why = judge.judge(span)  # type: ignore[arg-type]
            if ok:
                out.append(Unit(list(span), why))  # type: ignore[arg-type]
                i += L
                merged = True
                break
        if not merged:
            out.append(Unit([a]))
            i += 1
    return out
