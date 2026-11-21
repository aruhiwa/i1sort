"""Merge consecutive tokens into JMdict expressions (気がする, しょうがない, かもしれない).

For each position the longest span (2..N tokens) whose text is a JMdict
spelling wins. The last token of the span may be inflected: the key is the
earlier surfaces plus the last token's dictionary form (気が+し → 気がする), so
the auxiliaries after it stay separate. Spans never cross punctuation or names,
and pure noun runs are left to the compound stage.

Shape rules, so transparent combinations stay split:

* The entry must be an expression (``exp``) or a verb/adjective idiom; noun-,
  adverb- and particle-only entries (本当に, 目の前) are not merged.
* A kana span only matches entries whose spelling is kana or that are usually
  written in kana (``uk``); otherwise どう+し would match 同視 by its reading.
* *One word + function words* (これは, 誰も, 実は; 知らない, やった):
  - verb/adjective head: that is just inflection → not merged, unless the entry
    is an interjection (すみません, やった) — then it is merged and filtered.
  - other heads: merged only if the expression is at least as frequent as the
    head word alone (実は yes, これは no) — the compound frequency rule.
* Expressions made only of function words (ことがある, について, 方がいい) are
  merged but filtered as grammar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..dictdb import Entry
from ..textutil import is_kana, kata_to_hira
from .base import Morph, Word
from .resolve import Resolution, Resolver

CONTENT_POS = {"名詞", "動詞", "形容詞", "形状詞", "副詞", "代名詞", "連体詞"}
BREAK_POS = {"補助記号", "空白", "記号"}
_ALLOWED_FAMILIES = {"verb", "vs", "adj-i", "adj-na"}
_NOUNISH = {"名詞", "接頭辞", "接尾辞", "代名詞", "形状詞"}
SUBSIDIARY = frozenset(
    "いる 居る おく 置く しまう 仕舞う くる 来る いく 行く みる 見る あげる 上げる やる 遣る くれる 呉れる "
    "もらう 貰う いただく 頂く くださる 下さる ある 有る".split()
)

# Expressions built only from these (plus particles/auxiliaries) are grammar, not vocabulary.
FUNCTION_LEMMAS = frozenset(
    "こと 事 もの 物 ところ 所 わけ 訳 はず 筈 ため 為 よう 様 そう の ん とき 時 方 ほう "
    "ある 有る 在る いる 居る する 為る なる 成る できる 出来る いう 言う おく 置く しまう 仕舞う "
    "くる 来る いく 行く みる 見る つく 付く 就く よる 依る 因る とる 取る 対する 関する 際する おる "
    "ない 無い いい 良い よい".split()
)


# Set phrases built on a verb (済む, 下さる, 頂く...) that should merge into one unit and be
# treated like interjections: greetings and apologies, not vocabulary to schedule.
SET_PHRASES = frozenset(
    "すみません すいません すみませんでした ありがとうございます ありがとうございました 有難うございます "
    "おはようございます おめでとうございます いただきます 頂きます ごちそうさま ごちそうさまでした ご馳走様 "
    "失礼します 失礼しました 失礼いたします お疲れ様 お疲れさま お疲れ様です お疲れ様でした かしこまりました "
    "畏まりました よろしくお願いします よろしくお願いいたします 宜しくお願いします ごめんください "
    "ごめんなさい 御免なさい どういたしまして おやすみなさい お休みなさい いってきます 行ってきます "
    "いってらっしゃい 行ってらっしゃい ただいま おかえりなさい お帰りなさい".split()
)


@dataclass
class ExprMatch:
    end: int
    key: str
    res: Resolution | None
    interjection: bool = False


def _is_function(w: Word) -> bool:
    if w.pos0 in ("助詞", "助動詞"):
        return True
    if w.pos0 == "形状詞" and len(w.pos) > 1 and w.pos[1] == "助動詞語幹":
        return True
    if w.kind != "token":
        return False
    return w.lemma in FUNCTION_LEMMAS or (bool(w.morphs) and w.morphs[0].norm in FUNCTION_LEMMAS)


def _tails(span: list[Word]) -> list[bool]:
    """Mark words that only attach to a head: particles, auxiliaries, a subsidiary
    verb after て (帰ってくる, してやる), and a leading determiner (その時)."""
    out = []
    for k, w in enumerate(span):
        t = w.pos0 in ("助詞", "助動詞")
        if not t and k > 0 and w.kind == "token" and w.pos0 == "動詞" and span[k - 1].surface in ("て", "で") \
                and span[k - 1].pos0 == "助詞" and (w.lemma in SUBSIDIARY or (w.morphs and w.morphs[0].norm in SUBSIDIARY)):
            t = True
        if not t and k == 0 and w.pos0 == "連体詞":
            t = True
        out.append(t)
    return out


def _is_verbish(w: Word) -> bool:
    return w.pos0 in ("動詞", "形容詞") or (w.kind == "token" and bool(w.morphs) and w.morphs[0].pos[0] in ("動詞", "形容詞"))


# Grammar patterns built on a content word, so the all-function test below misses them:
# かも知れない is taught as grammar ("might"), not as the verb 知れる.
GRAMMAR_HEADWORDS = frozenset(
    "である であろう かも知れない かも知れん かもしれん に違いない に決まっている ざるを得ない に過ぎない かねない ねばならない "
    "なければならない なければいけない なくてはいけない なくてはならない てはいけない".split()
)


def _unknown_verb_head(span: list[Word]) -> bool:
    """A verb Sudachi has no kanji for (normalized form in kana), then only particles and
    auxiliaries: ほっ+とい is ほっとく (放っておく), not ほる (read as 掘る) + とく. A verb
    entry spelled like the whole span is better evidence than Sudachi's guess at the verb."""
    head = span[0]
    return (len(span) >= 2 and head.kind == "token" and head.pos0 == "動詞" and bool(head.morphs)
            and is_kana(head.morphs[0].norm) and all(w.pos0 in ("助動詞", "助詞") for w in span[1:]))


def is_grammar_expression(span: list[Word]) -> bool:
    return all(_is_function(w) for w in span)


def _span_keys(span: list[Word]) -> list[tuple[str, str]]:
    def rd(w: Word) -> str:
        if w.kind == "token" and w.morphs:
            return kata_to_hira(w.morphs[-1].reading)
        return w.reading

    head_s = "".join(w.surface for w in span[:-1])
    head_r = "".join(rd(w) for w in span[:-1])
    last = span[-1]
    lr = kata_to_hira(last.morphs[-1].lemma_reading) if last.morphs and last.kind == "token" else last.reading
    keys = [(head_s + last.lemma, head_r + lr)]
    if last.surface != last.lemma and (last.pos0 in ("助動詞", "助詞") or span[0].pos0 in ("助詞", "助動詞")):
        # すみ+ませ+ん (the auxiliary's lemma ぬ is not in the phrase); と+は+いえ (fixed form)
        keys.append((head_s + last.surface, head_r + rd(last)))
    # normalized spellings of content tokens (気がつく → 気が付く, しょうがない → 仕様がない)
    norm_s = "".join(
        (w.morphs[0].norm if w.kind == "token" and len(w.morphs) == 1 and w.pos0 not in ("助詞", "助動詞") else w.surface)
        for w in span[:-1]
    )
    nl = last.morphs[-1].norm if last.kind == "token" and last.morphs else last.lemma
    if norm_s + nl != keys[0][0]:
        keys.append((norm_s + nl, head_r + lr))
    return keys


class ExpressionMerger:
    def __init__(self, resolver: Resolver, rank: Callable[[str], float | None],
                 entry_rank: Callable[[int], float | None], max_tokens: int = 6, max_ratio: float = 6.0):
        self.resolver = resolver
        self.rank = rank
        self.entry_rank = entry_rank
        self.max_tokens = max_tokens
        self.max_ratio = max_ratio

    def _word_rank(self, w: Word) -> float | None:
        r = self.rank(w.lemma)
        if r is None and w.kind == "token" and w.morphs:
            r = self.rank(w.morphs[0].norm)
        return r

    def _entry_ok(self, e: Entry, span: list[Word], key: str) -> tuple[bool, bool]:
        """(accept, is_interjection)"""
        if key in SET_PHRASES or e.headword in SET_PHRASES:
            return True, True
        is_int = bool(e.senses) and "int" in e.senses[0].pos
        leading_function = span[0].pos0 in ("助詞", "助動詞")
        conj_ok = "conj" in e.pos and leading_function  # とはいえ, にしても
        if not ("exp" in e.pos or e.pos_family() & _ALLOWED_FAMILIES or "int" in e.pos or conj_ok):
            return False, False
        rank_e = self.entry_rank(e.seq)
        # Obscure entries matching a literal phrase (鬼になる): need frequency data or a common entry.
        if not e.common and rank_e is None:
            return False, False
        tails = _tails(span)
        heads = [w for w, t in zip(span, tails) if not t]
        if not leading_function and len(heads) == 1:
            # head word + particles/auxiliaries
            head = heads[0]
            if head.pos0 in ("動詞", "形容詞") or (head.kind == "token" and head.morphs and head.morphs[0].pos[0] in ("動詞", "形容詞")):
                if _unknown_verb_head(span) and "uk" in e.misc and e.pos_family() & {"verb"}:
                    return True, is_int  # ほっとく: a verb of its own Sudachi did not know
                return False, False  # just inflection (知らない, やった, 帰ってくる)
            r_head = self._word_rank(head)
            if rank_e is None or (r_head is not None and rank_e > r_head):
                return False, False  # transparent word + particle (これは, 誰も); 実は passes
            return True, is_int
        # Content verb + only light verbs/particles after it (見に行く, 食べに来る): the verb
        # is the word, the rest is grammar.
        if heads and not leading_function and _is_verbish(heads[0]) and all(_is_function(h) for h in heads[1:]):
            return False, False
        # Several content words (気がする, 手に入れる): an idiom is about as common as its
        # rarest content part; transparent phrases (電話をかける, 写真を撮る) are much rarer.
        content = [w for w in heads if not _is_function(w)]
        ranks = [r for r in (self._word_rank(w) for w in content) if r is not None]
        if self.max_ratio > 0 and ranks:
            if rank_e is None or rank_e / max(ranks) > self.max_ratio:
                return False, False
        return True, is_int

    def find(self, words: list[Word], i: int) -> ExprMatch | None:
        n = len(words)
        if words[i].pos0 in BREAK_POS:
            return None
        for j in range(min(n, i + self.max_tokens), i + 1, -1):
            span = words[i:j]
            if any(w.pos0 in BREAK_POS for w in span):
                continue
            if any(w.pos[:2] == ("名詞", "固有名詞") for w in span):
                continue
            if not any(w.pos0 in CONTENT_POS for w in span):
                continue
            if all(w.pos0 in _NOUNISH for w in span):
                continue
            for key, reading in _span_keys(span):
                if key in SET_PHRASES and j - i >= 2:
                    # set phrases JMdict lacks in this exact form (すみませんでした)
                    cands = self.resolver.resolve_joined(key, reading, allow_reading=is_kana(key))
                    if not cands:
                        return ExprMatch(j, key, None, True)
                kana_key = is_kana(key)
                for c in self.resolver.resolve_joined(key, reading, allow_reading=kana_key):
                    if kana_key and c.how.startswith("reading"):
                        # a kana span must be the entry's kana spelling, or its reading
                        # when the entry is usually written in kana (かもしれない)
                        term = c.how.split("→", 1)[1].split("【", 1)[0]
                        if not (term == key or "uk" in c.entry.misc):
                            continue
                        if term != key and not (span[0].pos0 in ("助詞", "助動詞") or "exp" in c.entry.pos
                                                or c.entry.common or _unknown_verb_head(span)):
                            continue
                    if span[0].pos0 in ("助詞", "助動詞") and not ({"exp", "conj"} & c.entry.pos):
                        continue  # か+し ≠ 掠る
                    ok, is_int = self._entry_ok(c.entry, span, key)
                    if ok:
                        # polite ending → prefer the plain-form entry (かもしれません → かもしれない)
                        if key.endswith("ません"):
                            plain = key[:-3] + "ない"
                            for pc in self.resolver.resolve_joined(plain, "", allow_reading=is_kana(plain)):
                                if "exp" in pc.entry.pos:
                                    return ExprMatch(j, plain, pc, is_int)
                        return ExprMatch(j, key, c, is_int)
        return None

    def merge_compound_verbs(self, words: list[Word]) -> list[Word]:
        """Verb stem + verb → a JMdict compound verb (言い+出す → 言い出す, 取り+出す → 取り出す)."""
        out: list[Word] = []
        i = 0
        while i < len(words):
            w = words[i]
            if (i + 1 < len(words) and w.kind == "token" and w.pos0 == "動詞" and len(w.pos) > 5
                    and w.pos[5].startswith("連用形") and words[i + 1].kind == "token" and words[i + 1].pos0 == "動詞"):
                nxt = words[i + 1]
                m2 = nxt.morphs[0]
                hit = None
                for key in dict.fromkeys([w.surface + nxt.lemma, w.surface + m2.norm]):
                    for c in self.resolver.resolve_joined(key, kata_to_hira(w.morphs[0].reading + m2.lemma_reading),
                                                          ("動詞",), allow_reading=False):
                        if c.entry.pos_family() & {"verb"}:
                            hit = (key, c)
                            break
                    if hit:
                        break
                if hit:
                    key, c = hit
                    out.append(Word(
                        surface=w.surface + nxt.surface, lemma=key,
                        reading=kata_to_hira(c.entry.reading) if is_kana(c.entry.reading) else "",
                        pos=nxt.pos, morphs=w.morphs + nxt.morphs, kind="compound", seq=c.seq,
                        note=f"compound verb {w.surface}+{nxt.lemma} → {c.entry.headword}", parts=[w, nxt],
                    ))
                    i += 2
                    continue
            out.append(w)
            i += 1
        return out

    def _clause_conjunction(self, words: list[Word], i: int) -> ExprMatch | None:
        """でも/では/だって at the start of a clause are conjunctions, not で+も."""
        if i > 0 and words[i - 1].pos0 not in BREAK_POS:
            return None
        for j in (i + 2, i + 3):
            span = words[i:j]
            if len(span) < 2 or j > len(words) or not all(w.pos0 in ("助詞", "助動詞") for w in span):
                continue
            key = "".join(w.surface for w in span)
            for c in self.resolver.resolve_joined(key, key, allow_reading=False):
                if "conj" in c.entry.pos:
                    return ExprMatch(j, key, c, False)
        return None

    def merge(self, words: list[Word]) -> list[Word]:
        words = self.merge_compound_verbs(words)
        out: list[Word] = []
        i = 0
        n = len(words)
        while i < n:
            m = self._clause_conjunction(words, i) or self.find(words, i)
            if m is None:
                out.append(words[i])
                i += 1
                continue
            span = words[i:m.end]
            morphs: list[Morph] = [mm for w in span for mm in w.morphs]
            w = Word(
                surface="".join(x.surface for x in span),
                lemma=m.key,
                reading=(kata_to_hira(m.res.entry.reading) if is_kana(m.res.entry.reading) else "") if m.res else m.key,
                pos=("表現", *span[-1].pos[1:]) if span[-1].pos else ("表現",),
                morphs=morphs,
                kind="expression",
                seq=m.res.seq if m.res else None,
                parts=list(span),
                note=f"expression {'+'.join(x.surface for x in span)} → "
                     + (f"{m.res.entry.headword} ({m.res.how})" if m.res else "set phrase (not in JMdict)"),
            )
            if m.interjection:
                w.filter, w.filter_detail = "interjection", "set phrase (JMdict int)"
            elif is_grammar_expression(span) or (m.res is not None and m.res.entry.headword in GRAMMAR_HEADWORDS):
                w.filter, w.filter_detail = "grammar", "grammatical expression"
            out.append(w)
            i = m.end
        return out
