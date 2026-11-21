"""The parsing pipeline: text → word units with JMdict IDs and filter reasons.

  clean (subtitle junk) → normalize → tokenize (Sudachi, mode C)
  → re-segment noun compounds → merge expressions → resolve JMdict entries
  → token filters

Parsing results (everything before the filters) are cached in SQLite, keyed by
a hash of the text plus the parser version, so re-runs on large decks are fast.
"""

from __future__ import annotations

import hashlib
import re
import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable

from ..config import Config
from ..dictdb import DictDB, open_dictdb
from ..textutil import has_kanji, is_kana, is_katakana, kata_to_hira
from .base import Backend, Morph, ParseResult, Word
from .clean import clean_text, katakana_speech, normalize_for_tokenizer
from .compounds import CompoundJudge, Unit, resegment
from .expressions import ExpressionMerger
from .resolve import Resolver, fallback_key

PARSER_VERSION = "0.4.33"  # 0.4.33: a ー that only stretches a known word stays (てよー); 0.4.31: はえー kept as a word, ほっとく; 0.4.30: frequency ranks without other words' kana counts; 0.4: word parts, colloquial ねー/すげー/おめえ, 金, 表, わし; 0.4.5: っ, 憎しみ, あたい, なされる; 0.3: X+さん kinship; 0.2: canonical keys


def make_backend(cfg: Config, db: DictDB | None = None) -> Backend:
    from .sudachi_backend import SudachiBackend

    pc = cfg["parser"]
    return SudachiBackend(pc.get("sudachi_dict", "full"), pc.get("split_mode", "C"))


def _word_from_morph(m: Morph) -> Word:
    return Word(
        surface=m.surface,
        lemma=m.lemma,
        reading=kata_to_hira(m.lemma_reading),
        pos=m.pos,
        morphs=[m],
    )


class ParseCache:
    def __init__(self, path: Path, version_key: str):
        self.path = path
        self.version_key = version_key
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS parse(h TEXT PRIMARY KEY, data TEXT)")
        self.lock = threading.Lock()
        self.pending: list[tuple[str, str]] = []

    def key(self, text: str) -> str:
        return hashlib.sha1((self.version_key + "\x00" + text).encode()).hexdigest()

    def get(self, text: str) -> list[Word] | None:
        r = self.con.execute("SELECT data FROM parse WHERE h=?", (self.key(text),)).fetchone()
        if not r:
            return None
        return [Word.from_json(d) for d in json.loads(r[0])]

    def put(self, text: str, words: list[Word]) -> None:
        self.pending.append((self.key(text), json.dumps([w.to_json() for w in words], ensure_ascii=False)))
        if len(self.pending) >= 500:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        with self.lock:
            self.con.executemany("INSERT OR REPLACE INTO parse VALUES (?,?)", self.pending)
            self.con.commit()
            self.pending.clear()


# Bare kanji Sudachi reads the wrong way regardless of context. A bare 金 is キン (gold)
# to Sudachi, but in dialogue it means money: all 19 キン readings in the test decks
# were かね (その金で, 金さえ出せば). Compounds (金色, 金曜) are separate tokens.
# Readings Sudachi picks for a lone noun that speech almost never means (表 ひょう "chart"
# vs おもて, 米 べい "America" vs こめ "rice").
_READING_FIX = {"金": "カネ", "表": "オモテ", "米": "コメ", "種": "タネ", "綿": "ワタ"}
# Verbs Sudachi reads as a classical (文語) verb in casual speech: 入んだ is はいるんだ.
_CLASSICAL_VERB_FIX = {"入る": "ハイル"}

_E2A = dict(zip("えけげせぜてでねへべぺめれ", "あかがさざただなはばぱまら"))
_E2O = dict(zip("えけげせぜてでねへべぺめれ", "おこごそぞとどのほぼぽもろ"))


class Parser:
    def __init__(self, cfg: Config, db: DictDB | None = None, backend: Backend | None = None,
                 use_cache: bool = True):
        self.cfg = cfg
        self.db = db or open_dictdb(cfg)
        self.backend = backend or make_backend(cfg, self.db)
        self.resolver = Resolver(self.db)
        self.judge = CompoundJudge(
            has_spelling=self.db.has_spelling,
            readings_of=self.db.readings_of_spelling,
            rank=lambda s: self.db.freq_rank(s),
            rank_by_reading=self.db.freq_rank_by_reading,
            is_common=self._common_word,
        )
        pc = cfg["parser"]
        self.expressions = ExpressionMerger(self.resolver, lambda s: self.db.freq_rank(s), self.db.entry_rank,
                                            int(pc.get("max_expression_tokens", 6)),
                                            float(pc.get("expression_max_ratio", 6.0)))
        self.version_key = "|".join([
            PARSER_VERSION,
            self.backend.version(),
            self.db.jmdict_title,
            json.dumps({k: pc.get(k) for k in sorted(pc)}, sort_keys=True),
        ])
        self.cache = ParseCache(cfg.data_dir / "parse_cache.sqlite", self.version_key) if use_cache else None

    def _frequent(self, seq: int) -> bool:
        e = self.db.entry(seq)
        if e is None:
            return False
        r = self.db.entry_rank(seq)
        return e.common or (r is not None and r <= 30000)

    def _common_word(self, t: str) -> bool:
        """A common/frequent JMdict word spelled t, or (kana) read t and usually written in kana (おばさん)."""
        if any(self._frequent(f.seq) for f in self.db.forms_by_term(t)):
            return True
        if is_kana(t):
            return any(self._frequent(f.seq) and "uk" in (self.db.entry(f.seq).misc)
                       for f in self.db.forms_by_reading(t) if f.reading == t)
        return False

    # ------------------------------------------------------------------ stages
    def _units_to_words(self, items: list) -> list[Word]:
        words: list[Word] = []
        for it in items:
            if isinstance(it, Morph):
                words.append(_word_from_morph(it))
                continue
            assert isinstance(it, Unit)
            if len(it.atoms) == 1:
                m = it.atoms[0].morph
                w = _word_from_morph(m)
                if m.a_units and len(m.a_units) > 1 and m.pos[:2] != ("名詞", "固有名詞"):
                    # kept whole (憎しみ, 魅力的, 高さ): its pieces let a known 憎い/魅力/高い cover it
                    w.parts = [_word_from_morph(u) for u in m.a_units]
                words.append(w)
                continue
            surface = it.surface
            joined_reading = "".join(kata_to_hira(a.morph.reading) for a in it.atoms)
            cands = self.resolver.resolve_joined(surface, joined_reading, ("名詞",), allow_reading=is_kana(surface))
            last = it.atoms[-1].morph
            w = Word(
                surface=surface,
                lemma=surface,
                reading=joined_reading,
                pos=("名詞", "普通名詞", "一般", "*", "*", "*") if last.pos[0] != "名詞" else last.pos,
                morphs=[a.morph for a in it.atoms],
                kind="compound",
                parts=[_word_from_morph(a.morph) for a in it.atoms],
                note=f"compound {'+'.join(a.surface for a in it.atoms)}: {it.reason}",
            )
            if cands:
                c = cands[0]
                w.seq = c.seq
                kana_readings = [f.reading for f in self.db.forms_by_term(surface) if is_kana(f.reading)]
                from ..textutil import readings_match

                match = [r for r in kana_readings if readings_match(r, joined_reading, ignore_voicing=False)]
                w.reading = kata_to_hira(match[0] if match else (kana_readings[0] if kana_readings else joined_reading))
            words.append(w)
        return words

    def _resolve(self, words: list[Word]) -> None:
        for i, w in enumerate(words):
            if w.parts:
                self._resolve(w.parts)
            if w.seq is None and w.kind == "token":
                m = w.morphs[0]
                if w.pos0 in ("補助記号", "空白", "記号"):
                    continue
                r = self._colloquial(m)
                nxt = words[i + 1] if i + 1 < len(words) else None
                prv = words[i - 1] if i > 0 else None
                if (r is None and m.lemma == "なれる" and m.norm == "なれる" and prv is not None
                        and (prv.surface in ("に", "と") or (prv.pos0 in ("形容詞", "副詞") and prv.surface.endswith("く")))):
                    # 先生になれる, 仲良くなれる: potential of なる, not 慣れる "get used to"
                    r = self.resolver.resolve("成る", "成る", "なる", "ナル", m.pos)
                    if r is not None:
                        r.how = f"potential なれる→成る; {r.how}"
                if (r is None and len(m.surface) == 1 and has_kanji(m.surface) and m.pos[0] == "名詞"
                        and nxt is not None and nxt.surface in ("っ", "ッ")):
                    # 寒っ, 怖っ: an adjective cut short, which Sudachi reads as a bare noun
                    adj = self.resolver.resolve(m.surface + "い", m.surface + "い", m.surface + "い", "", ("形容詞",))
                    if adj is not None and "adj-i" in adj.entry.pos:
                        r = adj
                        r.how = f"clipped adjective {m.surface}っ→{m.surface}い; {r.how}"
                if r is None and m.pos[0] == "連体詞" and m.surface.endswith("ような") and len(m.surface) == 5:
                    # どのような, このような: the こ/そ/あ/ど word (ような is grammar)
                    r = self.resolver.resolve(m.surface[:2], m.surface[:2], m.surface[:2], "", m.pos)
                if r is None and m.surface == "あり" and m.pos[0] == "名詞":
                    # 噂あり, 影あり, 異常あり: 有る (an ant is written アリ)
                    r = self.resolver.resolve("有る", "有る", "ある", "アル", ("動詞", "非自立可能"))
                    if r is not None:
                        r.how = f"あり (existence)→有る; {r.how}"
                if r is None and m.surface == "なさ" and m.norm == "成す" and nxt is not None and nxt.lemma == "れる":
                    # どうなされます: honorific なさる + れる, not 成す "to accomplish"
                    r = self.resolver.resolve("為さる", "為さる", "なさる", "ナサル", m.pos)
                    if r is not None:
                        r.how = f"honorific なされる→為さる; {r.how}"
                if r is None:
                    lr = m.lemma_reading
                    if m.pos[0] == "名詞":
                        lr = _READING_FIX.get(m.surface, lr)
                    elif m.pos[0] == "動詞" and m.pos[4].startswith("文語"):
                        lr = _CLASSICAL_VERB_FIX.get(m.norm, lr)
                    r = self.resolver.resolve(m.lemma, m.norm, m.surface, lr, m.pos)
                    if r is not None and lr != m.lemma_reading:
                        w.reading = kata_to_hira(lr)
                if r is None:
                    # casual spelling: バーカ → バカ, キラっ → キラ, すげー → すげ(ー)
                    for v in _spelling_variants(m.surface):
                        r = self.resolver.resolve(v, v, v, _kata(v), m.pos, strict_reading=True)
                        if r is not None:
                            r.how = f"variant {m.surface}→{v}; {r.how}"
                            break
                if r is None:
                    # repeated word (痛い痛い, 何何何): the word itself
                    mm = re.fullmatch(r"(.+?)\1+", m.surface)
                    if mm and (has_kanji(mm.group(1)) or len(mm.group(1)) >= 3):
                        u = mm.group(1)
                        r = self.resolver.resolve(u, u, u, "", None)
                        if r is not None:
                            r.how = f"repeated {u}; {r.how}"
                if r is not None:
                    w.seq = r.seq
                    w.note = r.how
            if w.seq is not None:
                e = self.db.entry(w.seq)
                if e is not None:
                    w.headword = e.headword
                    w.gloss = e.gloss
                w.key = f"j{self.db.canonical_seq(w.seq)}"
            else:
                m = w.morphs[0] if w.morphs else None
                w.key = fallback_key(m.norm if m and w.kind == "token" else w.lemma, w.pos)

    def _colloquial(self, m):
        """Slurred adjective/negative endings resolve to the standard word Sudachi normalizes
        them to (すげー → 凄い, ねー → 無い, うるせえ → 煩い), not to JMdict's separate
        colloquial entries: a learner who knows 凄い knows すげー."""
        from ..derive import first_sense_covered

        if m.pos[0] in ("助詞", "補助記号", "記号", "空白") or m.pos[:2] == ("名詞", "固有名詞"):
            return None
        own = None
        if m.norm and m.norm != m.lemma and re.search(r"[ーぇぃぉ]|[えい]$", m.lemma) \
                and m.pos[0] in ("形容詞", "助動詞") and (has_kanji(m.norm) or m.norm == "ない"):
            r = self.resolver.resolve(m.norm, m.norm, m.norm, "", m.pos)
            if r is not None:
                r.how = f"colloquial {m.lemma}→{m.norm}; {r.how}"
                return r
        elif m.norm and m.norm != m.lemma and has_kanji(m.norm) and m.pos[0] != "助動詞":
            # Sudachi's normalized form names the standard word (やっぱ → 矢張り, 愛す → 愛する);
            # JMdict's separate entry for the variant is used only if the meanings differ.
            own = self.resolver.resolve(m.lemma, m.lemma, m.surface, m.lemma_reading, m.pos)
            r = self.resolver.resolve(m.norm, m.norm, m.norm, m.lemma_reading, m.pos) \
                or self.resolver.resolve(m.norm, m.norm, m.norm, "", m.pos)
            if r is not None and own is not None and r.seq != own.seq and first_sense_covered(own.entry, r.entry):
                r.how = f"variant {m.lemma}→{m.norm}; {r.how}"
                return r
        if m.pos[0] not in ("形容詞", "助動詞", "代名詞"):
            return None
        # Sound changes Sudachi leaves alone: おめえ → おまえ, てめえ → てまえ (ae/ai/oi → ee).
        # The standard word must mean what JMdict's colloquial entry means, if it has one.
        mm = re.fullmatch(r"([ぁ-ゖ]*)([えけげせぜてでねへべぺめれ])[ーえぇ]", kata_to_hira(m.lemma))
        if not mm:
            return None
        head, c = mm.groups()
        if own is None:
            own = self.resolver.resolve(m.lemma, m.lemma, m.lemma, m.lemma_reading, m.pos)
        for v in (head + _E2A[c] + "え", head + _E2A[c] + "い", head + _E2O[c] + "い"):
            r = self.resolver.resolve(v, v, v, _kata(v), m.pos, strict_reading=True)
            if r is None or (own is not None and not first_sense_covered(own.entry, r.entry)):
                continue
            r.how = f"colloquial {m.lemma}→{v}; {r.how}"
            return r
        return None

    def _score_chunk(self, chunk: str) -> float:
        ms = self.backend.tokenize(chunk)
        bad = sum(1 for m in ms if m.pos[0] in ("補助記号", "記号") or m.oov)
        return len(ms) + 3 * bad

    def _stretch_only(self, chunk: str) -> bool:
        """Every ー stretches a sentence-final particle after a finished phrase (付き合ってよー,
        雑かよー, ないよー) or the colloquial negative ねー (じゃねーよ, 知らねー), and the
        tokenizer already reads it so. Respelling would let the chunk, scored without the rest of
        the sentence, merge into something else (付き合ってよう: てよう, then よう on its own).
        The tokenizer calls fragments particles too, so the particle must end the chunk (not
        待ちなさーい: さー + い), and the chunk must start with a word: not a filler or a stray
        kana (まさかー: ま + さ + かー; つよー = 強い), and not after は (おっはよー)."""
        ms = list(self.backend.tokenize(chunk))
        while ms and ms[-1].pos[0] in ("補助記号", "記号"):  # くだらねーっ
            ms.pop()
        if not ms or not any("ー" in m.surface for m in ms):
            return False
        first = ms[0]
        word = first.pos[0] in ("動詞", "形容詞", "形状詞", "名詞", "代名詞", "副詞")
        if not word or (not has_kanji(first.surface)
                        and (first.pos[0] == "名詞" or (len(first.surface) < 2 and first.pos[0] != "動詞"))):
            return False
        for i, m in enumerate(ms):
            if "ー" not in m.surface:
                continue
            if i == 0 or m.oov:
                return False
            prev = ms[i - 1]
            # after a particle, or a sentence-ending form (ないよー; not めんどく + さー, a cut adjective)
            ends = prev.pos[0] == "助詞" or (prev.pos[0] in ("助動詞", "形容詞")
                                             and prev.pos[5].startswith(("終止形", "意志推量形")))
            particle = m.pos[1] == "終助詞" and i == len(ms) - 1 and ends and prev.pos[1] != "係助詞"
            negative = m.surface == "ねー" and m.pos[0] in ("助動詞", "形容詞") and prev.pos[0] in ("動詞", "助動詞")
            if not (particle or negative):
                return False
        return True

    def _fix_elongation(self, text: str) -> str:
        """Pick a spelling for ー inside hiragana words (おいしーい → おいしい, そーなんだ → そうなんだ)
        by asking the tokenizer which variant segments most cleanly. Katakana ー is untouched."""
        from ..textutil import vowel_of

        def spelled(chunk: str, k: int) -> bool:
            """The run up to this ー is a dictionary spelling of a word (はえー = 早い): keep a
            vowel there. Only from the start of the run, and not a particle (みんなー: なー)."""
            for f in self.db.forms_by_term(chunk[:k + 1]):
                e = self.db.entry(f.seq)
                if e is not None and not e.pos <= {"prt", "int", "exp"}:
                    return True
            return False

        def variants(chunk: str) -> list[str]:
            out = [chunk]
            k = chunk.find("ー")
            while k > 0:
                prev = chunk[k - 1]
                keep = spelled(chunk, k)
                if "\u3400" <= prev <= "\u9fff" or prev == "々":
                    # 違ーう → 違う: a stretched kanji word just drops the ー
                    out = out + [c[:c.find("ー")] + c[c.find("ー") + 1:] for c in out if c.find("ー") > 0]
                elif "ぁ" <= prev <= "ゖ":
                    base = []
                    v = vowel_of(prev)
                    # full vowel first: on a tie the longer spelling wins (すげえ, not すげ = 菅)
                    alts = [v or "", ""]
                    if v == "お":
                        alts.insert(1, "う")
                    if v == "え":
                        alts.insert(1, "い")
                    if keep:
                        alts = [a for a in alts if a]  # はええよ, not はえよ (生える)
                    for c in out:
                        kk = c.find("ー")
                        if kk <= 0:
                            continue
                        for a in dict.fromkeys(alts):
                            base.append(c[:kk] + a + c[kk + 1:])
                    out = out + base
                # cap every step: each ー multiplies the spellings (うわーぁーぁー… had 3^18)
                out = list(dict.fromkeys(out))[:12]
                k = chunk.find("ー", k + 1)
            return out

        def fix(m: "re.Match[str]") -> str:
            chunk = m.group(0)
            if self._stretch_only(chunk):
                return chunk
            best, best_score = chunk, self._score_chunk(chunk) - 0.5  # small bias to the original
            for v in variants(chunk)[1:]:
                sc = self._score_chunk(v)
                if sc < best_score:
                    best, best_score = v, sc
            return best

        import re

        return re.sub(r"(?:[\u3400-\u9fff々]+[ぁ-ゖ]*|[ぁ-ゖ]+)ー[ぁ-ゖー]*", fix, text)

    def _kata_junk(self, text: str) -> int:
        """Katakana tokens that are no word: ンデイルトノ in 鬼ガ潜ンデイルトノ噂."""
        n = 0
        for m in self.backend.tokenize(text):
            s = m.surface
            if s and is_katakana(s) and (s[0] in "ンッァィゥェォャュョヮー" or m.oov or not self.db.has_spelling(s)):
                n += 1
        return n

    def _is_adj_i(self, spelling: str) -> bool:
        return any((e := self.db.entry(f.seq)) is not None and "adj-i" in e.pos for f in self.db.forms_by_term(spelling))

    def _fix_sokuon(self, text: str) -> str:
        """A trailing っ on a shouted word (行けっ, 待てっ, 逃げろっ) breaks the verb apart
        (行 + けっ). Drop it when the tokenizer segments the word more cleanly without it;
        interjections (あっ, くっ, ちょっ) segment the same either way and keep it."""

        def fix(m: "re.Match[str]") -> str:
            chunk = m.group(0)
            # 寒っ, うわ怖っ, ちょっと痛っ: an い-adjective cut short (not 思っ… or 行っ…, verb stems)
            cut = re.search(r"(?<![\u3400-\u9fff々])([\u3400-\u9fff])っ$", chunk)
            if cut and self._is_adj_i(cut.group(1) + "い"):
                return chunk[:cut.start()] + cut.group(1) + "い"
            if not has_kanji(chunk) and len(chunk) < 4:
                return chunk  # short kana shouts are interjections: あだっ, はえっ, うわっ
            return chunk[:-1] if self._score_chunk(chunk[:-1]) < self._score_chunk(chunk) else chunk

        return re.sub(r"[^\s　、。，．,.！？!?…‥・「」『』“”\"]+っ(?=[\s　、。，．,.！？!?…‥・」』”\"]|$)", fix, text)

    def parse_words(self, cleaned: str) -> list[Word]:
        """Segmentation + JMdict resolution (no filters). Cached."""
        if self.cache is not None:
            hit = self.cache.get(cleaned)
            if hit is not None:
                return hit
        text = normalize_for_tokenizer(cleaned)
        ks = katakana_speech(text)
        if ks is not None and (self._kata_junk(text) > self._kata_junk(ks) or self._score_chunk(ks) < self._score_chunk(text)):
            text = ks
        if "ー" in text:
            text = self._fix_elongation(text)
        if "っ" in text:
            text = self._fix_sokuon(text)
        morphs = self.backend.tokenize(text) if text else []
        pc = self.cfg["parser"]
        if pc.get("compound_resegment", True):
            items = resegment(morphs, self.judge)
        else:
            items = list(morphs)
        words = self._units_to_words(items)
        if pc.get("merge_expressions", True):
            words = self.expressions.merge(words)
        self._resolve(words)
        if self.cache is not None:
            self.cache.put(cleaned, words)
        return words

    def parse(self, text: str, token_filter=None, clean: bool = True) -> ParseResult:
        c = clean_text(text) if clean else None
        cleaned = c.text if c else text
        words = self.parse_words(cleaned)
        res = ParseResult(text=text, cleaned=cleaned, words=words,
                          has_music=c.has_music if c else False,
                          removed=c.removed if c else [], annotated=c.annotated if c else [])
        if token_filter is not None:
            token_filter.apply(words, annotated=c.annotated if c else ())
        return res

    def parse_many(self, texts: Iterable[str], token_filter=None) -> list[ParseResult]:
        out = [self.parse(t, token_filter) for t in texts]
        self.flush()
        return out

    def flush(self) -> None:
        if self.cache is not None:
            self.cache.flush()


def _kata(s: str) -> str:
    from ..textutil import hira_to_kata

    return hira_to_kata(s)


def _spelling_variants(s: str) -> list[str]:
    """Candidate standard spellings for casual/elongated subtitle forms."""
    out: list[str] = []
    t = s.rstrip("っッ")
    if t and t != s:
        out.append(t)
    if "ー" in s[1:]:
        u = s[0] + s[1:].replace("ー", "")
        out.append(u)
        if u.rstrip("っッ") != u:
            out.append(u.rstrip("っッ"))
    return [v for v in dict.fromkeys(out) if len(v) >= 2]
