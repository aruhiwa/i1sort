"""Token-level filters: words that never count as unknown.

Every filtered word gets a reason code (``Word.filter``) and a human-readable
detail (``Word.filter_detail``) so ``i1sort explain`` can show why.

Reason codes and their config switches ([filters] in config.toml). Loanwords are not
filtered: they count as known words (loanwords.py, ``[known] loanwords``), and so do
very frequent words (``[known] auto_known_rank``).

  particle      particles                       (particles)
  aux           auxiliary verbs                 (aux_verbs)
  symbol        punctuation / symbols           (symbols)
  whitespace    whitespace                      (whitespace)
  grammar       subsidiary verbs after て, する after a noun, そう/よう/みたい,
                grammatical expressions (ことがある)   (grammar)
  interjection  感動詞 incl. fillers (えっ, あの, まあ)  (interjections)
  name          proper nouns, JMnedict names, per-deck names   (names)
  honorific     さん/くん/ちゃん/様 ... after a name or noun   (honorifics)
  number        numerals and counters           (numbers)
  affix         lone prefixes/suffixes          (affixes)
  non_japanese  Latin text, digits, emoji       (non_japanese)
  onomatopoeia  JMdict on-mim words             (onomatopoeia, off by default)
  ignored       in an ignore_files list (exact text)   (ignore_list)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import Config
from .dictdb import DictDB
from .parse.base import Word
from .textutil import has_japanese, has_kanji, is_hiragana, is_kana, is_katakana, is_latin, kata_to_hira

SWITCH = {
    "particle": "particles",
    "aux": "aux_verbs",
    "symbol": "symbols",
    "whitespace": "whitespace",
    "grammar": "grammar",
    "interjection": "interjections",
    "name": "names",
    "honorific": "honorifics",
    "number": "numbers",
    "affix": "affixes",
    "non_japanese": "non_japanese",
    "onomatopoeia": "onomatopoeia",
    "ignored": "ignore_list",
}

HONORIFICS = frozenset(
    "さん 様 さま サマ 君 くん クン ちゃん チャン 氏 殿 どの たん 先輩 せんぱい 先生 嬢 坊 ちゃま 兄 姉 殿下 閣下 卿 たち 達 ら".split()
)
# Titles that are real words when they stand alone; only filtered right after a name.
TITLE_SUFFIXES = frozenset("先輩 せんぱい 先生 兄 姉 殿下 閣下 卿".split())

SUBSIDIARY_AFTER_TE = frozenset(
    "いる 居る る おる ある 有る おく 置く とく しまう 仕舞う ちゃう じゃう くる 来る いく 行く く みる 見る "
    "あげる 上げる やる くれる 呉れる もらう 貰う いただく 頂く くださる 下さる ほしい 欲しい いい 良い よい "
    "ちょうだい 頂戴 ございます 御座います".split()
)
# Sentence-initial particles Sudachi tags as conjunctions: で、 が、 と、 けど…
PARTICLE_CONJ = frozenset("で が と て けど けれど けれども し から だが ですが だけど でも だって じゃ じゃあ んじゃ んじゃあ では".split())
# Call words that are interjections when they stand alone between pauses (ねえ、 / なあ…)
ALONE_INTERJ = frozenset("ね ねえ ねー な なあ なー おい ほら あれ よ よし よーし よお よおっ よっ".split())
_NAME_MISC = frozenset("organization company product work char person place given surname fem masc serv "
                       "group creat dei ev fict leg myth obj oth relig ship station unclass doc".split())


FILLERS = frozenset("あの あのう あのー その そのー そのう まあ まぁ ええと えっと えと なんか ほら".split())
SURU_LEMMAS = frozenset("する 為る できる 出来る なさる 致す いたす される".split())


def load_ignore_set(cfg: Config) -> set[str]:
    """Every [filters] ignore_files list (exact text: surface, dictionary form or headword)."""
    from .config import as_list

    out: set[str] = set()
    for f in as_list(cfg["filters"].get("ignore_files")):
        out |= load_wordlist(cfg.resolve(f))
    return out


def load_wordlist(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        out.add(line.split(",")[0].split("\t")[0].strip())
    return out


class TokenFilter:
    def __init__(self, cfg: Config, db: DictDB, deck_names: Iterable[str] = (), ignore: Iterable[str] | None = None):
        self.cfg = cfg
        self.db = db
        self.sw = cfg["filters"]
        if ignore is None:
            ignore = load_ignore_set(cfg)
        self.ignore = set(ignore)
        self.deck_names = set(deck_names)

    def enabled(self, reason: str) -> bool:
        key = SWITCH.get(reason)
        return True if key is None else bool(self.sw.get(key, True))

    # ------------------------------------------------------------------ helpers
    def _is_suru_noun(self, w: Word) -> bool:
        if len(w.pos) > 2 and w.pos[0] == "名詞" and w.pos[2] == "サ変可能":
            return True
        if w.seq:
            e = self.db.entry(w.seq)
            if e and any(t.startswith("vs") for t in e.pos):
                return True
        return False

    def _is_name_token(self, w: Word) -> str | None:
        p = w.pos
        if w.kind == "expression":
            return None
        if len(p) > 2 and p[0] == "名詞" and p[1] == "固有名詞":
            if p[2] == "人名":
                return f"Sudachi proper noun (人名{'/' + p[3] if p[3] != '*' else ''})"
            if p[2] == "地名":
                # Countries and big cities (日本, 東京) are common JMdict words: count them.
                e = self.db.entry(w.seq) if w.seq else None
                if not (e and e.common):
                    return f"Sudachi proper noun (地名{'/' + p[3] if p[3] != '*' else ''})"
                # a common place word can still be this deck's character name (青山くん)
            # 固有名詞/一般 (organisations, products...): a name unless it is a common word
            e = self.db.entry(w.seq) if w.seq else None
            if not (e and e.common):
                return "Sudachi proper noun (一般)"
        if w.surface in self.deck_names or w.lemma in self.deck_names:
            return "deck name list"
        if w.seq is None and w.kind == "token" and p and p[0] == "名詞" and is_katakana(w.surface) \
                and len(w.surface.replace("ー", "")) >= 2:
            return "unknown katakana word (likely a name)"
        # JMnedict: nouns / unknown tokens in katakana or with kanji that are not common words.
        if p and (p[0] in ("名詞", "接頭辞") or w.morphs and w.morphs[0].oov) and w.kind == "token":
            s = w.surface
            if (is_katakana(s) and len(s) >= 2) or (has_kanji(s) and w.morphs[0].oov):
                names = self.db.names_by_term(s)
                if names:
                    from .loanwords import source_labels

                    e = self.db.entry(w.seq) if w.seq else None
                    # a common word, or a JMdict loanword (グリッチ), is not a name
                    if not (e and (e.common or source_labels(self.db, w.seq))):
                        types = sorted({t for n in names for t in n.types})
                        return f"JMnedict ({', '.join(types[:3])})"
        return None

    def _deck_name_spans(self, words: list[Word]) -> dict[int, str]:
        """Tokens covered by a multi-token deck name (流+河 for 流河)."""
        out: dict[int, str] = {}
        multi = [n for n in self.deck_names if len(n) >= 2]
        if not multi:
            return out
        starts = []
        pos = 0
        for w in words:
            starts.append(pos)
            pos += len(w.surface)
        text = "".join(w.surface for w in words)
        for name in multi:
            k = text.find(name)
            while k >= 0:
                end = k + len(name)
                idx = [i for i, st in enumerate(starts) if st >= k and st + len(words[i].surface) <= end]
                # only when the name lines up with token boundaries and spans 2+ tokens
                if len(idx) >= 2 and starts[idx[0]] == k and starts[idx[-1]] + len(words[idx[-1]].surface) == end:
                    for i in idx:
                        out.setdefault(i, f"deck name {name}")
                k = text.find(name, k + 1)
        return out

    def _sentence_names(self, words: list[Word], annotated: Iterable[str]) -> dict[int, str]:
        """Names visible only in this sentence:
        * kanji with an inline reading in the subtitle: 鬼舞辻無惨(きぶつじむざん)
        * a run of nouns right before an honorific that is not itself a word: 夜神+さん
        """
        out: dict[int, str] = {}
        ann = [a for a in annotated if a and not _common_word(self.db, a)]
        if ann:
            pos = 0
            for i, w in enumerate(words):
                start, end = pos, pos + len(w.surface)
                pos = end
                for a in ann:
                    k = "".join(x.surface for x in words).find(a)
                    if k >= 0 and start >= k and end <= k + len(a) and w.pos0 not in ("助詞", "助動詞", "補助記号", "空白"):
                        out[i] = f"annotated reading ({a})"
        for i, w in enumerate(words):
            if w.pos0 != "接尾辞" or w.surface not in HONORIFICS:
                continue
            j = i
            while j > 0 and words[j - 1].kind == "token" and words[j - 1].pos0 == "名詞" \
                    and words[j - 1].pos[1] != "数詞" and not is_hiragana(words[j - 1].surface):
                j -= 1
            run = words[j:i]
            if len(run) >= 2:
                joined = "".join(x.surface for x in run)
                if not _common_word(self.db, joined) and not self.db.has_spelling(joined):
                    for k in range(j, i):
                        out.setdefault(k, f"before honorific ({joined}{w.surface})")
        return out

    # ------------------------------------------------------------------ main
    def apply(self, words: list[Word], annotated: Iterable[str] = ()) -> list[Word]:
        sent_names = {**self._deck_name_spans(words), **self._sentence_names(words, annotated)} if self.enabled("name") else {}
        for i, w in enumerate(words):
            if i in sent_names and w.filter is None:
                w.filter, w.filter_detail = "name", sent_names[i]
                continue
            if w.filter is not None:  # set upstream (grammar expressions)
                if not self.enabled(w.filter):
                    w.filter, w.filter_detail = None, ""
                continue
            reason, detail = self._classify(words, i)
            if reason and self.enabled(reason):
                w.filter, w.filter_detail = reason, detail
                continue
            # Later checks apply to otherwise-counted words.
            if self.enabled("ignored") and self.ignore and (
                w.lemma in self.ignore or w.surface in self.ignore or (w.headword and w.headword in self.ignore)
            ):
                w.filter, w.filter_detail = "ignored", "in ignore list"
                continue
        # Short kanji nouns glued to a neighbouring noun: pieces of an unlisted compound.
        nounish = ("名詞", "接尾辞", "接頭辞")
        for i, w in enumerate(words):
            w.bound = ""
            if w.filter is not None or w.kind != "token" or w.pos[:2] != ("名詞", "普通名詞") \
                    or len(w.surface) > 2 or not has_kanji(w.surface) or not all(has_kanji(ch) for ch in w.surface):
                continue
            if i > 0 and words[i - 1].pos0 in nounish and words[i - 1].filter not in ("symbol", "whitespace"):
                w.bound = "suffix"
            elif i + 1 < len(words) and words[i + 1].pos0 in nounish and words[i + 1].filter not in ("symbol", "whitespace"):
                w.bound = "prefix"
        # Full names written with a middle dot: ディノ・カバネ, ミツルギ・キョウヤ
        if self.enabled("name"):
            for i, w in enumerate(words):
                if w.filter is None and is_katakana(w.surface) and w.pos0 == "名詞":
                    for d in (-2, 2):
                        j = i + d
                        if 0 <= j < len(words) and words[(i + j) // 2].surface in ("・", "･", "·", "=", "＝") \
                                and words[j].filter == "name":
                            w.filter, w.filter_detail = "name", f"joined to {words[j].surface} by ・"
                            break
        return words

    def _classify(self, words: list[Word], i: int) -> tuple[str | None, str]:
        w = words[i]
        p = w.pos
        p0 = w.pos0
        p1 = p[1] if len(p) > 1 else ""
        prev = words[i - 1] if i > 0 else None

        if p0 == "空白" or not w.surface.strip():
            return "whitespace", ""
        if p0 in ("補助記号", "記号"):
            return "symbol", ""
        if not has_japanese(w.surface):
            if any(ch.isdigit() for ch in w.surface) and p1 == "数詞":
                return "number", "numeral"
            return "non_japanese", "no Japanese characters"
        if p0 == "名詞" and p1 == "数詞":
            return "number", "numeral"
        if w.kind in ("token", "compound") and prev is not None and prev.pos[:2] == ("名詞", "数詞"):
            fp = w.morphs[0].pos if w.morphs else p
            if (fp[0] == "接尾辞" and fp[1] == "名詞的") or (len(fp) > 2 and fp[2] == "助数詞可能") or (
                w.seq and "ctr" in (self.db.entry(w.seq).pos if self.db.entry(w.seq) else ())
            ):
                return "number", f"counter after {prev.surface}"
        if p0 == "助詞":
            # ね/ねえ/な opening a sentence is a call for attention, not a final particle
            if p1 == "終助詞" and w.surface in ("ね", "ねえ", "ねー", "な", "なあ") and (
                prev is None or prev.pos0 in ("補助記号", "空白")
            ):
                return "interjection", "sentence-initial ね/な"
            return "particle", p1
        if p0 == "助動詞":
            return "aux", ""
        if p0 == "感動詞":
            return "interjection", p1 if p1 not in ("一般", "*") else ""
        if p0 == "接続詞" and w.surface in PARTICLE_CONJ:
            return "grammar", "particle-like conjunction"
        # A particle used as an adverb or conjunction (より多く, っと…): JMdict files these
        # under the particle's entry, and they are grammar, not vocabulary.
        if p0 in ("副詞", "接続詞") and w.kind == "token" and is_kana(w.surface) and len(w.surface.lstrip("っ")) <= 3 \
                and w.seq and "prt" in (self.db.entry(w.seq).pos if self.db.entry(w.seq) else ()):
            return "grammar", f"particle used as {'adverb' if p0 == '副詞' else 'conjunction'}"
        # Hesitation fillers tagged as other POS: あの、/ まあ、/ その…
        nxt = words[i + 1] if i + 1 < len(words) else None
        brk = ("補助記号", "空白")
        alone = (prev is None or prev.pos0 in brk) and (nxt is None or nxt.pos0 in brk)
        if w.surface in ALONE_INTERJ and alone:
            return "interjection", "call word standing alone"
        # み～, にぱ: a single kana between pauses is a sound, not a word
        if alone and w.kind == "token" and len(w.surface) == 1 and is_kana(w.surface) and p0 in ("名詞", "副詞", "代名詞"):
            return "interjection", "single kana standing alone"
        entry = self.db.entry(w.seq) if w.seq else None
        if w.kind == "token" and is_kana(w.surface):
            # よう！ へへ… きゃっ: a kana word standing alone that JMdict also files as an interjection
            if alone and p0 in ("副詞", "名詞", "接頭辞") and any("int" in (self.db.entry(f.seq).pos if self.db.entry(f.seq) else ())
                             for f in self.db.forms_by_term(kata_to_hira(w.surface))):
                return "interjection", "standing alone, JMdict interjection"
            # そこかっ！ お前かーっ: a sentence-final particle stretched by the speaker
            core = w.surface.rstrip("っッーぁぃぅぇぉ〜～")
            if core != w.surface and core in ("か", "な", "よ", "ね", "ぞ", "ぜ", "わ", "さ") \
                    and prev is not None and prev.pos0 not in brk and (nxt is None or nxt.pos0 in brk):
                return "particle", "stretched 終助詞"
            # か in 何もかも resolved to 蚊: a single kana is not written for a kanji word
            if len(w.surface) == 1 and w.headword and has_kanji(w.headword) and p0 not in ("動詞", "形容詞", "接頭辞", "接尾辞"):
                return "fragment", f"single kana read as {w.headword}"
        # copulas/auxiliaries Sudachi tags as verbs (すべきっす)
        if w.kind == "token" and entry is not None and entry.pos & {"aux-v", "cop"} \
                and entry.pos <= {"aux-v", "cop", "aux", "aux-adj", "exp"}:
            return "aux", f"JMdict auxiliary ({entry.headword})"
        # Stutters: カ… カズマ, そ　空, こ...凍る — a short kana fragment, a pause, then a word
        # starting with the same sound
        if w.kind == "token" and len(w.surface.rstrip("っッー")) <= 2 and is_kana(w.surface) and nxt is not None and nxt.pos0 in brk:
            j = i + 1
            while j < len(words) and words[j].pos0 in brk:
                j += 1
            if j < len(words):
                frag = kata_to_hira(w.surface.rstrip("っッー"))
                rd = kata_to_hira(words[j].morphs[0].reading if words[j].morphs else words[j].reading)
                if frag and (rd.startswith(frag) or kata_to_hira(words[j].surface).startswith(frag)):
                    return "interjection", f"stutter before {words[j].surface}"
        if w.surface in FILLERS and (nxt is None or nxt.pos0 in ("補助記号", "空白")):
            return "interjection", "filler"
        if p0 == "形状詞" and p1 == "助動詞語幹":
            return "grammar", "auxiliary stem (そう/よう/みたい)"
        # subsidiary verbs/adjectives after て/で (食べている, 見てもいい, 書いてある)
        if p0 in ("動詞", "形容詞") and p1 == "非自立可能" and w.kind == "token":
            j = i - 1
            while j >= 0 and words[j].pos0 == "助詞" and words[j].surface in ("は", "も"):
                j -= 1
            if j >= 0 and words[j].pos0 == "助詞" and words[j].surface in ("て", "で", "って") and (
                w.lemma in SUBSIDIARY_AFTER_TE or w.morphs[0].norm in SUBSIDIARY_AFTER_TE
            ):
                return "grammar", f"subsidiary {w.lemma} after て"
        # せっかくだし, 暇だし: copula だ + し, which Sudachi reads as the noun 出し (broth)
        if w.surface == "だし" and w.kind == "token" and prev is not None and prev.pos0 in ("名詞", "副詞", "形状詞", "代名詞"):
            return "grammar", "だし (copula + し)"
        # ロックじゃが, そうじゃが: old-speech copula じゃ + が, which Sudachi reads as the noun じゃが (potato)
        if w.surface == "じゃが" and w.kind == "token" and prev is not None and prev.pos0 in ("名詞", "形状詞", "代名詞", "副詞") \
                and (nxt is None or nxt.pos0 in brk):
            return "grammar", "じゃが (copula + が)"
        # prohibition: してはいけない, しちゃいけない, 見てはならない
        if p0 == "動詞" and w.kind == "token" and (
            w.lemma in ("いける", "行ける") or (w.lemma in ("なる", "成る") and w.surface == "なら")
        ):
            j = i - 1
            if j >= 0 and words[j].pos0 == "助詞" and (
                words[j].surface in ("ちゃ", "じゃ", "ちゃあ", "じゃあ")
                or (words[j].surface in ("は", "も") and j >= 1 and words[j - 1].surface in ("て", "で"))
            ):
                return "grammar", "prohibition (てはいけない)"
        # 〜なさい after a verb stem (言いなさい): honorific imperative
        if p0 == "動詞" and w.kind == "token" and w.lemma in ("なさる", "為さる") and prev is not None and prev.pos0 == "動詞":
            return "grammar", "なさい after a verb"
        # negation ない after an adverbial く-form or で/じゃ(+は/も): 走りたくない, 静かではない
        if p0 == "形容詞" and (w.lemma in ("ない", "無い") or w.morphs and w.morphs[0].norm == "無い") and w.kind == "token":
            j = i - 1
            while j >= 0 and words[j].pos0 == "助詞" and words[j].surface in ("は", "も"):
                j -= 1
            if j >= 0 and (
                (words[j].pos0 in ("形容詞", "助動詞") and words[j].surface.endswith("く"))
                or (words[j].pos0 in ("助動詞", "助詞") and words[j].surface in ("で", "じゃ", "では"))
            ):
                return "grammar", "negation after く/で"
        # する after に/く (静かにする, 早くする): "make/decide", not vocabulary
        if p0 == "動詞" and w.kind == "token" and (w.lemma in ("する", "為る") or w.morphs[0].norm == "為る"):
            if prev is not None and ((prev.pos0 == "助動詞" and prev.surface == "に") or
                                     (prev.pos0 == "形容詞" and prev.surface.endswith("く"))):
                return "grammar", f"する after {prev.surface}"
        # する after a suru-verb noun/adverb: 勉強する, ドキドキする (but not どうする, ことする)
        if p0 == "動詞" and w.kind == "token" and (w.morphs[0].norm in SURU_LEMMAS or w.lemma in SURU_LEMMAS):
            if prev is not None and prev.pos[:2] != ("名詞", "数詞") and self._is_suru_noun(prev):
                return "grammar", f"する-verb ({prev.surface}{w.lemma})"
        # うひひ → 狒々, ふわああ → 不和, にこ (a name) → 二胡: a two-kana noun/adverb matched by
        # reading alone to a kanji word that is rarely written in kana is a sound or name fragment
        # (hiragana only: katakana is the normal spelling of コツ, タコ, ノリ, ハゲ)
        if (w.kind == "token" and is_hiragana(w.surface) and len(w.surface) == 2 and p0 in ("名詞", "副詞")
                and w.seq is not None and w.headword and has_kanji(w.headword)
                # a real word takes a particle or an お/ご prefix: おわび, きめが細かい, 知るよしもなく
                and not (prev is not None and prev.pos0 == "接頭辞")
                and not (nxt is not None and nxt.pos0 == "助詞" and nxt.surface in ("が", "を", "に", "は", "も", "の", "と", "で", "として"))):
            kr = self.db.entry_kana_rank(w.seq, kata_to_hira(w.surface))
            is_verb = entry is not None and any(x.startswith("v") for x in entry.pos)  # 噂あり → 有る
            if (kr is None or kr > 10000) and not is_verb:
                return "fragment", f"two kana read as {w.headword} (rarely written in kana)"
        # unresolved grunts and laughter: ぐっ, がっ, フフフフ, ウハハハ
        if w.seq is None and w.kind == "token" and (
            (len(w.surface) <= 3 and w.surface[-1] in "っッ" and is_kana(w.surface))
            or (is_kana(w.surface) and len(set(w.surface.rstrip("っッー"))) <= 2 and len(w.surface) >= 3)
            or (is_kana(w.surface) and len(w.surface) >= 2 and len(set(w.surface.rstrip("っッー"))) == 1)  # ぐぐ
        ):
            return "interjection", "exclamation"
        # ヒヒッ, ヘヘッ, フフッ: a repeated syllable cut off by ッ is a laugh, whatever it resolves to
        core = w.surface.rstrip("っッー")
        if w.kind == "token" and is_kana(w.surface) and w.surface[-1:] in "っッ" and 2 <= len(core) <= 4 \
                and len(set(core)) == 1 and p0 not in ("動詞", "形容詞", "助動詞", "代名詞"):  # not ビビって, ここっ
            return "interjection", "laugh"
        # いてて, いてっ: "ouch", which Sudachi reads as 凍てる
        if w.kind == "token" and w.surface == "いて" and w.morphs and w.morphs[0].norm == "凍てる":
            return "interjection", "ouch (いてて)"
        # バシャン, ギューン, プシュッ: a kana "adverb" with no dictionary entry is a sound effect
        if w.seq is None and w.kind == "token" and p0 == "副詞" and is_kana(w.surface):
            return "interjection", "sound effect (no dictionary entry)"
        # names
        name = self._is_name_token(w)
        if name:
            return "name", name
        # honorifics after a name or noun
        if p0 == "接尾辞" or (w.surface in TITLE_SUFFIXES and prev is not None):
            if w.surface in HONORIFICS or w.lemma in HONORIFICS or w.morphs and w.morphs[0].norm in HONORIFICS:
                if w.surface not in TITLE_SUFFIXES or (prev is not None and prev.filter == "name"):
                    return "honorific", f"after {prev.surface}" if prev else ""
        if w.kind == "token" and p0 in ("接頭辞", "接尾辞"):
            return "affix", f"lone {'prefix' if p0 == '接頭辞' else 'suffix'}"
        if w.seq and w.kind == "token":
            e = self.db.entry(w.seq)
            if e and e.senses and all("int" in sn.pos for sn in e.senses):
                return "interjection", f"JMdict interjection ({e.headword})"
        if w.seq:
            e = self.db.entry(w.seq)
            if e and "on-mim" in e.misc and all("on-mim" in s.misc for s in e.senses):
                return "onomatopoeia", "JMdict on-mim"
        return None, ""


# ---------------------------------------------------------------------------- deck names

# Words that take honorifics but are not names.
_NOT_NAMES = frozenset(
    "父 母 兄 姉 弟 妹 お父 お母 お兄 お姉 じい 爺 ばあ 婆 おじ おば 叔父 叔母 伯父 伯母 祖父 祖母 "
    "先生 先輩 王 皆 みな みんな 皆さん お客 客 奥 旦那 坊 坊っ 神 神様 姫 王子 女王 お嬢 嬢 子 お子 "
    "誰 あなた 貴方 君 僕 俺 私 彼 彼女 奴 ちび 赤 坊や 隊長 社長 部長 課長 店長 局長 先方 相手 皆様 "
    "息子 娘 孫 主人 嫁 夫 妻 友達 刑事 警部 大家 魔王 女神 死神 お巡り 運転手 店員 "
    "村長 町長 市長 会長 委員長 所長 船長 団長 院長 校長 署長 理事長 領主 騎士 師匠 博士 監督 マスター 神父 シスター".split()
)
_HONORIFIC_AFTER = frozenset("さん くん 君 ちゃん 様 さま 殿 氏 先輩 先生 たん ちゃま".split())
_PERSONAL_HONORIFICS = frozenset("さん くん 君 ちゃん たん ちゃま".split())
# kinship terms that follow a given name: 尊兄さん, 隼姉ちゃん (Sudachi keeps 兄さん whole)
_KINSHIP_AFTER = frozenset("兄さん 兄ちゃん 兄貴 姉さん 姉ちゃん 姉貴 兄様 姉様 にいさん にいちゃん ねえさん ねえちゃん".split())


def _common_word(db: DictDB, s: str) -> bool:
    return any((e := db.entry(f.seq)) is not None and e.common for f in db.forms_by_term(s))


def detect_deck_names(results, db: DictDB, min_unresolved: int = 3, min_honorific: int = 2,
                      min_run: int = 4, min_kana_honorific: int = 5) -> dict[str, str]:
    """Guess character/place names for one deck from its parsed sentences.

    ``results`` are ParseResult objects (unfiltered is fine). Returns {name: evidence}.
    Signals:
      * speaker labels removed during cleanup: （カズマ）, （藤本） — unless the label
        is a common word (店主, 少女) or a sound caption (ため息, 風の音)
      * a word directly before an honorific (ライト君, 月くん); a common word
        (月, 滝) needs a JMnedict entry and ``min_honorific``+ occurrences
      * a katakana/kanji noun with no JMdict entry occurring ``min_unresolved``+ times (ダクネス)
      * a run of kanji nouns, one of them a single kanji, that is not a JMdict word and
        recurs ``min_run``+ times: names and show terms the tokenizer split (流+河, 雪+精)
    """
    import re
    from collections import Counter

    from .parse.clean import clean_text

    found: dict[str, str] = {}
    labels: Counter = Counter()
    before_hon: Counter = Counter()
    unresolved: Counter = Counter()
    runs: Counter = Counter()
    kana_hon: Counter = Counter()
    personal: Counter = Counter()  # seen before さん/くん/ちゃん or 兄さん, not only 様/殿 (titles)
    for r in results:
        ws_ = r.words

        def run_ok(x, first: bool) -> bool:
            if x.kind != "token" or not all("\u3400" <= c <= "\u9fff" or c == "々" for c in x.surface):
                return False
            if len(x.pos) > 2 and ("数詞" in x.pos[1] or "助数詞" in x.pos[2]):
                return False
            if x.pos0 == "名詞":
                return True
            return not first and x.pos0 == "接尾辞" and x.surface not in HONORIFICS  # 流+河(suffix)

        k = 0
        while k < len(ws_):
            j = k
            while j < len(ws_) and run_ok(ws_[j], j == k):
                j += 1
            after_number = k > 0 and ws_[k - 1].pos[:2] == ("名詞", "数詞")  # 三+年前
            if j - k >= 2 and not after_number and any(len(x.surface) == 1 for x in ws_[k:j]):
                joined = "".join(x.surface for x in ws_[k:j])
                if not db.has_spelling(joined):
                    runs[joined] += 1
            k = max(j, k + 1)
        for rem in clean_text(r.text).removed:
            inner = re.sub(r"^[（(【〔［≪＜\[]|[）)】〕］≫＞：:\]]$", "", rem.strip())
            inner = re.sub(r"[(（][^)）]*[)）]?", "", inner).strip()
            for part in re.split(r"[・･,、＆&／/]", inner):
                part = re.sub(r"(たち|達|ら)$", "", part.strip())
                if 2 <= len(part) <= 10 and not re.search(r"[\sぁ-ゖ。！？…Ａ-Ｚ０-９A-Z0-9]", part.replace("ん", "")):
                    labels[part] += 1
        ws = r.words
        for i, w in enumerate(ws):
            # ことりちゃん, にこちゃん: a hiragana name, counted apart (needs more evidence)
            if i + 1 < len(ws) and ws[i + 1].surface in ("ちゃん", "くん", "さん") and w.kind == "token" \
                    and is_hiragana(w.surface) and len(w.surface) >= 2 and not (i > 0 and ws[i - 1].pos0 == "接頭辞"):
                kana_hon[w.surface] += 1
            if i + 1 < len(ws) and (ws[i + 1].surface in _HONORIFIC_AFTER or ws[i + 1].surface in _KINSHIP_AFTER) \
                    and w.pos0 in ("名詞", "代名詞") \
                    and w.kind == "token" and w.surface not in _NOT_NAMES and not is_hiragana(w.surface) \
                    and not (i > 0 and ws[i - 1].pos0 == "接頭辞"):  # お館様, ご主人様: titles
                before_hon[w.surface] += 1
                if ws[i + 1].surface in _PERSONAL_HONORIFICS or ws[i + 1].surface in _KINSHIP_AFTER:
                    personal[w.surface] += 1
                # 尊兄+さん (Sudachi reads 尊兄 as one word): a JMnedict name before 兄/姉
                if len(w.surface) >= 2 and w.surface[-1] in "兄姉" and db.names_by_term(w.surface[:-1]):
                    before_hon[w.surface[:-1]] += 1
                    personal[w.surface[:-1]] += 1
                # multi-token names split by the tokenizer: 夜神+さん, 流+河+くん
                j = i
                while j > 0 and ws[j - 1].kind == "token" and ws[j - 1].pos0 == "名詞" and ws[j - 1].pos[1] != "数詞" \
                        and not is_hiragana(ws[j - 1].surface):
                    j -= 1
                # 尊兄さん, 隼姉ちゃん: the name is the part before 兄/姉
                if w.surface in ("兄", "姉") and i > 0 and ws[i - 1].kind == "token" and ws[i - 1].pos0 == "名詞" \
                        and not is_hiragana(ws[i - 1].surface):
                    before_hon[ws[i - 1].surface] += 1
                    personal[ws[i - 1].surface] += 1
                if j < i:
                    joined = "".join(x.surface for x in ws[j:i + 1])
                    if not db.has_spelling(joined):
                        before_hon[joined] += 1
            if w.seq is None and w.kind == "token" and w.pos0 == "名詞" and w.pos[1] != "数詞" \
                    and (is_katakana(w.surface) or has_kanji(w.surface)) and len(w.surface) >= 2:
                unresolved[w.surface] += 1
    for name, n in labels.items():
        if (not db.has_spelling(name) or db.names_by_term(name)) and not _common_word(db, name) \
                and not re.search(r"(音|声|息)$", name):
            found.setdefault(name, f"speaker label ×{n}")
    for name, n in before_hon.items():
        if db.has_spelling(name):
            # a dictionary word (騎士様, 領主様, 耳様) is a name only if JMnedict knows it as one
            # and it is used like a person's name: with さん/くん/ちゃん, not only as a title
            # (katakana with さん/ちゃん is a character name even without JMnedict: アオイさん)
            if n >= min_honorific and personal[name] and (db.names_by_term(name) or is_katakana(name)):
                found.setdefault(name, f"before honorific ×{n} (also a dictionary word)")
        else:
            found.setdefault(name, f"before honorific ×{n}")
    for name, n in kana_hon.items():
        # not おっぱいちゃん; not なっちゃん → なっ, which is also how なって (成る) is split
        if n >= min_kana_honorific and not db.has_spelling(name) and not name.endswith("っ"):
            found.setdefault(name, f"hiragana word before ちゃん/くん/さん ×{n}")
    for name, n in unresolved.items():
        if n >= min_unresolved:
            found.setdefault(name, f"unknown noun ×{n}")
    for name, n in runs.items():
        if n >= min_run:
            found.setdefault(name, f"recurring non-dictionary kanji run ×{n}")
    return found


# ---------------------------------------------------------------------------- sentence filters

_JA_RE = __import__("re").compile(r"[぀-ヿ㐀-鿿々]")
_LATIN_RE = __import__("re").compile(r"[A-Za-zＡ-Ｚａ-ｚ]")


def sentence_junk(res, cfg: Config, has_audio: bool = True) -> tuple[str, str] | None:
    """Sentence-level reasons a card is not worth learning, or None.

    Reason codes: empty, music, foreign, no_audio, no_words, too_short, too_long.
    (Duplicates are found per deck, see score.py.)
    """
    sc = cfg["sort"]
    text = res.cleaned
    ja = len(_JA_RE.findall(text))
    if ja == 0:
        return "empty", "nothing Japanese after cleanup"
    if res.has_music:
        return "music", "♪ song lyrics"
    latin = len(_LATIN_RE.findall(text))
    if latin > ja:
        return "foreign", f"{latin} Latin vs {ja} Japanese characters"
    if sc.get("require_audio", True) and not has_audio:
        return "no_audio", "audio field is empty"
    counted = res.counted
    if not counted:
        kinds = sorted({w.filter for w in res.words if w.filter not in (None, "symbol", "whitespace")})
        return "no_words", "only " + (", ".join(kinds) or "punctuation")
    real = len(counted) + sum(1 for w in res.words if w.filter == "name")
    if real < int(sc.get("min_words", 2)):
        return "too_short", f"{real} word(s) incl. names < {sc.get('min_words', 2)}"
    return None
