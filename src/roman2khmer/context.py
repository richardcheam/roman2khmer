"""Sentence-level spelling: choose each Latin word's Khmer spelling with the neighbouring words in mind.

The word model (`model.Transliterator`) gives a few candidate spellings per Latin word with log-probabilities.
Spelled one word at a time, *mok* can come out as មុខ (face) where the message means មក (come). This module scores
whole sentences instead: each candidate's model log-probability (times `alpha`) plus a Khmer word-bigram language
model over the result, including the Khmer words already in the text, and keeps the best sequence (Viterbi).

The language model is a table of Khmer word and word-pair counts from web text , smoothed by
interpolated absolute discounting. Words outside its vocabulary get a fixed small probability, so a misspelt
candidate is unlikely to win unless the word model is much more confident about it.

Example:
    >>> from roman2khmer import context, model                  # doctest: +SKIP
    >>> lm = context.load_lm("khmer_bigram.tsv.gz")            # doctest: +SKIP
    >>> context.convert("pa ban mok vinh hz", model.load("translit.npz"), lm)   # doctest: +SKIP
    'ប៉ា បាន មក វិញ ហើយ'
"""
from __future__ import annotations

import gzip
import math
import re
from pathlib import Path

BOS, EOS = "<s>", "</s>"
TOKEN = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|[ក-៓៝]+|\s+|.", re.DOTALL)
KHMER = re.compile(r"[ក-៓៝]+")
MAX_WORD = 24


class BigramLM:
    """Khmer word-bigram model with interpolated absolute discounting, stored as NumPy arrays.

    Words get integer ids; each kept word pair is one sorted 64-bit code (`previous id × vocabulary size + word id`)
    with its count, looked up by binary search. This keeps the 2.5 million pairs in about 30 MB instead of a Python
    dictionary of string pairs.

    Attributes:
        words: id → word (`BOS` included; it holds the number of sentences).
        ids: word → id.
        counts: id → word count (int64).
        codes: sorted pair codes (int64); `pair_counts`: their counts (int32).
        kept_sum, kept_n: per previous word, the summed count and number of its kept pairs (for the back-off weight).
        discount: the absolute discount `d` (default 0.75).
        oov_logp: log-probability of a word outside the vocabulary (default log 1e-8).
    """

    def __init__(self, words: list[str], counts, codes, pair_counts, discount: float = 0.75,
                 oov_logp: float = math.log(1e-8), n_vocab: int | None = None) -> None:
        import numpy as np
        self.words = list(words)
        self.ids = {w: i for i, w in enumerate(self.words)}
        self.counts = np.asarray(counts, dtype=np.int64)
        order = np.argsort(np.asarray(codes, dtype=np.int64), kind="stable")
        self.codes = np.asarray(codes, dtype=np.int64)[order]
        self.pair_counts = np.asarray(pair_counts, dtype=np.int32)[order]
        self.discount, self.oov_logp = discount, oov_logp
        n = len(self.words)
        prev = self.codes // n if n else self.codes
        self.kept_sum = np.bincount(prev, weights=self.pair_counts, minlength=n).astype(np.int64)
        self.kept_n = np.bincount(prev, minlength=n).astype(np.int64)
        bos = self.ids.get(BOS)
        n_sent = int(self.counts[bos]) if bos is not None else 0
        self.total = int(self.counts.sum())   # every word, plus one end of sentence per sentence (the BOS count)
        self._n_sent = n_sent
        self.n_vocab = len(self.words) if n_vocab is None else n_vocab   # ids past it (EOS) are not vocabulary
        self._denom = self.total + self.n_vocab + 1
        self.prefixes = {w[:k] for w in self.words for k in range(1, len(w) + 1)}
        self._cache: dict[tuple[str, str], float] = {}

    @classmethod
    def from_counts(cls, uni: dict[str, int], bi: dict[tuple[str, str], int], **kw) -> "BigramLM":
        """Build from plain dictionaries (handy for small examples and tests).

        Examples:
            >>> lm = BigramLM.from_counts({"<s>": 2, "ក": 2, "ខ": 1}, {("<s>", "ក"): 2, ("ក", "ខ"): 1})
            >>> lm.logp("ក", "ខ") > lm.logp("ក", "ក")
            True
        """
        words = list(uni) + sorted({w for pair in bi for w in pair} - set(uni))
        ids = {w: i for i, w in enumerate(words)}
        n = len(words)
        codes = [ids[p] * n + ids[w] for p, w in bi]
        return cls(words, [uni.get(w, 0) for w in words], codes, list(bi.values()), n_vocab=len(uni), **kw)

    def __contains__(self, w: str) -> bool:
        return self.ids.get(w, self.n_vocab) < self.n_vocab

    def _count(self, w: str) -> int:
        i = self.ids.get(w, self.n_vocab)
        return 0 if i >= self.n_vocab else int(self.counts[i])

    def unigram_logp(self, w: str) -> float:
        """log P(w), add-one smoothed; `oov_logp` for an unknown word. `EOS` counts once per sentence."""
        c = self._n_sent if w == EOS else self._count(w)
        if not c or w == BOS:
            return self.oov_logp
        return math.log((c + 1) / self._denom)

    def logp(self, prev: str, w: str) -> float:
        """log P(w | prev): discounted pair count plus the back-off share of the unigram probability.

        The count of `prev` covers every pair that starts with it, so the back-off share is the discounted mass of
        the kept pairs plus all the mass of the pairs cut from the table: the probabilities over the vocabulary sum
        to about one (up to the unigram smoothing).
        """
        key = (prev, w)
        r = self._cache.get(key)
        if r is not None:
            return r
        if w != EOS and self.ids.get(w, self.n_vocab) >= self.n_vocab:
            r = self.oov_logp
        else:
            pu = math.exp(self.unigram_logp(w))
            pi = self.ids.get(prev)
            cv = 0 if pi is None else int(self.counts[pi])
            if not cv:
                r = math.log(pu)
            else:
                backoff = (cv - int(self.kept_sum[pi]) + self.discount * int(self.kept_n[pi])) / cv
                c = 0
                wi = self.ids.get(w)
                if wi is not None:
                    code = pi * len(self.words) + wi
                    j = int(self.codes.searchsorted(code))
                    if j < len(self.codes) and int(self.codes[j]) == code:
                        c = int(self.pair_counts[j])
                r = math.log(max(c - self.discount, 0) / cv + backoff * pu)
        if len(self._cache) > 500_000:
            self._cache.clear()
        self._cache[key] = r
        return r

    def split(self, run: str) -> list[str]:
        """Greedy longest-match split of a Khmer run into vocabulary words; leftover characters form unknown pieces.

        Examples:
            >>> BigramLM.from_counts({"កខ": 1, "គ": 1}, {}).split("កខគឃ")
            ['កខ', 'គ', 'ឃ']
        """
        out, i, junk = [], 0, ""
        while i < len(run):
            best, j = 0, i + 1
            while j <= len(run) and j - i <= MAX_WORD and run[i:j] in self.prefixes:
                if self.ids.get(run[i:j], self.n_vocab) < self.n_vocab:
                    best = j
                j += 1
            if best:
                if junk:
                    out.append(junk)
                    junk = ""
                out.append(run[i:best])
                i = best
            else:
                junk += run[i]
                i += 1
        if junk:
            out.append(junk)
        return out

    def seq_logp(self, prev: str, words: list[str]) -> tuple[float, str]:
        """(log-probability of `words` after `prev`, last word)."""
        s = 0.0
        for w in words:
            s += self.logp(prev, w)
            prev = w
        return s, prev

    def save(self, path: str | Path) -> None:
        """Write the arrays to one `.npz` file (read back by `load_lm`)."""
        import numpy as np
        np.savez_compressed(path, words=np.array("\n".join(self.words)), counts=self.counts, codes=self.codes,
                            pair_counts=self.pair_counts, discount=np.float64(self.discount),
                            n_vocab=np.int64(self.n_vocab))


def load_lm(path: str | Path, min_pair: int = 1) -> BigramLM:
    """Read the word-pair model: a `.npz` written by `BigramLM.save`, or a gzip counts table
    (`1 word count types` / `2 prev word count`).

    Args:
        path: The `.npz` or `.tsv.gz` file.
        min_pair: For a counts table, ignore pairs seen fewer times (default 1: keep all that were written).

    Returns:
        The `BigramLM`.
    """
    import numpy as np
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            return BigramLM(str(z["words"]).split("\n"), z["counts"], z["codes"], z["pair_counts"],
                            discount=float(z["discount"]), n_vocab=int(z["n_vocab"]))
    words: list[str] = []
    ids: dict[str, int] = {}
    counts: list[int] = []
    pairs: list[tuple[str, str, int]] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] == "1":
                ids[p[1]] = len(words)
                words.append(p[1])
                counts.append(int(p[2]))
            elif int(p[3]) >= min_pair:
                pairs.append((p[1], p[2], int(p[3])))
    n_vocab = len(words)
    for a, b, _ in pairs:                   # pair-only words (the end of sentence): ids after the vocabulary
        for w in (a, b):
            if w not in ids:
                ids[w] = len(words)
                words.append(w)
                counts.append(0)
    n = len(words)
    codes = np.fromiter((ids[a] * n + ids[b] for a, b, _ in pairs), dtype=np.int64, count=len(pairs))
    pc = np.fromiter((c for *_, c in pairs), dtype=np.int32, count=len(pairs))
    return BigramLM(words, counts, codes, pc, n_vocab=n_vocab)


def _is_latin(t: str) -> bool:
    return t[:1].isascii() and t[:1].isalpha()


def join_keys(text: str) -> list[str]:
    """The joined spellings `convert(..., beta=...)` may try: each two Latin words separated by one space.

    Examples:
        >>> join_keys("tro troung nas, Bong")
        ['trotroung', 'troungnas']
    """
    toks = TOKEN.findall(text)
    return [(toks[i] + toks[i + 2]).lower() for i in range(len(toks) - 2)
            if _is_latin(toks[i]) and toks[i + 1] == " " and _is_latin(toks[i + 2])]


def convert(text: str, model, lm: BigramLM, alpha: float = 1.0, beam: int = 8,
            nbest: dict[str, list[tuple[str, float]]] | None = None, sep: str = " ",
            beta: float | None = None, chosen: dict[int, str] | None = None) -> str:
    """Replace every Latin word of `text` by Khmer, choosing the spellings jointly with the Khmer around them.

    Args:
        text: Any text: fully romanised (`"nhom tov sala"`) or Khmer with Latin words (`"ថ្ងៃនេះ Sokly មក"`).
        model: A `model.Transliterator` (only `nbest_batch` is used).
        lm: The Khmer bigram model.
        alpha: Weight of the word model's log-probability against the language model (1.0 = equal).
        beam: Candidates per Latin word.
        nbest: Optional precomputed `{lower-case word: nbest}` (skips the model for those words).
        sep: What replaces the spaces between two converted Latin words (default one space). Other text, including
            spaces next to Khmer, is kept as it is.
        beta: If set, two Latin words separated by one space may also be spelled as one word (*tro troung* →
            ទ្រទ្រង់), at a cost of `beta` per join. `None` (default) never joins.

        chosen: If given, filled with the choice per token: token index (in `TOKEN.findall(text)`) → Khmer written
            for it (a joined pair: the first token gets the joined spelling, the second ""). For diagnostics.

    Returns:
        The text with Khmer in place of each Latin word.
    """
    toks = TOKEN.findall(text)
    latin = {t.lower() for t in toks if _is_latin(t)}
    if beta is not None:
        latin |= set(join_keys(text))
    nb = dict(nbest or {})
    todo = sorted(w for w in latin if w not in nb)
    if todo:
        nb.update(zip(todo, model.nbest_batch([re.sub(r"[^a-z]", "", w) for w in todo], beam)))
    State = dict[str, tuple[float, dict[int, str]]]

    def relax(dst: dict[int, State], j: int, last: str, s: float, ch: dict[int, str]) -> None:
        cur = dst.setdefault(j, {})
        if last not in cur or s > cur[last][0]:
            cur[last] = (s, ch)

    # Viterbi over token positions; at each position the states are keyed by the last LM word and keep
    # (score, chosen text per token index). Edges: one Latin word, two joined Latin words, a Khmer run, a space,
    # or punctuation (the end of one sentence and the start of the next).
    at: dict[int, State] = {0: {BOS: (0.0, {})}}
    for i, t in enumerate(toks):
        states = at.pop(i, None)
        if not states:
            continue
        if _is_latin(t):
            spans = [(t.lower(), i + 1, {})]
            if beta is not None and i + 2 < len(toks) and toks[i + 1] == " " and _is_latin(toks[i + 2]):
                spans.append(((t + toks[i + 2]).lower(), i + 3, {i + 1: "", i + 2: ""}))
            for key, j, extra in spans:
                cost = 0.0 if j == i + 1 else beta
                cands = [(km, lp) for km, lp in nb[key][:beam] if km] or nb[key][:1]
                for km, lp in cands:             # an empty spelling would drop the word: never chosen if any other
                    words = lm.split(km) if km else []
                    for prev, (sc, ch) in states.items():
                        ls, last = lm.seq_logp(prev, words)
                        relax(at, j, last if words else prev, sc + alpha * lp + ls - cost, {**ch, i: km, **extra})
        elif KHMER.fullmatch(t):
            words = lm.split(t)
            for prev, (sc, ch) in states.items():
                ls, last = lm.seq_logp(prev, words)
                relax(at, i + 1, last, sc + ls, ch)
        elif t.isspace():
            for prev, (sc, ch) in states.items():
                relax(at, i + 1, prev, sc, ch)
        else:                                   # punctuation, digits, other scripts: a sentence break
            sc, ch = max(((sc + lm.logp(prev, EOS), ch) for prev, (sc, ch) in states.items()), key=lambda v: v[0])
            relax(at, i + 1, BOS, sc, ch)
    final = at[len(toks)]
    _s, choice = max(((sc + lm.logp(prev, EOS), ch) for prev, (sc, ch) in final.items()), key=lambda v: v[0])
    if chosen is not None:
        chosen.update(choice)
    out = []
    for i, t in enumerate(toks):
        if i in choice:
            out.append(choice[i])
        elif t.isspace() and i - 1 in choice and i + 1 in choice:
            out.append(sep)
        else:
            out.append(t)
    return "".join(out)
