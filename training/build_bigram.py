#!/usr/bin/env python3
"""Count Khmer word pairs in web text: the word-pair model of roman2khmer's sentence step.

    python training/build_bigram.py --shard data/fineweb2/000_00000.parquet --lexicon data/km_lexicon.tsv \
        --out runs/khmer_bigram.npz

Input: one FineWeb-2 `khm_Khmr` parquet shard (the first 200,000 documents are read) and Google's Khmer
pronunciation lexicon (only its headwords are used). The word list is the lexicon headwords plus every Khmer word
of the training pairs. Text is cut into sentences at anything that is not Khmer or a space; each Khmer run is split
into words by greedy longest match against the word list (unknown characters are skipped); each sentence counts as
`<s> w1 … wn </s>`. Pairs seen fewer than 2 times are dropped. The counts are saved with the smoothing settings of
`roman2khmer.context.BigramLM` (absolute discounting, d = 0.75) as `.npz`, plus the counts table as `.tsv.gz`.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from roman2khmer import context

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "config.yaml"
SENT = re.compile(r"[ក-៓៝ ]+")
RUN = re.compile(r"[ក-៓៝]+")
KHMER = re.compile(r"[ក-៿]")


def word_list(lexicon: Path, train_pairs: Path) -> frozenset[str]:
    """Lexicon headwords (first column, lines with a Khmer character) plus the Khmer side of the training pairs."""
    heads = {p[0] for p in (l.split("\t") for l in lexicon.read_text(encoding="utf-8").splitlines()
                            if l.strip() and not l.startswith("#")) if len(p) >= 2 and KHMER.search(p[0])}
    targets = {l.split("\t")[1] for l in train_pairs.read_text(encoding="utf-8").splitlines()[1:]}
    return frozenset(heads | targets)


def segment(run: str, prefixes: set[str], words: frozenset[str]) -> list[str]:
    """Greedy longest-match split of a Khmer run into known words; characters that start no word are skipped.

    Examples:
        >>> segment("កខគឃ", {"ក", "កខ", "គ"}, frozenset({"កខ", "គ"}))
        ['កខ', 'គ']
    """
    out, i = [], 0
    while i < len(run):
        best, j = 0, i + 1
        while j <= len(run) and run[i:j] in prefixes:
            if run[i:j] in words:
                best = j
            j += 1
        if best:
            out.append(run[i:best])
            i = best
        else:
            i += 1
    return out


def count(shard: Path, words: frozenset[str], n_docs: int):
    prefixes = {w[:k] for w in words for k in range(1, len(w) + 1)}
    uni: collections.Counter = collections.Counter()
    bi: collections.Counter = collections.Counter()
    n = 0
    for b in pq.ParquetFile(shard).iter_batches(batch_size=2000, columns=["text"]):
        for text in b.column("text").to_pylist():
            for sent in SENT.findall(text):
                ws = [w for run in RUN.findall(sent) for w in segment(run, prefixes, words)]
                if ws:
                    seq = [context.BOS] + ws + [context.EOS]
                    uni.update(seq[:-1])
                    bi.update(zip(seq, seq[1:]))
            n += 1
            if n >= n_docs:
                return uni, bi, n
    return uni, bi, n


def main() -> None:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=Path, required=True)
    ap.add_argument("--lexicon", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="the .npz to write (a .tsv.gz table goes next to it)")
    ap.add_argument("--docs", type=int, default=cfg["word_pairs"]["docs"])
    a = ap.parse_args()
    t0 = time.time()
    words = word_list(a.lexicon, ROOT / cfg["data"] / "spell_train.tsv")
    uni, bi, n = count(a.shard, words, a.docs)
    min_pair = cfg["word_pairs"]["min_pair"]
    types = collections.Counter(p for p, _w in bi)
    table = a.out.with_suffix(".tsv.gz")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(table, "wt", encoding="utf-8") as f:
        for w, c in uni.most_common():
            f.write(f"1\t{w}\t{c}\t{types[w]}\n")
        for (p, w), c in bi.items():
            if c >= min_pair:
                f.write(f"2\t{p}\t{w}\t{c}\n")
    context.load_lm(table).save(a.out)
    print(json.dumps({"docs": n, "word_list": len(words), "words_seen": len(uni), "tokens": sum(uni.values()),
                      "pairs_kept": sum(c >= min_pair for c in bi.values()),
                      "minutes": round((time.time() - t0) / 60, 1), "out": str(a.out)}))


if __name__ == "__main__":
    main()
