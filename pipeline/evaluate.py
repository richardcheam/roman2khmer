#!/usr/bin/env python3
"""Evaluate a model folder on every published test set and compare with the released system's numbers.

    python pipeline/evaluate.py                          # the released model (downloaded, pinned)
    python pipeline/evaluate.py --model my_model          # a folder from training/build_model_folder.py
    python pipeline/evaluate.py --model my_model --gamma 0 --out runs/eval_nocheck.json

What is measured (what each set was used for is in `test-sets/README.md`; development sets give optimistic numbers):
- **Messages B** (450 LLM-written chat messages, 30 generation calls; tune 285 / test 165): character error rate
  (CER) and the share of messages written exactly. CER = edit distance between output and reference over the
  reference length, pooled over messages, after keeping only Khmer and Latin letters (spaces, punctuation and digits
  removed). 95% interval: resample whole generation calls (10,000 resamples, seed 2025).
- **Messages A** (447 earlier LLM-written messages; fit 279 / check 168): the same measures, no interval.
- **Fresh word test** (148 words typed by the author, 179 spellings): each spelling converted alone; a word's score
  is the share of its spellings written exactly; reported as the mean over words, with a 95% interval resampling
  words, and over all spellings. Also the 225 sound-alike variants of the same words.
- **Public tests** (Chhunneng test split; Wikidata names): each Latin word converted alone, correct if the output is
  one of its attested Khmer spellings; only items whose Khmer targets never occur in the training data count.

`expected` values are the released system's (`pipeline/expected_results.json`). The released model reproduces
them exactly (NumPy inference is deterministic); a retrained model is expected to land near them, not on them.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS, RAW = ROOT / "test-sets", ROOT / "training-data" / "raw"
KEEP = re.compile(r"[ក-៓៝A-Za-z]")


def clean(s: str) -> str:
    """Scoring view: Khmer and Latin letters only."""
    return "".join(KEEP.findall(s))


def edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def tsv(path: Path) -> list[list[str]]:
    return [l.split("\t") for l in path.read_text(encoding="utf-8").splitlines()[1:] if l]


def messages(conv, msgs: list[dict]) -> list[tuple[int, int, bool]]:
    """Per message: (edits, reference length, exact)."""
    out = []
    for m in msgs:
        o, r = clean(conv.convert(m["text"])), clean(m["khmer"])
        out.append((edit(o, r), len(r), o == r))
    return out


def cer(rows) -> float:
    return 100 * sum(e for e, _, _ in rows) / sum(n for _, n, _ in rows)


def call_interval(rows, calls: list[int], n_boot: int = 10_000, seed: int = 2025) -> list[float]:
    """95% percentile interval of the pooled CER, resampling generation calls."""
    units = sorted(set(calls))
    E = {u: 0 for u in units}
    L = {u: 0 for u in units}
    for (e, n, _), c in zip(rows, calls):
        E[c] += e
        L[c] += n
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        pick = [units[rng.randrange(len(units))] for _ in units]
        draws.append(100 * sum(E[u] for u in pick) / sum(L[u] for u in pick))
    draws.sort()
    return [round(draws[int(0.025 * n_boot)], 2), round(draws[int(0.975 * n_boot) - 1], 2)]


def word_interval(scores: list[float], n_boot: int = 10_000, seed: int = 2025) -> list[float]:
    rng = random.Random(seed)
    n = len(scores)
    draws = sorted(100 * sum(scores[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return [round(draws[int(0.025 * n_boot)], 1), round(draws[int(0.975 * n_boot) - 1], 1)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="model folder (default: the released model, downloaded on first use)")
    ap.add_argument("--gamma", type=float, default=None, help="recheck weight (default: the model's, 1.0)")
    ap.add_argument("--out", type=Path, help="write the results as JSON")
    ap.add_argument("--quick", action="store_true", help="skip Messages A and the public tests (faster)")
    ap.add_argument("--train-data", type=Path, default=ROOT / "training-data" / "v0.1" / "spell_train.tsv",
                    help="the training pairs of the evaluated model, to decide which public-test items are unseen "
                         "(default: the release's; pass runs/data/spell_train.tsv for a model trained on other data)")
    a = ap.parse_args()
    from roman2khmer import Converter
    settings = {} if a.gamma is None else {"gamma": a.gamma}
    conv = Converter.from_dir(a.model, **settings) if a.model else Converter.from_pretrained(**settings)
    expected = json.loads((ROOT / "pipeline" / "expected_results.json").read_text(encoding="utf-8"))
    res: dict = {"model": a.model or "released (Hugging Face)", "gamma": conv.gamma}
    t0 = time.time()

    mb = jsonl(TESTS / "messages_b.jsonl")
    rows = messages(conv, mb)
    calls = [m["call"] for m in mb]
    part = lambda p: [r for r, m in zip(rows, mb) if m["part"] == p]  # noqa: E731
    res["messages_b"] = {"cer": cer(rows), "cer_interval_by_call": call_interval(rows, calls),
                         "exact_pct": 100 * sum(x for *_, x in rows) / len(rows), "n": len(rows),
                         "tune": {"cer": cer(part("tune")), "n": len(part("tune"))},
                         "test": {"cer": cer(part("test")), "n": len(part("test"))}}
    fresh = tsv(TESTS / "fresh_test.tsv")
    word_scores, spell_hits = [], []
    for _wid, km, typed in fresh:
        sp = [v.strip() for v in typed.split(",") if v.strip()]
        h = [clean(conv.convert(s)) == clean(km) for s in sp]
        word_scores.append(sum(h) / len(h))
        spell_hits += h
    var = tsv(TESTS / "fresh_variants.tsv")
    res["fresh_words"] = {"macro_pct": 100 * sum(word_scores) / len(word_scores),
                          "macro_interval_by_word": word_interval(word_scores), "words": len(word_scores),
                          "spellings_pct": 100 * sum(spell_hits) / len(spell_hits), "spellings": len(spell_hits),
                          "variants_pct": 100 * sum(clean(conv.convert(v)) == clean(k) for _w, k, _r, v in var)
                          / len(var), "variants": len(var)}
    if not a.quick:
        ma = jsonl(TESTS / "messages_a.jsonl")
        ra = messages(conv, ma)
        res["messages_a"] = {p: {"cer": cer([r for r, m in zip(ra, ma) if m["part"] == p]),
                                 "n": sum(m["part"] == p for m in ma)} for p in ("fit", "check")}
        train_km = {r[1] for r in tsv(a.train_data)}
        chh = tsv(RAW / "chhunneng_pairs.tsv")
        attested: dict[str, set[str]] = {}
        for lat, km, _s in chh:
            attested.setdefault(lat, set()).add(km)
        test_words = sorted({lat for lat, _k, s in chh if s == "test"})
        items = [(w, attested[w]) for w in test_words if not attested[w] & train_km]
        names: dict[str, set[str]] = {}
        for lat, km in tsv(TESTS / "names_test.tsv"):
            names.setdefault(lat, set()).add(km)
        nitems = [(w, k) for w, k in names.items() if not k & train_km]
        acc = lambda its: 100 * sum(conv.convert(w) in k for w, k in its) / max(1, len(its))  # noqa: E731
        res["public_tests"] = {"romanised_unseen_targets": {"accuracy_pct": acc(items), "items": len(items)},
                               "names_unseen_targets": {"accuracy_pct": acc(nitems), "items": len(nitems)}}
    res["minutes"] = round((time.time() - t0) / 60, 1)

    def show(label, got, exp_key, places=2):
        exp = expected.get(exp_key)
        line = f"  {label:<44} {got:8.{places}f}"
        if exp is not None:
            line += f"   released {exp:8.{places}f}   difference {got - exp:+.{places}f}"
        print(line)
    b = res["messages_b"]
    print(f"model: {res['model']} (recheck weight {res['gamma']})")
    show("Messages B, CER % (450, development)", b["cer"], "messages_b_cer")
    print(f"  {'   95% interval, resampling calls':<44} {b['cer_interval_by_call']}")
    show("Messages B, exact messages %", b["exact_pct"], "messages_b_exact_pct", 1)
    show("Messages B, CER % on the 165 former test", b["test"]["cer"], "messages_b_test_cer")
    f = res["fresh_words"]
    show("Fresh words, mean over 148 words %", f["macro_pct"], "fresh_words_macro_pct", 1)
    print(f"  {'   95% interval, resampling words':<44} {f['macro_interval_by_word']}")
    show("Fresh words, all 179 spellings %", f["spellings_pct"], "fresh_words_spellings_pct", 1)
    show("Fresh words, 225 sound-alike variants %", f["variants_pct"], "fresh_variants_pct", 1)
    if "messages_a" in res:
        show("Messages A fit, CER % (279, tuned on)", res["messages_a"]["fit"]["cer"], "messages_a_fit_cer")
        show("Messages A check, CER % (168)", res["messages_a"]["check"]["cer"], "messages_a_check_cer")
        p = res["public_tests"]
        show(f"Public romanised test, unseen ({p['romanised_unseen_targets']['items']}) %",
             p["romanised_unseen_targets"]["accuracy_pct"], "public_romanised_unseen_pct", 1)
        show(f"Public names test, unseen ({p['names_unseen_targets']['items']}) %",
             p["names_unseen_targets"]["accuracy_pct"], "public_names_unseen_pct", 1)
    print(f"  ({res['minutes']} minutes)")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
