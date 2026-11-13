#!/usr/bin/env python3
"""Prepare the training data: raw inputs per source -> the training files of the released model.

    python pipeline/prepare.py --out runs/data            # writes the training files
    python pipeline/prepare.py --out runs/data --check    # and compares them byte by byte with training-data/v0.1

Inputs (`training-data/raw/`, sources and licences in its README):
- `chhunneng_pairs.tsv`  public Chhunneng romanisation pairs with their train / val / test split;
- `mined_pairs.tsv`      name and loanword pairs mined from Khmer Wikipedia and FineWeb-2;
- `loanwords.tsv`        the author's loanword list with its train / val / test split;
- `author_examples.tsv`, `author_typing.tsv`, `author_checked_llm.tsv`  the author's spellings;
- `llm_spellings.tsv`    Gemini and GPT spellings of common words, already filtered;
- `common_words.txt`     the 500 commonest Khmer words of the FineWeb-2 shard;
- `aux_train.tsv`        the side task's pronunciation rows (prepared; see the README).
Held-out sets (`test-sets/`): every row that touches an evaluation or validation set is removed from training.

Steps, in order:
1. **Held-out sets.** Evaluation: the fresh word test (the author's spellings and sound-alike variants), the
   picked English loanwords, 41 development English words and the loanword-list test split; the public Chhunneng
   test and the names test are evaluated only on items whose target is absent from training, so they block nothing.
   Validation: romanised (Chhunneng val, without the 500 commonest words, which move to training), names, English
   loanwords. A validation item touching an evaluation set is dropped.
2. **Matching rule.** English sets match a Latin word by its suffix stems (*sharing* ~ *share*) or a Khmer target;
   the fresh test and the names and romanised validation sets match only their exact (Latin, Khmer) pairs.
3. **Training rows,** per source: lower-case; drop a row with characters outside a-z / Khmer or too long, then a
   row touching a held-out set, then duplicates within the source. The author's typed spellings give one joined row
   per spelling, plus one row per typed word when the Khmer word splits into as many known words.
4. **Recheck model files:** held-out Khmer words = romanised validation words absent from training; pool = every
   other (Khmer, Latin) training pair except web-mined rows.

The released data also excluded rows related to a sealed reserve of 120 English test words, which stays
unpublished; those rows (140 in all) are simply absent from the raw inputs, so this script reproduces the release
exactly without knowing the reserve.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, TESTS, RELEASED = ROOT / "training-data" / "raw", ROOT / "test-sets", ROOT / "training-data" / "v0.1"
MAX_IN, MAX_OUT, MAX_SRC, MAX_TGT = 48, 40, 48, 40
SUFFIXES = ("ings", "ing", "ers", "er", "ies", "ied", "es", "ed", "ly", "s")
STEM_SETS = ("pick_test", "dev41", "list_test", "val_english")         # matched by Latin stems or Khmer target
PAIR_SETS = ("val_names", "val_romanised", "fresh_test")                # matched by exact (Latin, Khmer) pair
KHMER = re.compile(r"[ក-៿]+")


# ---------------------------------------------------------------- reading
def tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pairs(path: Path, a: str = "latin", b: str = "khmer") -> list[tuple[str, str]]:
    return [(r[a].strip(), r[b].strip()) for r in tsv(path)]


def clean(spelling: str) -> list[str]:
    """Typed words, lower-case a-z only: "Puk ke" -> ["puk", "ke"]; "sa-at" -> ["saat"]."""
    return [w for w in (re.sub(r"[^a-z]", "", x) for x in spelling.lower().split()) if w]


def variants(typed: str) -> list[str]:
    return [v.strip() for v in typed.split(",") if v.strip()]


# ---------------------------------------------------------------- matching
def norm(word: str) -> str:
    """Matching form: accents removed, lower case, a-z only ("Wi-Fi" -> "wifi")."""
    w = unicodedata.normalize("NFKD", word).lower()
    return re.sub("[^a-z]", "", "".join(c for c in w if not unicodedata.combining(c)))


def stems(word: str) -> set[str]:
    """The word and its simple suffix-stripped stems: "sharing" -> {"sharing", "shar", "share"}."""
    out = {word}
    for suf in SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            base = word[: -len(suf)]
            out |= {base, base + "e", base + "y"} if suf in ("ies", "ied") else {base, base + "e"}
            if len(base) >= 4 and base[-1] == base[-2]:
                out.add(base[:-1])
    return {s for s in out if len(s) >= 3} | {word}


def match_sets(held: dict[str, list[tuple[str, str]]]) -> dict:
    """Per set: (Latin stems or exact pairs, Khmer targets) under the set's matching rule."""
    out = {}
    for n, ps in held.items():
        if n in PAIR_SETS:
            out[n] = ({(norm(a), b) for a, b in ps}, set())
        elif n in STEM_SETS:
            out[n] = ({s for a, _ in ps for s in stems(norm(a))}, {b for _, b in ps if b})
        else:                                        # public test panels: evaluated on unseen targets, block nothing
            out[n] = (set(), set())
    return out


def hits(latin: str, khmer: str, sets: dict) -> bool:
    w = norm(latin)
    st = stems(w)
    for n, (ms, km) in sets.items():
        if n in PAIR_SETS:
            if (w, khmer) in ms:
                return True
        elif (st & ms) or (khmer and khmer in km):
            return True
    return False


# ---------------------------------------------------------------- word splitting (author rows)
def segment(phrase: str, lexicon: set[str], max_word: int = 24) -> list[tuple[str, bool]]:
    """Greedy longest-match split into known words; unknown characters merge into runs."""
    out: list[tuple[str, bool]] = []
    i = 0
    while i < len(phrase):
        for j in range(min(len(phrase), i + max_word), i, -1):
            if phrase[i:j] in lexicon:
                out.append((phrase[i:j], True))
                i = j
                break
        else:
            if out and not out[-1][1]:
                out[-1] = (out[-1][0] + phrase[i], False)
            else:
                out.append((phrase[i], False))
            i += 1
    return out


def author_rows(answers: list[tuple[str, list[str]]], vocab: set[str]) -> list[tuple[str, str]]:
    """One joined row per typed spelling, plus per-word rows when the Khmer word splits into as many known words:
    *puk ke* for ពួកគេ gives ("pukke", "ពួកគេ"), ("puk", "ពួក"), ("ke", "គេ")."""
    out = []
    for km, spellings in answers:
        seg = segment(km, vocab - {km})
        parts = [p for p, _ in seg] if len(seg) > 1 and all(ok for _, ok in seg) else []
        for s in spellings:
            words = clean(s)
            if not words:
                continue
            out.append(("".join(words), km))
            if len(words) > 1 and len(words) == len(parts):
                out += list(zip(words, parts))
    return out


# ---------------------------------------------------------------- the build
def held_out(chh: list[dict], common: set[str], loans: list[dict]) -> dict[str, list[tuple[str, str]]]:
    fresh = [("".join(clean(s)), r["khmer"]) for r in tsv(TESTS / "fresh_test.tsv") for s in variants(r["typed"])]
    fresh += [("".join(clean(r["variant"])), r["khmer"]) for r in tsv(TESTS / "fresh_variants.tsv")]
    attested: dict[str, set[str]] = {}
    for r in chh:
        attested.setdefault(r["latin"], set()).add(r["khmer"])
    ev = {"romanised_test": sorted({(r["latin"], k) for r in chh if r["split"] == "test" for k in attested[r["latin"]]}),
          "names_test2": pairs(TESTS / "names_test.tsv"),
          "pick_test": pairs(TESTS / "pick_test.tsv", "word"), "dev41": pairs(TESTS / "dev41.tsv", "word"),
          "list_test": [(r["word"], r["khmer"]) for r in loans if r["split"] == "test"], "fresh_test": fresh}
    val = {"val_english": [(r["word"], r["khmer"]) for r in loans if r["split"] == "val"],
           "val_names": pairs(TESTS / "names_val.tsv"),
           "val_romanised": [(r["latin"], r["khmer"]) for r in chh if r["split"] == "val" and r["khmer"] not in common]}
    m = match_sets(ev)
    val = {n: [(a, b) for a, b in ps if not hits(a, b, m)] for n, ps in val.items()}
    return {**ev, **val}


def build(out: Path) -> dict:
    chh, loans = tsv(RAW / "chhunneng_pairs.tsv"), tsv(RAW / "loanwords.tsv")
    common = set((RAW / "common_words.txt").read_text(encoding="utf-8").split())
    held = held_out(chh, common, loans)
    sets = match_sets(held)
    typed = tsv(RAW / "author_typing.tsv")
    typing = [(r["khmer"], variants(r["typed"])) for r in typed if r["part"] == "train"] + \
             [(r["khmer"], variants(r["typed"])) for r in typed if r["part"] == "retired_test"]
    checked = [(r["khmer"], variants(r["typed"])) for r in tsv(RAW / "author_checked_llm.tsv") if variants(r["typed"])]
    vocab = {km for km, _ in typing + checked} | {r["khmer"] for r in chh if r["split"] == "train"}
    sources = {
        "mined": pairs(RAW / "mined_pairs.tsv"),
        "chhunneng": [(r["latin"], r["khmer"]) for r in chh if r["split"] == "train"],
        "list_train": [(r["word"], r["khmer"]) for r in loans if r["split"] == "train"],
        "owner_examples": pairs(RAW / "author_examples.tsv", "word"),
        "owner_typing": author_rows(typing, vocab),
        "owner_checked_llm": author_rows(checked, vocab),
        "llm_bulk": pairs(RAW / "llm_spellings.tsv"),
        "chhunneng_val_common": [(r["latin"], r["khmer"]) for r in chh if r["split"] == "val" and r["khmer"] in common],
    }
    spell, counts = [], {}
    for name, rows in sources.items():
        seen, n_kept = set(), 0
        drop = {"format": 0, "held_out": 0, "duplicate": 0}
        for a, b in rows:
            a = a.lower()
            if not re.fullmatch("[a-z]+", a) or len(a) > MAX_IN or not b or len(b) > MAX_OUT or not KHMER.fullmatch(b):
                drop["format"] += 1
            elif hits(a, b, sets):
                drop["held_out"] += 1
            elif (a, b) in seen:
                drop["duplicate"] += 1
            else:
                seen.add((a, b))
                spell.append((a, b, name))
                n_kept += 1
        counts[name] = {"rows_in": len(rows), "kept": n_kept, "dropped": drop}
    vals = {"romanised": held["val_romanised"], "names": held["val_names"], "english": held["val_english"]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "spell_train.tsv").write_text("latin\tkhmer\tsource\n" + "".join(f"{a}\t{b}\t{n}\n" for a, b, n in spell),
                                         encoding="utf-8")
    (out / "aux_train.tsv").write_bytes((RAW / "aux_train.tsv").read_bytes())
    for n, ps in vals.items():
        (out / f"val_{n}.tsv").write_text("latin\tkhmer\n" + "".join(f"{a}\t{b}\n" for a, b in ps), encoding="utf-8")
    # the recheck model's files
    train_km = {b for _a, b, _n in spell}
    hold: dict[str, set[str]] = {}
    for a, b in vals["romanised"]:
        if b not in train_km and len(b) + 1 <= MAX_SRC and len(a) <= MAX_TGT:
            hold.setdefault(b, set()).add(a)
    pool: dict[str, set[str]] = {}
    for a, b, n in spell:
        if n != "mined" and len(b) + 1 <= MAX_SRC and len(a) <= MAX_TGT and b not in hold:
            pool.setdefault(b, set()).add(a)
    (out / "reverse_heldout.tsv").write_text("khmer\tlatin_refs\n" + "".join(
        f"{k}\t{' '.join(sorted(v))}\n" for k, v in sorted(hold.items())), encoding="utf-8")
    (out / "reverse_pool.tsv").write_text("khmer\tlatin\n" + "".join(
        f"{k}\t{a}\n" for k, v in sorted(pool.items()) for a in sorted(v)), encoding="utf-8")
    aux = [l.split("\t") for l in (RAW / "aux_train.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    vocab_out = {"src": "".join(sorted({c for a, _b, _n in spell for c in a} | {c for w, _p in aux for c in w}
                                       | {c for ps in vals.values() for a, _ in ps for c in a.lower()})),
                 "tgt": "".join(sorted({c for _a, b, _n in spell for c in b})),
                 "aux": sorted({t for _w, p in aux for t in p.split()})}
    report = {"sources": counts, "rows": len(spell), "held_out_sizes": {n: len(p) for n, p in held.items()},
              "vocab": vocab_out}
    sha = lambda f: hashlib.sha256((out / f).read_bytes()).hexdigest()  # noqa: E731
    (out / "manifest.json").write_text(json.dumps({"vocab": vocab_out, "sources": counts}, ensure_ascii=False,
                                                  indent=1) + "\n", encoding="utf-8")
    (out / "reverse_manifest.json").write_text(json.dumps({"pool_words": len(pool), "heldout_words": len(hold),
                                                           "pool_sha256": sha("reverse_pool.tsv"),
                                                           "heldout_sha256": sha("reverse_heldout.tsv")}, indent=1)
                                               + "\n", encoding="utf-8")
    (out / "prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return report


FILES = ("spell_train.tsv", "aux_train.tsv", "val_romanised.tsv", "val_names.tsv", "val_english.tsv",
         "reverse_pool.tsv", "reverse_heldout.tsv")


def check(out: Path, report: dict) -> bool:
    """Compare every prepared file, and the character lists, with the released training data."""
    ok = True
    for f in FILES:
        same = hashlib.sha256((out / f).read_bytes()).digest() == hashlib.sha256((RELEASED / f).read_bytes()).digest()
        print(f"{'identical' if same else 'DIFFERENT'}  {f}")
        ok &= same
    rel = json.loads((RELEASED / "manifest.json").read_text(encoding="utf-8"))["vocab"]
    same = rel == report["vocab"]
    print(f"{'identical' if same else 'DIFFERENT'}  character lists (manifest.json vocab)")
    return ok and same


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="compare with training-data/v0.1")
    a = ap.parse_args()
    report = build(a.out)
    print(json.dumps({"rows": report["rows"], "kept_by_source": {k: v["kept"] for k, v in report["sources"].items()}}))
    if a.check:
        return 0 if check(a.out, report) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
