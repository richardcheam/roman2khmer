#!/usr/bin/env python3
"""Train the recheck model: given a Khmer word, how likely is a Latin spelling (roman2khmer's second stage).

    python training/train_reverse_model.py --seed 0 --device cuda --out runs/reverse_s0
    python training/train_reverse_model.py --seed 0 --device cuda --out runs/reverse_s0 --resume

It is the same encoder-decoder as the word model, run the other way: the input is a direction tag (U+E211, "write
Latin") followed by the Khmer characters, the output is restricted to the 26 Latin letters and the end mark. The
character list is shared: two direction tags, the Latin letters, then the Khmer characters (from `manifest.json`).

Data: `reverse_pool.tsv` (Khmer word, one real spelling per row; web-mined rows left out) and `reverse_heldout.tsv`
(held-out Khmer words with all their spellings). Each step takes 128 rows "word first": every pass visits each
Khmer word once in a shuffled order and picks one of its spellings at random, so frequent words with many spellings
do not dominate. Loss: cross-entropy with label smoothing 0.1 spread over the allowed outputs only. Every 782 steps
(and at the end) the model scores the held-out words' real spellings; the step with the lowest negative
log-likelihood (mean over a word's spellings, then over words) is kept. Output: `best.pt`, `last.pt`,
`train_log.json` and `reverse.npz`, which `roman2khmer.model.ReverseScorer` loads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

import nets

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "config.yaml"
TAGS = {"km": "", "lat": ""}           # direction tags: "write Khmer", "write Latin"


def load_data(folder: Path) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    """`(character list, pool: Khmer -> spellings, held-out: Khmer -> spellings)`, hashes checked."""
    man = json.loads((folder / "reverse_manifest.json").read_text(encoding="utf-8"))
    for f, k in (("reverse_pool.tsv", "pool_sha256"), ("reverse_heldout.tsv", "heldout_sha256")):
        if hashlib.sha256((folder / f).read_bytes()).hexdigest() != man[k]:
            raise SystemExit(f"{folder / f} differs from reverse_manifest.json")
    vocab = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["vocab"]
    V = [TAGS["km"], TAGS["lat"]] + list(vocab["src"]) + list(vocab["tgt"])
    pool: dict[str, list[str]] = {}
    for l in (folder / "reverse_pool.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        k, a = l.split("\t")
        pool.setdefault(k, []).append(a)
    held = {k: v.split() for k, v in (l.split("\t") for l in
                                      (folder / "reverse_heldout.tsv").read_text(encoding="utf-8").splitlines()[1:])}
    return V, pool, held


def latin_mask(V: list[str]) -> np.ndarray:
    """Allowed output ids: the end mark and the Latin letters (no tags, no Khmer)."""
    m = np.zeros(len(V) + 3, dtype=bool)
    m[nets.EOS] = True
    for i, c in enumerate(V):
        if c not in TAGS.values() and c.isascii():
            m[i + 3] = True
    return m


class WordFirst:
    """Rows drawn word first: each pass visits every Khmer word once (shuffled) with one of its spellings."""

    def __init__(self, pool: dict[str, list[str]], seed: int) -> None:
        self.words, self.pool = sorted(pool), pool
        self.rng = random.Random(20_000 + seed)
        self.order: list[tuple[str, str]] = []
        self.pos = 0

    def take(self, n: int) -> list[tuple[str, str]]:
        out = []
        while len(out) < n:
            if self.pos >= len(self.order):
                ws = list(self.words)
                self.rng.shuffle(ws)
                self.order = [(w, self.rng.choice(self.pool[w])) for w in ws]
                self.pos = 0
            k = min(n - len(out), len(self.order) - self.pos)
            out += self.order[self.pos:self.pos + k]
            self.pos += k
        return out


def ids(khmer: str, latin: str, index: dict[str, int], max_src: int, max_tgt: int) -> tuple[list[int], list[int]]:
    src = [index[TAGS["lat"]]] + [index[c] for c in khmer]
    if len(src) > max_src or len(latin) > max_tgt:
        raise ValueError(f"too long: {khmer!r} -> {latin!r}")
    return src, [nets.BOS] + [index[c] for c in latin] + [nets.EOS]


def masked_loss(logits, target, allowed, smoothing: float):
    """Cross-entropy over the allowed outputs, label smoothing spread over those outputs only; padding ignored."""
    lp = torch.log_softmax(logits.masked_fill(~allowed, float("-inf")), -1)
    keep = target != nets.PAD
    nll = -lp.gather(-1, target.masked_fill(~keep, nets.EOS)[..., None])[..., 0]
    smooth = -lp.masked_fill(~allowed, 0.0).sum(-1) / allowed.sum()
    return ((1 - smoothing) * nll + smoothing * smooth).masked_fill(~keep, 0.0).sum() / keep.sum()


def heldout_nll(model, held, index, allowed, dev, max_src, max_tgt) -> float:
    """Unsmoothed negative log-likelihood (nats, end mark included) of the held-out words' real spellings: mean over a
    word's spellings, then over words."""
    rows = [(k, a) for k in sorted(held) for a in held[k]]
    per: dict[str, list[float]] = {}
    was = model.training
    model.eval()
    with torch.no_grad():
        for i in range(0, len(rows), 512):
            ch = rows[i:i + 512]
            pairs = [ids(k, a, index, max_src, max_tgt) for k, a in ch]
            S, T = nets.pad_rows([s for s, _ in pairs]).to(dev), nets.pad_rows([t for _, t in pairs]).to(dev)
            lp = torch.log_softmax(model(S, T[:, :-1]).masked_fill(~allowed, float("-inf")), -1)
            y = T[:, 1:]
            keep = y != nets.PAD
            tok = lp.gather(-1, y.masked_fill(~keep, nets.EOS)[..., None])[..., 0].masked_fill(~keep, 0.0)
            for (k, _a), s in zip(ch, (-tok.sum(1)).tolist()):
                per.setdefault(k, []).append(s)
    model.train(was)
    return sum(sum(v) / len(v) for v in per.values()) / len(per)


def train(seed: int, out: Path, device: str = "cpu", resume: bool = False, steps: int | None = None,
          stop_after: int | None = None, cfg: dict | None = None, data=None) -> dict:
    """Train the recheck model; see the module docstring. `steps`, `stop_after`, `data` are for tests and pilots."""
    cfg = cfg or yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    c, m = cfg["reverse_model"], cfg["reverse_model"]["model"]
    steps = steps or c["steps"]
    torch.set_num_threads(min(12, os.cpu_count() or 1))
    V, pool, held = data or load_data(ROOT / cfg["data"])
    index = {ch: i + 3 for i, ch in enumerate(V)}
    dev = torch.device(device)
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed(seed)
    model = nets.Seq2Seq(len(V) + 3, len(V) + 3, c["max_src"], **m).to(dev)
    allowed = torch.tensor(latin_mask(V), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = nets.cosine_with_warmup(opt, c["warmup"], steps)
    stream = WordFirst(pool, seed)
    meta = {"vocab": V, "tags": TAGS, "max_src": c["max_src"], "max_tgt": c["max_tgt"], "layers": m["layers"],
            "heads": m["heads"]}
    out.mkdir(parents=True, exist_ok=True)
    last = out / "last.pt"
    log, best, best_step, step, tot, n, t0 = [], float("inf"), 0, 0, 0.0, 0, time.time()
    if resume and last.exists():
        ck = torch.load(last, weights_only=False, map_location="cpu")
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        stream.rng.setstate(ck["stream_rng"])
        stream.order, stream.pos = [tuple(x) for x in ck["stream_order"]], ck["stream_pos"]
        nets.set_rng_state(ck["torch_rng"], dev)
        step, best, best_step, log = ck["step"], ck["best"], ck["best_step"], ck["log"]
        t0 = time.time() - ck["elapsed_s"]
        print(json.dumps({"resumed_at_step": step}), flush=True)
    model.train()
    while step < steps:
        rows = stream.take(c["batch"])
        pairs = [ids(k, a, index, c["max_src"], c["max_tgt"]) for k, a in rows]
        S, T = nets.pad_rows([s for s, _ in pairs]).to(dev), nets.pad_rows([t for _, t in pairs]).to(dev)
        loss = masked_loss(model(S, T[:, :-1]), T[:, 1:], allowed, c["label_smoothing"])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        opt.step()
        sched.step()
        step += 1
        tot, n = tot + loss.item(), n + 1
        if step % c["eval_every"] == 0 or step == steps:
            nll = heldout_nll(model, held, index, allowed, dev, c["max_src"], c["max_tgt"])
            log.append({"step": step, "train_loss": round(tot / n, 4), "heldout_nll": round(nll, 5),
                        "minutes": round((time.time() - t0) / 60, 1)})
            tot, n = 0.0, 0
            print(json.dumps(log[-1]), flush=True)
            if nll < best:
                best, best_step = nll, step
                torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "step": step},
                           out / "best.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "stream_rng": stream.rng.getstate(), "stream_order": stream.order, "stream_pos": stream.pos,
                        "torch_rng": nets.rng_state(dev), "step": step, "best": best, "best_step": best_step,
                        "log": log, "elapsed_s": time.time() - t0}, out / "last.pt.tmp")
            (out / "last.pt.tmp").replace(last)
            if stop_after is not None and step >= stop_after and step < steps:
                return {"stopped_at_step": step, "log": log}
    ck = torch.load(out / "best.pt", weights_only=False, map_location="cpu")
    nets.export_npz(ck["state"], {**meta, "epoch": ck["step"]}, out / "reverse.npz")
    result = {"seed": seed, "steps": step, "pool_words": len(pool), "heldout_words": len(held),
              "best_step": best_step, "best_heldout_nll": round(best, 6), "device": str(dev), "log": log}
    (out / "train_log.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="the released model is seed 0")
    ap.add_argument("--out", type=Path, required=True, help="run folder")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--stop-after", type=int, default=None, help="stop after this many steps (quick checks)")
    a = ap.parse_args()
    r = train(a.seed, a.out, a.device, a.resume, stop_after=a.stop_after)
    print(json.dumps({k: v for k, v in r.items() if k != "log"}))


if __name__ == "__main__":
    main()
