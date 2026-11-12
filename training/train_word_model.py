#!/usr/bin/env python3
"""Train the word model: a Latin word in, Khmer spellings out (the first stage of roman2khmer).

    python training/train_word_model.py --seed 2 --device cuda --out runs/word_s2
    python training/train_word_model.py --seed 2 --device cuda --out runs/word_s2 --resume   # after an interruption

Data (`training-data/v0.1/`): `spell_train.tsv` (Latin, Khmer, source), `aux_train.tsv` (Latin-script lexicon
entries and their Khmer-adapted pronunciation as phone tokens), the three validation files, and the character
lists in `manifest.json`.

Training, per step: a batch of 128 (Latin, Khmer) pairs, teacher-forced cross-entropy with label smoothing 0.1. On
every 8th step a batch of pronunciation rows also goes through the same encoder and a second decoder, and its loss
is added with weight 0.5; this side task only shapes the encoder and is discarded at export. After every epoch the
model writes each validation word greedily; the epoch with the best mean exact match (over the romanised, names and
English sets) is kept. Output: `best.pt`, `last.pt` (for `--resume`), `train_log.json` and `translit.npz`, the
NumPy weights `roman2khmer` loads.

On the CPU, runs with the same seed are exactly repeatable; GPU runs differ slightly between repeats.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from fractions import Fraction
from pathlib import Path

import torch
import yaml
from torch import nn

import nets

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_data(folder: Path, repeated: set[str], repeat: int):
    """`(training pairs with repeats, side-task rows, validation sets, character lists)`."""
    rows = [l.split("\t") for l in (folder / "spell_train.tsv").read_text(encoding="utf-8").splitlines()[1:]]
    pairs = [(a, b) for a, b, s in rows if s not in repeated] + [(a, b) for a, b, s in rows if s in repeated] * repeat
    side = [(w, p.split()) for w, p in (l.split("\t") for l in
                                        (folder / "aux_train.tsv").read_text(encoding="utf-8").splitlines()[1:])]
    vals = {}
    for n in ("romanised", "names", "english"):
        lines = (folder / f"val_{n}.tsv").read_text(encoding="utf-8").splitlines()[1:]
        vals[n] = [(c[0].strip(), c[1].strip()) for c in (l.split("\t") for l in lines
                   if l and not l.startswith("#") and "\t" in l)]
    vocab = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["vocab"]
    return pairs, side, vals, vocab


def src_batch(words: list[str], chars: str, max_in: int) -> torch.Tensor:
    if any(len(w) > max_in for w in words):
        raise ValueError(f"input longer than {max_in} characters")
    return nets.pad_rows([[chars.index(c) + 3 for c in w] for w in words])


def tgt_batch(seqs: list[list[str]], tokens, max_len: int) -> torch.Tensor:
    if any(len(s) > max_len for s in seqs):
        raise ValueError(f"target longer than {max_len} tokens")
    index = {t: i + 3 for i, t in enumerate(tokens)}
    return nets.pad_rows([[nets.BOS] + [index[t] for t in s] + [nets.EOS] for s in seqs])


def exact_hits(model, pairs, src_chars: str, tgt_chars: str, max_in: int) -> int:
    """How many validation words greedy decoding writes exactly right."""
    model.eval()
    dev = next(model.parameters()).device
    hit = 0
    with torch.no_grad():
        for i in range(0, len(pairs), 512):
            chunk = pairs[i:i + 512]
            S = src_batch([a for a, _ in chunk], src_chars, max_in).to(dev)
            for ids, (_a, b) in zip(nets.greedy(model, S), chunk):
                hit += "".join(tgt_chars[t - 3] for t in ids if t >= 3) == b
    model.train()
    return hit


def train(seed: int, out: Path, device: str = "cpu", resume: bool = False, epochs: int | None = None,
          stop_after: int | None = None, cfg: dict | None = None, data=None) -> dict:
    """Train one word model; see the module docstring. `epochs`, `stop_after`, `data` are for tests and pilots."""
    cfg = cfg or yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    c, m = cfg["word_model"], cfg["word_model"]["model"]
    epochs = epochs or c["epochs"]
    torch.set_num_threads(min(12, os.cpu_count() or 1))
    pairs, side, vals, vocab = data or load_data(ROOT / cfg["data"], set(c["repeated_sources"]), c["repeat"])
    src_v, tgt_v, side_v = vocab["src"], vocab["tgt"], vocab["aux"]
    dev = torch.device(device)

    torch.manual_seed(seed)
    model = nets.Seq2Seq(len(src_v) + 3, len(tgt_v) + 3, c["max_in"], **m)
    keep = torch.get_rng_state()                          # building the side decoder must not move the RNG
    side_dec = nets.SideDecoder(len(side_v) + 3, c["side_task_max"], **m)
    torch.set_rng_state(keep)
    model.to(dev), side_dec.to(dev)
    if dev.type == "cuda":
        torch.cuda.manual_seed(seed)
    params = list(model.parameters()) + list(side_dec.parameters())
    opt = torch.optim.AdamW(params, lr=c["lr"], weight_decay=c["weight_decay"])
    B = c["batch"]
    total = epochs * math.ceil(len(pairs) / B)
    sched = nets.cosine_with_warmup(opt, c["warmup"], total)
    ce = nn.CrossEntropyLoss(ignore_index=nets.PAD, label_smoothing=c["label_smoothing"])
    st = c["side_task"]

    # three random streams: batch order, side-task row order, and the side task's own dropout
    rng_pairs, rng_side = random.Random(seed), random.Random(10_000 + seed)
    main_state = nets.rng_state(dev)
    torch.manual_seed(10_000 + seed)
    side_state = nets.rng_state(dev)
    nets.set_rng_state(main_state, dev)
    side_order, side_pos = [], 0

    out.mkdir(parents=True, exist_ok=True)
    log, best, best_epoch, step, start, t0 = [], Fraction(-1), 0, 0, 0, time.time()
    last = out / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, weights_only=False, map_location="cpu")
        model.load_state_dict(ck["model"]), side_dec.load_state_dict(ck["side"])
        opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        start, step, log = ck["epoch"], ck["step"], ck["log"]
        best, best_epoch = Fraction(*ck["best"]), ck["best_epoch"]
        rng_pairs.setstate(ck["rng_pairs"]), rng_side.setstate(ck["rng_side"])
        side_order, side_pos, side_state = ck["side_order"], ck["side_pos"], ck["side_state"]
        nets.set_rng_state(ck["torch_rng"], dev)
        t0 = time.time() - ck["elapsed_s"]
        print(json.dumps({"resumed_from_epoch": start}), flush=True)

    meta = {"src_vocab": src_v, "tgt_vocab": tgt_v, "heads": m["heads"], "layers": m["layers"],
            "phones": False, "max_in": c["max_in"], "tags": False}
    for ep in range(start, epochs):
        order = list(range(len(pairs)))
        rng_pairs.shuffle(order)
        tot, tot_side, n_side = 0.0, 0.0, 0
        for i in range(0, len(order), B):
            chunk = [pairs[j] for j in order[i:i + B]]
            S = src_batch([a for a, _ in chunk], src_v, c["max_in"]).to(dev)
            T = tgt_batch([list(b) for _, b in chunk], tgt_v, nets.MAX_OUT).to(dev)
            logits = model(S, T[:, :-1])
            loss = ce(logits.reshape(-1, logits.shape[-1]), T[:, 1:].reshape(-1))
            tot += loss.item() * len(chunk)
            if step % st["every"] == 0:                   # side task: pronunciation of Latin-script entries
                if side_pos + st["batch"] > len(side_order):
                    side_order = list(range(len(side)))
                    rng_side.shuffle(side_order)
                    side_pos = 0
                sb = [side[j] for j in side_order[side_pos:side_pos + st["batch"]]]
                side_pos += st["batch"]
                main_state = nets.rng_state(dev)
                nets.set_rng_state(side_state, dev)
                Sa = src_batch([w for w, _ in sb], src_v, c["max_in"]).to(dev)
                Ta = tgt_batch([p for _, p in sb], side_v, c["side_task_max"]).to(dev)
                memory, pad = model.encode(Sa)
                la = side_dec(memory, pad, Ta[:, :-1])
                loss_side = ce(la.reshape(-1, la.shape[-1]), Ta[:, 1:].reshape(-1))
                side_state = nets.rng_state(dev)
                nets.set_rng_state(main_state, dev)
                loss = loss + st["weight"] * loss_side
                tot_side, n_side = tot_side + loss_side.item(), n_side + 1
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, c["grad_clip"])
            opt.step()
            sched.step()
            step += 1
        hits = {n: exact_hits(model, p, src_v, tgt_v, c["max_in"]) for n, p in vals.items()}
        score = sum(Fraction(hits[n], len(p)) for n, p in vals.items()) / 3
        log.append({"epoch": ep + 1, "loss": round(tot / len(pairs), 4), "side_loss": round(tot_side / max(1, n_side), 4),
                    "val": {n: round(hits[n] / len(p), 4) for n, p in vals.items()}, "val_mean": round(float(score), 4),
                    "minutes": round((time.time() - t0) / 60, 1)})
        print(json.dumps(log[-1]), flush=True)
        if score > best:
            best, best_epoch = score, ep + 1
            torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": ep + 1,
                        "meta": meta}, out / "best.pt")
        torch.save({"model": model.state_dict(), "side": side_dec.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep + 1, "step": step, "log": log,
                    "best": (best.numerator, best.denominator), "best_epoch": best_epoch,
                    "rng_pairs": rng_pairs.getstate(), "rng_side": rng_side.getstate(), "side_order": side_order,
                    "side_pos": side_pos, "side_state": side_state, "torch_rng": nets.rng_state(dev),
                    "elapsed_s": time.time() - t0}, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(last)
        if stop_after is not None and ep + 1 >= stop_after and ep + 1 < epochs:
            return {"stopped_after_epoch": ep + 1, "log": log}
    ck = torch.load(out / "best.pt", weights_only=False, map_location="cpu")
    nets.export_npz(ck["state"], {**ck["meta"], "epoch": ck["epoch"]}, out / "translit.npz")
    result = {"seed": seed, "epochs": epochs, "steps": step, "pairs_per_epoch": len(pairs),
              "best_epoch": best_epoch, "best_val_mean": round(float(best), 6), "device": str(dev), "log": log}
    (out / "train_log.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2, help="the released model is seed 2")
    ap.add_argument("--out", type=Path, required=True, help="run folder")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--stop-after", type=int, default=None, help="stop after this many epochs (quick checks)")
    a = ap.parse_args()
    r = train(a.seed, a.out, a.device, a.resume, stop_after=a.stop_after)
    print(json.dumps({k: v for k, v in r.items() if k != "log"}))


if __name__ == "__main__":
    main()
