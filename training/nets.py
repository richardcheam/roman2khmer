"""The character encoder-decoder used by both trained models (PyTorch, training only).

A small pre-LayerNorm Transformer over characters: embeddings plus learned positions, `layers` encoder blocks and
`layers` decoder blocks (self-attention, cross-attention, GELU feed-forward), a final LayerNorm on each side and a
linear output layer. Parameter names match the NumPy inference in `roman2khmer.model`, so `export_npz` writes a file
the package loads directly.

Token ids: 0 = padding, 1 = start, 2 = end, and character `i` of a vocabulary is id `i + 3`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

PAD, BOS, EOS = 0, 1, 2
MAX_OUT = 40          # longest output; the decoder has MAX_OUT + 2 positions (start and end marks)


class Attn(nn.Module):
    """Multi-head attention with separate query, key, value and output projections."""

    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.heads, self.drop = heads, nn.Dropout(dropout)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Attend `q_in` (B, Tq, d) over `kv_in` (B, Tk, d); `mask` is True where attention is blocked."""
        B, Tq, d = q_in.shape
        h, dh = self.heads, d // self.heads
        q = self.q(q_in).view(B, Tq, h, dh).transpose(1, 2)
        k = self.k(kv_in).view(B, -1, h, dh).transpose(1, 2)
        v = self.v(kv_in).view(B, -1, h, dh).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(dh)
        if mask is not None:
            s = s.masked_fill(mask[:, None], -1e9)
        return self.o((self.drop(s.softmax(-1)) @ v).transpose(1, 2).reshape(B, Tq, d))


class Block(nn.Module):
    """One pre-LayerNorm block; `cross=True` adds attention over the encoder output (a decoder block)."""

    def __init__(self, d: int, heads: int, ff: int, dropout: float, cross: bool) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.self = Attn(d, heads, dropout)
        if cross:
            self.ln3, self.cross = nn.LayerNorm(d), Attn(d, heads, dropout)
        self.ff1, self.ff2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.is_cross, self.drop = cross, nn.Dropout(dropout)

    def forward(self, h, self_mask, memory=None, mem_mask=None):
        a = self.ln1(h)
        h = h + self.drop(self.self(a, a, self_mask))
        if self.is_cross:
            h = h + self.drop(self.cross(self.ln2(h), memory, mem_mask))
            ln_ff = self.ln3
        else:
            ln_ff = self.ln2
        return h + self.drop(self.ff2(nn.functional.gelu(self.ff1(ln_ff(h)))))


class Seq2Seq(nn.Module):
    """Character encoder-decoder.

    Args:
        n_src: Source vocabulary size including the 3 special ids.
        n_tgt: Target vocabulary size including the 3 special ids.
        max_in: Longest source (number of source positions).
        d, layers, heads, ff, dropout: Width, blocks per side, attention heads, feed-forward size, dropout.
    """

    def __init__(self, n_src: int, n_tgt: int, max_in: int, d: int = 192, layers: int = 3, heads: int = 4,
                 ff: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.src_emb, self.tgt_emb = nn.Embedding(n_src, d), nn.Embedding(n_tgt, d)
        self.src_pos, self.tgt_pos = nn.Embedding(max_in, d), nn.Embedding(MAX_OUT + 2, d)
        self.enc = nn.ModuleList(Block(d, heads, ff, dropout, False) for _ in range(layers))
        self.dec = nn.ModuleList(Block(d, heads, ff, dropout, True) for _ in range(layers))
        self.enc_ln, self.dec_ln = nn.LayerNorm(d), nn.LayerNorm(d)
        self.out = nn.Linear(d, n_tgt)
        self.drop = nn.Dropout(dropout)

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder output (B, Ts, d) and the padding mask (B, 1, Ts)."""
        pad = (src == PAD)[:, None, :]
        h = self.drop(self.src_emb(src) + self.src_pos.weight[: src.shape[1]])
        for blk in self.enc:
            h = blk(h, pad)
        return self.enc_ln(h), pad

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """Teacher-forced logits (B, Tt, n_tgt); `tgt_in` starts with the start id."""
        memory, pad = self.encode(src)
        L = tgt_in.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tgt_in.device), 1)[None]
        g = self.drop(self.tgt_emb(tgt_in) + self.tgt_pos.weight[:L])
        for blk in self.dec:
            g = blk(g, causal, memory, pad)
        return self.out(self.dec_ln(g))


class SideDecoder(nn.Module):
    """A second decoder of the same design reading the shared encoder (the pronunciation side task)."""

    def __init__(self, n_out: int, max_len: int, d: int = 192, layers: int = 3, heads: int = 4, ff: int = 512,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.emb, self.pos = nn.Embedding(n_out, d), nn.Embedding(max_len + 2, d)
        self.dec = nn.ModuleList(Block(d, heads, ff, dropout, True) for _ in range(layers))
        self.ln, self.out, self.drop = nn.LayerNorm(d), nn.Linear(d, n_out), nn.Dropout(dropout)

    def forward(self, memory: torch.Tensor, pad: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        L = tgt_in.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tgt_in.device), 1)[None]
        g = self.drop(self.emb(tgt_in) + self.pos.weight[:L])
        for blk in self.dec:
            g = blk(g, causal, memory, pad)
        return self.out(self.ln(g))


def pad_rows(rows: list[list[int]]) -> torch.Tensor:
    """Right-pad id lists with 0 into one (B, T) tensor."""
    T = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long)
    for r, s in enumerate(rows):
        T[r, :len(s)] = torch.tensor(s)
    return T


def greedy(model: Seq2Seq, src: torch.Tensor, allowed: torch.Tensor | None = None) -> list[list[int]]:
    """Greedy decoding of a batch; `allowed` (bool over output ids) restricts the outputs. Returns ids without
    start and end marks."""
    out = torch.full((src.shape[0], 1), BOS, dtype=torch.long, device=src.device)
    done = torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)
    for _ in range(MAX_OUT + 1):
        logits = model(src, out)[:, -1]
        if allowed is not None:
            logits = logits.masked_fill(~allowed, float("-inf"))
        nxt = logits.argmax(-1)
        out = torch.cat([out, nxt[:, None]], 1)
        done |= nxt == EOS
        if bool(done.all()):
            break
    return [r[: r.index(EOS)] if EOS in r else r for r in out[:, 1:].tolist()]


def export_npz(state: dict, meta: dict, path: Path) -> Path:
    """Write weights (float32) plus a `__meta__` JSON string, the format `roman2khmer.model` reads."""
    arrays = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in state.items()}
    np.savez(path, __meta__=np.array(json.dumps(meta, ensure_ascii=False)), **arrays)
    return path


def rng_state(device: torch.device) -> dict:
    """The torch random states that drive dropout (CPU, plus the GPU one when training there)."""
    st = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state(device)
    return st


def set_rng_state(st: dict, device: torch.device) -> None:
    torch.set_rng_state(st["cpu"])
    if device.type == "cuda" and "cuda" in st:
        torch.cuda.set_rng_state(st["cuda"], device)


def cosine_with_warmup(opt, warmup: int, total: int):
    """Linear warm-up over `warmup` steps, then cosine decay to 0 at `total` steps."""
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
