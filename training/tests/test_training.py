"""Training code: tiny runs on made-up data, and the package reads what the trainers export.

    python -m unittest discover -s training/tests        (needs training/requirements.txt)
"""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_bigram  # noqa: E402
import nets  # noqa: E402
import train_reverse_model as rev  # noqa: E402
import train_word_model as word  # noqa: E402
from roman2khmer import model as pkg  # noqa: E402

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8"))
PAIRS = [("ka", "កា"), ("kho", "ខោ"), ("ko", "គោ"), ("chi", "ចិ"), ("ta", "តា"), ("na", "ណា"), ("ba", "បា"),
         ("mae", "មេ"), ("la", "លា"), ("sa", "សា"), ("ha", "ហា"), ("kom", "កុំ")]
SRC = "abcdefghijklmnopqrstuvwxyz"
TGT = "".join(sorted(set("".join(b for _a, b in PAIRS))))


def small(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    for part in ("word_model", "reverse_model"):
        c[part]["model"] = {"d": 32, "layers": 1, "heads": 2, "ff": 64, "dropout": 0.0}
        c[part]["warmup"] = 10
        c[part]["lr"] = 0.01
        c[part]["batch"] = 12
    c["word_model"]["side_task"] = {"weight": 0.5, "every": 4, "batch": 4}
    c["reverse_model"]["eval_every"] = 50
    return c


class WordModel(unittest.TestCase):
    def test_learns_and_exports_for_the_package(self):
        side = [(a, list(a)) for a, _ in PAIRS]
        data = (PAIRS, side, {"romanised": PAIRS, "names": PAIRS[:4], "english": PAIRS[4:8]},
                {"src": SRC, "tgt": TGT, "aux": SRC})
        with tempfile.TemporaryDirectory() as d:
            torch.manual_seed(0)
            r = word.train(0, Path(d), epochs=150, cfg=small(CFG), data=data)
            self.assertGreaterEqual(r["best_val_mean"], 0.9)
            m = pkg.load(Path(d) / "translit.npz")
            self.assertEqual(m.nbest_batch(["kho"], 2)[0][0][0], "ខោ")

    def test_same_seed_same_result_on_cpu(self):
        side = [(a, list(a)) for a, _ in PAIRS]
        data = (PAIRS, side, {"romanised": PAIRS, "names": PAIRS, "english": PAIRS}, {"src": SRC, "tgt": TGT, "aux": SRC})
        with tempfile.TemporaryDirectory() as d:
            a = word.train(1, Path(d) / "a", epochs=3, cfg=small(CFG), data=data)
            b = word.train(1, Path(d) / "b", epochs=3, cfg=small(CFG), data=data)
        self.assertEqual([x["loss"] for x in a["log"]], [x["loss"] for x in b["log"]])


class ReverseModel(unittest.TestCase):
    def test_scores_real_spelling_higher(self):
        V = [rev.TAGS["km"], rev.TAGS["lat"]] + list(SRC) + list(TGT)
        pool = {b: [a] for a, b in PAIRS}
        held = {b: [a] for a, b in PAIRS[:3]}
        with tempfile.TemporaryDirectory() as d:
            torch.manual_seed(0)
            rev.train(0, Path(d), steps=400, cfg=small(CFG), data=(V, pool, held))
            scorer = pkg.ReverseScorer(Path(d) / "reverse.npz")
            good, bad = scorer.score([("ខោ", "kho"), ("ខោ", "sa")])
        self.assertGreater(good, bad)

    def test_mask_allows_latin_and_end_only(self):
        V = [rev.TAGS["km"], rev.TAGS["lat"], "a", "b", "ក"]
        self.assertEqual(rev.latin_mask(V).tolist(), [False, False, True, False, False, True, True, False])


class Bigram(unittest.TestCase):
    def test_segment(self):
        self.assertEqual(build_bigram.segment("កខគឃ", {"ក", "កខ", "គ"}, frozenset({"កខ", "គ"})), ["កខ", "គ"])


class Nets(unittest.TestCase):
    def test_parameter_names_match_the_numpy_reader(self):
        m = nets.Seq2Seq(5, 6, max_in=8, d=8, layers=1, heads=2, ff=16)
        names = set(m.state_dict())
        for k in ("src_emb.weight", "tgt_pos.weight", "enc.0.self.q.weight", "dec.0.cross.o.bias", "dec_ln.weight",
                  "out.weight"):
            self.assertIn(k, names)
        self.assertTrue(np.isfinite(m(torch.tensor([[3, 4]]), torch.tensor([[nets.BOS]])).detach().numpy()).all())


if __name__ == "__main__":
    unittest.main()
