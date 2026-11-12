"""Tests. The model tests need the model folder: set ROMAN2KHMER_MODEL_DIR (skipped otherwise)."""
import doctest
import os
import unittest
from pathlib import Path

from roman2khmer import Converter, context, model

DIR = os.environ.get("ROMAN2KHMER_MODEL_DIR")


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(context))
    tests.addTests(doctest.DocTestSuite(model))
    return tests


class ContextTests(unittest.TestCase):
    def test_bigram_prefers_seen_pair(self):
        lm = context.BigramLM.from_counts({"<s>": 2, "ក": 2, "ខ": 1}, {("<s>", "ក"): 2, ("ក", "ខ"): 1})
        self.assertGreater(lm.logp("ក", "ខ"), lm.logp("ក", "ក"))

    def test_save_and_load_keep_probabilities(self):
        import tempfile
        lm = context.BigramLM.from_counts({"<s>": 2, "ក": 2, "ខ": 1}, {("<s>", "ក"): 2, ("ក", "ខ"): 1, ("ខ", "</s>"): 1})
        with tempfile.TemporaryDirectory() as d:
            lm.save(Path(d) / "lm.npz")
            back = context.load_lm(Path(d) / "lm.npz")
        for prev, w in [("ក", "ខ"), ("ខ", "</s>"), ("<s>", "ក"), ("ក", "គ"), ("គ", "ក")]:
            self.assertEqual(lm.logp(prev, w), back.logp(prev, w))
        self.assertNotIn("</s>", back)

    def test_join_keys(self):
        self.assertEqual(context.join_keys("tro troung nas, Bong"), ["trotroung", "troungnas"])


class RecheckTests(unittest.TestCase):
    """The recheck arithmetic with a stand-in reverse model (no model files needed)."""

    class Fake:
        def score(self, pairs):
            return [-1.0 if k.startswith("ម") and a == "mok" else -5.0 for k, a in pairs]

    def test_adds_gamma_times_reverse_score_and_keeps_empty(self):
        c = Converter(word_model=None, lm=None, reverse=self.Fake(), gamma=0.5)
        got = c._recheck({"mok": [("មុខ", -1.2), ("ទឹក", -1.0), ("", -9.0)]})
        self.assertEqual(got["mok"], [("មុខ", -1.2 - 0.5), ("ទឹក", -1.0 - 2.5), ("", -9.0)])

    def test_gamma_zero_or_no_model_changes_nothing(self):
        nb = {"mok": [("មុខ", -1.2)]}
        self.assertIs(Converter(word_model=None, lm=None, reverse=self.Fake(), gamma=0)._recheck(nb), nb)
        self.assertIs(Converter(word_model=None, lm=None)._recheck(nb), nb)


@unittest.skipUnless(DIR, "set ROMAN2KHMER_MODEL_DIR to run the model tests")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = Converter.from_dir(Path(DIR))

    def test_common_words(self):
        self.assertEqual(self.c.candidates("pteah", 4)[0][0], "ផ្ទះ")
        self.assertEqual(self.c.convert("nham bay"), "ញ៉ាំបាយ")

    def test_context_picks_meaning(self):
        self.assertIn("មក", self.c.convert("pa ban mok vinh hz"))

    def test_khmer_and_punctuation_kept(self):
        out = self.c.convert("ថ្ងៃនេះ nham bay, ok?")
        self.assertTrue(out.startswith("ថ្ងៃនេះ "))
        self.assertIn(",", out)

    @unittest.skipUnless(DIR and (Path(DIR) / "reverse.npz").exists(), "no reverse.npz in the model folder")
    def test_reverse_scores_follow_the_typed_latin(self):
        r = self.c.reverse
        good, bad = r.score([("ផ្ទះ", "pteah"), ("ផ្ទះ", "mok")])
        self.assertGreater(good, bad)
        self.assertLess(good, 0)

    @unittest.skipUnless(DIR and (Path(DIR) / "reverse.npz").exists(), "no reverse.npz in the model folder")
    def test_gamma_zero_is_the_plain_pipeline(self):
        plain = Converter(self.c.word_model, self.c.lm)
        off = Converter(self.c.word_model, self.c.lm, reverse=self.c.reverse, gamma=0.0)
        for text in ("pa ban mok vinh hz", "knhom tov sala", "ថ្ងៃនេះ Sokha mok pteah"):
            self.assertEqual(plain.convert(text), off.convert(text))


if __name__ == "__main__":
    unittest.main()
