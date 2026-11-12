"""roman2khmer: romanised Khmer (Khmer typed in Latin letters) → Khmer script.

    >>> from roman2khmer import Converter                       # doctest: +SKIP
    >>> c = Converter.from_dir("path/to/model")                 # doctest: +SKIP
    >>> c.convert("bong mok nham bay nov")                      # doctest: +SKIP
    'បងមកញ៉ាំបាយនៅ'
    >>> c.convert("ថ្ងៃនេះ Sokha mok pteah")                      # doctest: +SKIP
    'ថ្ងៃនេះ សុខាមកផ្ទះ'
    >>> c.candidates("mok", 3)                                  # doctest: +SKIP
    [('មក', -1.06), ('ម៉ក', -2.60), ('មុខ', -2.72)]  (rounded)

Three parts: a small character model that proposes Khmer spellings for each Latin word (`model`); a recheck that
adds how well each candidate explains the typed Latin, from a model of the other direction (Khmer → Latin,
`model.ReverseScorer`); and a sentence step that chooses among the candidates with a Khmer word-pair model, so that
the spelling fits the words around it (`context`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import context, model

__all__ = ["Converter", "context", "model"]
__version__ = "0.1.0"

WEIGHTS = "translit.npz"
WORD_PAIRS = "khmer_bigram.npz"
REVERSE = "reverse.npz"


@dataclass
class Converter:
    """A word model plus the Khmer word-pair model, with the settings chosen on development messages.

    Attributes:
        word_model: The loaded `model.Transliterator`.
        lm: The loaded `context.BigramLM`.
        alpha: Weight of the word model against the word-pair model.
        beta: Cost of spelling two neighbouring Latin words as one Khmer word (`None`: never).
        beam: Candidates per Latin word.
        reverse: The recheck model (`None`: no recheck).
        gamma: Weight of the recheck, `log P(typed Latin | Khmer candidate)`, added to each candidate's score
            (0: no recheck).
        cache_size: How many words' candidates to remember between calls (0: none).
    """
    word_model: model.Transliterator
    lm: context.BigramLM
    alpha: float = 1.0
    beta: float | None = 8.0
    beam: int = 8
    reverse: model.ReverseScorer | None = None
    gamma: float = 1.0
    cache_size: int = 50_000
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_dir(cls, path: str | Path, **settings) -> "Converter":
        """Load `translit.npz`, `khmer_bigram.npz` and, if present, the recheck model `reverse.npz` from one folder.

        Args:
            path: The model folder.
            **settings: Overrides for `alpha`, `beta`, `beam`, `gamma`.

        Returns:
            A ready `Converter`.

        Raises:
            FileNotFoundError: If either file is missing.
        """
        p = Path(path)
        rev = model.ReverseScorer(p / REVERSE) if (p / REVERSE).exists() else None
        return cls(model.load(p / WEIGHTS), context.load_lm(p / WORD_PAIRS), reverse=rev, **settings)

    def convert(self, text: str, sep: str = "") -> str:
        """Write every Latin word of `text` in Khmer, choosing spellings with the neighbouring words in mind.

        Khmer already in the text is kept and used as context; punctuation, digits and spacing next to Khmer are
        kept as they are.

        Args:
            text: Romanised Khmer, or Khmer mixed with Latin words.
            sep: What replaces a space between two converted Latin words (default none, as Khmer is written).

        Returns:
            The text in Khmer script.
        """
        keys = {t.lower() for t in context.TOKEN.findall(text) if context._is_latin(t)}
        if self.beta is not None:
            keys |= set(context.join_keys(text))
        todo = sorted(k for k in keys if k not in self._cache)
        if todo:
            if len(self._cache) + len(todo) > self.cache_size:
                self._cache.clear()
            got = dict(zip(todo, self.word_model.nbest_batch([re.sub(r"[^a-z]", "", k) for k in todo], self.beam)))
            got = self._recheck(got)
            if self.cache_size:
                self._cache.update(got)
        else:
            got = {}
        nb = {k: self._cache.get(k) or got[k] for k in keys}
        return context.convert(text, self.word_model, self.lm, alpha=self.alpha, beam=self.beam, nbest=nb, sep=sep,
                               beta=self.beta)

    def _recheck(self, nb: dict[str, list[tuple[str, float]]]) -> dict[str, list[tuple[str, float]]]:
        """Add `gamma · log P(Latin | candidate)` to every non-empty candidate (empty ones keep their score)."""
        if self.reverse is None or self.gamma == 0:
            return nb
        pairs = sorted({(km, k) for k, v in nb.items() for km, _ in v if km})
        sc = dict(zip(pairs, self.reverse.score([(km, re.sub(r"[^a-z]", "", k)) for km, k in pairs])))
        return {k: [(km, lp + self.gamma * sc[km, k]) if km else (km, lp) for km, lp in v] for k, v in nb.items()}

    def candidates(self, word: str, n: int = 8) -> list[tuple[str, float]]:
        """The word model's `n` best Khmer spellings of one Latin word, with log-probabilities, best first."""
        return self.word_model.nbest_batch([word], n)[0]
