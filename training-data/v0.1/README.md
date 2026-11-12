# roman2khmer v0.1 training data

The exact files the released v0.1 models were trained on. With them, `TRAINING.md` (repository root) retrains the
word model and the recheck model.

| File | Rows | Columns | Used for |
|---|---|---|---|
| `spell_train.tsv` | 109,088 | `latin`, `khmer`, `source` | word model: (romanised spelling, Khmer word) pairs |
| `aux_train.tsv` | 6,772 | `latin`, `phones` | word model's training-only side task: Khmer-adapted pronunciation of Latin-script lexicon entries |
| `val_romanised.tsv`, `val_names.tsv`, `val_english.tsv` | 1,576, 320, 179 | `latin`, `khmer` | checkpoint selection (greedy exact match, mean of the three) |
| `reverse_pool.tsv` | 70,763 | `khmer`, `latin` | recheck model (Khmer → Latin) training pool |
| `reverse_heldout.tsv` | 662 | `khmer`, `latin_refs` | recheck model checkpoint selection (held-out words, all their spellings) |
| `manifest.json`, `reverse_manifest.json` | | | character lists, counts per source, held-out rule, sha256 of every file |

## Sources of `spell_train.tsv`
| `source` | Rows | What it is |
|---|---|---|
| `llm_bulk` | 48,057 | Gemini and GPT spellings of common Khmer words, kept only if they pass fixed rules, with no hand checking: letters a–z only and at most 3 × (Khmer length) + 2 letters; dropped if trusted data (the author's typing, the Chhunneng pairs) uses the same spelling for another word; a spelling suggested for several words is kept only for the most common one. Claude-only spellings excluded |
| `mined` | 34,345 | name and loanword pairs mined from Khmer Wikipedia and FineWeb-2 (bracketed Latin spellings), filtered |
| `chhunneng` | 24,016 | the public Chhunneng romanisation pairs (MIT) |
| `list_train` | 1,480 | the author's loanword list (English and foreign words in Khmer spelling, drafted with LLM help and checked by the author), training split |
| `owner_checked_llm` | 580 | LLM-suggested spellings the author checked |
| `owner_typing` | 408 | the author's own typing |
| `owner_examples` | 143 | the author's example spellings |
| `chhunneng_val_common` | 59 | Chhunneng validation rows of the 500 commonest Khmer words, moved into training |

The author-made sources (`list_train`, `owner_*`; "owner" = the author) are repeated 4 times per epoch in training
(`manifest.json`, `repeated_4x`).

**Held out.** Every row whose Khmer target or Latin spelling belongs to an evaluation set was removed before these
files were written (`manifest.json`, `held_out`). The evaluation sets themselves (the author's typed
test words, the reserve, the LLM-written development messages) are not published.

## Licence
CC BY-SA 4.0, the same as the model files, because part of the data comes from Khmer Wikipedia (CC BY-SA). The
upstream sources keep their own terms; generated spellings are subject to their providers' terms. Sources, versions
and notices: `NOTICE.md` in this folder.
