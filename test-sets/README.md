# Test sets

What each set is, and what it was used for while the model was built. A set that guided a choice gives an
optimistic number for the model it chose; none of these sets is an untouched test any more. New claims need new
test data, ideally real chat from many writers.

| File | Size | What it is | Used for during development | Score with `pipeline/evaluate.py` |
|---|---|---|---|---|
| `messages_b.jsonl` | 450 messages, 30 generation calls | chat messages written by an LLM (Claude, `claude-opus-5-5`): 300 romanised, 150 mixing Khmer and Latin; the reference is the same LLM's Khmer version, unreviewed. `part`: `tune` (285, 19 calls) / `test` (165, 11 calls), split by call | `tune` chose the recheck weight and compared recheck models; `test` was read once to decide whether to keep the recheck; afterwards all 450 compared the released candidates and chose the released training seed | character error rate (95% interval resampling calls), exact messages |
| `messages_a.jsonl` | 447 messages | earlier LLM-written chat messages (Claude) with LLM references; `part`: `fit` (279) / `check` (168), split by hash | `fit` chose the sentence step's weights; `check` was reported, then reused in later comparisons | character error rate |
| `fresh_test.tsv` | 148 words, 179 spellings | less common Khmer words and how the author types them (`typed`: comma-separated), typed 2026-11-07; the words were chosen before typing | compared training recipes and guarded release decisions | mean over words of the share of spellings written exactly (95% interval resampling words), and over all spellings |
| `fresh_variants.tsv` | 225 | sound-alike variants of the author's spellings made by fixed rules (e.g. *j* → *ch*), frozen | as above | share written exactly |
| `names_test.tsv` | 228 word pairs | Cambodian person names from Wikidata (English and Khmer labels), aligned word by word | regression check | accuracy on names whose Khmer words never occur in training |
| `names_val.tsv` | 320 word pairs | the same kind of names, other people | validation (checkpoint selection) | not scored |
| `../training-data/raw/chhunneng_pairs.tsv`, `split = test` | 2,240 rows | the public Chhunneng pairs' test split | regression check | accuracy on words whose Khmer forms never occur in training |
| `pick_test.tsv`, `dev41.tsv` | 150, 41 words | English loanword panels from earlier English-word work | kept out of training only | not scored here |

**Not included:** the blinded human comparisons (one judge, the author). Their verdicts belong to the outputs that
were judged, not to a retrained model, which would need new judgments.

**Normalisation.** Character error rate keeps only Khmer and Latin letters (spaces, punctuation and digits removed)
and pools edits and reference lengths over messages. Exact match compares the same view.

**Licence.** CC BY-SA 4.0, like the training data (`../training-data/v0.1/NOTICE.md`). The names come from Wikidata
(CC0); the Chhunneng split from MIT-licensed pairs; the chat messages and their references were generated with
Claude (Anthropic) and are published for evaluation only, consistent with keeping Claude output out of training.
