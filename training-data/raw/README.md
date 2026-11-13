# Raw inputs of the training data

`pipeline/prepare.py` turns these files into the training files of the released model (`training-data/v0.1/`),
byte for byte (`--check` verifies it). Sources, licences and notices: `../v0.1/NOTICE.md`.

| File | Rows | What it is | Origin |
|---|---|---|---|
| `chhunneng_pairs.tsv` | 28,060 (24,178 train, 1,642 val, 2,240 test) | romanisation pairs (`latin`, `khmer`) with the project's frozen `train` / `val` / `test` split | Chhunneng, *khmer-text-transliteration* (GitHub, commit fe8028f1, MIT); mostly dictionary-style |
| `mined_pairs.tsv` | 35,211 | name and loanword pairs mined from bracketed spellings, e.g. "ចនសុន (Johnson)", already filtered | Khmer Wikipedia dump 2026-10-01 and FineWeb-2 `khm_Khmr` shard 000_00000. The mining used an older in-house model to choose and align pairs, which is not published, so mining is not part of this pipeline |
| `loanwords.tsv` | 1,850 | the author's loanword list (`word`, `khmer`) with its `train` / `val` / `test` split | drafted with LLM help and checked by the author |
| `author_examples.tsv` | 149 | the author's example spellings | the author |
| `author_typing.tsv` | 320 | Khmer words and how the author types them (`typed`: comma-separated variants); `part` = `train` or `retired_test` (an earlier test, later moved into training) | the author, typed on a local page |
| `author_checked_llm.tsv` | 400 | LLM-suggested spellings the author checked or corrected | the author |
| `llm_spellings.tsv` | 48,871 | spellings of common Khmer words from Gemini (`gemini-3.6-flash`, API, 2026-11-03 to 06) and GPT (`gpt-6.1-sol`, Codex CLI, 2026-11-06), kept by fixed rules; spellings only Claude gave are excluded | regenerating them needs API calls and gives different answers, so the filtered result is shipped |
| `common_words.txt` | 500 | the 500 commonest Khmer words of the FineWeb-2 shard; their Chhunneng validation rows move to training | counted from FineWeb-2 |
| `aux_train.tsv` | 6,772 | the side task: Latin-script entries of Google's Khmer lexicon with their pronunciation (phone tokens) | Google `language-resources` `km/data/lexicon.tsv` (commit bc72dd65, CC BY 4.0), filtered |

**What is not here.** The released data also left out rows related to a sealed reserve of 120 English test words,
kept unpublished so it can still serve as a test. Those rows are absent from these files (6 Chhunneng training rows,
1 Chhunneng validation row, 80 mined, 11 + 1 loanword-list, 1 example, 40 LLM; 140 in all), so `prepare.py`
reproduces the release without knowing the reserve. `aux_train.tsv` is shipped already filtered for the same
reason. This keeps the reserve's words unpublished; it cannot fully hide them, since comparing these files with
their public sources can hint at which entries were removed.
