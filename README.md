# roman2khmer

Turns **romanised Khmer** (Khmer typed in Latin letters, as in chat) into **Khmer script**:

| input | output |
|---|---|
| `bong mok nham bay nov` | បងមកញ៉ាំបាយនៅ |
| `knhom tov sala rean` | ខ្ញុំទៅសាលារៀន |
| `ថ្ងៃនេះ Sokha mok pteah` | ថ្ងៃនេះ សុខាមកផ្ទះ |

There is no standard way to romanise Khmer: *tror trung*, *tro troung* and *tro trung* all mean ទ្រទ្រង់. The goal is
the right Khmer word for what the writer meant, so that the output reads correctly in any application (search,
messaging, text-to-speech, ...).

**Status: v0.1.0, research preview.** Model files: [ric912/roman2khmer](https://huggingface.co/ric912/roman2khmer) on Hugging Face.
Technical report: [`report/roman2khmer-technical-report.pdf`](report/roman2khmer-technical-report.pdf).

## How it works
1. **Word model:** a small character Transformer (2.6M parameters, about 10 MB, NumPy only at run time) proposes up
   to 8 Khmer spellings for each Latin word, with probabilities.
2. **Recheck:** a model of the other direction (Khmer → Latin, same size) scores how well each candidate explains
   the Latin that was typed, and adds that to the candidate's score. It removes candidates that drop or add a
   syllable, or that a common word would otherwise win.
3. **Sentence step:** a Khmer word-pair model counted from web text chooses one spelling per word so that the whole
   message is likely Khmer (*mok* can be មក "come" or មុខ "face"; the neighbouring words decide). Two neighbouring
   Latin words may also become one Khmer word (*pon man* → ប៉ុន្មាន). Khmer already in the text is kept and used as
   context.

## Use
**Python**
```bash
pip install "roman2khmer[hub]"          # NumPy + huggingface_hub; the model (about 27 MB) downloads on first use
```
```python
from roman2khmer import Converter

c = Converter.from_pretrained()          # pinned model release, sha256-checked, cached after the first call
c.convert("pa ban mok vinh hz")          # 'ប៉ាបានមកវិញហើយ'
c.convert("ថ្ងៃនេះ Sokha mok pteah")       # Khmer already in the text is kept and used as context
c.candidates("mok", 3)                   # the word model's best spellings with log-probabilities
```
Without `[hub]`, download the files yourself (`hf download ric912/roman2khmer --local-dir model`) and use
`Converter.from_dir("model")`. Load the converter once and reuse it: loading takes under a second and about 300 MB of
RAM. A short message whose words are all new takes about 0.9 second on a laptop CPU (0.4 second with
`gamma=0`, which turns the recheck off); words seen before are cached.

**Command line**
```bash
roman2khmer convert "bong mok nham bay nov"                 # បងមកញ៉ាំបាយនៅ
printf 'knhom tov sala rean\npa ban mok vinh hz\n' | roman2khmer convert   # one message per line
roman2khmer convert --offline "bong mok"                    # only the local cache, no network
```

**HTTP API**
```bash
pip install "roman2khmer[serve]"
roman2khmer serve --port 8000                               # this machine only, by default
curl -s localhost:8000/convert -H 'Content-Type: application/json' -d '{"text": "bong mok nham bay nov"}'
# {"khmer": "បងមកញ៉ាំបាយនៅ"}
```
One model in memory; messages are converted one at a time, so a busy server answers 503 ("try again") rather than
queuing without limit. Limits: `--max-chars` (default 1000), `--max-waiting` (8), `--timeout` (30 s). There is no
authentication: put a reverse proxy with TLS, authentication and rate limits in front before exposing it.

**Docker**
```bash
docker build -t roman2khmer .            # fetches the pinned model files at build time
docker run --rm -p 8000:8000 roman2khmer
```

## How well it works
There is no public multi-writer benchmark of real romanised chat yet, so these figures come from the sets the
author could build (details, intervals and caveats in the report):

| Measure | Result |
|---|---|
| Character error rate on 450 LLM-written chat messages (spaces removed; references LLM-written, unreviewed). These messages were used to choose the decoding settings and the released training seed, so the figure is optimistic. | 8.95% |
| Fresh word test: 148 less common words typed by one native speaker (179 spellings), each converted alone | 58.2% (share of each word's spellings written exactly, averaged over words) |
| Recheck on vs off (earlier word model), 165 held-out messages scored once | error 10.56% → 9.37%; one judge preferred the recheck in 17 differing messages, the version without it in 7 |

**Limitations.** All human judgments and typed test words come from one native speaker; message results use
LLM-written messages. Remaining errors include context choices (ព្រុស "bark" vs the more common ព្រោះ "because"),
abbreviations that stand for whole words or names (*fb*, *BKK*), and occasional Khmer strings that are not words.

## Train it yourself
[`TRAINING.md`](TRAINING.md) retrains all three parts from scratch with the code in [`training/`](training/) and
the exact training data of v0.1 in [`training-data/v0.1/`](training-data/v0.1/README.md): about two hours on a
laptop GPU, or longer on a CPU.

## Repository
| Path | What it is |
|---|---|
| `src/roman2khmer/` | the package: `Converter` (`__init__.py`), the sentence step and word-pair model (`context.py`), the character Transformer in NumPy (`model.py`) |
| `tests/` | package tests (`python -m unittest discover -s tests`; set `ROMAN2KHMER_MODEL_DIR` for the model tests) |
| `training/` | training code and settings (PyTorch), with its own tests |
| `training-data/v0.1/` | the training data of the released models, with sources and notices |
| `report/` | the technical report (PDF) |

## Licence
Code: MIT (`LICENSE`). Training data and model files: CC BY-SA 4.0, because part of the data comes from Khmer
Wikipedia; the upstream sources keep their own terms (`training-data/v0.1/NOTICE.md`).

## Citation
```bibtex
@techreport{cheam2026roman2khmer,
  title  = {Chat-Romanised Khmer to Khmer Script: Candidate Generation, Reverse Scoring and Contextual Decoding},
  author = {Cheam, Richard},
  year   = {2026},
  url    = {https://github.com/richardcheam/roman2khmer}
}
```
