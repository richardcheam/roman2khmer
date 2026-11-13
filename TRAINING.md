# Reproducing roman2khmer: data, training, evaluation

The whole pipeline, from raw inputs to the published numbers:

| Step | Script | What it does | Time on a laptop (GTX 1050 Ti, 4 GB) |
|---|---|---|---|
| 1. Prepare data | `pipeline/prepare.py` | raw inputs per source → training files, checked byte for byte against the release | 2 seconds |
| 2. Word model | `training/train_word_model.py` | Latin word → Khmer spellings (beam search gives 8 candidates) | about 1 hour |
| 3. Recheck model | `training/train_reverse_model.py` | how likely a Latin spelling is for a Khmer word | about 40 minutes |
| 4. Word-pair model | `training/build_bigram.py` | Khmer word-pair counts from web text, for the sentence step | about 10 minutes (or download it) |
| 5. Pack | `training/build_model_folder.py` | the three files into one model folder | seconds |
| 6. Evaluate | `pipeline/evaluate.py` | every published test set, compared with the released model | about 6 minutes (CPU) |

All settings are in `training/config.yaml` and are the ones used for the release. Everything trains from scratch on
one machine: no pretrained model, no paid service. A CPU works too, about 5 times slower for training.

## 0. Set up
Python 3.10 or newer.
```bash
git clone https://github.com/richardcheam/roman2khmer.git
cd roman2khmer
python -m venv .venv && source .venv/bin/activate
pip install -e '.[hub]' -r training/requirements.txt \
  --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
# CUDA 12.6 build of PyTorch; it also runs on the CPU. Without a GPU you may use .../whl/cpu instead.
python -m unittest discover -s tests && python -m unittest discover -s training/tests
```

## Everything in one command
```bash
pipeline/run_all.sh                    # on the GPU; DEVICE=cpu pipeline/run_all.sh on a CPU
BIGRAM=rebuild pipeline/run_all.sh     # also rebuild the word-pair model from FineWeb-2 (2.3 GB download)
```
It runs steps 1–6 into `runs/` and continues where it stopped if started again. The last step prints each measure
next to the released model's value.

## 1. Prepare the training data
```bash
python pipeline/prepare.py --out runs/data --check
```
Reads the raw inputs in `training-data/raw/` (what each is and where it comes from: its README) and the held-out
sets in `test-sets/`, and writes the training files: training pairs, side-task rows, validation sets, the recheck
model's pool and held-out words. Every pair that touches an evaluation or validation set is removed, so no test word
leaks into training. `--check` compares each file with `training-data/v0.1/`: all must say `identical`.

## 2. Train the word model
```bash
python training/train_word_model.py --seed 2 --device cuda --data runs/data --out runs/word_s2
```
Each epoch prints the training loss, the side-task loss and the share of validation words written exactly right
(romanised chat spellings, names, English loanwords). The epoch with the best mean is kept and exported as
`runs/word_s2/translit.npz`. The released run (seed 2) kept epoch 28 with a mean of 0.402; other seeds differ by up
to about 2 points. If a run stops, add `--resume`.

How it trains: batches of 128 (Latin, Khmer) pairs, cross-entropy with label smoothing 0.1, AdamW with a learning
rate warming up over 500 steps and then decaying to 0 along a cosine. The author's own rows are shown 4 times per
epoch. On every 8th step a second decoder also learns the pronunciation of Latin-script dictionary entries from the
same encoder (loss weight 0.5); it helps the encoder and is thrown away afterwards.

## 3. Train the recheck model
```bash
python training/train_reverse_model.py --seed 0 --device cuda --data runs/data --out runs/reverse_s0
```
The same network, trained the other way round (Khmer → Latin), 31,265 steps. Every 782 steps it measures how likely
it finds the real spellings of 662 held-out words, and keeps the best point. Output: `runs/reverse_s0/reverse.npz`.

## 4. The word-pair model
Download the released table, or rebuild it from one FineWeb-2 Khmer shard and Google's Khmer lexicon:
```bash
hf download ric912/roman2khmer khmer_bigram.npz --revision 67eacafe5bb9592622ebcdd6d64695eb3b6b69f6 --local-dir runs/release
# or
hf download HuggingFaceFW/fineweb-2 --repo-type dataset --include "data/khm_Khmr/train/000_00000.parquet" --local-dir runs/src
curl -L -o runs/src/km_lexicon.tsv \
  https://raw.githubusercontent.com/google/language-resources/bc72dd6527aee6963910b9afc69a57f7b9aacc1d/km/data/lexicon.tsv
python training/build_bigram.py --shard runs/src/data/khm_Khmr/train/000_00000.parquet \
  --lexicon runs/src/km_lexicon.tsv --out runs/khmer_bigram.npz
```
The released table's word list also held a few hundred words from training rows later removed, so a rebuild gives
the same probabilities for common word pairs but not an identical file.

## 5. Pack the model folder
```bash
python training/build_model_folder.py --word runs/word_s2/translit.npz --reverse runs/reverse_s0/reverse.npz \
  --bigram runs/khmer_bigram.npz --out runs/model
```

## 6. Evaluate
```bash
python pipeline/evaluate.py --model runs/model --out runs/eval.json
python pipeline/evaluate.py                       # the released model, for comparison
```
| Measure (released model) | Value | 95% interval |
|---|---|---|
| Messages B: character error rate (450 LLM-written messages) | 8.95% | 7.75–10.23 (resampling generation calls) |
| Messages B: messages written exactly | 36.9% | |
| Fresh word test: mean over 148 words of the share of spellings written exactly | 58.2% | 50.3–66.0 (resampling words) |
| Fresh word test: all 179 spellings | 60.3% | |
| Sound-alike variants (225) | 52.0% | |
| Messages A: character error rate, fit (279) / check (168) | 1.95% / 3.43% | |
| Public Chhunneng test, words with unseen Khmer forms (1,180) | 45.6% | |
| Public names test, names with unseen Khmer words (38) | 21.1% | |

**How to read them.** Evaluating the released model reproduces these values exactly (inference is deterministic).
A model you retrain lands near them, not on them: training on a GPU is not bit-for-bit repeatable, and retraining
the same recipe with three seeds moved the fresh word test with a standard deviation of 2.4–3.0 points
(ranges up to 5.8) and the message error rate by 0.6–1.0 points. The intervals cover the sampling
of test items for one fixed model, not retraining. The message sets and the fresh word test also guided choices
while the model was built (`test-sets/README.md`), so they are optimistic for this model; they are useful for
comparing a change against the released model on equal terms.

A check run of `pipeline/run_all.sh` on a GTX 1050 Ti (2026-11-13), from the raw inputs, gave: Messages B 8.42%
(released 8.95%), exact messages 36.7% (36.9%), fresh words 59.5% (58.2%), variants 52.4% (52.0%), Messages A
check 2.83% (3.43%), public romanised test 46.3% (45.6%), names 23.7% (21.1%): near the release, as expected.

## Changing things
- **Your own word pairs:** add rows to a file in `training-data/raw/` (or a new source in `pipeline/prepare.py`),
  run `prepare.py` without `--check`, and train on `runs/data`. Keep the words you want to test on out of training.
- **Model size, epochs, learning rate, side task:** `training/config.yaml`.
- **Decoding weights** (word model, word pairs, joins, recheck): arguments of `Converter.from_dir`, e.g.
  `Converter.from_dir("runs/model", gamma=0)` turns the recheck off; `evaluate.py --gamma 0` measures it.

Why the system is built this way: `report/roman2khmer-technical-report.pdf`.
