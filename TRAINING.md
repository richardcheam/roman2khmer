# Training roman2khmer from scratch

This retrains the three parts of the released v0.1 system from the data in `training-data/v0.1/` and packs them
into a model folder that `roman2khmer.Converter` loads:

| Part | Script | What it learns | Time on a laptop GPU (GTX 1050 Ti, 4 GB) |
|---|---|---|---|
| Word model | `training/train_word_model.py` | Latin word → Khmer spellings (beam search gives 8 candidates) | about 1 hour (40 epochs) |
| Recheck model | `training/train_reverse_model.py` | how likely a Latin spelling is for a Khmer word | about 40 minutes |
| Word-pair model | `training/build_bigram.py` | Khmer word-pair counts from web text, for the sentence step | about 5 minutes (CPU) |

All settings are in `training/config.yaml`; they are the ones the released models were trained with. Everything
trains from scratch: no pretrained model, no paid service. A CPU works too, about 5 times slower.

## 1. Set up
Python 3.10 or newer.
```bash
git clone https://github.com/richardcheam/roman2khmer.git
cd roman2khmer
python -m venv .venv && source .venv/bin/activate
pip install -e .                                          # the roman2khmer package (NumPy only)
pip install -r training/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu126   # PyTorch with CUDA 12.6; use the CPU index without a GPU
python -m unittest discover -s training/tests              # quick checks of the training code (under a minute)
```

## 2. Train the word model
```bash
python training/train_word_model.py --seed 2 --device cuda --out runs/word_s2
```
Each epoch prints one line: training loss, side-task loss and the share of validation words written exactly right
(romanised chat spellings, names, English loanwords). The epoch with the best mean is kept and exported as
`runs/word_s2/translit.npz`. The released run (seed 2) kept epoch 28 with a mean of 0.402 (romanised 0.511, names
0.359, English 0.335); other seeds differ by up to about 2 points. If a run stops, add `--resume`.

On a GPU, two runs with the same seed are close but not identical, so a retrain gives a model of the same quality
with some different choices. A check run of this code reached 0.403 (released: 0.402).

How it trains, briefly: batches of 128 (Latin, Khmer) pairs, cross-entropy with label smoothing 0.1, AdamW with a
learning rate warming up over 500 steps and then decaying to 0 along a cosine. The author's own rows are shown 4
times per epoch. On every 8th step, a second decoder also learns the pronunciation of Latin-script dictionary entries
from the same encoder (loss weight 0.5); it helps the encoder and is thrown away afterwards.

## 3. Train the recheck model
```bash
python training/train_reverse_model.py --seed 0 --device cuda --out runs/reverse_s0
```
The same network, trained the other way round (Khmer → Latin), 31,265 steps. Every 782 steps it measures how
likely it finds the real spellings of 662 held-out words, and keeps the best point. Output: `runs/reverse_s0/reverse.npz`.

## 4. Build the word-pair model
Download one FineWeb-2 Khmer shard (about 2.3 GB) and Google's Khmer lexicon, then count:
```bash
pip install huggingface_hub
hf download HuggingFaceFW/fineweb-2 --repo-type dataset --include "data/khm_Khmr/train/000_00000.parquet" --local-dir data/fineweb2
curl -L -o data/km_lexicon.tsv \
  https://raw.githubusercontent.com/google/language-resources/bc72dd6527aee6963910b9afc69a57f7b9aacc1d/km/data/lexicon.tsv
python training/build_bigram.py --shard data/fineweb2/data/khm_Khmr/train/000_00000.parquet \
  --lexicon data/km_lexicon.tsv --out runs/khmer_bigram.npz
```
The released table's word list also held a few hundred words from training rows that were later removed, so a
rebuild differs slightly from it. To use the released table instead:
`hf download ric912/roman2khmer khmer_bigram.npz --local-dir runs/release`.

## 5. Pack and use
```bash
python training/build_model_folder.py --word runs/word_s2/translit.npz --reverse runs/reverse_s0/reverse.npz \
  --bigram runs/khmer_bigram.npz --out my_model
```
```python
from roman2khmer import Converter
c = Converter.from_dir("my_model")
c.convert("bong mok nham bay nov")       # 'បងមកញ៉ាំបាយនៅ'
```

## Changing things
- **Add your own word pairs:** append rows (`latin`, `khmer`, `source`) to a copy of `spell_train.tsv`, point `data`
  in `training/config.yaml` to the copy, and retrain. Keep the words you want to test on out of training.
- **Model size, epochs, learning rate, side task:** `training/config.yaml`.
- **Decoding weights** (word model, word pairs, joins, recheck): arguments of `Converter.from_dir`, e.g.
  `Converter.from_dir("my_model", gamma=0)` turns the recheck off.

How well the released system works, how it was evaluated and why it is built this way: `README.md` and
`report/roman2khmer-technical-report.pdf`.
