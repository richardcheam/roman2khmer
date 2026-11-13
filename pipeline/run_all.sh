#!/usr/bin/env bash
# The whole pipeline: prepare data -> train both models -> word-pair model -> pack -> evaluate.
#   pipeline/run_all.sh                 # GPU (cuda); about 2 hours on a 4 GB laptop GPU
#   DEVICE=cpu pipeline/run_all.sh      # CPU; several hours
#   BIGRAM=rebuild pipeline/run_all.sh  # also rebuild the word-pair model from FineWeb-2 (downloads about 2.3 GB)
# Every step writes into runs/; a stopped run continues where it stopped when started again.
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE=${DEVICE:-cuda}
BIGRAM=${BIGRAM:-release}
mkdir -p runs

echo "== 1. prepare the training data (and check it equals training-data/v0.1)"
python pipeline/prepare.py --out runs/data --check

echo "== 2. word model (seed 2, as released)"
[ -f runs/word_s2/translit.npz ] || \
  python training/train_word_model.py --seed 2 --device "$DEVICE" --data runs/data --out runs/word_s2 --resume

echo "== 3. recheck model (seed 0, as released)"
[ -f runs/reverse_s0/reverse.npz ] || \
  python training/train_reverse_model.py --seed 0 --device "$DEVICE" --data runs/data --out runs/reverse_s0 --resume

echo "== 4. word-pair model ($BIGRAM)"
if [ "$BIGRAM" = rebuild ] || [ ! -f runs/khmer_bigram.npz ]; then
  if [ "$BIGRAM" = rebuild ]; then
    mkdir -p runs/src
    hf download HuggingFaceFW/fineweb-2 --repo-type dataset --include "data/khm_Khmr/train/000_00000.parquet" \
      --local-dir runs/src/fineweb2
    curl -L -o runs/src/km_lexicon.tsv \
      https://raw.githubusercontent.com/google/language-resources/bc72dd6527aee6963910b9afc69a57f7b9aacc1d/km/data/lexicon.tsv
    python training/build_bigram.py --shard runs/src/fineweb2/data/khm_Khmr/train/000_00000.parquet \
      --lexicon runs/src/km_lexicon.tsv --out runs/khmer_bigram.npz
  else
    hf download ric912/roman2khmer khmer_bigram.npz --revision 67eacafe5bb9592622ebcdd6d64695eb3b6b69f6 \
      --local-dir runs/release
    cp runs/release/khmer_bigram.npz runs/khmer_bigram.npz
  fi
fi

echo "== 5. pack the model folder"
python training/build_model_folder.py --word runs/word_s2/translit.npz --reverse runs/reverse_s0/reverse.npz \
  --bigram runs/khmer_bigram.npz --out runs/model

echo "== 6. evaluate, and compare with the released model"
python pipeline/evaluate.py --model runs/model --train-data runs/data/spell_train.tsv --out runs/eval.json
