#!/usr/bin/env python3
"""Pack the three trained parts into a model folder that `roman2khmer.Converter.from_dir` loads.

    python training/build_model_folder.py --word runs/word_s2/translit.npz --reverse runs/reverse_s0/reverse.npz \
        --bigram runs/khmer_bigram.npz --out my_model

Writes `translit.npz`, `reverse.npz`, `khmer_bigram.npz` and `manifest.json` (each file's sha256 and the decoding
settings from `config.yaml`), then converts one example message as a check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import yaml
from roman2khmer import Converter

CONFIG = Path(__file__).resolve().parent / "config.yaml"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--word", type=Path, required=True, help="translit.npz from train_word_model.py")
    ap.add_argument("--reverse", type=Path, required=True, help="reverse.npz from train_reverse_model.py")
    ap.add_argument("--bigram", type=Path, required=True, help="khmer_bigram.npz from build_bigram.py or the release")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for src, name in ((a.word, "translit.npz"), (a.reverse, "reverse.npz"), (a.bigram, "khmer_bigram.npz")):
        shutil.copyfile(src, a.out / name)
    files = {f: hashlib.sha256((a.out / f).read_bytes()).hexdigest()
             for f in ("translit.npz", "reverse.npz", "khmer_bigram.npz")}
    settings = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["decoding"]
    (a.out / "manifest.json").write_text(json.dumps({"files": files, "settings": settings}, indent=1) + "\n",
                                         encoding="utf-8")
    example = "bong mok nham bay nov"
    print(json.dumps({"folder": str(a.out), "files": files,
                      "example": {example: Converter.from_dir(a.out).convert(example)}}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
