# Changelog

## 0.2.0
- `Converter.from_pretrained()`: downloads the released model files on first use (pinned commit of
  `ric912/roman2khmer`, sha256-checked), with the optional extra `roman2khmer[hub]`.
- `roman2khmer` command: `convert` (arguments or stdin, line by line), `download`, `serve`.
- `roman2khmer serve`: a small HTTP JSON API (`POST /convert`, `GET /health`) with the optional extra
  `roman2khmer[serve]` (FastAPI, uvicorn); one conversion at a time, bounded waiting, size limits, local by default.
- Dockerfile for the API; GitHub Actions for tests and releases (PyPI Trusted Publishing).
- The model files are unchanged (v0.1.0); conversions are identical to 0.1.0.

## 0.1.0
- First release: word model, recheck, sentence step; `Converter.from_dir`.
