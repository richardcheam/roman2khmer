"""Download the released model files from Hugging Face, pinned to one commit and checked by sha256.

Needs the optional extra `roman2khmer[hub]` (huggingface_hub). Nothing is downloaded at import or install time; the
files are fetched on the first `download()` call and then read from the local cache.

    >>> from roman2khmer import hub                       # doctest: +SKIP
    >>> folder = hub.download()                           # doctest: +SKIP

The model files are licensed CC BY-SA 4.0 (attribution and notices: `NOTICE.md` in the model repository), separately
from this package's MIT code.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO = "ric912/roman2khmer"
REVISION = "67eacafe5bb9592622ebcdd6d64695eb3b6b69f6"     # model files of release v0.1.0
FILES = {                                                 # file -> sha256
    "translit.npz": "f107a711d02e597dfe8c81f3cf214b97f2c5395348736b1fb0169c16c54e13ac",
    "reverse.npz": "70bffa7c4b2c88644ed62eb01b71f0d3f47b8e91709c46c3c0c2e994bfd9b514",
    "khmer_bigram.npz": "8ecf489cc1e356d148f75f0d9b7a3fb018aee99ac89405eb6dd16c7408dfcd46",
}
NOTICES = ("NOTICE.md", "LICENSE")                        # kept next to the model files


class IntegrityError(RuntimeError):
    """A downloaded model file does not have the expected sha256."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(folder: str | Path, files: dict[str, str] | None = None) -> Path:
    """Check that `folder` holds every model file with its expected sha256.

    Raises:
        FileNotFoundError: A file is missing.
        IntegrityError: A file's sha256 differs.
    """
    folder = Path(folder)
    for name, want in (files or FILES).items():
        p = folder / name
        if not p.exists():
            raise FileNotFoundError(f"model file missing: {p}")
        got = sha256_file(p)
        if got != want:
            raise IntegrityError(f"{p}: sha256 {got} differs from the expected {want}")
    return folder


def download(cache_dir: str | Path | None = None, local_files_only: bool = False, repo: str = REPO,
             revision: str = REVISION) -> Path:
    """The local folder of the pinned model files, downloading them on first use.

    Args:
        cache_dir: Hugging Face cache folder (default: the user's standard cache).
        local_files_only: Use only files already in the cache (no network).
        repo: Model repository. Change it only together with `revision` and the expected hashes.
        revision: Full commit hash of the model repository.

    Returns:
        The folder holding `translit.npz`, `reverse.npz` and `khmer_bigram.npz`, verified.

    Raises:
        ImportError: huggingface_hub is not installed (`pip install "roman2khmer[hub]"`).
        IntegrityError: A downloaded file has an unexpected sha256.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError('downloading the model needs huggingface_hub: pip install "roman2khmer[hub]" '
                          "(or download the files yourself and use Converter.from_dir)") from e
    folder = snapshot_download(repo_id=repo, revision=revision, cache_dir=cache_dir,
                               local_files_only=local_files_only, allow_patterns=[*FILES, *NOTICES])
    return verify(folder) if (repo, revision) == (REPO, REVISION) else Path(folder)
