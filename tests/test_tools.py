"""Download pinning, the command line and the HTTP API, without the network.

The server tests need the [serve] extra (skipped otherwise); tests that convert real text need
ROMAN2KHMER_MODEL_DIR=path/to/model.
"""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from roman2khmer import cli, hub

DIR = os.environ.get("ROMAN2KHMER_MODEL_DIR")
try:
    import fastapi  # noqa: F401
    from fastapi.testclient import TestClient
    HAVE_SERVE = True
except ImportError:
    HAVE_SERVE = False


class Pinning(unittest.TestCase):
    def test_pinned_to_a_full_commit_and_three_files(self):
        self.assertEqual(len(hub.REVISION), 40)
        self.assertEqual(sorted(hub.FILES), ["khmer_bigram.npz", "reverse.npz", "translit.npz"])

    def test_verify_rejects_a_changed_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.npz"
            p.write_bytes(b"abc")
            hub.verify(d, {"a.npz": hub.sha256_file(p)})
            p.write_bytes(b"abd")
            with self.assertRaises(hub.IntegrityError):
                hub.verify(d, {"a.npz": "0" * 64})
            with self.assertRaises(FileNotFoundError):
                hub.verify(d, {"missing.npz": "0" * 64})


class FakeConverter:
    def convert(self, text: str) -> str:
        return text.upper()


@unittest.skipUnless(HAVE_SERVE, "needs the [serve] extra")
class Server(unittest.TestCase):
    def setUp(self):
        from roman2khmer.server import create_app
        self.c = TestClient(create_app(FakeConverter(), max_chars=10, max_body=100))

    def test_convert_and_health(self):
        self.assertEqual(self.c.post("/convert", json={"text": "bong"}).json(), {"khmer": "BONG"})
        self.assertEqual(self.c.get("/health").json()["status"], "ok")

    def test_limits_and_errors(self):
        self.assertEqual(self.c.post("/convert", json={"text": "x" * 11}).status_code, 422)
        self.assertEqual(self.c.post("/convert", content=b"x" * 101).status_code, 413)
        self.assertEqual(self.c.post("/convert", content=b"{not json").status_code, 400)
        self.assertEqual(self.c.post("/convert", json={"text": 3}).status_code, 422)
        self.assertEqual(self.c.post("/convert", json=["bong"]).status_code, 422)

    def test_no_cors(self):
        r = self.c.post("/convert", json={"text": "a"}, headers={"Origin": "https://example.com"})
        self.assertNotIn("access-control-allow-origin", r.headers)


class CommandLine(unittest.TestCase):
    def test_missing_model_folder_is_a_clean_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cli.main(["convert", "--model", "/nonexistent", "bong"])
        self.assertEqual(code, 2)
        self.assertIn("error", err.getvalue())

    @unittest.skipUnless(DIR, "set ROMAN2KHMER_MODEL_DIR")
    def test_convert_argument(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["convert", "--model", DIR, "bong", "mok"])
        self.assertEqual((code, out.getvalue().strip()), (0, "បងមក"))


if __name__ == "__main__":
    unittest.main()
