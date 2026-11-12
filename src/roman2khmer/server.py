"""A small HTTP JSON API around one `Converter` (needs `pip install "roman2khmer[serve]"`).

    POST /convert   {"text": "bong mok nham bay nov"}  ->  {"khmer": "បងមកញ៉ាំបាយនៅ"}
    GET  /health    ->  {"status": "ok", "version": "..."}

Design, for a model that takes about a second per message on a CPU:
- **One conversion at a time.** The converter is loaded once and shared; conversions are serialised, so its word
  cache is never changed by two threads at once and memory stays at one copy of the model.
- **Bounded waiting.** At most `max_waiting` requests wait; more get HTTP 503 (try later) instead of piling up.
- **Limits before work.** Bodies larger than `max_body` bytes are refused (413) before JSON parsing; messages longer
  than `max_chars` characters are refused (422).
- **Timeouts.** A request that has not finished after `timeout_s` seconds gets HTTP 504. The conversion itself
  cannot be interrupted and finishes in the background before the next one starts.
- **No CORS, local by default.** Browsers on other sites cannot call it unless you add CORS yourself; the command
  line binds to 127.0.0.1. Put authentication, TLS and rate limits in a proxy before exposing it to other people.
"""
from __future__ import annotations

import asyncio
import json
import threading

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool


def create_app(converter, max_chars: int = 1000, max_body: int = 16_384, max_waiting: int = 8,
               timeout_s: float = 30.0) -> FastAPI:
    """The FastAPI app for one loaded `Converter` (see the module docstring for the limits)."""
    from roman2khmer import __version__
    app = FastAPI(title="roman2khmer", version=__version__, docs_url=None, redoc_url=None)
    lock = threading.Lock()
    state = {"in_flight": 0}

    def run(text: str) -> str:
        with lock:
            return converter.convert(text)

    def error(code: int, msg: str) -> JSONResponse:
        return JSONResponse({"error": msg}, status_code=code)

    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__}

    @app.post("/convert")
    async def convert(request: Request):
        size = request.headers.get("content-length")
        if size is not None and (not size.isdigit() or int(size) > max_body):
            return error(413, f"body larger than {max_body} bytes")
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > max_body:
                return error(413, f"body larger than {max_body} bytes")
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return error(400, 'send JSON like {"text": "bong mok"}')
        text = data.get("text") if isinstance(data, dict) else None
        if not isinstance(text, str):
            return error(422, '"text" must be a string')
        if len(text) > max_chars:
            return error(422, f'"text" longer than {max_chars} characters')
        if state["in_flight"] > max_waiting:               # one running + max_waiting waiting
            return JSONResponse({"error": "busy, try again shortly"}, status_code=503, headers={"Retry-After": "2"})
        state["in_flight"] += 1
        try:
            khmer = await asyncio.wait_for(run_in_threadpool(run, text), timeout=timeout_s)
        except asyncio.TimeoutError:
            return error(504, f"not finished after {timeout_s} s")
        finally:
            state["in_flight"] -= 1
        return {"khmer": khmer}

    return app
