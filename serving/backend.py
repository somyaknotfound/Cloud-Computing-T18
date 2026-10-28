"""Backends the scheduling proxy can dispatch to.

Owned by: Somyak Mohanta (241CS257)

Two implementations behind one interface:

`MockBackend` needs no GPU and no model. It sleeps for exactly as long as the calibrated
cost model says the work would take, including the batching effect: it knows how many
requests are decoding right now, so each token costs `decode_time(batch_size)`. Adding a
second request therefore slows every request down a little and raises total throughput a
lot -- the same behaviour the simulator models, happening in real wall-clock time.

`OllamaBackend` talks to a real Ollama server. Same interface, real tokens. Nobody needs it
to run the project; it exists so the results can be checked against an actual model when a
machine with Ollama is available.

Both stream token by token, because the metric that matters most (TTFT) is only observable
if you can see when the *first* token arrives.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cost_model import CostModel  # noqa: E402


class MockBackend:
    """Simulated GPU: sleeps according to the calibrated cost model, no model required."""

    name = "mock"

    def __init__(self, cost_model: CostModel | None = None):
        self.cm = cost_model or CostModel.load()
        self.decoding = 0          # sequences currently in the decode batch

    async def generate(self, prompt_tokens: int, max_tokens: int) -> AsyncIterator[str]:
        await asyncio.sleep(self.cm.prefill_time([(0, prompt_tokens)]))
        self.decoding += 1
        try:
            for i in range(max_tokens):
                if i > 0:
                    # Cost per token depends on how many sequences share the pass, so a
                    # busier server makes every individual stream slower while the total
                    # token rate climbs. That is continuous batching, in real time.
                    await asyncio.sleep(self.cm.decode_time(self.decoding))
                yield f"t{i} "
        finally:
            self.decoding -= 1


class OllamaBackend:
    """Real generation through a local Ollama server (http://localhost:11434)."""

    name = "ollama"

    def __init__(self, model: str = "qwen2.5:0.5b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    async def generate(self, prompt_tokens: int, max_tokens: int) -> AsyncIterator[str]:
        import json

        import httpx

        # The workload is synthetic, so there is no real prompt text -- we ask for a
        # prompt of roughly the right token count so the prefill cost is representative.
        prompt = "word " * max(1, prompt_tokens)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream("POST", f"{self.host}/api/generate", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("response"):
                        yield chunk["response"]
                    if chunk.get("done"):
                        return


def build_backend(kind: str = "mock", **kwargs):
    if kind == "mock":
        return MockBackend(**kwargs)
    if kind == "ollama":
        return OllamaBackend(**kwargs)
    raise KeyError(f"unknown backend {kind!r}; use 'mock' or 'ollama'")
