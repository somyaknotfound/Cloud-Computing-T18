"""A real scheduling proxy, running the same policies as the simulator.

Owned by: Somyak Mohanta (241CS257)

    uvicorn serving.proxy:app --port 8000
    POLICY=mlfq uvicorn serving.proxy:app --port 8000

Why this exists: the simulator is a closed world, and a scheduling policy that only ever
works inside its own simulator has not been shown to work at all. This is an actual HTTP
server -- concurrent connections, real clocks, real async scheduling -- and it imports
`src/policies.py` **unmodified**. If a policy behaves the same way here as it does in the
simulator, the simulator's version of it was not an artefact of the simulator.

The design is the smallest thing that is still a real scheduler:

    POST /generate  ->  request joins `waiting`, then blocks on its own asyncio.Event
    dispatcher      ->  whenever capacity frees, sorts `waiting` with the policy's
                        order_key and wakes the winner
    handler         ->  streams tokens from the backend, then releases its slot

Preemption is not implemented, so only the non-preemptive policies (FCFS, SJF-Oracle,
SJF-Pred, Priority-SLO) can be served; the proxy refuses the others at startup.

`max_batch_size` is what creates the queue. Without a concurrency cap every request would
start instantly, nothing would ever wait, and the policy would never get to make a decision
-- the same reason the simulator only shows differences above saturation.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI                          # noqa: E402
from fastapi.responses import StreamingResponse      # noqa: E402
from pydantic import BaseModel                       # noqa: E402

from serving.backend import build_backend            # noqa: E402
from src.policies import SchedulerState, build       # noqa: E402
from src.workload import Request, load_workload_config  # noqa: E402

MAX_BATCH = int(os.environ.get("MAX_BATCH", "8"))
POLICY_NAME = os.environ.get("POLICY", "fcfs")
BACKEND_KIND = os.environ.get("BACKEND", "mock")


class GenerateRequest(BaseModel):
    rid: int
    prompt_tokens: int
    max_tokens: int
    sla_class: str = "interactive"
    predicted_len: int | None = None


@dataclass
class Pending:
    """A request waiting for a slot. `admitted` is what the dispatcher fires."""

    req: Request
    admitted: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class Record:
    rid: int
    sla_class: str
    prompt_len: int
    output_len: int
    arrival: float
    start: float
    first_token: float
    finish: float


class Scheduler:
    def __init__(self, policy_name: str, backend, max_batch: int):
        self.policy = build(policy_name)
        if self.policy.preemptive:
            # This proxy only orders admissions; it never evicts a running stream. Serving
            # a preemptive policy here would silently test a different policy.
            raise SystemExit(f"{self.policy.name} is preemptive and the proxy does not "
                             f"implement preemption; validate it in the simulator instead")
        self.backend = backend
        self.max_batch = max_batch
        self.slo = load_workload_config()["slo"]

        self.waiting: list[Pending] = []
        self.running = 0
        self.records: list[Record] = []
        self.t0 = time.perf_counter()
        self._wake = asyncio.Condition()

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def _state(self) -> SchedulerState:
        return SchedulerState(now=self.now(), kv_blocks_free=0, kv_blocks_total=0,
                              running_count=self.running, slo=self.slo)

    async def enqueue(self, pending: Pending) -> None:
        async with self._wake:
            self.waiting.append(pending)
            self.policy.on_arrival(pending.req, self._state())
            self._wake.notify_all()

    async def release(self) -> None:
        async with self._wake:
            self.running -= 1
            self._wake.notify_all()

    async def dispatch_loop(self) -> None:
        """Wakes on every arrival and every completion, and admits whoever the policy picks."""
        while True:
            async with self._wake:
                while not self.waiting or self.running >= self.max_batch:
                    await self._wake.wait()

                state = self._state()
                ordered = self.policy.order_waiting([p.req for p in self.waiting], state)
                by_rid = {p.req.rid: p for p in self.waiting}

                while self.waiting and self.running < self.max_batch and ordered:
                    chosen = by_rid[ordered.pop(0).rid]
                    self.waiting.remove(chosen)
                    self.running += 1
                    chosen.req.start_time = self.now()
                    chosen.admitted.set()


app = FastAPI(title="LLM scheduling proxy")
scheduler: Scheduler | None = None


@app.on_event("startup")
async def _startup():
    global scheduler
    scheduler = Scheduler(POLICY_NAME, build_backend(BACKEND_KIND), MAX_BATCH)
    asyncio.create_task(scheduler.dispatch_loop())
    print(f"proxy up: policy={scheduler.policy.name} backend={BACKEND_KIND} "
          f"max_batch={MAX_BATCH}")


@app.get("/health")
async def health():
    return {"status": "ok", "policy": scheduler.policy.name,
            "backend": scheduler.backend.name, "max_batch": scheduler.max_batch,
            "running": scheduler.running, "waiting": len(scheduler.waiting),
            "completed": len(scheduler.records)}


@app.get("/metrics")
async def metrics():
    """Per-request timings, for loadgen.py to pull at the end of a run."""
    return {"policy": scheduler.policy.name, "backend": scheduler.backend.name,
            "max_batch": scheduler.max_batch,
            "records": [r.__dict__ for r in scheduler.records]}


@app.post("/reset")
async def reset():
    scheduler.records.clear()
    scheduler.t0 = time.perf_counter()
    return {"status": "reset"}


@app.post("/generate")
async def generate(body: GenerateRequest):
    req = Request(
        rid=body.rid,
        arrival=scheduler.now(),
        prompt_len=body.prompt_tokens,
        output_len=body.max_tokens,
        sla_class=body.sla_class,
        predicted_len=body.predicted_len or body.max_tokens,
    )
    pending = Pending(req=req)
    await scheduler.enqueue(pending)

    async def stream():
        await pending.admitted.wait()
        first_token = None
        count = 0
        try:
            async for tok in scheduler.backend.generate(req.prompt_len, req.output_len):
                if first_token is None:
                    first_token = scheduler.now()
                count += 1
                yield tok
        finally:
            finish = scheduler.now()
            scheduler.records.append(Record(
                rid=req.rid, sla_class=req.sla_class, prompt_len=req.prompt_len,
                output_len=count, arrival=req.arrival, start=req.start_time,
                first_token=first_token if first_token is not None else finish,
                finish=finish,
            ))
            await scheduler.release()

    return StreamingResponse(stream(), media_type="text/plain")
