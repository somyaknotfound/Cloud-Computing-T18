# Presentation outline — 15 slides

Detailed presentation, ~3rd week of November 2026. 8 of the 25 marks are for
*understanding and presentation*, and the questions will be about **why**, not what.

Speaker assignments follow code ownership, so each person presents what they built and can
answer follow-ups on it. Every number below is printed by `python report/report_numbers.py`
(results) or `python calibration/fit_cost_model.py` (calibration).

---

### 1. Title  *(Somyak)*
Problem statement, group, mentor. One line: *"We measured what it costs to schedule LLM
requests seven different ways — and found that the cost of evicting a request matters more
than the policy."*

### 2. Why this is not the scheduling you were taught  *(Somyak)*
A CPU scheduler picks the next job. An LLM server runs **dozens of requests in every forward
pass**, so the question is *who is in the batch* — and, when the batch or GPU memory is full,
*who gets evicted, and what that costs*.

> Anticipated question: *"Is this just SJF vs FCFS again?"*
> Answer: the mechanism is the same, but the constraint is memory rather than CPU, job length
> is unknowable in advance (slide 12), and evicting a job is expensive in a way that decides
> which policies work at all (slide 11).

### 3. Prefill and decode  *(Ronak)*
Prefill = whole prompt in one pass. Decode = one token per pass, dominated by a fixed cost.
**Measured on a T4: decoding 32 sequences costs 1.11× decoding 1, not 32×** — that is why
continuous batching exists.

### 4. Continuous batching, the KV cache, and two ways to evict  *(Ronak)*
Diagram: the running batch, requests joining and leaving between iterations. Eviction is
either **recompute** (vLLM: discard the cache, re-prefill prompt + generated tokens later,
stalling the batch) or **swap** (FastServe: copy the cache to host memory and back). On the
T4, recompute costs **35–80×** a swap round trip.

### 5. Calibration: the GPU disagreed with us  *(Ronak)*
Colab T4, Qwen2.5-0.5B, 161 measurements. We planned straight lines accepted at R² ≥ 0.9.
Reality: prefill is **flat (~37 ms) then quadratic**; the line had R² 0.92 but a **55% median
error** and *negative* times for short prompts. Decode is so flat that R² was 0.02 and two
noisy points made the slope negative. New model: launch floor + linear + quadratic attention,
robust (Theil–Sen) fits, judged by relative error → **1.7% / 2.5% / 3.8%**. Swap measured at
13 GB/s, matching our paper estimate from KV size and PCIe 3.0.

**Present the R² trap deliberately:** a fit can score 0.92 and still be wrong by half.

### 6. A policy is two sort keys  *(Chirs)*
`order_key` and `victim_key`; everything else is shared code, so any difference is caused by
the policy alone. Show FCFS (one line) next to MLFQ-LAS.

### 7. The seven policies  *(Chirs)*
One table: preemptive or not, reads a predictor or not. FCFS and MLFQ-LAS need **no
prediction**; SJF-Oracle reads the truth and exists only as a bound.

### 8. Workload, metrics and statistics  *(Pranav)*
Lognormal lengths, Poisson arrivals, seeded. Why **normalized latency**. Why **max wait**
counts every stint in the queue (a preemptive policy admits at once and starves later). Every
comparison is a one-minute overload burst at 2.7× capacity; E7 repeats it on **ten seeds at
three burst lengths** with 95% CIs.

### 9. Capacity first  *(Pranav)*  — **figure 1**
3.70 req/s. Below it, all seven policies are identical. *This is the control.*

### 10. The classic trade-off  *(Somyak)*  — **figures 2 and 3**
SJF: **5.9× better** mean latency, **64×** better median TTFT, **1.6× worse** worst case. Over
ten traces: 0.19 ± 0.02 of FCFS's latency.

> Anticipated question: *"Is throughput the same for every policy?"*
> Answer: not exactly — and we say so. Non-preemptive policies differ by ~10% because SJF
> leaves the longest jobs for last and drains with a near-empty batch. What theory guarantees
> is that no ordering beats the engine's ceiling, and we assert that.

### 11. The main result: preemption cost decides  *(Somyak)*  — **figure 8**
Under recompute every preemptive policy loses throughput; predictor-free MLFQ is **2× worse
than FCFS** on every trace. Switch to swap: **SRTF draws level with SJF** (never >1% behind on
11 traces) with the fastest first tokens; both MLFQs keep an **FCFS-level worst case**. 10×
dearer swaps already flip it back — so FastServe's KV management is what makes its scheduler
viable.

### 12. How good must a length predictor be?  *(Chirs)*  — **figure 4**
Rank correlation 0.68 → **87%** of the oracle's benefit. **Ordering matters, accuracy mostly
does not.** MLFQ-LAS is a flat line: it never reads the prediction, and a test proves it.

### 13. The bug that taught us the most  *(Chirs)*  — **figure 5**
Our first MLFQ "needed no predictor" and "captured 90% of the oracle's benefit". Then we read
our own code: its skip-join used `predicted_len`, which was **exact** in that experiment.
Without it, MLFQ was *worse than FCFS* — a non-preemptive feedback queue cannot demote anything
already running. Fix: MLFQ-LAS became preemptive and predictor-free, MLFQ-Pred became its own
policy, and a test asserts bit-identical results with a perfect and a useless predictor.

Then the **ageing trap**, twice: a threshold below the actual queueing delay turns MLFQ into
FCFS. Calibration doubled the delays (~62 s), so the 30 s default now gives MLFQ-Pred 43% of the
oracle's benefit; **45–60 s gives 89–96% at a worst case within 9% of FCFS**. Rule: set it from
the observed delay. We did not re-tune the default after seeing results — say why.

### 14. Why should you believe any of this?  *(Pranav)*  — **figure 9**
(a) calibration on a real GPU; (b) theory gate — only claims that held on all 11 traces, run
on three; (c) invariants; (d) **byte-for-byte reproduction** with `--verify`; (e) a FastAPI
proxy running the same policy code against a *mock* backend — a consistency check, not
hardware validation. State the limitations before being asked.

### 15. What we would deploy  *(Somyak)*
- Below saturation: nothing matters; buy capacity instead.
- Recompute-only engine: **non-preemptive SJF with any reasonable predictor.** Avoid
  preemptive policies.
- Cheap preemption (KV swap): **SRTF** for first-token latency at SJF's mean; **MLFQ-Pred**
  with ageing set near the observed delay for near-oracle latency at an FCFS-level worst case;
  **MLFQ-LAS** if there is no predictor and first-token latency is what matters.
- Don't reach for priority classes first: plain SJF gave interactive traffic **57.5%** SLO
  attainment vs **6.9%** for Priority-SLO. Reduce queueing before reallocating it.
- Keep FCFS if you are judged on the worst case; it genuinely wins there.

---

## Demo (2 minutes, have it pre-run as a fallback)

```bash
pytest tests/test_theory.py -q                        # the theory gate passing
python calibration/fit_cost_model.py                  # the T4 fit and its errors, live
python -m src.runner e6 --jobs 8                      # preemption-cost sweep, ~5 s
python -m src.runner e0 --verify                      # "identical" -- reproducibility, live
```

If asked to prove the policy code runs outside the simulator:
`python serving/loadgen.py --policies fcfs sjf-pred --repeats 1` starts a real server and
replays the trace over HTTP (~4 minutes; start it before you need it).

## Questions to rehearse

| Question | Short answer |
|---|---|
| "Are your absolute numbers real?" | They come from 161 measurements on a Colab T4, fitted to within a few percent. But they describe Hugging Face eager mode, whose fixed per-step cost is far above vLLM's — so we lean on relative results, which every policy shares. |
| "You changed the cost model after seeing the data — isn't that cheating?" | The planned model failed its own test (55% error, negative times). The new one isn't looser, it's physically motivated — launch floor, linear work, quadratic attention — and cuts the error to 1.7%. The raw CSV is committed so anyone can refit it. We did *not* re-tune the load or the ageing threshold afterwards. |
| "Why not just use vLLM and measure it?" | Seven policies × two preemption mechanisms × ten load levels × ten seeds is 800+ runs on identical traces; calibrate-then-simulate is the standard way to make that affordable and reproducible. |
| "Isn't your simulator just confirming your assumptions?" | It contradicted three of ours: that MLFQ needs no predictor, that throughput is policy-independent, and that the cost laws are straight lines. All are in the report's appendix. |
| "Is SJF-Oracle cheating?" | Yes, deliberately. It is the bound that makes slide 12 interpretable. |
| "Why does MLFQ-LAS need no predictor?" | It infers length from service received: everything starts at the top and is demoted as it generates tokens. That only works if demoted running requests can be evicted — so it is preemptive, and its usefulness depends on preemption being cheap. |
| "Why does SRTF lose to SJF under recompute?" | Every eviction throws away the victim's cache and later re-prefills it while the whole batch stalls: ~300 evictions cost 10% of throughput. With swaps the same policy draws level with SJF on every trace. |
| "Why 10 req/s?" | It was fixed before calibration as a clearly overloaded point; on the measured engine it is 2.7× capacity. Below capacity all seven policies are identical (slide 9), and E1 shows every load from 2 to 16. |
| "How do you know the results aren't one lucky trace?" | E7: ten fresh seeds, three burst lengths, 95% CIs; the theory gate only admits claims that held on all eleven traces — one SRTF claim was demoted because it held on nine. |
