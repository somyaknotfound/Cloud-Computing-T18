# BUILD.md — specification and implementation notes

The spec this project was built against, plus a running record of every deviation from it and
why. Read `README.md` first for what the project *is*; this file is for whoever has to change
it.

---

## 1. Goal

Answer, with evidence: **which scheduling policy should an LLM inference server use, what
does it cost, and how would we know?**

Hard constraints that shaped every decision:

1. **Must run with no GPU, no model, no Ollama.** Ollama is not installed on the primary dev
   machine and already blocked a previous project. Any design where the benchmark cannot run
   without it is wrong.
2. **Four teammates must be able to defend their own module in a live presentation.** Every
   module stays under ~250 lines and does one thing.
3. **Every claim must be checkable.** Results come from committed CSVs; figures regenerate
   from those CSVs; properties that must hold are asserted in tests, not eyeballed.

## 2. Architecture

```
calibration/  colab notebook -> CSV -> fit_cost_model.py -> cost_model.json
                                                                  |
config/*.yaml ------------------------------------------------+   |
                                                              v   v
src/workload.py --(requests)--> src/engine.py <--(policy)-- src/policies.py
                                     |
                                 src/metrics.py --> src/runner.py --> results/*.csv
                                                                          |
                                                        src/plots.py <----+--> report/
serving/proxy.py  imports src/policies.py UNMODIFIED  <-- the consistency check
```

The one-way dependency that matters: **`policies.py` imports nothing from `engine.py`.** A
policy is pure decision logic, which is what allows the same file to drive both the simulator
and the FastAPI proxy.

## 3. Component specifications

### 3.1 `src/cost_model.py` — seven constants

```
t_prefill(segments) = max(floor, sum over sequences of b*k + q*((s+k)^2 - s^2))
t_decode(B)         = c + d*B
t_hybrid(seg, B)    = t_decode(B) + the same prefill sum   # chunked prefill, floor paid once
t_swap(k)           = s0 + e*k                              # one way, swap preemption only
```

A prefill segment is (s, k): a sequence's tokens already prefilled and the tokens in this pass.
Attention is quadratic per sequence, so a token total is not enough; chunks add up exactly to
the whole prompt. The only link between the simulator and physical hardware. Loaded from
`calibration/cost_model.json`, fitted from `calibration/calibration_t4.csv` (measured on a
Colab T4, 2026-10-03).

### 3.2 `src/workload.py` — traces

Poisson arrivals, lognormal prompt/output lengths, seeded. The invariant that matters:
**`output_len` is ground truth and policies must not read it.** They see `predicted_len`,
which carries controllable error. Only `SJFOracle` reads the truth, deliberately.

### 3.3 `src/policies.py` — two sort keys

```python
order_key(req, state)    # waiting requests admitted in increasing order
victim_key(req, state)   # running request with the LARGEST value is evicted first
preemptive = False       # may evict a strictly worse-ranked running request to admit a
                         # better waiting one -- for a batch slot, KV pages, or both
```

Optional hooks `on_arrival` and `on_iteration_end` for policies with per-request bookkeeping
(only the MLFQs use them). Preemptive: SRTF, MLFQ-LAS, MLFQ-Pred.

### 3.4 `src/engine.py` — the simulator

One iteration:
1. Admit in policy order, subject to `max_batch_size` and free KV pages; a preemptive
   policy may evict to make room.
2. Form the batch: prefill-only (vLLM default, stalls decodes) or fused prefill+decode
   (chunked, Sarathi-style).
3. Charge `dt` from the cost model, advance the clock.
4. Emit tokens; a completing prefill emits the first token.
5. Grow KV; evict via `victim_key` if the cache is full.
6. Retire finished requests and free their pages.

Eviction follows `preemption_mode`: `recompute` (discard KV, re-prefill prompt + generated on
return) or `swap` (copy KV to host and back, progress kept, both copies charged to the next
pass).

### 3.5 `src/metrics.py` — TTFT, TPOT, E2E, normalized latency, goodput, Jain, max wait

`max_wait` is the longest *total* time in the queue, across every preemption — the starvation
witness. `max_queue_delay` (first admission only) is kept but no longer used for that.

### 3.6 `serving/` — the consistency-check path

FastAPI proxy with a real queue and a concurrency cap, plus a mock backend that sleeps
according to the cost model. `loadgen.py` starts one proxy per policy, replays the trace over
HTTP three times, and compares against the simulator. The proxy orders admissions but never
evicts, so it refuses preemptive policies; only the four non-preemptive ones are checked.

### 3.7 `src/runner.py` — experiments E0–E7

`--jobs N` runs points in parallel; results are collected in task order, so the CSVs are
identical to a serial run. `--verify` re-runs and compares every CSV byte for byte. CSVs are
written with `\n` line endings and `.gitattributes` pins them to LF, so the bytes match on
every OS.

## 4. Implementation notes — deviations from the spec, and why

Recorded as they happened. These are the things a reader would otherwise have to rediscover.

**4.1 KV accounting was wrong in the first draft.** Blocks were reserved for the full prompt
at admission, but freed based on `prefilled + generated`, which is smaller during a partial
prefill — so the cache slowly leaked. Fixed by making `Request.blocks` the single source of
truth for what a request holds, with `kv_tokens() = prompt_len + generated` as the single
definition of what it needs. `test_kv_accounting_balances` now asserts the cache is empty
once the queue drains.

**4.2 `prefill_done` needed an explicit target.** Defining it as `prefilled >= prompt_len`
breaks after a preemption: a request that lost its cache after generating G tokens must
re-prefill `prompt_len + G`, not `prompt_len`. Added `prefill_target`, set by the engine on
every admission.

**4.3 Preemption is decided before it is executed.** The first version evicted victims one at
a time and rolled back if it could not free enough — but rollback had to un-do
`reset_for_recompute()`, which is not cleanly reversible. Now `_preempt_for` computes the
victim set and only commits if it will succeed.

**4.4 MLFQ's defaults were silently degenerate.** With 4 levels of base quantum 32, nearly
every request landed in the same bucket; and with `starvation_s=10` against real queueing
delays of ~30 s, *every* queued request was promoted to level 0. The result was a policy
numerically indistinguishable from FCFS. Changed to 8 levels, base quantum 16, ageing at 30 s
— and added experiment E4 to sweep it, because the failure was invisible without the sweep.

**4.5 TPOT is NaN, not 0, for single-token answers.** A one-token answer has no gap between
tokens. Recording 0.0 would drag every percentile down and flatter every policy. Such rows
are excluded from TPOT statistics and pass the TPOT half of the SLO check by default.

**4.6 E3 originally stopped at sigma=2.0** and concluded "prediction always helps", which was
an artefact of not degrading the predictor far enough. Extended to sigma=8.0, and the x-axis
switched from our internal sigma to **Spearman rank correlation**, which is what could
actually be measured for a real predictor.

**4.7 E5 originally measured nothing.** Chunked prefill was tested at chunk size 512 on a
workload whose median prompt is 244 tokens — so no prompt was ever chunked. Moved to the
bimodal (long-prompt) workload and swept chunk size instead of toggling on/off.

**4.8 The cost model was PROVISIONAL until 2026-10-03.** Until the Colab notebook had been run,
`cost_model.json` held literature-derived placeholders and `runner.py` printed a warning on
every invocation. The warning is still there, keyed on `source` starting with `PROVISIONAL`;
it went quiet when the measured file replaced the placeholder (4.17).

**4.9 Absolute sim-vs-proxy latencies do not match, by design.** The mock backend has no KV
cache and does not stall decodes behind prefills. `loadgen.py` therefore compares *rankings*
and FCFS-relative ratios, not absolute values, and says so in its docstring.

**4.10 MLFQ-LAS was reading the oracle (found in the 2026-10-02 audit).** Its skip-join
placed arrivals by `predicted_len`, which is the true length at sigma=0 — the setting of the
headline experiment. The "predictor-free" policy that "captured 90% of the oracle's benefit"
was quietly an oracle; with no length information it scored NL 0.300 against FCFS's 0.288.
Fixed by splitting it: MLFQ-LAS starts every request at level 0, and the skip-join variant is
a separate policy, MLFQ-Pred. `test_mlfq_ignores_the_predictor` now asserts MLFQ-LAS's output
is bit-identical at sigma=0 and sigma=8.

**4.11 A non-preemptive MLFQ is FCFS.** Every waiting request has received no service, so
all of them sit at level 0 and admission order is arrival order. Demotion only matters if a
demoted *running* request can be evicted for a fresh one, so both MLFQs are preemptive.

**4.12 "Preemptive" SRTF never preempted for a slot.** `_admit` stopped as soon as the batch
was full, so preemptive policies could only evict under KV pressure. Fixed: `_preempt_for`
frees a batch slot and/or KV pages. This is what made SRTF and the MLFQs genuinely
preemptive — and exposed that recompute preemption makes them thrash (E6).

**4.13 `preemption_mode` was a dead config key.** It was loaded and ignored. Swap is now
implemented and every headline experiment runs under both modes. Swap is not always cheaper:
for a 70-token victim, re-prefilling costs about one decode step *and* yields the next token,
so recompute can win (`test_swap_is_cheaper_than_recompute_for_a_large_victim`).

**4.14 The starvation witness ignored preemption.** `max_queue_delay` only measured the wait
for *first* admission; a preemptive policy admits at once and starves later. Added
`wait_time` (accumulated across every stint in the queue) and `max_wait`.

**4.15 Two theory-gate claims held for trace 42 only.** Over ten seeds, "all policies reach the
same throughput (within 5%)" failed on 4 of 11 traces even for non-preemptive policies —
under continuous batching the token rate depends on batch occupancy, and SJF drains with a
near-empty batch. "MLFQ beats FCFS on the mean" failed on 2 of 10. Both were removed from the
gate (with the reason in its docstring), and every remaining robust claim is checked on three
seeds. This is also why E7 exists.

**4.16 The calibration notebook measures swap too.** A cell times copying a real KV cache to
pinned host memory and back. Before it was run, we derived 1.0 us/token from the model's KV
size (12,288 bytes/token) and PCIe 3.0 bandwidth; the T4 measured 0.94 us/token plus 0.26 ms
per copy, i.e. 13 GB/s. E6 sweeps it from 0.1x to 100x. The notebook can also clone the repo
and commit its results back itself, given a `GITHUB_TOKEN` Colab secret (README).

**4.17 The real calibration broke the planned cost model.** We had planned straight lines for
prefill and decode, accepted at R^2 >= 0.9. On the T4:
- Prefill is flat (~37 ms, launch-bound) up to ~400 tokens, then superlinear (attention). The
  line scored R^2 = 0.92 yet had a 55% median error and returned *negative* times below 63
  tokens, which would have run the simulator's clock backwards.
- Decode is nearly flat (32 sequences cost 1.11x one), so R^2 is meaningless for it; and two
  slow settings (batch 4 and 8) made the least-squares slope negative. R^2 = 0.02.

Fix: prefill = max(floor, b*n + q*n^2) fitted as a segmented model, decode and swap as
Theil-Sen lines, each judged by median relative error <= 10% (1.7%, 2.5%, 3.8%). Changing a
model's form after seeing data needs saying out loud: the linear model was the prior and failed
its own criterion; the replacement is physically motivated, not merely looser, and the raw CSV
is committed. `test_committed_cost_model_is_what_the_committed_csv_fits_to` guards against
anyone hand-editing the constants. The 10 req/s operating point was *not* re-tuned after
calibration (it is now 2.7x capacity, not 1.7x), and neither was MLFQ's 30 s ageing default --
which is now below the ~62 s queueing delay, so E4's knee moved to 45-60 s. That is reported,
not fixed, because tuning it to the result would be fitting the study to its outcome.

**4.18 One more gate claim fell to the measured constants.** "SRTF beats SJF with swaps" held
on 9 of 11 traces. It became "swaps close SRTF's gap to SJF to within 5%" (11 of 11; the
largest ratio is 1.006).

## 5. Verification

```bash
pytest tests/ -q                         # 114 tests, ~30 s
python -m src.runner --all --jobs 8      # ~2 min
python -m src.runner --all --verify      # byte-for-byte reproduction check
python -m src.plots
python serving/loadgen.py                # ~25 min
python report/report_numbers.py          # every number the report quotes
python report/build_pdf.py
```

`tests/test_theory.py` is the important one. It asserts, on three traces, what scheduling
theory or the modelled mechanism *requires* — SJF beats FCFS on the mean and loses on the
worst case; no policy exceeds the engine's capacity ceiling; recompute costs preemptive
policies throughput; a perfect predictor reproduces the oracle; MLFQ-LAS ignores the
predictor; all seven policies converge at low load. **If a change breaks one of these, the
change is wrong, not the test** — unless you can explain why the theory does not apply, as
4.15 did.

## 6. Extending it

- **A new policy**: add a class to `policies.py` with two methods, register it in `REGISTRY`.
  Nothing else changes; it is picked up by every experiment automatically. Give it a colour
  and marker in `plots.py`.
- **A different GPU or model**: re-run the Colab notebook, re-run `fit_cost_model.py`. All
  results rescale.
- **A new workload**: add a block to `config/workloads.yaml`.
- **A new metric**: add it in `metrics.py::aggregate`; it appears in every results CSV.

Things deliberately *not* built, and why: multi-replica routing and prefill/decode
disaggregation (out of scope — a different scheduling problem, at the cluster rather than the
worker level); overlapping swap transfers with compute (we charge them in full, which makes
swap slightly pessimistic); preemption in the proxy; Prometheus/Grafana (a dashboard would add moving
parts without adding evidence, and matplotlib figures are what the report needs).
