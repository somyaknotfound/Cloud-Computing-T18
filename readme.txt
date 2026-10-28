================================================================================
BENCHMARKING LLM INFERENCE SCHEDULING POLICIES
Cloud Computing - Research Assignment
Group 18, Dept. of Computer Science and Engineering, NITK Surathkal
Mentor: Balaji (balajink252is009@nitk.edu.in)
================================================================================

TEAM
--------------------------------------------------------------------------------
  241CS149   Ronak Maharia              Simulator engine, cost model and GPU
                                        calibration, recompute and swap preemption
  241CS222   Chirs Tony                 All seven scheduling policies (MLFQ-LAS,
                                        MLFQ-Pred, SRTF)
  241CS243   Pranav Shaji               Workload generation, metrics, multi-seed
                                        statistics (E7)
  241CS257   Somyak Priyadarshi Mohanta Serving proxy and consistency check,
                                        experiment runner, figures, report (LEADER)


WHAT IS IN THIS SUBMISSION
--------------------------------------------------------------------------------
  llm-scheduling-benchmark/   the project folder (source code)
  report.pdf                  the report
  readme.txt                  this file


WHAT THE PROJECT DOES
--------------------------------------------------------------------------------
An LLM inference server does not run one request at a time. It runs "continuous
batching": dozens of requests share every forward pass of the model. So the
scheduler does not choose who runs next - it chooses who is in the batch, and,
when the batch or GPU memory (the KV cache) is full, who gets evicted.

We implement seven scheduling policies inside one continuous-batching engine
model, calibrated on a real GPU, and compare them on identical request traces
under two ways of evicting a request: "recompute" (vLLM: discard its cache,
rebuild it later) and "swap" (FastServe-style: park its cache in host memory).

  FCFS           first come first served      (what vLLM and Orca ship)
  SJF-Oracle     shortest true output length  (impossible; a theoretical bound)
  SJF-Pred       shortest predicted length    (the realistic version)
  SRTF           shortest remaining, preemptive
  MLFQ-LAS       feedback queue, preemptive, needs NO length predictor
  MLFQ-Pred      the same queue, placing arrivals by predicted length
  Priority-SLO   interactive class first, earliest deadline within class


CALIBRATION
--------------------------------------------------------------------------------
The engine's timing comes from 161 measurements of Qwen2.5-0.5B on a Google Colab
T4 GPU (calibration/calibration_t4.csv). On that GPU, decoding 32 sequences
costs only 1.11x decoding one. The straight-line cost model we had planned
failed on the real data (55% median error, negative times for short prompts), so
it was replaced by a model that fits every measured setting to a median error
under 4%. The constants describe Hugging Face transformers on a T4, not a
production engine.

To re-run the calibration (needs only a browser; it clones the repo, measures on
a free Colab T4, and commits the results back to GitHub):
  https://colab.research.google.com/github/somyaknotfound/Cloud-Computing-T18/blob/main/calibration/colab_calibrate.ipynb


MAIN RESULTS   (10 requests/s, i.e. 2.7x the engine's capacity)
--------------------------------------------------------------------------------
  1. The engine saturates at 3.70 requests/second. BELOW that point all seven
     policies perform identically - the scheduler never has a decision to make.

  2. Shortest-job-first improves mean normalized latency 5.9x over FCFS and
     median time-to-first-token 64x, but makes the worst-case wait 1.6x worse.
     Over ten fresh traces the latency ratio is 0.19 +/- 0.02 (95% CI).

  3. The cost of preemption decides which policies work. Under recompute,
     every preemptive policy loses throughput, and the predictor-free MLFQ is
     about 2x WORSE than FCFS on every trace. With cheap swaps, SRTF draws level
     with SJF and gives the fastest first tokens, and both MLFQs keep a worst
     case close to FCFS.

  4. MLFQ's ageing threshold must follow the actual queueing delay. At the 30 s
     default, MLFQ-Pred gets 43% of the oracle's benefit; at 45-60 s it gets
     89-96% with a worst case within 9% of FCFS.

  5. Length prediction is forgiving. A predictor with a Spearman rank
     correlation of only 0.68 still captures 87% of a perfect oracle's benefit.

  6. Strict class priority is a BAD way to protect interactive traffic. Plain
     SJF gave interactive requests 57.5% SLO attainment against 6.9% for the
     SLO-aware policy, and was better for batch traffic too.


HOW TO RUN IT
--------------------------------------------------------------------------------
Requires Python 3.13 (tested with 3.13.12). NO GPU, NO MODEL AND NO OLLAMA ARE
NEEDED.

  python -m venv .venv
  .venv\Scripts\activate            (Windows)
  source .venv/bin/activate         (Linux / macOS)
  pip install -r requirements-lock.txt

  pytest tests/ -q                       114 tests, about 30 seconds
  python -m src.runner --all --jobs 8    all experiments -> results/*.csv (~2 min)
  python -m src.runner --all --verify    re-run and check results byte for byte
  python -m src.plots                    all figures     -> figures/*.png
  python serving/loadgen.py              proxy consistency check     (~25 min)
  python report/report_numbers.py        every number quoted in the report
  python report/build_pdf.py             rebuild report.pdf from report.md

Every simulation is seeded and deterministic, and two policies are always
compared on the identical request trace.


HOW WE CHECKED THE RESULTS
--------------------------------------------------------------------------------
  1. CALIBRATION. The timing model fits 161 real GPU measurements; a test checks
     that refitting the committed measurements reproduces the committed model.

  2. THEORY GATE. tests/test_theory.py asserts, on three different traces,
     properties the theory or mechanism requires - SJF must beat FCFS on the
     mean and lose on the worst case, no policy may exceed the engine's
     capacity, recompute must cost preemptive policies throughput, MLFQ-LAS
     must produce identical results with a perfect and a useless predictor,
     and at low load all seven policies must converge. Only claims that held
     on all eleven traces we checked are asserted.

  3. TEN SEEDS. Experiment E7 repeats the comparison on ten fresh traces at
     three burst lengths and reports 95% confidence intervals.

  4. DETERMINISM. `--verify` re-runs every experiment and checks each CSV byte
     for byte; parallel and serial runs give identical files.

  5. CONSISTENCY CHECK. serving/proxy.py is an actual FastAPI server that
     imports the policy code UNMODIFIED. Its backend is a MOCK that sleeps
     according to the cost model (no model, no GPU), so this checks that the
     policies rank requests the same way in a real asynchronous server - it is
     not a hardware validation, and absolute latencies are not expected to
     match. Only the four non-preemptive policies are checked, because the
     proxy does not implement preemption.

  6. INVARIANTS. No token is created or lost, the KV cache is never
     oversubscribed, the cache is empty once the queue drains, time is
     monotonic, and causality holds for every request.


KNOWN LIMITATIONS (stated in Section 9 of the report)
--------------------------------------------------------------------------------
  - One calibration session, of Hugging Face transformers in eager mode.
  - Single GPU worker: no tensor parallelism, multi-replica routing, or
    prefill/decode disaggregation.
  - Swap preemption is charged in full and never overlapped with compute.
  - Above capacity there is no steady state; results depend on burst length.
  - Output lengths are drawn from a distribution rather than generated.


SOURCE CODE
--------------------------------------------------------------------------------
  GitHub: https://github.com/somyaknotfound/Cloud-Computing-T18
  See README.md in the project folder for full documentation, and BUILD.md for
  the specification and a record of every implementation decision.
