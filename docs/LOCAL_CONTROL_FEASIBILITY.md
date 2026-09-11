# Local Robot Control Feasibility

Measured 2026-09-11 on this workstation. **Recommendation: pursue a VLM plus conventional, persistent feedback-controlled skills. The local GPU can run a compact planner while motion executes, but the current rendering/worker arrangement does not meet dependable 20-50 Hz perception or 100 Hz wall-clock motor-control targets. Planner usefulness remains unvalidated.** No production controller or frontend was changed.

## Actual Hardware And Runtime

| Component | Observed |
| --- | --- |
| GPU | NVIDIA RTX 3080, **10,240 MiB / 10 GiB**, compute capability 8.6; not the proposed 12 GB variant |
| Available GPU memory at initial inspection | 7,110 MiB free; about 2,944 MiB already used by the desktop/apps |
| System memory | 63.92 GiB RAM total, 27.80 GiB free; separate from VRAM |
| CPU / OS | Ryzen 7 3700X, 8 cores / 16 threads; Windows 11 Enterprise, build 26200 |
| Driver / CUDA | NVIDIA 610.88, WDDM; driver reports CUDA UMD 13.3. This is driver compatibility, not an installed CUDA toolkit. `nvcc` absent from PATH |
| Simulation | Python 3.11.16, PyBullet 3.2.5, CPU DIRECT physics and TinyRenderer RGB; 240 fixed simulated steps/s, 100 solver iterations |
| GPU viewer | Existing Three.js app via ANGLE / RTX 3080 / D3D11; continuous draw activity verified in a separate port-8003 viewer |
| Local inference | Ollama 0.34.0; all three tested models reported full GPU residency at 4,096 context |
| Other runtimes | No torch, transformers, bitsandbytes or vLLM in the physics environment. Only stopped docker-desktop WSL2 distribution present |

The 12 GB card is **not measured** here. More VRAM would not remove the observed CPU-rendering stalls.

## Measured Planner Performance

Each row has **five excluded warm-ups followed by 100 requests**, batch one, one RGB image, temperature zero, no conversation history, 4,096 context, maximum 96 output tokens. Strict JSON schema: `skill`, `target_visible`, normalized `target_x`. The skills are proposals in shadow mode; these model outputs do not actuate the robot.

| Model / precision | Input and concurrent load | p50 / p95 seconds | Peak total GPU MiB |
| --- | --- | --- | --- |
| Gemma 4 E2B / Q4_0 | 320x240 replay, inference only | 0.511 / 0.646 | 5,902 |
| Gemma / Q4_0 | 640x480 replay, inference only | 0.531 / 0.671 | 6,162 |
| Gemma / Q4_0 | 320 replay + motion + 640 camera | 0.586 / 0.759 | 5,893 |
| Gemma / Q4_0 | 640 replay + motion + 640 camera + WebGL | 0.574 / 0.716 | 6,446 |
| Gemma / Q4_0 | 320 replay + motion + **320 camera** + WebGL | 0.573 / 0.693 | 6,407 |
| Cosmos Reason2 / Q4_K_M | 320 replay, inference only | 0.533 / 0.600 | 5,985 |
| Cosmos Reason2 / Q4_K_M | 320 replay + motion + 640 camera + WebGL | 0.622 / 0.777 | 5,897 |
| Cosmos Reason2 / Q8_0 | 320 replay, inference only | 0.628 / 0.694 | 6,472 |
| Cosmos Reason2 / Q8_0 | 320 replay + motion + 640 camera + WebGL | 0.763 / 0.958 | 6,718 |
| Cosmos Reason2 / Q4_K_M | **Fresh live camera** + motion + 640 camera + WebGL | **0.960 / 1.247** | **6,208** |

All rows: **100/100 structurally valid**, 100 distinct image hashes, no measured steady-state request exceeded the two-second benchmark deadline. This is not 100% task accuracy. The final live run's maximum was 1.343 s; acquisition alone was 0.324 / 0.490 s p50/p95. An earlier live repetition measured 0.934 / 1.081 s, illustrating run-to-run variability. Quantiles of components must not be added to infer an end-to-end quantile.

Replay timing starts before PNG decode, resizing and encoding and ends after complete response parsing and validation. It includes HTTP, server vision encoding, prompt processing and all generated tokens. Live timing additionally includes worker queuing and fresh observation/capture. Server-side vision and reasoning stages are not independently exposed; no separate stage timings are invented. Cosmos reported 1,150 input tokens and 20-33 output tokens, Gemma 165/247 input tokens at 320/640 and 24 output tokens; provider token accounting is not necessarily comparable.

`think=false` was requested; no separate thinking field was returned. Cosmos advertises vision/completion/tools rather than an Ollama thinking capability. These are **short direct structured-output tests**, not evidence that long physical reasoning fits the same latency budget. Early startup outliers included 14.95 s for Gemma, 13.71 s for Cosmos Q8 and 32.67 s in the initial Cosmos Q4 smoke test. Warm and health-check the planner before allowing motion.

Memory is whole-device `nvidia-smi` usage sampled at roughly 250 ms plus query overhead, including warm-up and desktop processes. It is not a precise process allocation peak, and short peaks may be missed. Run order was sequential, not randomized; other workstation load was not isolated. WebGL ran in a separate existing-app viewer, not a synchronized display of the benchmark worker. Draw counters confirmed activity, not display FPS. Do not infer a WebGL speedup from these results.

### Caching And Accuracy

Replay images are 100 distinct real head-camera views from bench, park, apartment and kitchen/bathroom scenes; warm-up uses five other views. Scene/head placement for corpus construction is scripted and privileged. Each request includes only the RGB image and fixed instruction, never scene labels, coordinates, evaluator state or depth. The live run also produced 100 distinct images. Adjacent views remain correlated; this is a representative timing corpus, not 100 independent task trials.

Runtime prefix/vision caching defaults were retained. The system prefix is identical; the corpus is reused across configurations. Distinct hashes prevent exact within-run repeated-image tests but do not prove zero cache reuse. No cache-hit counter was exposed. Results are warmed application performance, not cache-disabled throughput.

**Model selection is blocked on semantic validation.** Cosmos Q4 and Q8 disagreed about red-cube visibility on **71/100 paired images**, despite identical imported templates and stop settings. Q4 chose approach 10 times; Q8 81 times. A manual spot-check of frame 0005 found Q8 claiming a visible red cube where none was apparent. This is a warning, not a labelled accuracy score. Q4 also generated fewer tokens, so its latency advantage is not a pure kernel-speed comparison. Neither precision has demonstrated reliable navigation, grasp selection or progress assessment. Require held-out labelled visibility/localization tests and closed-loop success rates before selecting a planner.

### Verified Cosmos Configuration

Ollama successfully downloaded and ran `hf.co/apolo13x/Cosmos-Reason2-2B-GGUF:Q4_K_M` and `:Q8_0`, both with the F16 vision projector. `/api/show` identifies `qwen3vl` and the respective quantization. `/api/ps` reports equal loaded size and GPU size, approximately 1.98/2.70 GB respectively; use sampled whole-device usage above for workstation headroom.

Public conversion revision: `6efd567dd99d4cf9d54cc80ea38e878ae25700dc`. Ollama digests: Q4 `5b13749607f29cb2070db9ca52edd78f6771bf3c3121066ea9287035767d280d`; Q8 `7bae3c09fd4bc1a9c883dd8c383e99849e4d43e11bc866a3c9d4bf5dd56623f4`. Gemma digest: `07ea59a474013479c8b6b802bef095c40e964a1d776ba02f264c0e30e1aede0c`.

This verifies this community conversion/runtime on this Ampere GPU. It does not verify the current upstream weights: the conversion predates NVIDIA's March 2026 update. NVIDIA's official card documents Linux, BF16 testing and a 24 GB minimum for its published setup. The measured short-context GGUF configuration is a different deployment. FP16/BF16 and other inference engines were not benchmarked. Native FP8/FP4 speedups must not be assumed on Ampere; choose a supported implementation such as INT4 W4A16/GGUF and measure it.

## Control And Responsiveness

The existing worker separates asynchronous inference from motion ownership, but [worker.py](../backend/worker.py) performs snapshot/camera work on the physics thread. [navigation.py](../backend/navigation.py) executes 12-tick slices and local clearance checks; [simulation.py](../backend/simulation.py) renders and PNG-encodes the head camera synchronously. Nominal 240 Hz simulated integration is not evenly paced wall-clock actuation.

| Measured load | Worker publication Hz | Publication p95 gap | Camera captures/s | Worst physics-step gap |
| --- | --- | --- | --- | --- |
| No inference, 640 camera, 20 s | 12.76 | 316 ms | 4.97 | 343 ms |
| No inference, 320 camera, 20 s | 18.84 | 92 ms | 8.32 | 106 ms |
| Cosmos Q4 replay + 640 camera + WebGL | 12.52 | 355 ms | 4.46 | 695 ms |
| Gemma replay + 320 camera + WebGL | 18.38 | 116 ms | 7.67 | 196 ms |
| Cosmos Q4 fresh live feedback + WebGL | **9.17** | **519 ms** | **4.83** | **970 ms** |

Final live run: publication absolute jitter p50/p95 36.9/469.1 ms relative to 50 ms; 269/909 publication gaps exceeded 50 ms. Physics ran in bursts: median/p95 step gap 0.95/4.16 ms, yet **314/10,919 gaps exceeded 10 ms**. The harness counts 8,526 missing 10-ms opportunities using `ceil(gap/period)-1`; these are gap-budget diagnostics, not deadline misses of a separately scheduled 100 Hz controller (none exists here). Low p95 step gaps hide the long pauses.

Physics advanced during all 100 Cosmos requests in both concurrent replay and final live runs. Head velocity exceeded 0.02 rad/s in 67% of final live publication samples. This demonstrates overlapping execution, not unbroken motion: trajectories settle, reverse, and occasionally wait for refills. The benchmark uses sustained head trajectories; it does not establish sustained base-plus-arm task execution. Median 640-camera capture cost was about 145 ms without inference; 320-camera capture was about 37 ms. Resolution reduction helps substantially but still does not establish 20-50 Hz visual feedback or 100 Hz actuation.

### Perturbation Trials

One exploratory trial of each case ran with Gemma and again with Cosmos Q4 inference in shadow mode. These are **scripted control tests using real physics**, not statistically meaningful VLM task-success rates.

| Case, Cosmos run | Responsiveness / result | Task outcome and qualification |
| --- | --- | --- |
| Moving red target | RGB centroid + head encoder tracker; accepted update p50/p95 185/310 ms; 4 stale replacements rejected | Final three image errors <15% image width; worst error 28.7%. Partial tracking success, not navigation. Target moved discretely by fixture; command acceptance is not motor-response latency |
| Unexpected obstacle | 1.57 ms fixture-insertion-to-command-brake; no contact | Safety stop succeeded; destination not reached. Insertion was serialized just before a safety check, so this is favorable-phase detection, not a worst-case reaction bound or physical stopping time |
| Three-second delayed response | Bounded head motion braked after 1.03 s from the post-submission observation; late update rejected as `STALE_PLAN` | Correct safe pause, not continuous task completion |
| Empty/failed grasp | Contact false on both fingers, zero load; close-to-returned-feedback 896 ms | Failed grasp detected and scripted reopen completed. No autonomous regrasp or successful pickup demonstrated |

Obstacle prevention uses existing proximity sensors **and privileged simulator whole-body collision lookahead**. Gripper contacts/load are simulated sensor proxies. No depth was sent to the model. Fixture body poses are used to inject events and cannot be counted as camera-only perception. Existing focused physics tests also verify same-velocity buffer replacement without a velocity reset, stale/Stop rejection, collision braking and real assisted grasp/lift behavior. They do not measure learned chunk blending or prove sim-to-real grasp transfer.

## Architecture Decision

| Option | Assessment |
| --- | --- |
| VLM plus conventional controllers | **Recommended.** Local short-output planning and concurrent execution are measured feasible; robust semantic planning and fast wall-clock control still need work. Reuse the worker, contracts, IK, trajectory and safety code |
| VLM plus learned action policy | **Blocked for Milo today.** No compatible Milo checkpoint or synchronized demonstration dataset is present. Published GR00T embodiments are not this wheeled, dual-6DOF-arm, head-camera robot. Action-vector reshaping is not sufficient adaptation |

Start with a planner at roughly **one decision per second**, event-triggered for failures, with a two-second response budget. This is a latency target supported by compact-output measurements, not demonstrated one useful decision per second. Keep conservative speed limits until perception/reaction is measured. Do not lengthen motion leases simply to hide inference latency.

Make navigation/grasp/scan **persistent controller-owned skills** with explicit preconditions, completion and failure feedback. The planner chooses semantic goals/target image regions; camera-calibrated perception/tracking supplies target estimates. Conventional local planning, IK and trajectory tracking then update continuously without asking a language model for wheel/joint commands. Monocular image coordinates alone do not provide reliable metric grasp depth: use calibrated geometry/multiview estimation, or explicitly add and evaluate real depth sensing.

Move head-camera rendering/PNG encoding and expensive telemetry away from the motion execution path, using a snapshot-fed separate renderer rather than concurrently accessing the same Bullet client. Target a measured 20 Hz local perception/safety loop first; raise toward 50 Hz only where task testing demonstrates value. Retain 240 Hz fixed-step physics for fidelity, and separately instrument a 100 Hz wall-clock controller. A physical robot should use an independent motor controller/MCU and hardware watchdog; Windows scheduling and Python tests do not certify hard real time.

Keep one inference outstanding, timestamp observations at acquisition, retain only the latest frame, and attach episode/skill revision, observation sequence, deadline and validity region to proposals. Reject expired or drifted actions; Stop, takeover, reset, disconnect and local safety failure must invalidate pending motion. Continue the current safe skill while planning runs; if its validity horizon or watchdog expires, brake locally without waiting for the model. This harness tests those existing buffer paths but does not install the proposed persistent skill controller.

For future learned chunks, evaluate 0.2/0.5/1.0 s execution horizons, overlap/refill timing, joint velocity/acceleration continuity and event-to-first-changed-motion latency. Compare against the same conventional controller and perturbations. A longer queue can conceal slow inference while worsening visual reaction. Current tests cover velocity preservation for conventional buffers, not learned action stitching.

### Learned-Policy Prerequisites

GR00T's current documented standard inference minimum is **16 GB VRAM**; below that, including either 10 or 12 GB, is experimental and unvalidated here. Its guide recommends **40 GB+ for fine-tuning**. Co-residency with a separate VLM would further reduce headroom. No GR00T weights were installed or policy performance claimed.

A concrete learned-policy pilot would be a single-arm grasp/recovery skill if conventional vision/IK proves inadequate. Record synchronized head RGB, calibrated joint/gripper states, commands and timestamps, with successful and failed grasps and recovery demonstrations. Define joint order, units, base/EEF frames, absolute versus delta actions, execution rate, normalization, camera extrinsics and gripper semantics. Use held-out objects/layouts and randomized dynamics. A small ACT/diffusion policy needs Milo demonstrations and training; its fit and training cost remain unmeasured. GR00T requires a new-embodiment modality configuration, post-training and access to the larger training GPU budget. No defensible demonstration count or training-time estimate can be made from this task.

For physical transfer, replace privileged clearance with realizable sensing, calibrate latency/friction/backlash and coordinate frames, validate force/current limits and independent e-stop, and test staged low-speed motion. The simulator's assisted grasp constraints are not physical evidence.

## Reproduce

Run from repository root with the existing physics interpreter. The standalone [benchmark_control.py](../scripts/benchmark_control.py) does not change app configuration or contact cloud models. Q4/Q8 downloads remain in Ollama's model cache. Raw per-request JSONL, summaries, warmups, GPU samples and source PNGs are under `.runtime/control-benchmark/`; the final full-path evidence is [concurrent-320.json](../.runtime/control-benchmark/cosmos-live-final/concurrent-320.json).

```powershell
./.runtime/env/python.exe -m scripts.benchmark_control --stage capture
ollama pull hf.co/apolo13x/Cosmos-Reason2-2B-GGUF:Q4_K_M
./.runtime/env/python.exe -m scripts.benchmark_control --stage alone --model hf.co/apolo13x/Cosmos-Reason2-2B-GGUF:Q4_K_M --output .runtime/control-benchmark/repeat-alone
./.runtime/env/python.exe -m scripts.benchmark_control --stage concurrent --live-camera --model hf.co/apolo13x/Cosmos-Reason2-2B-GGUF:Q4_K_M --output .runtime/control-benchmark/repeat-live
./.runtime/env/python.exe -m scripts.benchmark_control --stage scenarios --model hf.co/apolo13x/Cosmos-Reason2-2B-GGUF:Q4_K_M --output .runtime/control-benchmark/repeat-scenarios
./.runtime/env/python.exe -m scripts.benchmark_control --stage control --camera-width 320
```

Use `--camera-width 320` to change the isolated simulator renderer; `--width` controls model image size. For Q8, pull/use `:Q8_0` and unload the other benchmark model first. `--provider openai --endpoint http://127.0.0.1:8080 --model <served-model>` supports an already-running local OpenAI-compatible server; that adapter is mock-tested, not live-runtime-validated. Choose a new output directory to retain prior evidence. For WebGL contention, run an isolated existing-app server on an unused port and keep its viewer visible; `--graphics-load` records a description but does not launch or verify the viewer.

Validation: **23 focused tests passed**, including eight benchmark tests and selected navigation/grasp regressions. No frontend changes, rebuild or full application regression suite were needed. Pylance's editor interpreter does not resolve PyBullet; the documented physics interpreter runs it successfully. This evaluation does not certify hard real time, broad task success, long reasoning, a learned policy, or physical-robot safety.

## External References

- [Cosmos Reason2 official model card](https://huggingface.co/nvidia/Cosmos-Reason2-2B): official setup, March update, reasoning and memory guidance; not a 3080 latency forecast.
- [Tested community GGUF](https://huggingface.co/apolo13x/Cosmos-Reason2-2B-GGUF): Q4/Q8 files and separate F16 projector. Model provenance/quality must be validated beyond the name.
- [vLLM quantization support](https://docs.vllm.ai/en/latest/features/quantization/): Ampere supports several INT4/GGUF implementations; implementation support is not measured model performance.
- [GR00T requirements and embodiment workflow](https://github.com/NVIDIA/Isaac-GR00T): 16 GB inference, recommended 40 GB+ training, new-embodiment adaptation.
- [LeRobot async inference](https://github.com/huggingface/lerobot/tree/main/src/lerobot/async_inference): server latest-only observation queue, timed actions, rejection of already-executed timesteps, early refill and overlap aggregation. Reuse these concepts, not its robot-specific adapters or trusted pickle transport. Add this project's epoch/revision/expiry and local-safety authority; queueing alone does not guarantee fresh motion.
- [Figure Helix 02](https://www.figure.ai/news/helix-02): slower semantic reasoning, 200 Hz visuomotor targets, 1 kHz learned whole-body control on Figure's hardware/data. Reference architecture only; neither checkpoint availability nor RTX 3080 performance evidence.