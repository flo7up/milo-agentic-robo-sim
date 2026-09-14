# Luna + SmolVLA

The operator UI offers `luna_navigation` (Luna supervising resident SmolVLA primitives) and `luna_continuous` (Luna selecting observed destinations for conventional local tracking). Continuous is the default; saved selection is retained. Manual controls remain available. Legacy single-step, voice and seven-axis arm controller APIs remain for compatibility/testing but are no longer offered by the main autonomous panel. Historical sections below describe earlier experiments, not the current UI.

**Current status:** The local default is `.runtime/navigation-recovery-6500/checkpoint`, trained for instruction-conditioned forward, backward, left/right turn and hold actions. Real supervised parking has succeeded, but **two distinct challenge completions have not been demonstrated**. Recharge and kitchen/bathroom attempts remain unsuccessful. Model action-learning loss is not a task-success metric. The arm checkpoint is unchanged and has no verified pickup. Training remained native Windows CUDA; evaluation invoked the user's configured Luna deployment and did not provision cloud resources.

## Observed Local-Goal Experiment: 2026-09-13

The isolated docking-v1 runner now accepts `--conditioning task|waypoint`. Waypoint conditioning acquires a goal from the head-camera green marker and updates robot-relative forward/left text using measured wheel odometry. It does not supply hidden target coordinates or substitute conventional velocity outputs. With identical preexisting task-waypoints-12500 weights and seed 716, six fixed cases passed **4/6 with waypoint text versus 0/6 with task-only text**. The same one-second buffers, stop votes, measured rest, full-bay containment and 60-second deadline apply. No retraining, live promotion or learned action sequence was added; the live SmolVLA primitive adapter is unchanged. [Evidence and limitations](VALIDATION.md#adaptive-navigation-and-goal-conditioning-2026-09-13).

```powershell
./.runtime/env/python.exe -m scripts.audit_architecture --stage docking --source smolvla --checkpoint .runtime/navigation-task-waypoints-12500/checkpoint --conditioning waypoint --seeds 716 --output .runtime/my-waypoint-docking
```

Use a separate output directory and `--conditioning task` for the matched ablation. Keep the 1500 parking checkpoint as a learned positive control. These small structured docking results do not establish reliable natural-language room navigation.

## Time-Bounded Navigation Tests

The isolated `scripts/evaluate_supervised.py` evaluator now defaults to **1,800 seconds of monotonic wall time with no overall request-count limit**. Set `--time-limit-s` to change the time budget (positive, at most 3,600 seconds). Model preload and scene initialization are measured separately and excluded; controller startup, warm-up, inference, feedback waits and motion are included. Saved reports distinguish termination reason, physics success, completion source and unverified model claims. Shutdown/report collection can add a small amount beyond the budget.

```powershell
./.runtime/env/python.exe -m scripts.evaluate_supervised --checkpoint .runtime/navigation-recovery-8500/checkpoint --challenges kitchen_bathroom recharge --time-limit-s 1800 --reasoning high --output .runtime/my-timed-evaluation
```

For staged comparisons, use explicit `--turns 80 --session-limit-s 1200`, then `--turns 120 --session-limit-s 1800`, with fresh output directories. `--turns` accepts 1-1,000 supervision turns and cannot be combined with `--time-limit-s`. A supervision turn can request up to eight local motion steps; reported `model_commands` counts completed local commands, not Luna requests or every attempted prediction. The default time-only mode reports both request limits as null rather than inventing a large count cap.

This is an **evaluation-only** budget supplied internally to `AgentController`, not a new public Start API field. The browser retains its existing 80-turn/20-minute limit and live checkpoint. Local-only waypoint benchmarks and `start.ps1 -NavigationTest` retain their existing separate budgets. Per-request inference timeouts, eight-step subgoal bounds, 60-second skill deadlines, motion leases, velocity/clearance/freshness checks, Stop/takeover/reset/disconnect and serialized inference remain intact. Time-only means success, model termination or a safety fault can still stop before the deadline.

Six real tests used the same recovery-8500 weights and supervisor prompt: 80 turns, 120 turns, then time-only, on kitchen and recharge. All six failed physics completion. Time-only runs passed the old count caps (135 and 138 turns), ending on an unverified completion claim and battery depletion respectively, not on request limits. **No extra fine-tuning was performed in this budget experiment:** the traces need corrected planning/docking examples; increased runtime alone did not establish a training benefit. [Results and qualifications](VALIDATION.md#navigation-budget-expansion-2026-09-12).

## Task-Skill Training: 2026-09-12

Completed **4,000 additional CUDA optimizer updates** from recovery-8500, producing experimental checkpoints at 10,500 and 12,500 total updates. Neither is promoted: the new training reduced imitation loss but did not produce a full-route or autonomous-task improvement in the tests below. The live default and resident process were left unchanged.

`scripts/task_navigation.py` records docking, Kitchen to Bathroom doorway routes, an obstacle detour to the recharge survey zone, and retreat/re-approach routes. Each frame pairs the real head image and 20 measured state values with a short robot-relative waypoint instruction and the demonstrated velocity. Teacher routes use known fixture coordinates offline; the relative waypoint is explicit teacher assistance, not camera-only route discovery. Only paired images, measured state, instruction and actions enter training. No evaluator labels, private scene geometry, task-stage metadata or world poses are input features. The existing live supervisor still sends its five canonical primitive instructions; the new waypoint text is currently used only in the isolated training/benchmark pipeline.

- **Data:** 16 recorded episodes, 997 new frames, all replayed exactly (maximum state error 0.0). Twelve episodes satisfy the full parking/bathroom physical destination scorer; four recharge episodes satisfy only the survey objective, not return-and-recharge. Rejected northern-route clearance and battery-depleted round-trip pilots remain excluded and retained under `.runtime/task-waypoint-pilot-v1` and `-v2`.
- **Retention:** appended 704 replay-verified primitive frames from `.runtime/recovery-demonstrations-v2`, matched by episode split. Export has 1,701 frames: 1,278 training and 423 held out; loss samples 143 held-out frames. RGB pixels, states, actions and instructions were read back exactly from LeRobot. Train episodes 0-11 and validation 12-15 remain separate. Offsets are nearby variants of the same layouts, not generalization proof.
- **Training:** batch four, frozen vision/language backbone, 99,880,992 trainable parameters. First 2,000 updates at 1e-5 reduced held-out loss 3.63850 -> 0.60064; next 2,000 at 3e-5 reduced it 0.60064 -> 0.23599. Both saved checkpoints reloaded strictly; each used a new AdamW optimizer state and about 3,321 MiB peak allocated VRAM. Total training wall time was about 23.8 minutes.

| Test | Recovery-8500 | Task-10500 | Task-12500 |
| --- | --- | --- | --- |
| Teacher-waypoint routes completed, four evaluation-only offset cases | 0/4 | 0/4 | 0/4 |
| Individual waypoints completed across those cases | 0 | 0 | 2 |
| Full Luna-supervised kitchen/recharge tasks, 40 turns each | 0/2 | Not run | 0/2 |
| Primitive movement signs correct, three samples per command on three scenes | 36/36 | Not run | 36/36 |
| Hold predictions within existing stop deadband | 6/9 | Not run | 7/9 |

The 10,500-step model reached a physically valid parking position in one assisted trial but did not finish the requested waypoint; a safety stop ended the run. It remains a benchmark failure. The 12,500-step model completed one detour waypoint and one retreat waypoint, then timed out or stopped early. The supervised comparison used the unchanged five-primitive Luna flow, not the new waypoint policy interface. Same settings do not fix cloud sampling variation; one run per task/checkpoint cannot establish a ranking.

**Next training decision:** do not add more updates to this dataset unchanged. Improve goal representation and examples near stopping/turning boundaries, including corrective states visited by the learned policy. Goal vectors supplied as numeric features are a candidate experiment; they are not implemented or validated here. Any later integration must use Luna-selected or observed local goals, never the offline teacher's hidden route.

Artifacts and full qualifications: [validation record](VALIDATION.md#task-skill-training-and-tests-2026-09-12).

```powershell
./.runtime/env/python.exe -m scripts.task_navigation --output .runtime/new-task-demos
./.runtime/smolvla-env/Scripts/python.exe -m scripts.task_navigation --stage export --source .runtime/new-task-demos --retain-primitives .runtime/recovery-demonstrations-v2 --output .runtime/datasets/new-task-waypoints
./.runtime/env/python.exe -m scripts.task_navigation --stage evaluate --checkpoint .runtime/navigation-task-waypoints-12500/checkpoint --cases 16 17 18 19 --output .runtime/new-task-skill-evaluation
```

Use fresh output directories. Cases 16-19 are evaluation-only and rejected by recording. This benchmark explicitly supplies teacher waypoints, advances only through learned stop decisions within 0.07 m, and preserves the original one-second segments, stop deadband, two-second freshness, 60-second skill deadline and validated worker clearance. It must never be described as autonomous challenge success.

## Live Browser Navigation

Run `./start.ps1`, configure **Luna connection**, load the chosen challenge, and click **Start LLM control**. The same worker shown in the browser owns all physics. The app loads the local Python 3.12 CUDA checkpoint once, validates its hash/contract, warms up without motion, and then asks Luna for bounded navigation decisions. No server on port 8085 is required. Camera movements follow supervisor decisions; drive velocities come from SmolVLA, not Luna or a scripted route.

Luna receives the user goal, current and labeled initial head images, measured AgentObservation, its own compact route memory and execution feedback. It must call `guide_navigation`; raw wheel velocities/code are not accepted. It selects 1-8 local steps, a movement intention, visible floor-pixel/marker target, or retracing of the measured encoder path. Pixel projection assumes a flat floor and the calibrated upright camera; it has no access to the hidden map or goal centers. Marker segmentation is a camera-only geometric assist, not a learned SmolVLA skill. Encoder retracing follows previously measured positions, not an oracle path.

SmolVLA receives the paired head image/AgentObservation and the selected canonical movement instruction. It generates both velocity components, subject to unchanged 0.15 m/s and 0.5 rad/s limits, acceleration bounds, two-second observation age, per-buffer leases and whole-robot clearance checks. A predicted clearance obstruction brakes and invalidates tickets, then requests a different supervisor subgoal; actual contact, instability, expired sensor feedback and operator Stop remain terminal. Each new supervisor subgoal has its own bounded travel/time window; total supervision is bounded by 80 turns and 20 minutes. Near-zero predictions pause and return to Luna instead of silently ending the task. Stop, takeover, reset and disconnect still cancel the run and discard late results while retaining weights. Luna's completion statement is marked agent-reported; the independent physics scorer determines verified success.

## Recovery Continuation

`scripts/recovery_policy.py` recorded 16 fixed demonstration episodes across park/recharge fixtures: 704 paired frames, 528 training frames (episodes 0-11), 176 validation frames (12-15). All 16 episodes replayed with maximum measured-state error 0.0. Each repeats bounded forward/reverse/turn/hold intentions in varied starting views; these are **primitive-action demonstrations, not successful full-task routes**. Raw episode metadata and private fixture positions are not training features. LeRobot export readback verifies tasks, actions, measured states and image pixels. Original task evaluation starts were not demonstration starts. Repeated task trials informed supervisor changes, so those results are development evaluations, not an untouched independent benchmark.

Three actual AdamW continuations from the 1,500-step checkpoint trained 1,500 + 1,500 + 2,000 further steps, batch size four. The last stage used learning rate 1e-5, earlier stages 1e-4. About 99.9 million of 450.0 million parameters were trainable. Fixed validation loss fell **6.6263 -> 0.0780 -> 0.0555 -> 0.01438**; every checkpoint was saved and strictly reloaded. This does not prove route planning or recovery success.

| Candidate | New Updates | Wall Time | Weight SHA-256 |
| --- | --- | --- | --- |
| recovery-3000 | 1,500 | 549 s | `7f55850b661a8fa827072f282645ba6631d698d4b46c055bb64adfa32d85e4a1` |
| recovery-4500 | 1,500 | 542 s | `08c1ebe8a113cf47b75dd4bd5a4a3860cf9392010286ccdad4682ef7f65ad0ef` |
| recovery-6500 | 2,000 | 689 s | `b1e5f8f533bd0f2a982ee5bf39b12bef5ab98953f42013d17d3b1abeaff9c08f` |

**Measured task results:** the recovery-6500 candidate passed **Park in the Bay** in `.runtime/supervised-challenges-6500-final/park/report.json`: 19 local commands, six Luna turns, 151.3 wall seconds, zero manual placements. An earlier recovery-4500 run also passed. Other parking trials failed due to premature Luna completion claims. **Remember and Recharge has not passed**; one calibrated trial completed its survey stage but timed out on return. The latest no-repeat recovery trial exhausted 80 supervisor turns. Kitchen to Bathroom did not pass its trial. All first outcomes, traces and camera captures remain under `.runtime/supervised-*`; no failed records were overwritten. No reliable aggregate success rate is claimed.

Repeat an isolated task evaluation with the same production controller (this invokes paid Luna inference):

```powershell
./.runtime/env/python.exe -m scripts.evaluate_supervised --checkpoint .runtime/navigation-recovery-6500/checkpoint --output .runtime/my-supervised-evaluation --challenges park recharge --turns 80 --reasoning high
```

Further progress needs successful multi-stage route/recovery demonstrations and stronger validation of Luna's spatial decisions, not just more updates on the repeated primitive dataset. The two-challenge acceptance target remains open. Historical experiments below retain their original results and qualifications.

### Resident Model Lifetime

The browser backend owns one `ResidentNavigationModel`. Per-run sessions share it under a serialization lock. Loading begins on the first Start and continues if that run is stopped during startup; the stopped robot never receives an action from that abandoned session. Cancelled in-flight predictions are drained and discarded before a new session may reset or use the protocol stream. Requests queued for a closed session are skipped. The existing model ticket and robot-episode checks still apply to all returned actions.

A strict motion-free reset message clears policy state and reseeds the existing model for every new run, preserving repeatability without reloading weights. The UI reports GPU residency independently of run completion. Explicit **Unload local model** is allowed only while no robot controller or motion is active; application shutdown also cancels pending service work and closes the process tree. A failed request or dead worker is discarded and the next Start reloads it. Ordinary Stop, reset, challenge changes, voice takeover and disconnect keep the weights resident. The optional CLI `-NavigationTest` remains a separate short-lived process and does not use this browser model service.

The saved tokenizer pads to 48 tokens and originally truncated longer goals. Navigation inference now overrides truncation only: the short parking input remains identical, and the recharge instruction retains all 124 tokens in the real saved preprocessing pipeline. Weights and normalization files are unchanged. Cold model startup may take up to five minutes while the robot remains stationary; loading stages are written to `.runtime/local-navigation/*.log`. This does not extend the two-second observation-age limit or motion leases.

**Experimental scope:** recovery-action training does not establish general-navigation ability. Long-route planning, charging-station memory, multi-stage completion, manipulation and reliable recovery remain unverified. The 3D view is live physical motion, not a replay or separate CLI scene. References to selectable legacy modes below are historical.

## Navigation Stopping Experiment

The original navigation dataset has only **24 exact-stop labels among 206 training samples (11.7%)**. The known failure in validation case 15 alternated between near-zero and moving predictions after entering the bay. To test whether emphasizing stopping helps, `--balance-stops` adds opt-in weighted sampling with replacement: the stopped and moving classes each receive 50% probability. A stop label means both recorded action values equal zero; the helper does not change action labels, images, normalization statistics or split membership. It rejects invalid labels and datasets missing either class. Manipulation training rejects this option, and ordinary uniform sampling remains the default.

The new candidate continues the 1,000-step navigation weights for **500 additional real optimizer updates**, batch size 4, with new AdamW state and otherwise unchanged settings. All samples come from training episodes 0-11; the same validation set is excluded from gradients and normalization. The sampler drew **1,010 stopped examples out of 2,000 draws (50.5%)**. Fixed validation loss fell **0.49531 -> 0.31876**. The complete run took **244.6 s**, with **3,320.7 MiB** peak allocated GPU memory; strict checkpoint reload passed. [Training report](../.runtime/navigation-stop-balanced-1500/report.json).

Candidate: `.runtime/navigation-stop-balanced-1500/checkpoint/`. Weight SHA-256: `8bf69d9e08777d06a45a589b891d47a27246e1948eb774a0b951f56b35d6a119`. The original dataset, 1,000-step checkpoint and failed results remain intact. This experiment changes **both additional training and sampling**, so it cannot attribute the outcome to balancing alone.

### Measured Results

The previously failed case 15, seed 716, now completes the same parking and two-consecutive-stop criterion after **20 requests**, versus the baseline exhausting 30. It travels 1.28 m, with 7 saturated velocity requests and no final contact readings. [Retest report](../.runtime/navigation-stop-balanced-case15/report.json), [driving timelapse](../.runtime/navigation-stop-balanced-case15/motion.gif). This case motivated the experiment and is therefore a diagnostic retest, not independent validation.

Four new cases (16-19) were fixed after training and before either model was evaluated, with disjoint target offsets from all original training/validation cases. They remain in a separate `EVALUATION_CASES` list excluded from recording/export/training. Each checkpoint runs sequentially once per case with matched seeds 816-819, the same 30-request budget, the same scripted setup scans, and unchanged controller/stop rules. The comparison keeps unsuccessful and premature-stop outcomes rather than retrying them.

| Case / Target / Posts | Baseline Success / Requests | Candidate Success / Requests | Saturated Requests, Baseline / Candidate |
| --- | --- | --- | --- |
| 16 / (1.05, -0.40) / No | Yes / 21 | Yes / 16 | 6 / 5 |
| 17 / (1.05, 0.40) / No | Yes / 18 | Yes / 16 | 5 / 6 |
| 18 / (1.55, -0.18) / Yes | Yes / 21 | Yes / 21 | 5 / 9 |
| 19 / (1.55, 0.18) / Yes | Yes / 21 | Yes / 21 | 4 / 7 |

**Both checkpoints pass 4/4**, with no premature stops in these four runs. Total commands fell **81 -> 74**; requests requiring velocity saturation increased **20 -> 27**. These are bounded-controller results, not raw-action reliability. No broad success-rate gain is demonstrated, and four nearby offsets are not evidence of reliable warehouse navigation. [Complete paired results](../.runtime/navigation-stopping-comparison/comparison.json). The final comparison code additionally emits aggregate command/saturation totals, a completeness flag and the additional-training confound; the original saved reports are not rewritten.

Physics scoring is evaluated only after the model's stop decision; its hidden bay coordinates are not passed into the controller. The existing two-stop rule, 0.15 m/s and 0.5 rad/s velocity bounds, acceleration/clearance checks, observation freshness, leases and Stop protections were not relaxed. Case 19 [before/after](../.runtime/navigation-stopping-comparison/candidate-case19/comparison.png) was inspected. GIFs remain 250 ms per decision snapshot, not real-time playback.

### Reproduce

Use a new output directory for each command. These run locally and do not change the live application:

```powershell
./.runtime/env/python.exe -m scripts.navigation_policy --stage evaluate --checkpoint .runtime/navigation-stop-balanced-1500/checkpoint --case 15 --seed 716 --output .runtime/navigation-stopping-retest
./.runtime/env/python.exe -m scripts.navigation_policy --stage compare --baseline .runtime/navigation-training-1000/checkpoint --checkpoint .runtime/navigation-stop-balanced-1500/checkpoint --seed 800 --requests 30 --output .runtime/navigation-stopping-paired-repeat
```

Reproduce the bounded continuation:

```powershell
./.runtime/smolvla-env/Scripts/python.exe -m scripts.train_milo --embodiment navigation --dataset .runtime/datasets/milo-navigation --base .runtime/navigation-training-1000/checkpoint --balance-stops --steps 500 --batch-size 4 --output .runtime/navigation-stopping-training-repeat
```

The candidate is available explicitly by checkpoint path and through live local navigation. Single step and planning remain selectable; the browser's initial mode is now local navigation and explicit user choices are remembered. Broader starts/layouts, false-stop testing when no bay is present, and raw action-bound reliability remain open validation work for experimental use beyond parking.

## Local Navigation Fine-Tune

This is actual supervised fine-tuning from the pinned public SmolVLA base, **not the arm checkpoint**, a prompted LLM, or replay presented as learned control. [train_milo.py](../scripts/train_milo.py) has an explicit `--embodiment navigation` option and refuses continuation from an arm-model training report. The original manipulation mode remains the default and its checkpoints/data are unchanged.

**Contract:** `milo-navigation-v1` takes one paired 320x240 head RGB image, a task string, and 20 measured values: head yaw/pitch, wheel-derived linear/angular speeds, eight range distances and eight range statuses. A clear beam encodes 2 m with status 0, a hit its measured distance with status 1, and self-occlusion distance 0 with status -1. No evaluator poses, target locations, floor plans or case IDs enter inference. State and image normalization are identity; the two action outputs use mean/std statistics from training episodes only.

Outputs are `[linear_mps, angular_radps]`, representing the setpoint of a **one-second buffered drive segment**. `chunk_size=1`, dataset `fps=1`; this is a decision/action timebase, not one inference per wall-clock second. The existing worker ramps wheel velocity, checks whole-body clearance and floor support, caps commands at 0.15 m/s and 0.5 rad/s, and stops on faults. Physics integration remains 240 Hz simulation time. Arms and head do not receive learned commands in this pilot.

### Dataset And Training

[navigation_policy.py](../scripts/navigation_policy.py) recorded 16 simple green-bay approaches: straight, left/right offset targets and approaches through two widely spaced posts (1.95 m clear opening). A privileged geometric demonstrator chooses labels **only during recording**; every move goes through `NavigationRuntime` on an isolated `SimulationWorker`. Two scripted head scans satisfy the existing inspect/locate prerequisites before recorded driving begins. Each episode ends with two stopped samples.

All 16 recordings succeeded and all saved-action replays matched the 20-value sensor vectors exactly (maximum error 0.0). The exporter validates schema, camera path/hash, action limits, time alignment, successful replay, unique case ordering and robot-asset hash. Real LeRobot export/readback verified all **276 paired samples**, including exact float32 state/action values and RGB pixels. Data is in `.runtime/datasets/milo-navigation/`; [export report](../.runtime/datasets/milo-navigation/navigation-export.json).

Episodes 0-11 provide **206 training samples**. Episodes 12-15 contain **70 held-out samples** at excluded target offsets within the same simple layout family. They are validation cases, not a broad independent benchmark. No training sample contains the private `case` setup or target coordinates.

After a successful one-step smoke test, the model ran **1,000 optimizer updates**, batch size 4, float32, AdamW learning rate 1e-4 and gradient clipping at 10. The vision/language backbone stays frozen; 99,880,992 of 450,046,176 parameters are trainable. Loss on 26 fixed-noise validation samples decreased **3.40054 -> 0.49546**. Total run time including load/validation/reload was **395.4 s**, peak allocated GPU memory **3,323.9 MiB**. Strict checkpoint reload produced finite `[1,1,2]` outputs. Loss is not a navigation success metric.

Checkpoint: `.runtime/navigation-training-1000/checkpoint/`. [Training report](../.runtime/navigation-training-1000/report.json). SHA-256 of saved weights: `ae48c721b3bb0a48186a39a7e3972464b5717ad472bd6e6f8bebabf2f16547a2`.

### Real Model Tests

The evaluator launches a separate native Windows CUDA inference process and verifies the training report, weight hash and navigation feature contract. It submits only paired `PolicyRequest` camera/sensors/instruction. Subsequent wheel setpoints come from model predictions, **not** the demonstrator, target coordinates or evaluator feedback. Commands are saturated to existing navigation limits with raw/bounded values and affected axes logged. Small outputs (`abs(v)<0.025`, `abs(w)<0.05`) execute an explicit zero-velocity segment. Two consecutive such predictions end the attempt; physics then checks full bay containment, rest and ground contact. Reaching the bay alone does not force a model stop or count as success.

Each test starts with the same scripted inspect/locate scans, then the model controls the approach. The evaluator waits for each one-second buffer to brake before sending fresh feedback. It preserves navigation ticket/revision checks, Stop, the 60 s skill deadline, travel limits, 1.5 s segment lease and safety checks. It rejects an observation older than 2 s before submission. The private planner's geometry is used for safety, never as learned-policy input. These tests do **not** validate continuous refill, model-generated plans or camera-only collision avoidance.

| Held-Out Case / Seed | Layout | Learned Commands | Wheel Travel | Outcome |
| --- | --- | ---: | ---: | --- |
| 12 / 713 | Left-offset bay | 16 | 1.00 m | Parked; two consecutive stop predictions |
| 13 / 714 | Right-offset bay | 20 | 1.11 m | Parked after small forward/back corrections |
| 14 / 715 | Left-offset bay beyond wide posts | 19 | 1.25 m | Parked; two consecutive stop predictions |
| 15 / 716 | Right-offset bay beyond wide posts | 30 | 1.62 m | Reached bay, but oscillating stop/move predictions exhausted the request budget; **failed** |

All four finished physically inside their bay with no final contact readings or navigation safety failure. **Three of four** met the full learned-stop success criterion. The model occasionally requests excessive speed, so these are **bounded-controller results**, not unmodified raw-action validation. Do not present this small set as reliable warehouse/multi-room navigation or a generalized 75% success rate. The fourth-case failure is retained, not relabeled as success.

Reports: [case 12](../.runtime/navigation-eval-1000-case12/report.json), [case 13](../.runtime/navigation-eval-1000-case13/report.json), [case 14](../.runtime/navigation-eval-1000-case14/report.json), [case 15](../.runtime/navigation-eval-1000-case15/report.json). [Driving timelapse](../.runtime/navigation-eval-1000-case14/motion.gif) and [before/after](../.runtime/navigation-eval-1000-case12/comparison.png) show actual changes to the robot camera as the base moves. GIF timing is 250 ms per decision snapshot, **not wall-clock playback**.

### Repeat Locally

Use a fresh output directory. This starts an isolated physics scene, does not contact the live app or Luna, and does not change either UI control mode:

```powershell
./.runtime/env/python.exe -m scripts.navigation_policy --stage evaluate --case 12 --seed 713 --output .runtime/navigation-local-retest
```

Reproduce data and training with distinct new directories:

```powershell
./.runtime/env/python.exe -m scripts.navigation_policy --stage record --output .runtime/navigation-new
./.runtime/env/python.exe -m scripts.navigation_policy --stage replay --source .runtime/navigation-new
./.runtime/smolvla-env/Scripts/python.exe -m scripts.navigation_policy --stage export --source .runtime/navigation-new --output .runtime/datasets/navigation-new
./.runtime/smolvla-env/Scripts/python.exe -m scripts.train_milo --embodiment navigation --dataset .runtime/datasets/navigation-new --steps 1000 --batch-size 4 --output .runtime/navigation-training-new
```

Do **not** load this checkpoint into the existing `Luna + SmolVLA` policy server: that API intentionally supports seven-axis left-arm manipulation only. The navigation model is currently accessed through this CLI test harness, not the UI model picker. Arbitrary task strings, maze routes, unexpected obstacles, varied starts, moving targets and long-range navigation require additional data and evaluation.

### Single Step And Planning

**Retain both.** Single step remains useful for precise actions, manipulation, calibration, fault diagnosis and controlled recovery. Planning is the appropriate higher-level mode for multi-stage navigation and bounded motion buffers. They can share the same validated worker and cancellation rules without removing either interface. Eventually a planner could select a destination/skill and hand short-range execution to a trained navigation policy, but this pilot does not integrate that handoff. No automatic mode switch or replacement of the current default was made.

## Windows Policy Timing Fix

The worker's former per-tick relative pacing was the measured bottleneck: it requested waits of approximately 4.2 ms but Windows slept for a median **18.9 ms**. Median policy-tick computation was **0.92 ms**, collision validation **1.37 ms**, and render-update work **2.14 ms**. In a five-chunk scripted benchmark with real physics and the snapshot renderer, every accepted 0.4 s trajectory expired after just **28-31 of 96 physics ticks**. The original 0.55 s wall-clock lease was working as designed. Inclusive timing categories overlap and must not be summed.

`PolicyPacer` now uses high-resolution `perf_counter()` absolute deadlines instead of adding a full relative wait on every tick. Oversleep is compensated by subsequent ticks without an extra wait; every tick still checks Stop, the skill deadline and the chunk lease. Catch-up is bounded: a lag above 50 ms or a changed motion revision rebases pacing, and an empty buffer clears the pacing state. No global Windows timer settings, busy-spin loop, extra physics thread or deadline extension was introduced. Manual and navigation pacing are unchanged.

The identical scripted benchmark now completes **5/5 chunks, all 96 ticks each**, in **0.406-0.422 s** within the same 0.55 s lease. Maximum final joint-target error was about **0.000138 rad**. Effective aggregate cadence rose from **51.5 to 230.2 Hz**, but individual tick spacing remains uneven (post-fix p95 gap **17.5 ms**, maximum **20.8 ms**, including boundaries). This is improved trajectory throughput, **not a 240 Hz hard-real-time guarantee**. [Before report](../.runtime/policy-timing-before/report.json), [after report](../.runtime/policy-timing-after/report.json).

The same three real-model scene/seed/request-count trials were repeated without changing weights, adapter targets, motion limits or safety leases:

| Trial | Completed Chunks | Expired Chunks | Policy Physics Before / After | Maximum Joint Change After | Pickup |
| --- | ---: | ---: | ---: | ---: | --- |
| Held-out (0.36, 0.26), seed 713 | 10/10 | 0 | 1.571 / 4.000 s | 0.1847 rad | No |
| (0.35, 0.25), seed 714 | 20/20 | 0 | 3.179 / 8.000 s | 0.4720 rad | No |
| (0.37, 0.25), seed 715 | 40/40 | 0 | 5.883 / 16.000 s | 1.0469 rad | No |

All **70/70 accepted adapter-assisted chunks completed**, with no expiry or assisted attachment. The policy still did not lift/place the cube in any trial. This removes the observed execution bottleneck; it does not fix raw-policy action quality. The test uses a stop-and-observe adapter and nearby fixed scenes, so it does not establish continuous-motion refill performance, broad generalization, or Luna supervision.

Across the three trials, inference p95 ranged from 0.561 to 0.618 s and the largest observed camera age before adapter validation was 0.703 s. The clean full backend suite passed **251 tests**, with two existing dependency deprecation warnings, and three affected browser workflows passed. Stop, stale-response and forced-expiry regressions remain covered.

Reports: [held-out](../.runtime/milo-paced-heldout-1/report.json), [second](../.runtime/milo-paced-second-1/report.json), [third](../.runtime/milo-paced-third-1/report.json). [Updated before/after image](../.runtime/milo-paced-third-1/comparison.png) and [timelapse](../.runtime/milo-paced-third-1/motion.gif) show the arm reaching the cube area but no lift. Animation timing remains 250 ms per request snapshot, not wall-clock playback.

Reproduce in fresh output directories, without restarting or contacting the live app:

```powershell
./.runtime/env/python.exe -m scripts.benchmark_control --stage policy-timing --policy-repeats 5 --output .runtime/policy-timing-repeat
./.runtime/env/python.exe -m scripts.evaluate_milo --checkpoint .runtime/milo-training-1100/checkpoint --motion-adapter --timing --cube-xy .37 .25 --seed 715 --requests 40 --output .runtime/milo-paced-repeat
```

`--timing` is opt-in test instrumentation: computation/wait costs and physics cadence go only into the diagnostic report, never model inputs. Per-chunk execution records distinguish accepted, completed and expired trajectories. The existing live app must be restarted to load the worker code; no existing session was reset for these experiments. The isolated adapter remains outside production policy serving.

## Adapter-Assisted Movement

The original 100-step weights were continued for **1,000 updates at batch size 4**, after a separate batch-fit smoke test. AdamW state was restarted; this is weight continuation, not optimizer resume. Training/held-out episode split and normalization stayed unchanged. Mean loss on eight fixed held-out samples fell from **0.73946 to 0.05972**. The complete run took **557.7 s**, peaking at **3,321.5 MiB** allocated GPU memory. [Training report](../.runtime/milo-training-1100/report.json). The checkpoint is `.runtime/milo-training-1100/checkpoint/`.

Raw evaluation remained unsuccessful in three fixed scene/seed pairs: all were rejected before motion for gripper-range violations. The first two cases also contained requested joint speeds of approximately 4.2 and 5.0 rad/s, above the unchanged 0.8 rad/s limit. Lower training loss alone does not establish executable actions.

The approved `--motion-adapter` option exists **only in the isolated evaluator**. It takes the raw model's fixed index-19 target, saturates it to Milo's actual joint/gripper bounds, then scales the displacement and uses a 0.4-second quintic trajectory from rest with speed/acceleration margin. It has no object coordinates, IK-generated waypoints or alternate model. Reports retain every raw prediction, selected target, saturated axis, progress fraction and adapted chunk. The original observation ticket/capture age is preserved; stale input, Stop, joint/speed/acceleration violations and collision validation still use the existing worker checks. This intentionally changes execution semantics and must not be counted as raw-policy success.

Initial real-model, real-physics tests before the pacing fix above:

| Cube XY / Seed | Accepted Targets | Maximum Joint Change | Gripper Change | Policy Physics Time | Pickup |
| --- | ---: | ---: | ---: | ---: | --- |
| (0.36, 0.26) / 713, held-out position | 10/10 | 0.1133 rad | 4.9 mm | 1.571 s | No |
| (0.35, 0.25) / 714, training position | 20/20 | 0.2459 rad | 6.1 mm | 3.179 s | No |
| (0.37, 0.25) / 715, training position | 40/40 | 0.2446 rad | 53.7 mm | 5.883 s | No |

All three initial trials ended at their request budget without lift or assisted attachment. Trajectories repeatedly expired before all eight planned samples executed; the worker braked and awaited fresh feedback. Subsequent instrumentation traced this specific throughput defect to Windows per-tick wait overshoot, fixed above. Acceptance does not mean a full trajectory completed. No lease or motion limit was relaxed.

Reports: [held-out trial](../.runtime/milo-adapter-heldout-1/report.json), [second trial](../.runtime/milo-adapter-second-1/report.json), [third trial](../.runtime/milo-adapter-third-1/report.json). The third trial's [before/after image](../.runtime/milo-adapter-third-1/comparison.png) shows the left gripper moving with the right gripper fixed. Its [camera animation](../.runtime/milo-adapter-third-1/motion.gif) contains 41 snapshots at 250 ms each: a **timelapse, not wall-clock playback**. The final image differs from the initial image on 2.88% of pixels at a 12-level RGB threshold; physical joint readings independently confirm movement. These are small repeated scenes, not a generalization or pathfinding benchmark.

Repeat with the existing checkpoint and a fresh output directory:

```powershell
./.runtime/env/python.exe -m scripts.evaluate_milo --checkpoint .runtime/milo-training-1100/checkpoint --motion-adapter --cube-xy .37 .25 --seed 715 --requests 40 --output .runtime/milo-adapter-retest
```

Omit `--motion-adapter` to test unmodified chunks. CPU/GPU environments remain separated, no app session is reset, and the candidate is not enabled in the live UI. **43 demonstration/training/adapter/policy tests pass**, including input privacy, checkpoint integrity, saturation accounting, stale-ticket rejection, speed/acceleration bounds, visual evidence and existing Stop protections. Next work is held-out task progress, broader demonstrations, and timing diagnosis with the existing limits preserved.

## Windows Training And Local Test

[train_milo.py](../scripts/train_milo.py) uses the cached pinned base weights with strict loading, Milo's seven-axis state/action features and one 320x240 head camera. LeRobot's internal 32-value padding allows weight reuse without reshaping the six-value base output. Training uses episodes 0-3 (2,880 frames); episode 4 is held out. Fresh state/action mean/std statistics are computed from training frames only; images retain identity normalization. Model batches allow only paired camera/state/action/task fields and the action-padding mask.

The action expert and state projection are trainable; the vision encoder and remaining VLM are frozen. A one-step forward/backward/save/reload test passed before the 100-step run. The pilot uses batch size 1, float32, AdamW at 1e-4 and gradient clipping at 10. No experiment tracker or Hub upload is enabled.

**Measured 2026-09-11:** 99,880,992 trainable parameters out of 450,046,176; all 100 losses and gradient norms finite. Mean flow-matching loss across eight fixed samples in the held-out episode decreased from **1.02030 to 0.73974** with fixed evaluation noise. Peak allocated GPU memory was **3,320.2 MiB (3.24 GiB)**, reserved 3,452 MiB; the complete run including loading, validation and checkpoint reload took **114.8 s**. The held-out cube position differs by only a centimeter, so this is not generalization evidence.

Checkpoint and saved processors: `.runtime/milo-training-pilot-100/checkpoint/`. [Training report](../.runtime/milo-training-pilot-100/report.json). The checkpoint reloaded strictly and returned finite `[1,50,7]` chunks. Saved processors round-tripped known dataset actions with maximum absolute error **1.19e-7**.

[evaluate_milo.py](../scripts/evaluate_milo.py) loads that candidate in a separate native-Windows CUDA subprocess, verifies its training report and weight hash, and submits only paired `AgentObservation`, head RGB and instruction. A fresh `SimulationWorker` recreates the held-out cube at (0.36,0.26), sets the recorded head pose and warms up inference without actuation. The experiment then uses the unchanged `SkillRuntime` freshness, speed, acceleration, collision and tracking checks. The in-memory candidate metadata identifies an actually fine-tuned experimental model; it is not saved as a production readiness manifest. No scripted actions or clipping replace predictions.

**Actual result:** the first measured prediction took **0.409 s**. **39 of 50 gripper values exceeded 0.11 m**, reaching **0.2417 m**. `PolicyChunk` rejected the entire response (`INVALID_POLICY_CHUNK`), and the worker stopped. All seven arm/gripper state values were unchanged, with **zero simulated policy-motion seconds**, no lift and no task success. Camera setup moved the head before the test; it is not learned action. The scene and grippers were visible in the saved head image. [Evaluation report](../.runtime/milo-policy-evaluation-100/report.json) and [head-camera input](../.runtime/milo-policy-evaluation-100/frame-000.png).

Reproduce the original pilot locally, using fresh output directories. The default is 100 steps; continued experiments now permit at most 3,000 additional steps per invocation:

```powershell
./.runtime/smolvla-env/Scripts/python.exe -m scripts.train_milo --steps 100 --output .runtime/milo-training-repeat
./.runtime/env/python.exe -m scripts.evaluate_milo --checkpoint .runtime/milo-training-repeat/checkpoint --output .runtime/milo-evaluation-repeat
```

Test the existing candidate without retraining:

```powershell
./.runtime/env/python.exe -m scripts.evaluate_milo --output .runtime/milo-policy-retest
```

The evaluator returns a report even for a safety rejection; process exit zero is not task success. This original candidate is **not enabled in the live UI**. The later, explicitly approved experiment above demonstrates movement using the continued checkpoint and a separate trajectory adapter. Do not confuse those results with raw-policy execution or weaken gripper/motion limits.

## Pretrained Smoke Tests

The standalone [probe_smolvla.py](../scripts/probe_smolvla.py) loads public `lerobot/smolvla_base` revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205` with strict weight-key validation. Backbone configuration/tokenizer revision is `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. Full policy weights contain the VLM weights; `load_vlm_weights=False` during construction avoids downloading an additional backbone weight copy, after which the full policy checkpoint is strictly loaded. Neither the script nor its imports include the robot executor.

**Input:** "Pick up the red cube." (7 tokens including processor formatting, within the 48-token cap), synthetic six-value zero state, and one saved Milo head-camera PNG repeated into all three base-checkpoint camera slots. Each slot is resized to its declared 256x256 input shape; the model's original 512x512 padded preprocessing is retained. This is an interface smoke test, **not calibrated three-camera input, Milo joint mapping, or a task-quality evaluation**.

**Measured 2026-09-11:** CPU float32, eight PyTorch threads, 450,046,176 parameters, original 10 denoising steps and cache enabled. Download took 14.96 s and model/processor loading took 51.61 s. One warm-up took 5.55 s; the following two complete image/state-to-postprocessed-action predictions took **5.32 s and 5.54 s**. All three outputs had shape `[1,50,6]` and finite values. Peak process working-set RAM was **2,620.8 MiB (2.56 GiB)**, not VRAM. With only two measured repeats of identical input, no p50/p95 benchmark or visual accuracy claim is made; the policy's sampled noise produces differing actions.

The native Windows CUDA 12.8 smoke used the same pinned model, float32, inputs and strict loading path. The RTX 3080 completed one warm-up in **0.869 s** and one measured prediction in **0.490 s**. Peak allocated VRAM was **1,789.1 MiB** and peak process working-set RAM was **3,170.2 MiB**. This single measured request is functional evidence, not a latency distribution or task-quality result. Full evidence is in [report.json](../.runtime/smolvla-base-smoke-cuda-windows/report.json).

The action values are in the base checkpoint's representation, not Milo's seven-axis command contract. No output was sent to the policy server or robot, no fine-tuning occurred, and existing app sessions were unchanged. The CPU latency exceeds the current one-second freshness budget. Do not relax that budget to present this as responsive control or extrapolate these numbers to the RTX 3080.

Full evidence and numeric outputs: [report.json](../.runtime/smolvla-base-smoke-run2/report.json). Public files are cached under `.runtime/smolvla-base-cache/`. An initial attempt failed on a long Windows temporary-download path before inference; shortened cache directory names resolved it without any access/TLS bypass. That failed report is separate from the successful run.

To repeat (choose a new output directory; no robot connection is made):

```powershell
./.runtime/smolvla-env/Scripts/python.exe -m scripts.probe_smolvla --output .runtime/smolvla-base-smoke-repeat
./.runtime/smolvla-env/Scripts/python.exe -m scripts.probe_smolvla --device cuda --requests 1 --output .runtime/smolvla-base-smoke-cuda-repeat
./.runtime/env/python.exe -m pytest -q tests/test_probe_hybrid.py
```

The probe tests pass (9 tests), including rejection of wrong action dimensions and nonfinite outputs. Broader CUDA timing and memory measurements can now proceed; useful Milo motion still requires the matching embodiment, task training and validation.

## Ownership

| Component | Responsibility |
| --- | --- |
| Luna / selected Foundry profile | Choose a supported task, inspect camera/sensor feedback, cancel or report completion |
| Local task manager | Own the active instruction, revisions, timeouts and independent policy requests |
| SmolVLA policy process | Preprocess camera/state/instruction, predict numeric action chunks, denormalize with checkpoint processors |
| Physics worker | Validate future actions, track targets with bounded motors, brake on expiry/fault/Stop |
| Snapshot renderer process | Render a copy of a timestamped physics snapshot without blocking motion with camera rendering |

The supervisor has only `start_skill`, `cancel_skill`, `complete_skill`, `observe`, and `stop`. It cannot submit joint arrays, select arbitrary checkpoints, call local shell commands, or bypass the worker. Selecting **Luna + SmolVLA** prefers the Luna cloud profile and sets a one-second supervision interval; any configured Foundry profile may be used. Local Ollama supervision is rejected in this mode to keep the intended cloud/local split explicit.

`start_skill` takes the current `expected_revision`, `skill="pick_place"`, a concise `instruction`, and `timeout_s` from 2 to 60 (default 30). Acceptance starts independent policy inference, not task completion. `observe` does not brake an active skill. `complete_skill` brakes and records a **supervisor-reported** semantic assessment, separate from private scenario scoring. A text-only supervisor response ends the session and stops motion. To change an instruction, cancel the current skill first.

## First Embodiment

`milo-left-arm-v1` supports one bounded manipulation policy, with the base parked and head/right arm held. Prepare the head angle and arm pose manually before starting. The camera must match the training setup and see the workspace. This is not a general navigation, bimanual, or mobile-manipulation policy.

State and absolute action order:

```text
left_joint_1, left_joint_2, left_joint_3,
left_joint_4, left_joint_5, left_joint_6,
left_gripper_opening_m
```

The six joints are radians, gripper aperture is meters, and the checkpoint action sampling rate is **20 Hz**. This is an action timebase, not a claim of 20 model inferences or visual reactions per second. The single camera feature is `observation.images.head`; the state feature is `observation.state` with shape `[7]`, output `action` has shape `[7]`, and `n_obs_steps=1`. The model may predict at most 50 actions. The adapter uses the checkpoint's saved preprocessing and action-denormalization pipeline, never the SO100 normalizer or an arbitrary six-to-seven-axis reshape.

SmolVLA's published base checkpoint declares a different action/camera layout. It must be adapted and fine-tuned using synchronized Milo demonstrations. Roughly 50 diverse episodes can be a starting point for a bounded pick/place task, not a promise of success. Record measured state, target actions, camera frames, task strings and timing; retain held-out object positions and recovery cases. Five physical-only pilot episodes have been recorded, replayed and exported to LeRobot as described below. The 100-step fine-tuning pilot above passed the training checks but failed its first motion-contract evaluation. The exchange feed is not a ready-made training dataset.

## Demonstration Pilot

Implemented [record_milo.py](../scripts/record_milo.py) and [milo_dataset.py](../scripts/milo_dataset.py). No production simulation/controller code was changed for this pilot, and running operator episodes were not reset.

**Measured:** five out of five scripted recordings and five out of five saved-action replays completed pickup, transfer, release and settled placement. Each episode has 720 paired RGB/state/action records at 20 Hz simulated time: **3,600 frames and 180 simulated seconds total**. Peak cube-bottom height was approximately 0.20 m, with simultaneous contact from both fingers during lift. No Bullet constraint or assisted attachment was present. All seven recorded pre-action state values matched fresh-worker replay exactly (maximum absolute difference 0.0 for each episode).

The five starting cube centers are (0.36,0.25), (0.35,0.25), (0.37,0.25), (0.36,0.24), and (0.36,0.26) meters. The target square is centered at (0.36,0.34). This is a **very narrow, scripted mechanics/data pilot**, not a learned-policy success rate, sufficient training corpus, held-out generalization result, or physical-robot validation. Camera samples were inspected at start, lift and final placement; the workspace, cube and gripper are visible.

Recordings are in `.runtime/milo-demonstrations-pilot/`. See the [recording summary](../.runtime/milo-demonstrations-pilot/summary.json) and [replay summary](../.runtime/milo-demonstrations-pilot/replay-summary.json). Every episode stores:

- `episode.json`: fixed action/state names, clock/alignment description, camera setup, robot asset hash, and **private fixture setup**. This metadata is not a model input.
- `frames.jsonl` and `images/*.png`: measured seven-value state and paired 320x240 head RGB **before** the next 50 ms motion, plus the **commanded endpoint**, task, timestamps and image hash. Actions are not copied from measured positions. States/commands keep full precision.
- `terminal.png`, `terminal.json`, and `result.json`: final observation and mechanical outcome, kept separate from training features.
- `replay.json`: fresh-worker physical result, state comparison, source-frame-file hash and robot asset hash. Replay loads the saved actions; it does not regenerate them using object coordinates or IK.

The demonstrator uses privileged target positions to generate trajectories, then submits them through the existing `SkillRuntime` inside an isolated `SimulationWorker`. Initial joint-space descent was collision-rejected; Cartesian vertical waypoints, a slow gripper transition and a smooth raised joint-space transfer passed the existing limits. Rejected early probes remain separately under `.runtime/milo-demo-*-probe/`; they are not training episodes. No speed, acceleration, collision or tracking guard was relaxed. The internal scripted-runtime metadata enables the test executor only; it is not a trained checkpoint and is never installed into the production policy server.

**Clock qualification:** recording and replay deliberately use fixed **simulation time**, without model/network latency, and disable preview publication in their private worker. Offline capture can therefore be slower than real time without changing action timing. These results do not establish wall-clock lease/refill performance. Deployment still uses the ordinary real-time expiry and cancellation checks.

Reproduce with the existing physics environment (choose a fresh output directory for recording):

```powershell
./.runtime/env/python.exe -m scripts.record_milo --probe --episodes 5 --output .runtime/milo-demo-check
./.runtime/env/python.exe -m scripts.record_milo --episodes 5 --output .runtime/milo-demonstrations-new
./.runtime/env/python.exe -m scripts.record_milo --replay --episodes 5 --output .runtime/milo-demonstrations-new
./.runtime/env/python.exe -m scripts.milo_dataset --validate-only --source .runtime/milo-demonstrations-new
./.runtime/env/python.exe -m pytest -q tests/test_demonstrations.py tests/test_policy.py
```

The validator checks complete episodes, exact schema/action ordering, finite state/action values, 20 Hz alignment, image dimensions/hashes/path containment, limits and matching successful replay evidence. The export adapter refuses failed/unreplayed episodes or an existing output directory. It passes only head RGB, measured state, commanded action and task to LeRobot's official `create/add_frame/save_episode/finalize` writer. It does not upload data or include fixture/evaluator state. The writer integration is now **verified with installed LeRobot 0.6.1**, in addition to mock regression coverage.

The approved-feed installation and export commands below were executed successfully. The dataset now exists; a new export must use a different output directory. Verification can be repeated without rewriting it:

```powershell
uv pip install --python .runtime/smolvla-env/Scripts/python.exe -r scripts/requirements-smolvla.txt "lerobot[training]==0.6.1"
./.runtime/smolvla-env/Scripts/python.exe -m scripts.milo_dataset --source .runtime/milo-demonstrations-pilot --output .runtime/datasets/milo-pilot --repo-id local/milo-pilot
./.runtime/smolvla-env/Scripts/python.exe -m scripts.milo_dataset --verify-export --source .runtime/milo-demonstrations-pilot --output .runtime/datasets/milo-pilot --repo-id local/milo-pilot --report .runtime/milo-export-validation.json
```

### Verified LeRobot Export

Local dataset: `.runtime/datasets/milo-pilot/`, identifier `local/milo-pilot`. Evidence: [milo-export-validation.json](../.runtime/milo-export-validation.json). It contains five contiguous 720-frame episodes with float32 seven-axis state/action fields, image-backed RGB and LeRobot's standard timestamp/index/task metadata. The original recordings and their full-precision values remain unchanged.

- Every reopened image matches its paired source PNG exactly after conversion to channel-first RGB floats in [0,1] (maximum pixel difference **0.0**).
- Every state and action matches the expected float32 conversion exactly. The largest difference from original float64 values is approximately **5.96e-8**, solely representation rounding. Task strings, per-episode frame indices and timestamps match all 3,600 records.
- Recorded state and commanded action remain distinct. Seven-axis min/max/mean/std statistics were recomputed from the raw source both in float64 and using LeRobot's own statistics implementation. The saved statistics match the latter within strict tolerances; no axis has standard deviation below 1e-4 in this pilot.
- LeRobot's float32 `mean(x*x) - mean(x)**2` variance computation differs slightly from direct float64 standard deviation. Maximum std differences are **3.57e-5** for state and **2.90e-5** for action. Verification uses absolute tolerance 1e-6 plus relative 0.1% for std (0.01% for other statistics); the LeRobot source recomputation is checked at absolute 1e-8 plus relative 1e-7. No exported statistics were overwritten. Image statistics are sampled by LeRobot and were checked for finite per-channel shape/range, not equated to an all-pixel population statistic.
- Fifteen **50-action window** checks at episode starts, penultimate frames and last frames confirm correct future-action alignment, last-action padding and `action_is_pad` masks. No window leaks actions from the next episode.
- Seventeen recorder/export regression tests pass, including rejection of altered pixels/actions, shifted timestamps, wrong episode boundaries, corrupted statistics and unexpected private features.

Installed extras include datasets 4.8.5, PyArrow 25.0.1, pandas 2.3.3, PyAV 15.1.0, TorchCodec 0.11.1 and wandb 0.27.2. TorchCodec's native libraries could not load (missing FFmpeg/dependencies reported); LeRobot selected its supported PyAV fallback on export. Reopening explicitly uses `video_backend="pyav"`. This is an **image-backed dataset**, so video decoding was not needed. TorchCodec/video workflows remain unverified. Installing the training extras did not log into wandb, upload recordings or start training.

The raw recordings remain usable independently of GPU availability. The next data milestone is broader successful demonstrations plus held-out validation and recovery cases. The later 100-step training pilot uses the pinned base revision with Milo's seven-axis features, single head camera and fresh dataset normalization; it does not establish usable manipulation.

## Installation Diagnosis

**Resolved package route:** the machine-wide `C:\ProgramData\pip\pip.ini` already configures `https://packagefeedproxy.microsoft.io/pypi/simple/`. [uv does not read pip configuration files](https://docs.astral.sh/uv/pip/compatibility/#configuration-files-and-environment-variables), so the earlier uv commands unintentionally attempted direct public PyPI downloads. The SmolVLA requirements now explicitly contain that existing approved index. The successful install resolved 64 packages through this route, without a public fallback or TLS bypass. The package-download blocker is resolved; the earlier network symptoms below do not mean the approved feed was unavailable.

Verified installed versions: LeRobot 0.6.1, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Transformers 5.5.4, NumPy 2.2.6, Python 3.12.10 x64. `uv pip check` passes, `SmolVLAPolicy` imports, and CUDA matmul, torchvision CUDA NMS, and the pinned SmolVLA smoke run all execute on the RTX 3080. GPU inference success is not training or model-quality validation.

The official native Windows CPython 3.12 wheels are cached under `.runtime/wheels/cu128/`. They were downloaded directly from `download-r2.pytorch.org`, verified against hashes published by the PyTorch index, and installed together with `--no-deps` so the existing CFS-installed dependency set was not re-resolved through blocked public PyPI hosts:

```text
torch-2.11.0+cu128-cp312-cp312-win_amd64.whl
SHA256 7c78215c3af4f62e63f2b2e360f1722fc719b0853c7ac22666483d9810613a4c
torchvision-0.26.0+cu128-cp312-cp312-win_amd64.whl
SHA256 8c0d1c4fbb2c9a4d5d41d0aaa87da20e525bcb2a154ce405725b0be59456804b
```

On 2026-09-11, `uv 0.9.21`, Windows curl (Schannel), and PowerShell all failed TLS negotiation to `files.pythonhosted.org`. Both the LeRobot `.whl.metadata` URL and actual `.whl` URL failed; using the Windows certificate store had already failed. The PyPI index, Hugging Face and the actual `https://download.pytorch.org/whl/cu128/torch/` index returned HTTP 200. A 403 from the PyTorch hostname root was not treated as proof its package index was blocked. Windows Internet Settings reported no enabled explicit proxy or configured PAC; this does not rule out managed network interception.

Exact failing artifact:

```text
https://files.pythonhosted.org/packages/58/c0/8b17580cfae9d970edc1b9138a2231fb5416eb2e8e03e3c79f4355f978c2/lerobot-0.6.1-py3-none-any.whl
PyPI-advertised SHA256: 1894516040c65f80a45bd9741f8174aae90ed5d93da0627ab4f1a85fd8d75e90
Windows curl: SEC_E_ILLEGAL_MESSAGE (fatal TLS handshake alert)
PowerShell: HttpRequestException, SSL connection could not be established
```

The direct public wheel was not downloaded during those probes, so its advertised hash was not locally verified then. These historical transport failures were not evidence of GPU incompatibility. The subsequent package install used CFS; public Hugging Face base-model weights were later downloaded for the separate non-actuating CPU smoke test. No proxy enforcement or package-host restriction was bypassed and no untrusted mirror was substituted.

## Run The Policy Server

**Browser scope:** `Luna + SmolVLA` connects to a `milo-left-arm-v1` pick/place service, not either of the two-output navigation checkpoints. An offline message means that arm service cannot be reached; starting it with a navigation checkpoint is not a fix. For local driving use **Local SmolVLA navigation**, which starts its own process automatically. Single step and Navigation plan use the selected LLM, not the local navigation weights. See [Live Browser Navigation](#live-browser-navigation) or [Navigation Stopping Experiment](#navigation-stopping-experiment). Restart the app backend and refresh the page after updating.

Keep LeRobot separate from `.runtime/env`, whose PyBullet setup uses NumPy 1.x. The published LeRobot 0.6.1 requires Python 3.12 and NumPy 2.x. The isolated Windows environment at `.runtime/smolvla-env` now contains the CUDA 12.8 runtime and has passed import, GPU execution, model smoke, and dependency checks.

Reproduce the approved-feed installation from the repository root; starting the server additionally requires a compatible trained checkpoint:

```powershell
uv pip install --python .runtime/smolvla-env/Scripts/python.exe -r scripts/requirements-smolvla.txt
./.runtime/smolvla-env/Scripts/python.exe -m backend.smolvla_server --checkpoint .runtime/checkpoints/milo-pick-place --device cpu --dtype float32 --port 8085
```

Use a trusted local **fine-tuned checkpoint directory**, including its LeRobot configuration, weights and saved pre/postprocessor statistics. The cached base model is not a Milo-compatible checkpoint. The command above selects CPU explicitly as a conservative server example; it is not a real-time performance recommendation. CPU and CUDA float32 base-model inference pass in the standalone smoke test, but this server's trained-checkpoint execution path has only scripted integration coverage. CUDA/BF16 server execution still requires a Milo checkpoint and separate validation.

The checkpoint must also contain `milo-policy.json`, populated by the trainer/operator after validating its embodiment and task coverage:

```json
{
  "backend": "smolvla",
  "checkpoint": "milo-pick-place-reviewed-revision",
  "embodiment": "milo-left-arm-v1",
  "action_names": [
    "left_joint_1", "left_joint_2", "left_joint_3",
    "left_joint_4", "left_joint_5", "left_joint_6",
    "left_gripper_opening_m"
  ],
  "camera_key": "observation.images.head",
  "fps": 20,
  "trained_for_milo": true,
  "skills": ["pick_place"]
}
```

The manifest is an explicit operator declaration, not proof of training or a cryptographic safety certificate. Setting the flag on base weights does not make them suitable. The server additionally checks the checkpoint's actual state/action/camera feature dimensions. Pin and record the training/model revision separately.

In the UI, select **Luna + SmolVLA**, retain the configured Foundry endpoint/deployment, and set **SmolVLA endpoint** to `http://127.0.0.1:8085`. Enter an instruction in the checkpoint's trained task distribution and start. **Local manipulation** shows the active instruction, checkpoint label, buffer, policy latency/request count and rejected chunks. The **Policy** exchange filter shows paired observations, proposed chunks, acceptance and rejection. Global Stop and **Cancel policy skill** remain available.

Selecting this mode or changing its endpoint now runs a non-inference **Policy connection** check. Offline/unreachable, incompatible metadata and untrained checkpoints have distinct messages; **Start LLM control** and **Send chat message** remain disabled until the check succeeds. Use the refresh icon to retry after starting the service. The start API checks again before creating a supervisor session, and Stop/disconnect during that check invalidates the start. The check does not load models, train weights, or prove task accuracy.

If the service is offline, **Single step** and **Navigation plan** remain available without SmolVLA. Installing the runtime and supplying a compatible trained checkpoint are separate prerequisites; changing the port or inference settings cannot substitute for them. An older running backend may report that an update is required for readiness checks even if its static UI has been rebuilt.

## Execution And Freshness

- Policy inference is single-flight and independent of the supervisor request. The server rejects concurrent requests with 429 rather than accumulating old frames. Network/decode/validation failures stop the active skill; there is no automatic cloud-policy fallback.
- Every policy request binds episode ID, epoch, motion revision and observation sequence. Input is limited to the canonical `AgentObservation`, its paired head-camera PNG, and the active instruction. Neither model receives render snapshots, depth, spectator geometry or evaluator state.
- The renderer has latest-only bounded queues. It reconstructs physics snapshots in a separate unstepped Bullet process and returns 320x240 RGB with the matching sensor timestamp. Its privileged geometry is internal rendering data, never policy input. The initial frame is captured while motion is stopped; continuous policy-mode camera work occurs off the physics thread. Other modes retain their original renderer.
- The worker drops action samples whose scheduled time has already elapsed, rejects responses older than one second or with no future actions, and installs at most one second of future targets. Validation-time expiry also rejects the response. Expiry/exhaustion clears the buffer and invalidates pending motion without changing the active task. Three stale rejections stop for review.
- Each 20 Hz waypoint is linearly interpolated across 12 fixed physics substeps. Adjacent waypoint velocities are limited to 0.8 rad/s and gripper opening to 0.06 m/s; inter-waypoint velocity changes are limited to 4 rad/s squared and 0.3 m/s squared. The transition is rejected rather than silently clipped. These are discrete setpoint checks, not a jerk-limited trajectory or a hard-real-time motor guarantee.
- Whole-arm and gripper preflight uses the existing private collision client, then checks the next segment again at execution. This is **simulator-ground-truth safety**, not camera-only collision avoidance. Unexpected contact, more than 2 cm base displacement or 0.05 rad orientation drift, more than 0.08 rad fixed-joint drift, or more than 0.10 rad arm tracking error stops motion. Arm/gripper force targets are bounded at 20.
- The buffer has a duration-plus-0.15-second wall-clock lease. Stop is checked independently of inference. Skill timeout, model failure, reset, mode switch, takeover and final operator disconnect invalidate pending actions. Cancellation closes the policy client and snapshot process; manual takeover receives a fresh observation.
- Gripper outputs control physical fingers directly; this mode creates **no assisted grasp attachment**. Contact feedback is available, but load currently measures assisted-constraint force and may remain zero during physical-only grasps. Do not infer failed/successful pickup from load alone. Same-arm self-collision and carried-object sweep gaps inherited from the simulator remain limitations.

## Validation

`./.runtime/env/python.exe -m pytest -q tests/test_policy.py` covers strict contracts, wrong embodiment, NaN/limits, stale tickets, real bounded arm movement, stationary base, supervisor/policy independence, completion provenance, Stop, timeout, inference failure, unsafe future actions, renderer shutdown and absence of synchronous camera rendering on the physics motion path.

The Edge test uses scripted supervisor and policy implementations against real physics, checks policy traces and 320-pixel camera delivery, verifies nonblank desktop/mobile canvases and layout, and cancels/resets through the UI. It does not load or impersonate real SmolVLA weights. Production has no scripted-policy fallback.

Windows CUDA and a bounded training/checkpoint pipeline are verified. The first real candidate-policy test was safety-rejected, not successful manipulation. Real Luna-plus-SmolVLA task success, joint supervisor/policy performance, post-change wall-clock jitter and sim-to-real transfer remain unverified. Next gates are broader data, further training, and held-out manipulation tests with timing and failure-response measurements.