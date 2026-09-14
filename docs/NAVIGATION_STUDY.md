# Wheeled Household Navigation Study

Date: 2026-09-13. This is a local FastAPI/PyBullet study, not a ROS migration, physical-robot qualification, or new Luna benchmark. No walking, gait learning, balancing policy, retraining, cloud provisioning, or existing live-session reset was performed.

## Decision

Keep **Luna + mission executive + observed spatial memory + conventional planning/tracking** as the default. Retain **NavDP as an experimental local RGB-D trajectory proposer** and **NoMaD as an RGB-only policy alternative**, with the same worker owning motors and stopping. Do not replace the simulator to run either policy. Do not promote the current SmolVLA primitive controller as a driving dependency; preserve it as a reproducible research baseline. Restrict SmolVLA manipulation to experimental use until it demonstrates physical task success.

This recommendation is provisional: conventional local driving and simple return work, but detours, kitchen exploration, fresh sensing under load, and dependable long-range recovery remain unresolved. NavDP produced actual useful wheel motion, including two doorway successes; neither learned policy demonstrated reliable exploration/return on this robot. A fast model is not equivalent to a fast or successful mission.

Implemented production follow-up: continuous navigation retains 0.5 m/s open-space transit but caps requested speed at 0.2 m/s when a forward/diagonal range hit is within 1 m. This addresses the observed parking-post clearance-stop at the higher cap, without altering the independent braking threshold, acceleration or freshness checks. Existing backends need restart to load changes; none of the shared live sessions was restarted. Learned backends are available only through the isolated study CLI, not the UI.

## Final Verification

- **34 focused backend tests passed** in 165.86 s, including contract/codec/frame checks, real-wheel curves/door passage, pedestrian braking, stale goals, ownership and disconnect. Artifact: `.runtime/navigation-study/final-regressions.xml`.
- After the near-obstacle profile change, **16 motion/safety checks passed** in 74.01 s. The narrow parking fixture took 6.5/6.4 simulated seconds at 0.35/0.5 transit caps; the wide fixture retained 6.1/5.75 s. These replace neither the archived speed experiment nor the matched 0.2 m/s model comparison.
- Frontend build passed. **Four enhanced-backend Playwright workflows passed** in 1.3 minutes: camera-selected navigation with no buffer stops, kitchen preparation/Stop, and goal redirection in both continuous and original SmolVLA modes. Models are scripted in browser tests, physics is real. The first browser run had a range-clearance stop and a persisted spatial-setting leak into the following test; the approach profile and explicit test-only setting reset fixed those observed failures without weakening camera or arrival assertions.
- No all-green full-backend/full-browser claim: full suites were not rerun. Timing-related stale sensing still occurs in retained study trials. Successful focused reruns do not erase those reliability limitations.

## Hardware And Scope

- **Detected:** NVIDIA RTX 3080, 10,240 MiB VRAM, driver 610.88; Ryzen 7 3700X, 8 cores/16 threads; 68,637,376,512 bytes system RAM (63.93 GiB); native Windows. Initial whole-device allocation was 8,300 MiB. Existing previews and workloads remained running.
- **Target only:** RTX 5080 / 16 GB. It is not installed, and no latency/throughput/VRAM result here was measured on it. The installed PyTorch 2.11.0+cu128 build reports `sm_86` and `sm_120` support, which is binary compatibility evidence, not performance validation.
- Physics uses `.runtime/env/python.exe` (Python 3.11). Inference uses the existing isolated Python 3.12 CUDA interpreter with additional packages in `.runtime/navigation-study/packages`, not installed into physics or the SmolVLA environment.
- New study stages preserve the live 0.5 m/s continuous default but use a process-local **0.2 m/s comparison cap**, 0.5 rad/s turn cap, 0.3 m/s squared acceleration, unchanged 1.8 m local path/60 s local skill budgets, one-second policy-frame validity, and original safety guards. Initial v1 runs used the earlier default cap and are diagnostic, not matched v2 results.
- Camera capture, map processing, recorder I/O, policy transport, and other active desktop workloads influence wall-clock timing. Repeats are fixed-layout development trials; they are not an independent home-navigation benchmark.

## Current Architecture

### Instruction To Wheels

1. The browser submits the user goal and execution mode to [app.py](../backend/app.py). `AgentController` in [agent.py](../backend/agent.py) owns the session, budgets, tool validation, cancellation, and serialized inference.
2. In `luna_continuous`, [continuous_supervisor.py](../backend/continuous_supervisor.py) supplies Luna the user goal, allowed measured observations, head images, sensor-derived reachable candidates, floor regions, and bounded history. `guide_continuous` chooses a candidate/region or requests look, turn, scan, wait, arrival inspection, continuation, or finish. It does not supply arbitrary code or motor speeds. Candidate IDs and calibrated depth projection ground coordinates; a model's description of a room remains unverified.
3. [worker.py](../backend/worker.py) projects a selected paired RGB-depth pixel or uses an already observed-map target, obtains whole-robot footprint, and invokes the conventional backend. The new [navigation_backends.py](../backend/navigation_backends.py) wraps the existing map planner with explicit goal/frame/proposal validation.
4. `ObservedMap.plan` in [spatial.py](../backend/spatial.py) uses a rolling 8 x 8 m, 5 cm occupancy grid, conservative footprint inflation, sparse Dijkstra and checked line-of-sight simplification. Unknown cells are blocked. An explicit start-footprint bootstrap bridges self-occlusion; it is not proof of observed floor.
5. `ContinuousNavigation` in [continuous_navigation.py](../backend/continuous_navigation.py) follows XY waypoints using measured wheel odometry. It rotates before large direction changes, slows on approach and for range clearance, and refills short velocity buffers. The path is geometric and retimed, not a timestamped motor command replay.
6. `NavigationRuntime.tick` in [navigation.py](../backend/navigation.py) ramps body velocity and writes the two Bullet wheel motors. Geometry in [robot.py](../backend/robot.py) is **differential drive**, wheel radius 0.09 m, track 0.38 m, with passive support. Wheel targets are `(v - 0.19*w)/0.09` and `(v + 0.19*w)/0.09` rad/s. Lateral translation is not an available actuator.
7. Execution status/encoder feedback returns to the executive. The independent challenge scorer uses simulator truth; a Luna completion assertion does not establish physics success.

There is no verified physical motor/encoder/camera driver in this path. `SimulationWorker` is the single authoritative owner of all simulated actuator groups. Policy subprocesses own inference only; frontend, HTTP and ROS adapters cannot directly write motors. This does not yet provide separate hardware wheel/arm/head drivers.

### Observations And Privilege

| Observation | Implemented source | Limitation |
| --- | --- | --- |
| Head RGB | Calibrated Bullet TinyRenderer or paired enhanced Three.js snapshot | Synthetic appearance; rotating/tilting head; not a fixed navigation camera |
| Aligned metric depth | Same camera/projection and capture state, 160x120 | Axial depth, invalid/self pixels are null; current usable range must come from calibration, not assumed fixed |
| Wheel encoder state/odometry | Bullet joint positions/velocities, radius/track integration | No external localization or loop closure; slip/contact can cause drift |
| Joint/head state | Simulated joint sensors | Assumes nominal calibration and flat/upright chassis for projection |
| Range/bumpers | Eight simulated 2 m rays and physical contact samples | Low sparse beams miss objects above/between rays; not a full 3D safety sensor |
| Gripper | Aperture, per-finger contact; assisted-constraint load proxy | Load is not a general real tactile/force sensor; no verified grasp-identity perception |
| IMU | No general IMU observation in `AgentObservation` | Runtime tilt/support checks use simulator truth, not a hardware IMU fusion stack |
| ROS LiDAR | Opt-in synthetic 720-ray, 8 m scan plus RGB-D | Exists only in the ROS simulation bridge; not evidence of a fitted LiDAR |
| Global pose, complete colliders, object/task identity | Evaluator and simulator safety only | Never sent to learned policies or Luna as observations |

**Important fairness limit:** all pathways retain the simulator-privileged full-body clearance shield, including unseen geometry, floor support and moving-obstacle prediction. NoMaD's policy sees RGB only, but its robot still uses the common RGB-D map and privileged shield. Zero recorded collisions does not demonstrate camera-only learned collision avoidance or hardware safety.

### Memory, Recovery And Timing

`NavigationMemory` retains bounded measured positions, viewed heading sectors, actions, failed routes, unverified semantic notes and successful route endpoints. `MapHistory` retains expiring footprints/sightings. Neither is SLAM, a durable cross-episode semantic graph, or calibrated localization uncertainty. The new experimental image memory retains 32 goal views and up to 64 successful directed view connections, marks wheel-odometry uncertainty and requires fresh clearance. It does not yet search a global topological graph or persist across process restart; the helper is not a new autonomous room-search system.

The existing supervisor performs bounded adaptive scans and recovery. The study's optional two-rescan executive failed its detour/kitchen pilots and remains opt-in. No failed edge is recorded as traversable. The study executive is deterministic sensor-candidate selection, not Luna; it cannot establish semantic object finding, task adherence, or kitchen recognition.

Physics integrates at 240 **simulated** Hz; navigation checks run in 12-tick slices, while local steering targets 50 ms wall-clock updates. These are not guaranteed 240/20 Hz real-time rates. RGB-D capture remains synchronous on the motion worker; map processing is one outstanding asynchronous job. The enhanced path uses a separate process for map processing. A warmed single-map probe took 0.118 s; the first unprepared probe took 0.892 s, largely imports. Live capture/scheduling still produced stale maps. Stop, takeover, reset, disconnect, expired leases and stale frames invalidate motion rather than extending deadlines.

## Upstream Verification

Exact Git revisions and local source hashes are saved in every study `manifest.json`. NavDP was inspected at `bff5eb8857f50e082243fb455d312409daabc3aa`; the checkpoint comes from Hugging Face revision `7cee38a8d8308874d2b8488783c612f42060ac41`.

| Pathway | Released inputs and preprocessing | Output and integration | Access/licensing |
| --- | --- | --- | --- |
| Conventional / Nav2 | Current baseline uses aligned depth + encoder map. Nav2 needs valid TF, odometry, footprint, obstacle sources and map/localization or SLAM | Existing custom planner/tracker is working locally. Nav2 is an optional future backend, not required to benchmark policies | No learned weights. Nav2 packages have Apache/BSD/LGPL terms; inspect package licenses |
| NavDP | **Eight RGB frames**, current depth, point/explore plus image/pixel interfaces in release. 224 square resize/pad; HTTP RGB converts to BGR before ImageNet normalization. Depth outside 0.1-5 m becomes zero; HTTP PNG is meters x10,000 | 24 cumulative local XY/yaw waypoints from denoising increments divided by four; critic-ranked trajectories. Released benchmark transforms camera-local XY into world space and uses a separate MPC tracker, ignoring predicted yaw | Original README code license CC BY-NC-SA 4.0; original latest checkpoint link is a form. Public `navdp_pretrain.ckpt` found in authors' X-NavDP repository, downloaded and strictly loaded with four auxiliary-head tensors explicitly retained. Weight license is not separately established; clarify before production/redistribution |
| NoMaD | Four RGB observations (context=3 plus current), optional real goal image or masked-goal exploration; PIL resize to 96x96, ImageNet normalization; no depth/point vector | Eight 2D relative waypoints; unnormalize deltas using min [-2.5,-4], max [5,4], cumulative sum, then the deployment's `max_v/frame_rate=0.2/4=0.05 m` scale. No yaw prediction. Released sample 0 and first three waypoints used here | Official Drive checkpoint downloaded, strict load: 19,049,675 parameters. Code MIT; downloaded weights have no separately verified license statement. Deployment example is Ubuntu/ROS Noetic, not proof of Windows support; isolated inference actually ran on Windows |
| SmolVLA | Paired RGB, state and language; release recommends task/embodiment fine-tuning | Continuous action chunks; Milo has distinct navigation and seven-axis manipulation contracts | Existing pinned base/checkpoints preserved. LeRobot code and model weights must be reviewed separately; model card is not a hardware-performance guarantee |

Installed LeRobot 0.6.1 metadata declares Apache-2.0. The pinned SmolVLA base revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205` returned no weight-card license field. The NoMaD repository MIT file covers its code; do not silently treat that as a separately verified license for the Drive weights. No model artifacts were republished.

Nav2's inspected Jazzy MPPI source distinguishes `DiffDriveMotionModel` (`isHolonomic=false`) from `OmniMotionModel`; the regulated pure-pursuit implementation transforms plans to base coordinates, regulates curvature/cost/approach speed, rotates to heading, and collision-checks velocities. Our [ROS bridge](../ros/bridge.py) still serializes HTTP capture, sensor publication and command submission in one timer callback, uses capture wall timestamps for TF, and can miss its 0.5 s watchdog. The earlier actual Nav2 attempts remain unsuccessful; this study did not relaunch ROS or claim a repaired Nav2 stack.

NoMaD weight SHA256: `70f79b8262527e20e56ced64a3e3d7ef91855bc9e7c3fa348d78edcb83c6a333`.
NavDP weight SHA256: `3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947`, **135,725,066 parameters** including the retained auxiliary heads. Strict loading found no remaining missing/unexpected keys. The original wrapper's `strict=False` would have hidden these extra tensors. Its low-critic lateral fallback is deliberately not executed by this adapter; raw critic values are recorded and the common safety layer remains authoritative.

The NavDP paper's current v3 (2025-12-24) has multi-frame RGB, 0.1-5 m depth and a larger training dataset. Earlier discussion based on v2's single-frame/0.1-3 m description is superseded. The released image/pixel branches exist, but this public checkpoint's image-goal quality was not validated; the implemented study adapter advertises only point/explore. Mapless inference does not supply persistent place memory, and updating point goals still requires a coordinate estimate.

X-NavDP was inspected only because it supplied an accessible NavDP pretraining artifact and documents embodiment modulation/back-out behavior relevant to observed failures. Its post-trained weights were **not** loaded or evaluated. No Isaac/Isaac Lab/acados installation or RL training was performed. Its original-code MIT statement must not be generalized to all vendored code or weights.

### Microduck Lessons

Inspected the actual `robotd/src/main.rs`, model API/version handling, and design documents in [Microduck](https://github.com/pollen-robotics/microduck). The useful pattern is one authoritative motor loop, replaceable policy modules, explicit recurrent state/model API, independent media/transports, and health measured by deadline performance rather than mere liveness. Its design document labels portions as a target architecture; distinguish that from implemented code. No legged-control policy or hardware-specific behavior was imported.

## Shared Boundary And Camera

Implemented internally in [navigation_backends.py](../backend/navigation_backends.py) and [worker.py](../backend/worker.py):

- Capabilities explicitly declare point/image/explore support, sensor requirements, temporal history, and fixed-head requirement. NoMaD rejects point goals.
- Goals have identity, sensor-evidence category, measured odometry point or retained image ID, actual 0.04 m local arrival tolerance and bounded timeout. Evidence strings alone are not proof: the live producer resolves existing depth/map candidates, and experimental callers are trusted internal code, not a public arbitrary-coordinate endpoint.
- Frames bind run/epoch, monotonic capture time, sequence, goal ID, Stop revision, wheel-odometry frame, head angles and optical frame. No evaluator geometry is part of the policy payload.
- Proposals contain finite meter-based XY paths, optional radian headings, and explicit geometric-retiming semantics. Strict pose trajectories reject lateral-only translation. Paths are sampled densely against observed clearance, restricted to the local travel budget and checked for starting-pose drift.
- Learned routes require forward head yaw near zero and pitch 0.3 rad down, then pause if head movement exceeds 0.02 rad. Camera history resets on goal/episode change, head movement or a gap above 0.75 s; insufficient fresh history fails explicitly. This threshold is an evaluation constraint, not an upstream timing guarantee.
- `navdp_camera_goal`/`navdp_camera_proposal` use the calibrated camera-to-head-to-base-to-odometry transform, including translation and pitch. Optical axes are right/down/forward; NavDP camera-local axes are forward/left/up. NoMaD's deployed XY uses base-relative forward/left conventions, with camera mounting mismatch remaining an embodiment assumption to evaluate.
- `capture_navigation_frame` and `start_navigation_proposal` are internal experimental APIs. They share the same wheel-path installer as conventional navigation, serialize work, reject conflicting controllers, reject superseded goals/Stop revisions, and enforce current depth clearance. There is no learned backend UI selector or live Luna integration yet.

The study deliberately uses a stopped base while acquiring fresh histories and predicting, followed by bounded path tracking. This is **not** upstream asynchronous refill performance. A permanently forward navigation camera would simplify temporal context and goal-view matching, but would be a new sensor/configuration requiring calibration and comparison. Rotating the current head cannot reconstruct previously unseen pixels; camera transforms alone do not remove the policy's visual distribution shift.

Loaded transport remains unsupported: existing assisted grasps are rejected by buffered navigation, and the robot-only footprint is not a payload swept volume. Add a verified carrying posture, payload retention sensing and payload-aware clearance before enabling transport. Keep arm deployment/manipulation and base travel in separate verified phases.

## Closed-Loop Evidence

Primary aggregate: [.runtime/navigation-study/decision-v1/summary.json](../.runtime/navigation-study/decision-v1/summary.json). Each trial has raw predictions, source hashes, a privileged evaluator recording and local HTML replay. Failed startup, early v1 interface errors and the initial 0.5 m/s diagnostic trials remain in separate directories; they are not silently retried into successes.

| Scenario | Conventional | NavDP | NoMaD |
| --- | --- | --- | --- |
| Nearby grounded point | 2/2, 8.96-9.09 s | 1/2, 13.55-14.49 s; one local path ended outside goal tolerance | Point goal unsupported |
| Turning to observed candidate | 2/2, 8.12-8.80 s | Not run in final matched set | Point goal unsupported |
| Doorway passage | 2/2, 8.98-9.04 s | 2/2, 13.89-15.07 s | Point goal unsupported |
| Target beyond obstacle | 0/2, no observed continuation | 0/2, predicted hold | Point goal unsupported |
| Simple return | 2/2, 26.12-26.16 s | Not run | 0/2 image-goal return; short/invalid trajectory or decision budget |
| Separate two-room doorway round trip | 2/2, 25.98-26.96 s; physical path checked independently | Not run | Not run |
| Complex kitchen exploration/return | 0/2, no reachable observed route | Not run | Not run |
| Exploration, four-decision/45 s budget | Two trials: stale sensing or budget end | Two trials: progress, then deadline/stale observation | Two trials: progress, then insufficient fresh history |
| Dead end | 0/2 recovery; stopped before obstruction | Not run | Not run |
| Cancel during movement | 2/2 safe stops | 1/1 | 1/1 |
| Superseded goal | 2/2 old proposals rejected | 1/1 | 1/1 |
| Injected delayed observation/response | 2/2 stale rejection | Fresh-history failure before injection; not a completed real-policy delay test | 1/1 stale rejection |

These times are control wall time including selection/history/prediction; preparation/model warmup is separately recorded. Different goal modalities and retiming mean this is a bounded integration comparison, not a pure neural-policy ranking. Candidate selection uses the same sensor-grounded rule but is recalculated from each trial's actual frames; exact numerical endpoints can differ slightly. The four-decision limit also favors longer trajectories over NoMaD's short executed prefix. Exploration has no binary destination, so its aggregate success is null and travel/coverage/loops are reported instead. Endpoint novelty is a coarse revisit measure, not proof of loop-free exploration.

The 45 primary trials comprise 27 conventional (including three failed opt-in rescan pilots), 11 NavDP and 7 NoMaD. All had **zero sampled contact episodes**; all recordings retained their samples. This is not a physical-robot safety certification, and short contacts between recorder samples are not excluded. Minimum clearance is sampled every 60 physics ticks against simulator geometry and is evaluator-only. The rescan pilots did not improve detour/kitchen success and are not promoted.

### Compute And Latency

| Measured on RTX 3080 | NoMaD | NavDP |
| --- | ---: | ---: |
| Real closed-loop policy predictions | 12 | 14 |
| Inference median / observed range | 0.274 / 0.222-0.479 s | 0.698 / 0.498-0.859 s |
| JSON transport round-trip median / max | 0.276 / 0.481 s | 0.730 / 0.897 s |
| Peak PyTorch allocated tensors | 93.73 MiB | 584.63 MiB |
| Highest policy process RSS | 1420.46 MiB | 2179.12 MiB |

These are small mixed-case samples, not stable tail-latency estimates. NavDP observation age reached 1.031 s and was rejected by the one-second gate. New history acquisition is included in control time but excluded from per-prediction latency; the first warmup is excluded from these inference distributions. Nomad probe used repeated saved images only and is not task evidence. Tensor allocation excludes CUDA context, driver, other apps and renderer memory; sampled whole-device allocation is recorded separately and is not a process peak. Host CPU seconds/RSS are recorded per trial, with parent versus child scope explicitly labelled. Full system/renderer peak RAM and a 5080 benchmark remain unmeasured.

## SmolVLA Responsibility

The original `luna_navigation` path remains: Luna selects one of five canonical intentions; the actual fine-tuned **450,046,176-parameter** model predicts `[linear_mps, angular_radps]` from one RGB image and 20 measured values. Its output really reaches the wheel executor after clipping/deadband/clearance checks; it is not bypassed in that mode. The current default recovery-6500 checkpoint and learned parking checkpoints were not overwritten. New conventional/NavDP/NoMaD experiments do not load SmolVLA.

Manipulation is a different seven-state/seven-action, 50-sample chunk contract (six left-arm radians plus gripper meters). Existing training froze the vision/language backbone and trained about 99.9M parameters; fine-tuning did not shrink the model or produce interchangeable task adapters. The [previous audit](ARCHITECTURE_AUDIT.md) retains measured primitive and parking results; [SMOLVLA.md](SMOLVLA.md) records unsuccessful raw/adapted pick-and-place tests. No broad claim about SmolVLA's general capability follows from those limited data.

**New manipulation diagnostic:** the existing geometric/IK demonstrator completed one full 720-frame, 36-simulated-second pick-and-place with real opposing finger contact, zero assisted attachment, 0.2013 m peak cube-bottom height, supported release and final speed 0.0000124 m/s. Artifact: `.runtime/navigation-study/geometric-manipulation/episode-000/result.json`. It uses privileged known object positions and a simulation clock: it proves mechanical feasibility, not autonomous visual manipulation or equal wall-clock performance.

A fresh GPU SmolVLA trial was **not run**: only 1,755 MiB VRAM was free at its preflight, below the prior 1,789 MiB inference allocation before safety headroom. Existing user workloads were not stopped. Thus there is no new fair manipulation superiority comparison; existing no-pickup evidence supports restricting SmolVLA to research, not deleting its checkpoint or declaring it intrinsically incapable. The most informative manipulation follow-up is a frozen held-out grasp set comparing the current checkpoint with sensor-grounded geometric control, with enough free GPU memory.

## Reproduce

The overlay installation deliberately uses `--no-deps` to preserve the existing verified CUDA PyTorch/torchvision runtime and its installed transitive dependencies.

From repository root, retain separate output directories. Code snapshots used for this study are under `.runtime/navigation-study/upstream`; exact revisions are in the manifests. Do not update them between matched runs. Recreate missing source folders with shallow clones of the primary repositories and `real-stanford/diffusion_policy`, then checkout recorded revisions. No upstream package needs installation into physics.

```powershell
uv pip install --python .runtime/smolvla-env/Scripts/python.exe --target .runtime/navigation-study/packages --no-deps -r scripts/requirements-navigation-study.txt
./.runtime/env/python.exe -m pytest -q tests/test_navigation.py -k 'navigation_backend_boundary or navigation_images_reset'
./.runtime/env/python.exe -m scripts.navigation_comparison --stage study --backend conventional --speed .2 --repeats 2 --decisions 4 --time-limit 45 --output .runtime/my-conventional
./.runtime/env/python.exe -m scripts.navigation_comparison --stage study --backend nomad --cases explore return --speed .2 --repeats 2 --decisions 4 --time-limit 45 --output .runtime/my-nomad
./.runtime/env/python.exe -m scripts.navigation_comparison --stage study --backend navdp --checkpoint .runtime/navigation-study/navdp-weights/navdp_pretrain.ckpt --cases straight doorway obstacle explore --speed .2 --repeats 2 --decisions 4 --time-limit 45 --output .runtime/my-navdp
./.runtime/env/python.exe -m scripts.record_milo --episodes 1 --output .runtime/my-geometric-grasp
```

The inference subprocess defaults to CUDA and the physics renderer defaults to TinyRenderer; use `--rendering enhanced` for a separately labelled renderer comparison. `--stage probe --backend nomad --image <paired-image> --output <new-directory>` performs no actuation. NavDP probes additionally need `--depth <paired-spatial.json>` and the checkpoint path. The public NoMaD file ID is `1YJhkkMJAYOiKNyCaelbS_alpUpAJsOUb`; download it with the isolated `gdown` helper. The public NavDP artifact is `InternRobotics/X-NavDP/navdp_pretrain.ckpt` at the pinned revision above. Check both hashes before evaluating.

The existing Luna + SmolVLA baseline remains reproducible through `scripts.evaluate_supervised --mode luna_navigation --checkpoint .runtime/navigation-recovery-6500/checkpoint` with explicit fresh output and budgets. That command invokes paid Luna inference; it was not run in this local-policy study. Original app settings and checkpoints are unchanged.

## Next Experiment And Remaining Gaps

**Single most informative next experiment:** a fixed two-room obstacle-and-return test with the same sensor-grounded waypoint graph, same 0.2 m/s tracker, and conventional versus NavDP local proposals, after isolating one simulator/renderer and measuring fresh-sensor availability. Use fixed forward camera calibration, frozen goals/starts, at least ten paired trials, and score task completion, frame age, stops and coverage. This separates a missing mission-memory/recovery layer from local-policy quality; more fine-tuning would not answer that question.

Remaining blockers: no physical drivers or calibrated localization; no persistent global place graph/uncertainty model; synchronous sensor capture and stale-map stops; no loaded-payload safety model; no reliable complex-room exploration; missing explicit weight-license confirmation; no 5080 measurement; no fresh GPU SmolVLA manipulation comparison; no production learned-backend UI/Luna integration; and no complete equal-goal NoMaD-versus-point-policy comparison. The Nav2 bridge remains experimental with previously observed TF/command-watchdog failures, not a working default. Repair its transport and time alignment only when committing to ROS hardware integration, rather than forcing this benchmark through it.

Primary sources: [NavDP code](https://github.com/InternRobotics/NavDP), [NavDP paper v3](https://arxiv.org/abs/2505.08712), [NoMaD code](https://github.com/robodhruv/visualnav-transformer), [NoMaD paper](https://arxiv.org/abs/2310.07896), [Nav2](https://github.com/ros-navigation/navigation2), [SmolVLA documentation](https://huggingface.co/docs/lerobot/smolvla), [Microduck runtime](https://github.com/pollen-robotics/microduck), [public NavDP pretraining artifact](https://huggingface.co/InternRobotics/X-NavDP).