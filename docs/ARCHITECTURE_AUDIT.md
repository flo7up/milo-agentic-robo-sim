# Robot Control Architecture Audit

For the newer conventional/NavDP/NoMaD implementation study, current hardware checks, closed-loop results and limitations, see [NAVIGATION_STUDY.md](NAVIGATION_STUDY.md). The dated audit below remains unchanged as historical Luna + SmolVLA comparison evidence.

Date: 2026-09-12. Scope: native-Windows FastAPI/PyBullet application, preserved navigation-recovery-6500 checkpoint and existing Luna Responses integration. No retraining, ROS migration, production controller replacement, or live-session interruption.

## Findings

The fine-tuned navigation policy demonstrably causes motion and follows its five primitive instructions. In 45 closed-loop tests, both the policy and a five-entry velocity lookup passed every trial; the learned version took longer. The matched Luna missions produced 0/4 physical endpoint successes with SmolVLA6500 and 1/4 with lookup. Neither variant demonstrated verified, self-terminated mission completion. These small, shared-machine trials do not establish conventional navigation superiority.

The strongest diagnosis is a narrow learned role and training objective: the supervisor already computes a direction, and the policy approximates five fixed velocity labels. Separately, the preserved parking-specific checkpoint1500 achieved 3/4 isolated visual-parking successes where6500 achieved 0/4. This supports a specialization/retention concern, not a claim that SmolVLA generally cannot navigate or produce sequences. No manipulation competence was established.

The archived baseline is `.runtime/architecture-audit-20260912/baseline/`: selected immediate source files from backend, scripts, tests, assets and frontend/src, four root configuration/instruction files, and SHA256 hashes. It excludes secrets, weights, datasets, documentation, frontend tests and nested assets; it is not a complete environment archive. Each experimental batch has source hashes and, for learned runs, a checkpoint hash. Final comparison found only tests/test_recording.py changed among baseline files. Existing production code, datasets and checkpoints were not edited. The default checkpoint SHA256 is `b1e5f8f533bd0f2a982ee5bf39b12bef5ab98953f42013d17d3b1abeaff9c08f`.

## Actual Responsibilities

| Component | Current responsibility | What it does not do |
|---|---|---|
| Luna in `luna_navigation` | Interprets goal; reads current and initial RGB, AgentObservation, recent execution history and episode memory; selects primitive, pixel/marker/relative target, steps, head angles, assessment | Does not generate wheel speeds; does not establish objective success by assertion |
| Navigation supervisor | Projects RGB floor pixels using nominal camera geometry, detects colored markers, retains target and encoder breadcrumbs; converts target error into five primitive labels | Does not send numeric target to the live SmolVLA policy; has no route planner around obstacles |
| Navigation SmolVLA | Current paired RGB + text primitive + 20-value sensor state -> two body-velocity targets | No map, spatial memory, mission state, explicit target vector, arm commands, or multi-action chunk |
| NavigationRuntime / worker | Exclusive validated actuation; ticket/revision checks, clipping, acceleration, braking, geometric collision vetoes, deadlines, Stop | Not evidence the neural policy learned obstacle avoidance |
| `luna_continuous` | Luna chooses observed depth-map candidates; conventional planning/tracking follows paths and replans locally | SmolVLA is disabled; not a matched policy-only ablation because sensors/prompts/actions/posture differ |
| SpatialMap / NavigationMemory | Depth-derived local occupancy and wheel-odometry routes; bounded positions, view sectors, failed actions, labelled model notes | No SLAM loop closure, semantic place graph, uncertainty-calibrated localization, persistent cross-reset memory or autonomous frontier executive |
| Challenge evaluator | Physical task predicates from simulator geometry, support, rest and ordered progress | Not available as a policy observation; privileged evaluation is not hardware perception |

Sources: [navigation supervisor](../backend/navigation_supervisor.py), [local policy client](../backend/local_navigation.py), [worker](../backend/worker.py), [continuous supervisor](../backend/continuous_supervisor.py), [spatial map](../backend/spatial.py), [episode memory](../backend/navigation_memory.py).

### Instruction To Actuation

1. AgentStart selects `luna_navigation`; AgentController checks configuration and acquires a local resident session. A hold inference warms the model but does not execute motion.
2. Luna receives the user goal separately from sensor/history input, returns validated `GuideNavigation`. Scene text is untrusted. Look angles only directly govern the head; movement centers it at yaw 0 / pitch 0.45.
3. The executive begins a bounded local subgoal and executes a geometric head segment. For a grounded target, `intention_to_target()` calculates a discrete forward/backward/left/right/hold intention and 0.5 or 1-second duration from encoder odometry.
4. The policy receives a fresh paired image/observation and the corresponding fixed English primitive. It never receives the full mission goal in this supervised loop.
5. Inference postprocessing denormalizes two outputs. `bounded_velocity()` clips to +/-0.15 m/s and +/-0.5 rad/s. The supervisor applies the joint deadband `abs(v)<0.025 AND abs(w)<0.05`, replacing both with zero. It does not substitute the teacher action when prediction fails.
6. `replace_motion_buffer` carries run/epoch/observation/revision/action IDs. The worker checks validity, ownership and clearance. Acceptance is queue acceptance, not completed execution; the pre-existing trace field named `executed_action` actually describes this requested target.
7. At 240 Hz simulated physics, acceleration limits 0.3 m/s^2 and 1 rad/s^2 plus braking-to-buffer-end modify the target. Wheel targets are `(v - 0.19*w)/0.09` and `(v + 0.19*w)/0.09` rad/s, applied by Bullet velocity motors with force limit 5. Runtime safety is checked every 12 ticks; no magic base teleport is used.
8. The buffer is drained before the next policy inference or Luna call. Measured odometry and execution status return to Luna. Normal inference does not run concurrently with another model's actuator commands.

### Exact Navigation Contract

`observation.images.head`: one RGB camera, 320x240 bytes decoded to float32 [0,1], CHW (3,240,320), model resize-with-padding 512x512. No camera-order ambiguity because there is one key. `task`: primitive text. `observation.state`: ordered 20-vector:

```text
head_yaw_rad, head_pitch_rad, wheel_linear_mps, wheel_angular_radps,
distance_front_m, distance_front_left_m, distance_left_m, distance_rear_left_m,
distance_rear_m, distance_rear_right_m, distance_right_m, distance_front_right_m,
range_status_front, range_status_front_left, range_status_left, range_status_rear_left,
range_status_rear, range_status_rear_right, range_status_right, range_status_front_right
```

Range hit uses measured meters, clear=2, occluded=0; status hit=1, clear=0, occluded=-1. Wheel velocities are converted with radius 0.09 m and separation 0.38 m. Head angles are radians; positive yaw is left. Outputs `[linear_mps, angular_radps]` are body-frame forward/left-turn velocities, not wheel speeds, positions, joint targets, deltas, or end-effector commands. State/image normalization is identity after pixel scaling; actions use training-only mean/std and inverse postprocessing. Checkpoint shape is [1,1,2], chunk_size=n_action_steps=n_obs_steps=1, float32 inference with 10 flow steps; policy state is reset on every prediction. See [encoding/inference](../scripts/navigation_policy.py#L20) and [training](../scripts/train_milo.py).

AgentObservation itself additionally contains run/epoch/sequence/timestamps/frame reference, all measured joint positions/velocities, gripper aperture/contact/load, odometry, bumpers, battery, proximity and runtime feedback. The navigation tensor drops most of these. The local ticket validates the observation identity; it cannot cryptographically prove external hardware image synchronization.

### Simulation And Hardware Gap

There is no physical robot driver in this execution path. RGB/depth, encoders, head position, battery and bumper/gripper readings have hardware analogues, but simulated readings omit real calibration drift, depth artifacts, latency and tactile uncertainty. Wheel odometry is not perfect world localization, and maps use flat-floor/nominal-height assumptions.

**Critical:** `NavigationRuntime.check_clearance()` calls `sim._sync_planner()` and checks the complete simulator collision world with `getClosestPoints`, including unseen obstacles. Tilt and floor-support checks also use simulator truth. Both experimental arms retain the SAME shield. This is a privileged safety baseline, not realistic perception-only collision avoidance. No hidden map is sent to Luna or SmolVLA, but their action acceptance depends on privileged geometry. Hardware would need an observed collision model and conservative unknown-space handling, not removal of the safety shield.

## Competing Explanations

| Finding | Evidence and confidence | Smallest discriminating experiment |
|---|---|---|
| Role mismatch dominates the current division | High: live policy receives fixed primitives; geometric code already computes target direction; no mission/target/history in tensors | Matched lookup substitution under unchanged Luna prompt/worker; compare physical missions |
| Primitive training succeeds but recovery label is overstated | High: recovery-v1 = 16 episodes x44 frames, 12 train /4 validation; identical 11-step sequence repeated four times; each instruction always has one fixed action | Vary obstacles requiring different safe actions under same goal; compare learned adaptation before shield intervention |
| Split is disjoint by episode but weak for generalization | High: close poses in the same two layouts, fixed objects/lighting; variant0 here is reserved recovery pose12, not a new environment | New layouts with disjoint geometry and randomized initial headings, fixed beforehand |
| Action integration gross mismatch is unlikely | High for primitive signs/units: all45 physical skill tests pass, shared training/deployment encoding and motor conversion | Tensor/action golden sample plus saved demonstration replay; long-horizon accuracy can still fail |
| Timing adds structural pauses | High: one-second labels and execution, but inference is serialized between drained buffers; observed policy median0.813s; Luna adds further waits | Compare sampled wheel duty cycle and time under identical decisions, lookup vs policy |
| No ordinary Luna/Smol actuator race identified | Medium: live-identity checks and resident serialization; all16 delayed-reply cancellation cases and both live-instruction cases passed in this audit | Broader real-load interruption testing remains useful; scripted race checks do not prove every scheduling interleaving |
| Task adherence can be misread | High: navigation supervisor sets outcome completed from guide.status even if local_model.success is false; evaluator remains separate | Force favourable model completion at start; assert physical failure and count false report |
| Conventional continuous alternative has its own limitations | High from code: nominal camera geometry, finite local grid/horizon, heuristic colors/finish guard, domain-specific prompt, no systematic frontier search | Fixed room-search suite; separately test perception grounding, route execution and finish validity |

No evidence yet justifies broader retraining, a larger policy, or a wholesale rewrite. Novel waypoint checkpoints were trained offline with relative-waypoint text but are not the live6500 primitive policy. Their results cannot be attributed to this live architecture without an explicit checkpoint/adapter comparison.

## Diagnostic Protocol And Results

All counts below are new experiments from this audit unless marked historical. Raw artifacts live under `.runtime/architecture-audit-20260912/`. No cloud model is used by primitive tests; they use real checkpoint inference and real physics. Ground-truth pose and challenge status are evaluator-only.

### Isolated Primitives And Instruction Sensitivity

Five feasible, training-represented instructions, each from the SAME initialized scene; two closed-loop one-second actions, with a fresh image/state before each. Seeds716/717/718 reset Torch sampling per trial. Three variants: reserved pose[-0.25,-0.1], modest +0.06m x/y and +0.1rad initial head pitch, and shifted pose with an additional off-path0.16m obstacle. Preparation centers head pitch0.45, so the head-pitch perturbation is erased and NOT evidence for camera-pitch robustness. The obstacle does not demand avoidance; this is modest visual/pose shift, not a recovery benchmark.

Postconditions were defined before runs: signed translation>0.12m, lateral error<0.1m and yaw error<0.25rad; turns>0.35rad with translation<0.06m; hold translation<0.02m and yaw<0.08rad; no observed contacts or safety failure. This tests intended direction and execution, not precise goal tracking.

| Source | Passes | Median wall time per trial | Range | Local inference median/p95 |
|---|---:|---:|---:|---:|
| SmolVLA6500 |45/45|3.937s|3.672-4.672s|0.813/1.078s,90calls|
| Fixed primitive lookup |45/45|2.219s|2.125-3.234s|Below timer resolution,90lookups|
| Zero-action ablation, nominal scene |1/5|2.172s|2.125-2.204s|No model|

Zero succeeds only on hold; all movement postconditions fail. SmolVLA therefore contributes causal motion and instruction sensitivity, but no measured advantage over lookup on these tasks. Repeated seeds are not45 independent environments. Timing includes shared-machine contention and first-inference overhead; the policy/lookup median difference is1.718s for two motions, not a mission speedup estimate.

```powershell
./.runtime/env/python.exe -m scripts.audit_architecture --stage primitives --source smolvla --output .runtime/audit-new-primitives-a
./.runtime/env/python.exe -m scripts.audit_architecture --stage primitives --source lookup --output .runtime/audit-new-primitives-b
./.runtime/env/python.exe -m scripts.audit_architecture --stage primitives --source zero --seeds 716 --variants 0 --output .runtime/audit-new-zero
```

### Matched Mission Ablation

A uses the current Luna+SmolVLA6500 architecture. B0 replaces only the injected local policy with the exact five-entry teacher velocity lookup. Same Luna deployment/prompt/tools/reasoning(high), goal, scene, RGB/proprioception, head behavior, geometric target controller, runtime limits and180s mission budget. Smol weights preload outside the budget, as expected for residency. No cloud seed control; local seed is recorded. Neither arm gets a perfect map as input; both share the privileged safety shield described above. The preserved prompt still names SmolVLA in B0, intentionally holding semantics fixed; source-labelled audit logs are authoritative.

This B0 is a minimal conventional-skills ablation, NOT the richer existing B1=`luna_continuous` system. B1 changes depth access, planning, compact posture and prompt; compare it as a system package, not as causal evidence about removing SmolVLA.

Two repetitions per scene, labelled716/717. Reported turns include a dispatched request even when its reply is cancelled. Physical success below is the final evaluator predicate, not Luna's assessment. Accepted/requested counts refer to drive buffers and include holds.

| Variant | Scene / repetition | Physical endpoint success | Wall seconds | Travel meters | Drive buffers accepted/requested | Sampled nonzero drive, simulated seconds |
|---|---|---:|---:|---:|---:|---:|
| A: Luna +6500 | Park /716 | No |180.046|2.029|44/44|36.50|
| A | Park /717 | No |180.234|2.169|54/54|43.50|
| A | Kitchen /716 | No |180.063|0.996|36/37|28.85|
| A | Kitchen /717 | No |180.250|1.502|43/44|40.05|
| B0: Luna +lookup | Park /716 | No |30.578|0.000006|0/0|0.00|
| B0 | Park /717 | Yes, at deadline stop |180.078|1.916|59/59|47.00|
| B0 | Kitchen /716 | No |180.141|1.974|62/62|54.00|
| B0 | Kitchen /717 | No |180.203|1.545|54/54|49.15|

Seven runs ended at the evaluation time limit. B0 Park716 ended early with the controller's generic "Inference or session timed out" error after three returned Luna decisions and no drive request; the saved trace does not identify the underlying timeout operation. It is retained in the denominator, not discarded or credited as fast completion. B0 Park717 still issued movement decisions and ended as `limited`, so its physical endpoint success is not evidence of reliable arrival recognition or autonomous termination. All eight report zero manual placements and zero false *terminal completion* claims. This does not grade every intermediate semantic assertion. No drive contact samples or final contacts were recorded; sampling and the privileged shield limit that safety claim.

Both A kitchen runs had one command rejected for clearance. B0 Kitchen717 had a later whole-robot clearance stop despite all54 commands being initially accepted. Observed feedback revisions containing buffer-expiry reasons numbered6/8 for A kitchen716/717 and1/1 for B0 kitchen716/717. These are deduplicated feedback revisions, not an exhaustive count of independent stop episodes. Normal buffer exhaustion is the deliberate drain-and-brake execution pattern and is distinct from wall-clock expiry.

Per-run Luna response medians were4.523-5.531s in A and4.172-5.070s in B0; per-run p95 values ranged8.734-21.843s and6.891-11.547s respectively. A local-policy medians were0.828-0.875s. A produced85 clipped proposals, including any warmup proposals, and eight recorded deadband overrides; B0 produced neither. Raw proposals, actual sampled wheel rates, runtime targets and odometry remain separate in the logs. These data show inference/actuation overhead and some timing loss, but differing decisions and trajectories prevent a causal mission speedup estimate.

Provider-reported usage: A380,893 input /24,531 output tokens; B0405,691 /25,995. Cancelled calls may have unreported usage. Deployment was `gpt-5.6-luna`; underlying model identity and monetary cost were not independently verified. Cloud sampling is uncontrolled, order was A716, B0716/717, A717, and the GPU was shared with untouched user processes. Cold loading was unusually slow in the first A batch and excluded from mission budgets; residency was reused across scenes in the second A batch. This is a diagnostic comparison, not a statistically powered or hardware-real-time benchmark.

Artifacts: [A716](../.runtime/architecture-audit-20260912/mission-a-716/summary.json), [A717](../.runtime/architecture-audit-20260912/mission-a-717/summary.json), [B0](../.runtime/architecture-audit-20260912/mission-b0/summary.json), [derived per-trial metrics](../.runtime/architecture-audit-20260912/results.json). B1 was inspected in code but not rerun in this audit. C was not benchmarked as a working system because its learned manipulation component is unqualified.

```powershell
./.runtime/env/python.exe -m scripts.audit_architecture --stage missions --source smolvla --seeds 716 717 --seconds 180 --output .runtime/audit-new-missions-a
./.runtime/env/python.exe -m scripts.audit_architecture --stage missions --source lookup --seeds 716 717 --seconds 180 --output .runtime/audit-new-missions-b0
```

### Parking Without Luna

Same two pre-existing held-out open-bay cases16/17, targets[1.05,-0.40] and[1.05,0.40], seeds716/717, maximum30 requests, unchanged60s runtime skill deadline and two consecutive deadband-stop votes. Preparation contains two scripted head scans; subsequent driving is learned. No fixture coordinates are passed to the policy. This is a task-transfer/retention comparison between checkpoints, not the primitive lookup ablation.

| Checkpoint | Case16 /716 | Case17 /716 | Case16 /717 | Case17 /717 | Final physical successes |
|---|---|---|---|---|---:|
| Recovery6500 | Fail,11 calls | Fail,2 calls | Fail,11 calls | Fail,2 calls |0/4|
| Parking1500 | Pass,18 calls | Pass,18 calls | Fail,15 calls | Pass,16 calls |3/4|

All eight ended on two stop votes, without recorded command rejection or contact.6500 travelled less than0.09m in every trial;1500 travelled0.902-0.975m. The1500 failure stopped before full containment. Its successful wall times were30.563-34.593s. Checkpoint1500 hash is `8bf69d9e08777d06a45a589b891d47a27246e1948eb774a0b951f56b35d6a119`. The difference is consistent with loss of parking specialization after primitive-only continuation; changed weights and action normalization are bundled, so it does not isolate a neural forgetting mechanism. Two familiar layouts and four trials cannot establish generalization.

Artifacts: [6500](../.runtime/architecture-audit-20260912/parking-6500/summary.json), [1500](../.runtime/architecture-audit-20260912/parking-1500/summary.json). Repeat with `--stage parking --source smolvla --seeds 716 717 --variants 0 1 --steps 30`, an explicit `--checkpoint`, and a fresh `--output` directory.

### Training Artifact Checks

The [inventory](../.runtime/architecture-audit-20260912/inventory/inventory.json) records current metadata/parquet hashes and matches all discovered checkpoint weight hashes to their reports. It verified all16 recovery-v1 raw episodes /704 frames with the existing strict loader, including image/asset hashes, dimensions and state/action schema. Exact train/validation image-hash overlap was zero. Each of its five instructions has exactly one unique action label. Similar layouts and neighboring poses still make this a weak generalization split.

| Stored dataset | Episodes | Frames | Distinct task strings |
|---|---:|---:|---:|
| milo-navigation |16|276|1|
| milo-pilot, arm |5|3600|1|
| milo-recovery-v1 |16|704|5|
| milo-recovery-v2 |16|704|5|
| milo-task-waypoints-v1 |16|1701|439|

These are stored metadata counts, not independent tasks or successful learned episodes. Only recovery-v1 received a fresh exhaustive raw-frame validation here. The training code explicitly selects episode splits even where exported metadata labels the full dataset as train. The6500 report records2000 additional updates from4500, a fresh AdamW state,528 train frames and60 sampled validation frames; validation loss fell0.05555 to0.01438. This proves optimization and reload integrity, not closed-loop recovery. Reports do not fully bind historical training to dataset content hashes; today's inventory cannot reconstruct missing provenance. The439 waypoint strings largely parameterize targets, not439 distinct mastered skills.

### Manipulation

The separate [arm adapter](../backend/smolvla_server.py) and [SkillRuntime](../backend/policy.py) implement `pick_place`: RGB plus `[left_joint_1..6, left_gripper_opening_m]` -> up to50 absolute7-value targets at20Hz. Six radians plus aperture meters, not Cartesian poses. Base/head/right arm remain fixed. Old chunk prefixes are skipped by wall age and accepted future horizon is bounded to1s; stale/invalid chunks are rejected. This is a materially different timing contract from navigation.

[Demonstrations](../scripts/record_milo.py) use privileged IK targets in an offline red-cube fixture, four training episodes and one nearby validation episode, fixed destination and fixed instruction. A single constant task string cannot establish language-conditioned manipulation. Training loss or metadata `trained_for_milo` cannot establish grasp/placement success. C must remain unqualified until an independent, contact-based held-out skill succeeds. No navigation result here is a manipulation result.

The saved1100-step arm checkpoint has no required `milo-policy.json` deployment manifest. That is an intentional qualification boundary, not a reason to manufacture one. No new arm rollout or successful grasp is claimed in this audit. Existing historical arm results are documented separately in [docs/SMOLVLA.md](SMOLVLA.md).

## Recommended Ownership

| Responsibility | Owner and interface | Required measurable contract |
|---|---|---|
| Language and semantic hypothesis | Luna proposes grounded subgoal or requests clarification | Reference observation IDs; distinguish a past sighting/hypothesis from current confirmation |
| Mission state, constraints, retries, Stop and completion | Extend existing AgentController executive, not a second actuator owner | Goal revision, bounded retries/time, explicit termination reason, independently checked postcondition |
| Spatial memory/localization | SpatialMap + NavigationMemory, incrementally extended | Pose frame/age/uncertainty, observed free/unknown, visited coverage, semantic sightings with provenance |
| Exploration | Reachable-viewpoint selector feeding current local route planner | `explore`: new observed coverage or explicit no-frontier/failure; bounded progress/time |
| Navigation | Conventional local planner/controller behind worker | `navigate_to`: observed reachable target, tolerance/rest, cancellation, blocked/stale/timeout failures |
| Inspect and search | Geometric head/turn control plus Luna perception | `inspect` supplies fresh views; `search_object` confirms current evidence, not a remembered assertion |
| Straightforward arm/head motions | Existing geometry and validated worker | Joint/Cartesian limits, contacts, timeouts, no ungrounded object coordinates |
| Learned skills | Explicit `execute_manipulation_skill` or other proven policy plugin | Qualified checkpoint, preconditions, progress, cancellation, postcondition and measured advantage against geometric baseline |

Nav2 is not needed for this Windows/PyBullet diagnostic. No ROS graph, hardware transport or Nav2 integration currently exists in this path. Reuse the local abstractions to test ownership; revisit a hardware navigation stack when a robot/ROS deployment target exists.

## Changes And Uncertainty

Added [scripts/audit_architecture.py](../scripts/audit_architecture.py), this report, and two audit checks in [tests/test_recording.py](../tests/test_recording.py). Production controllers, checkpoints and datasets remain unchanged. The runner separates raw proposals, clipping/deadband, accepted/rejected commands and sampled executed velocity/odometry; later logging adds actual wheel measurements. Pre-existing `executed_action` names are not used as proof of motion. Logs are evaluator artifacts and are never model input.

Verification:21 focused tests passed in76.10s, covering scoring,16 cancellation-resistant late-reply cases, two live-instruction/Stop cases and two physical demonstration/replay checks. After adding summary stop accounting, both audit tests passed in6.09s. [JUnit results](../.runtime/architecture-audit-20260912/regression.xml) retain the21-test run. These are scripted control tests, separate from the real-model trials above. Edited Python files have no editor diagnostics. No frontend changes, frontend suite or full backend suite rerun was needed for this audit-only slice.

Measurement limits: execution is sampled every12 physics ticks, not continuously recorded; initial primitive batches lack the later wheel/wall-time fields, and first A missions lack sampled evaluator completion. Primary trace images are saved, but not every secondary historical image batch. No exact all-tick collision count, worst-case Stop latency, calibrated hardware safety, complete energy budget or all-run time-to-first-success claim is made. Image dependence was not causally isolated by masking/shuffling; correct primitive direction does not prove vision was necessary. The runner refuses existing rollout output directories; its summarize stage intentionally regenerates only derived results.

## Decision And Next Experiment

Use conventional primitives as the engineering baseline for this five-command interface. Preserve the learned checkpoints and Luna-supervised policy path for research. Do not spend more updates teaching the same five constant labels, and do not promote the arm checkpoint or declare the richer continuous controller reliable from these results. A conventional navigation executive plus qualified learned skills is a sensible ownership direction, but C is currently a proposal with an unproven manipulation component.

The most informative next experiment is a sensor-grounded local-goal benchmark with Luna removed from the inner loop: freeze the same reachable target, starting state, observations, safety shield and budget for a conventional visual/odometric controller and a goal-conditioned learned policy. Include different required actions under the same goal, changed initial headings, off-trajectory recovery states and unseen layouts. First retain the preserved1500 parking skill as a positive control. Compare final tolerance/rest, autonomous stopping, clearance interventions, action continuity and wall time; preserve all failures. If a learned candidate cannot beat this baseline, investigate goal encoding, corrective data, normalization and temporal execution separately before further training. A chunked policy is a separate adapter/training experiment because the live navigation checkpoint has chunk size1.

In parallel with that research direction, mission reliability needs grounded arrival confirmation, bounded exploration/retry state and sensor-based navigation progress. Those changes can improve the executive regardless of which local policy wins. They were not implemented during this audit, so the comparison remains against the preserved architecture.