# Milo Use-Case Specification

Specification ID: `milo-use-cases-v1.0`  
Created: 2026-09-16  
Scope: local FastAPI/PyBullet robot, observed-map navigation, model supervision and operator cockpit.

## Purpose And Authority

This is the reference catalog for what Milo should do and how we decide whether it did it. Use its stable `UC-*` IDs in issues, tests, design reviews and evaluation reports. A use case is not an implementation claim, and passing a software test is not proof of autonomous task completion.

- This document owns use-case intent, required outcomes and traceability. It consolidates existing requirements; it does not select a new architecture or authorize implementation, training, paid inference or a live-scene reset.
- [HOUSEHOLD_BENCHMARK.md](HOUSEHOLD_BENCHMARK.md) and its frozen suite/map/scorer fingerprints own the existing numerical benchmark contract. This document does not revise that series.
- [DESIGN_PERFORMANCE.md](DESIGN_PERFORMANCE.md) owns dated measurements, changes and failures. The evidence snapshot below is dated, not a continuously updated success claim.
- [MECHANICS.md](MECHANICS.md) and [HOME_MAPPING.md](HOME_MAPPING.md) explain runtime behavior and limitations. [TASK_CURRICULUM.md](TASK_CURRICULUM.md) preserves the broader task roadmap; [UNIFIED_MISSION_PLAN.md](UNIFIED_MISSION_PLAN.md) preserves the implementation plan, not current implementation status.
- Where an acceptance profile is missing, mark qualification **Pending profile**. Define and review its thresholds before running it; do not infer success from a different scenario's score.

## Quick Reference

| Reference | Use case | Primary evidence/profile |
| --- | --- | --- |
| [UC-01](#uc-01-start-stop-and-restore-a-session) | Session lifecycle | AP-UI and lifecycle/API checks |
| [UC-02](#uc-02-reuse-and-localize-on-a-saved-map) | Saved-map localization | AP-HF1 `localize` |
| [UC-03](#uc-03-reach-an-observed-destination) | Destination and doorway navigation | AP-HF1 `navigate`, AP-DOOR |
| [UC-04](#uc-04-return-home-after-a-task) | Return Home | AP-HF1 `return_home` |
| [UC-05](#uc-05-reject-or-recover-from-a-blocked-route) | Blockage and bounded recovery | AP-HF1 `blocked_route`, AP-UNIT |
| [UC-06](#uc-06-explore-an-unknown-or-partial-map) | Local exploration | AP-HF1 `explore` |
| [UC-07](#uc-07-find-identify-and-inspect-an-object) | Object search and inspection | AP-SCENE; generic semantics unqualified |
| [UC-08](#uc-08-find-identify-and-enter-a-room) | Room identity and arrival | AP-HF1 `room_to_room`; semantic profile pending |
| [UC-09](#uc-09-stop-take-over-and-reject-late-motion) | Stop and cancellation | AP-HF1 `cancel`, AP-REVIEW |
| [UC-10](#uc-10-continue-safely-during-delayed-supervision) | Delayed supervision | AP-REVIEW, AP-UNIT |
| [UC-11](#uc-11-circle-a-selected-object) | Object circuit | AP-SCENE `furniture_circuit` |
| [UC-12](#uc-12-park-at-a-destination) | Parking | AP-SCENE `park` |
| [UC-13](#uc-13-return-to-a-charger-and-recharge) | Charging round trip | AP-SCENE `recharge` |
| [UC-14](#uc-14-yield-to-a-crossing-obstacle) | Dynamic-obstacle yielding | AP-SCENE `pedestrian_crossing` |
| [UC-15](#uc-15-pick-up-and-place-supported-objects) | Pick/place and sorting | AP-SCENE `tidy`, `sort`, `workshop` |
| [UC-16](#uc-16-redirect-a-running-task) | Live instruction changes | Instruction/API checks, mode-specific |
| [UC-17](#uc-17-sustain-a-real-model-mission-for-five-minutes) | Five-minute real-model mission | Pending profile |
| [UC-18](#uc-18-inspect-and-explain-robot-behavior) | Operator diagnostics | AP-UI and telemetry checks |
| [UC-19](#uc-19-future-task-extensions) | Figure eight, people, transport and other extensions | Pending profiles |

## Shared Contract

The following requirements apply to every use case unless the frozen fixture explicitly specifies an evaluator-only setup step.

| ID | Requirement |
| --- | --- |
| INV-01 | Only the validated worker may actuate the robot. Controllers must not compete for wheel or arm ownership. Model inference stays serialized. |
| INV-02 | Stop, takeover, reset, disconnect and incompatible instruction changes invalidate pending motion and late replies. Resume, reconnect, restoring preferences or restarting the server never resumes an old task automatically. |
| INV-03 | Fresh sensing, valid localization when required, observed whole-footprint clearance and unexpired motion authority are required for motion. A clear distance ray or a model's reassurance is insufficient. Unknown space is not free space. |
| INV-04 | Exploration-objective authorization and short motion-buffer authorization are separate. Renewal, retries and subgoal changes must not reset the original task deadline or revive an expired command. |
| INV-05 | Model input is limited to AgentObservation and its permitted paired head-camera/history/map imagery. Map imagery may render only the typed observed-map snapshot. Scenario-derived room labels, hidden target coordinates and spectator/evaluator geometry are not model evidence. The privileged simulator collision shield remains a separate safety backstop. |
| INV-06 | Target identity, physical arrival and full-task completion are distinct claims. Tentative sightings, confidence values and controller completion messages are not independent verification. A target must not be silently replaced by an easier one. |
| INV-07 | Skipping map reuse preserves previous maps. Draft observations, reviewed labels and saved geometry remain distinguishable. Loading a saved map requires fresh localization before map-frame motion. |
| INV-08 | Freeze time, travel, request and token budgets before an evaluation. A token threshold is checked against reported usage after a response and may be crossed by that response; it is not an exact billing cap. Missing usage is unknown, not zero. |
| INV-09 | Every stopped or rejected operation must retain its actual reason and available evidence. Separate a local route failure from the final mission outcome. Timing-only retries preserve the selected frontier within the existing bounds; this does not force another attempt after authority expires. |
| INV-10 | Simulation time measures physical execution/dwell; process monotonic time measures local deadlines; Unix time correlates records. Never mix clocks or compare process-monotonic timestamps across restarts. Sensor capture age, receipt age and accumulated-map integration age have different meanings. |
| INV-11 | Every evaluation retains failures, timeouts, prerequisite blocks, assistance and missing evidence. No altered fixture, hidden teleport, cherry-picked retry or unrelated score may turn a failed task into a pass. Simulation success does not certify real hardware safety. |

## Evidence And Outcomes

Evidence must be labelled independently of the requested use case:

| Evidence class | Establishes | Does not establish |
| --- | --- | --- |
| Unit contract | Deterministic decision, boundary or validation behavior with controlled inputs | Physical success, sensor accuracy or real-time scheduling |
| Scripted physical feasibility | Real simulated actuation can complete supplied commands/routes, sometimes using known geometry | Perception, autonomous route selection or model competence |
| Scripted controller evaluation | Sensor-driven controller completes a frozen task; semantic replies may be scripted | Real-model understanding or inference-load reliability |
| Real-model evaluation | A specified model/controller/configuration attempted the task using permitted inputs | General reliability outside the frozen scenarios/repetitions |
| Operator-assisted evaluation | Measured behavior with all interventions recorded | Unassisted autonomy |

Each trial has two independent results: **task outcome** and **safety response**. Stopping correctly on sensor loss can pass a safety test while the navigation task fails. A deliberately timed exploration trial can pass its coverage criteria while its controller reports budget exhaustion.

Record the provider, configured deployment, verified underlying model identity when available, and assistance separately. A deployment name alone does not establish model identity. Keep mechanical grasp assistance distinct from operator intervention.

Use the existing report outcomes: **Passed**, **Failed**, **Blocked prerequisite**, **Invalid evidence**, **Not run**. An obstacle encountered during execution is not a missing prerequisite. Report partial milestones without calling the full instruction complete. A renderer or worker failure after launch remains a failed attempt, even if its stop was correct.

## Acceptance Profiles

### AP-HF1: Frozen Household Foundation

Existing, unchanged profile: `household-foundation-v1`. Three initial x offsets (0, +0.10, -0.10 m), Shared Apartment V1, built-in mapped controller, enhanced renderer, frozen source map and independent scorer. Setup has a separate 60-second budget. All passes require complete recording, no sampled contact episodes or manual relocation, unchanged source and unchanged original map.

| Case key | Use case | Task budget | Existing acceptance criteria |
| --- | --- | --- | --- |
| `localize` | UC-02 | 20 s | Hinted scan localization within 0.15 m and 0.15 rad of evaluator pose; at most 0.03 m wheel travel. |
| `navigate` | UC-03 | 60 s | Reach the fixed observed checkpoint within 0.15 m and remain grounded/stable for at least 0.5 simulated seconds. |
| `return_home` | UC-04 | 90 s total | Both checkpoint and Home meet the arrival/dwell checks under the same budget. |
| `blocked_route` | UC-05 | 20 s | Fresh evidence rejects the occupied checkpoint; at most 0.05 m travel and no contact. This is rejection, not detour qualification. |
| `explore` | UC-06 | 30 s | At least 0.5 m actual travel and 1 m2 added observed free area; empty buffers at task end. Existing final acknowledgement allowance does not extend exploration. |
| `cancel` | UC-09 | 30 s | Trigger Stop after at least 0.10 m travel; acknowledge within 0.5 wall seconds; freeze ticks/buffers for 0.75 s; reject old commands after Resume. |
| `room_to_room` | UC-08 | 90 s total | Two saved named rooms at least 2 m apart; independently verify arrival/dwell at both. Missing annotations remain blocked prerequisites. |

Full protocol and immutable input identities: [HOUSEHOLD_BENCHMARK.md](HOUSEHOLD_BENCHMARK.md). Do not establish a new baseline or alter its map to remove blocked prerequisites without explicitly creating a new series.

### Other Existing Profiles

| Profile | Binding and limits |
| --- | --- |
| AP-DOOR | Existing doorway fixture in [test_home_mapping.py](../tests/test_home_mapping.py), `test_local_doorway_route_rounds_observed_corner_without_luna`: three fixed offset/heading approaches, 45 s task budget, endpoint error at most 0.15 m, travel over 1.5 m and no contacts. Separate Stop and closed-door rejection variants. Scripted destination, not semantic navigation. |
| AP-REVIEW | Existing six variants of `test_delayed_review_keeps_one_enhanced_physics_task_and_revokes_late_motion` in [test_mission_review.py](../tests/test_mission_review.py): continue, Stop, objective expiry, stale timestamp, actual sensor loss and blocked path. The phase-aligned scripted trigger does not replace a natural-cadence endurance trial. |
| AP-SCENE | Existing scenario-specific scorers in [challenges.py](../backend/challenges.py) and physics checks in [test_challenges.py](../tests/test_challenges.py). Freeze the chosen scenario/version/parameters and its actual scorer thresholds before testing. A mechanical demonstration and a real-model attempt remain different evidence classes. |
| AP-UNIT | Decision and invariant tests in [test_navigation.py](../tests/test_navigation.py), including `continuous_policy`, and objective/worker tests. Validate velocities, guards and bounded recovery, not full task competence. |
| AP-UI | Existing [frontend tests](../frontend/tests) with real isolated physics and scripted inference. Covers operator controls, disclosures, sensor/image provenance and zone rendering. It does not invoke the configured real model. |

## Use-Case Catalog

Every entry defines the actor's goal, prerequisites/trigger, expected flow, acceptance and required negative cases. The shared contract applies throughout. Stable IDs describe outcomes, not a particular UI button, planner or model.

### UC-01: Start, Stop And Restore A Session

**Goal:** the operator loads a scene, enters an instruction and starts one bounded mission.

**Preconditions/trigger:** compatible backend, available sensors/controller and required model configuration; explicit Start. Map reuse or a fresh draft is selected deliberately.

**Flow:** validate configuration and budgets, prepare sensing/posture, establish the required map frame, then execute. Scene selection alone, preference restore or model residency must not initiate work.

**Acceptance:** one motion owner and one mission; the selected scene/goal/settings are respected; preparation failure is reported; Stop is reachable throughout. Reconnect/restart restores settings or a fresh episode without motion/inference. Before a deliberate server restart, preserve the current run/draft where supported. Use AP-UI and lifecycle/API tests.

**Negative cases:** unsupported controller/backend, missing configuration, failed renderer, Stop during preparation, reload/disconnect and stale-episode commands.

### UC-02: Reuse And Localize On A Saved Map

**Goal:** use prior observed geometry without treating the saved pose as current truth.

**Preconditions/trigger:** matching saved-map identity/frame; fresh scan; explicit reuse/localization request.

**Flow:** load without overwriting, estimate pose from scan evidence, validate the result, and offer map-frame destinations only with valid localization.

**Acceptance:** AP-HF1 `localize`; preserve geometry and distinguish scan-based correction from ongoing fixed-pose consistency. The seeded drift diagnostic must not be reported as global recovery or continuous SLAM.

**Negative cases:** wrong map/frame, ambiguous or missing returns, stale scan, discontinuous odometry, lost localization and declined map reuse.

### UC-03: Reach An Observed Destination

**Goal:** the operator or supervisor selects a sensor-supported destination and the local controller reaches it.

**Preconditions/trigger:** reachable observed destination, current footprint, fresh required sensors and valid authority. A model target uses an offered or validated observed reference, not evaluator coordinates.

**Flow:** validate the footprint corridor, follow a bounded route, slow/turn as needed, and verify stopped arrival. Preserve useful motion through compatible updates; do not stop merely because another model review is pending.

**Acceptance:** AP-HF1 `navigate` for mapped checkpoints; AP-DOOR for the corner/doorway variant. Report travel, arrival error/dwell and every intermediate stop/recovery. Passing a recovered route is not uninterrupted-motion qualification.

**Negative cases:** occupied/unknown corridor, unavailable depth, changed posture, expired buffer, unreachable goal, premature arrival at a loop start and sharply incompatible continuation.

### UC-04: Return Home After A Task

**Goal:** return to the previously established Home reference after the requested work.

**Preconditions/trigger:** Home belongs to the current map/frame; outward subgoal completed or explicitly abandoned; return requested in the instruction.

**Flow:** retain Home and task history, navigate back under the original total budget, verify arrival and stopped dwell.

**Acceptance:** AP-HF1 `return_home` for the fixed two-leg route. Composite missions also require their outward task's identity/inspection receipt. Returning Home does not retroactively complete an unsuccessful search, and Home arrival is not charging success.

**Negative cases:** invalid localization, blocked return path, insufficient remaining budget, wrong Home reference and completion after only one leg.

### UC-05: Reject Or Recover From A Blocked Route

**Goal:** avoid unsafe motion without declaring all routes impossible after one failed attempt.

**Preconditions/trigger:** new obstacle, missing corridor support or route-execution failure.

**Flow:** brake when required; record the initiating guard and measurements; revalidate a bounded alternative when allowed. Timing-only failures retain the frontier/target within the existing retry budget. Geometric rejection may invalidate that route; it must not be misreported as a sensor or model failure.

**Acceptance:** AP-HF1 `blocked_route` proves safe rejection; AP-UNIT and AP-DOOR cover bounded retries and retained target/deadline. A new detour capability needs its own frozen destination, layout and evaluator profile.

**Negative cases:** repeating an expired buffer, unlimited retries, silently changing the target, treating a single clear beam as a free corridor and turning a safe abort into task success.

### UC-06: Explore An Unknown Or Partial Map

**Goal:** obtain useful new sensor coverage while making bounded physical progress.

**Preconditions/trigger:** local-only exploration or a mission-authorized exploration objective; fresh sensing; valid draft or localized map. Geometry exploration must not require Luna inference for every local update.

**Flow:** choose observed reachable frontiers, execute, integrate accepted observations, retain attempted frontiers and review on relevant changes. Stop at the declared budget or safety boundary; saving remains explicit.

**Acceptance:** AP-HF1 `explore`. Record added observed area and actual travel separately. Coverage is not map accuracy, room identification or proof the entire home was searched. An object/room search still needs UC-07/UC-08 evidence.

**Negative cases:** no reachable frontier, stale sensing, repeated timing failures, localization loss, retry exhaustion and accidental saved-map overwrite.

### UC-07: Find, Identify And Inspect An Object

**Goal:** locate the requested object, approach a suitable observed viewpoint and verify inspection.

**Preconditions/trigger:** target description, ambiguity policy, search budget and frozen fixture/scorer. Optional return invokes UC-04.

**Flow:** search from permitted observations; retain tentative image-backed sightings; reject decoys; choose a depth/footprint-validated approach; obtain fresh stationary evidence. Clarify ambiguity rather than substituting a target.

**Acceptance:** score **discovery**, **identity**, **approach** and **inspection** separately. AP-SCENE apartment/gallery search requires the correct visible target within 0.9 m and a stationary one-second inspection. The generic object-approach contract additionally uses its own chosen standoff/facing/arrival rules in [HOME_MAPPING.md](HOME_MAPPING.md#object-approach). Neither scorer alone proves arbitrary semantic recognition.

**Negative cases:** wrong box/object, occluded or historical-only target, failed approach, false completion, disappeared target and budget exhausted without a find. A bounded failed search does not prove absence.

### UC-08: Find, Identify And Enter A Room

**Goal:** identify a requested room from observed evidence, enter it fully and optionally return.

**Preconditions/trigger:** room description or saved named-room reference, search budget and valid localization where required. Saved room-to-room trials need distinct real annotations; an unseen-room search is a different profile.

**Flow:** explore, report a tentative camera-backed room hypothesis, validate its identity separately, traverse a supported doorway, then verify full entry and stopped dwell. Save room evidence with frame/pose/time/confidence and review status.

**Acceptance:** report **discovery**, **identity** and **physical arrival** independently. AP-SCENE governs physical Kitchen/Bathroom arrival; its geometry scorer does not grade the language report. AP-HF1 `room_to_room` covers only the named-map workflow. A full semantic-room qualification profile is still pending and must define independent identity assessment before evaluation.

**Negative cases:** generic fixture misidentification, doorway-only arrival, wrong room, missing named-room prerequisites, stale hypothesis, ungrounded pose and completing before a required return.

### UC-09: Stop, Take Over And Reject Late Motion

**Goal:** the operator can halt or replace control without a delayed reply restarting it.

**Preconditions/trigger:** Stop, takeover, reset, disconnect, power-off or a superseding incompatible instruction at any phase.

**Flow:** revoke authority, discard pending commands, brake/hold, publish the reason and retain diagnostic history. Keep old replies invalid even after Resume.

**Acceptance:** AP-HF1 `cancel`, AP-REVIEW and lifecycle/API tests. Measure actual Stop acknowledgement separately from overall task time. Verify during preparation, sensing, inference, motion and arrival inspection. Power-on/reconnect requires a new explicit start.

**Negative cases:** cancellation race, queued old commands, stale scene IDs, retry after revocation and hidden auto-resume. Passing one injected phase does not qualify every interruption path.

### UC-10: Continue Safely During Delayed Supervision

**Goal:** allow useful local motion while a semantic review is pending, without unbounded autonomy.

**Preconditions/trigger:** an active compatible exploration objective, fresh footprint corridor and live trajectory authority; serialized model review is pending.

**Flow:** the local controller refreshes bounded motion independently; review triggers do not automatically become stop commands. Apply only a still-valid reply. Stop for actual safety/authority expiry or an action that requires a stopped view.

**Acceptance:** AP-REVIEW verifies same-task continuation and the five invalidation variants; AP-UNIT covers exact timing boundaries. Record moving/holding/inference intervals, accepted/refused renewals and sensor age. Preserve safety even when the movement objective fails.

**Negative cases:** sensor loss, path obstruction, objective or buffer expiry, Stop during review, late target selection and overlapping model requests. Natural-cadence endurance remains UC-17, not an inference from these unit tests.

### UC-11: Circle A Selected Object

**Goal:** circle the correct observed object in the requested direction, then stop.

**Preconditions/trigger:** object identity, clockwise/counterclockwise direction, lap count, permissible clearance band and budget fixed by the selected profile.

**Flow:** identify, approach the permitted circuit region, maintain the same target through bounded arcs/continuations, complete the directed lap and verify stopped closure.

**Acceptance:** AP-SCENE `furniture_circuit`: correct object/direction, measured winding and translation, radius-band compliance, lap closure and grounded stationary finish for 0.5 simulated seconds. The existing five-run table/clockwise curriculum profile remains specific to its frozen controller/configuration. Known-center tests starting on the annulus do not qualify model identification or approach; historical passes do not automatically qualify current code.

**Negative cases:** spin in place, wrong object/direction, partial lap, early loop-start completion, contact and invalidated observed arc.

### UC-12: Park At A Destination

**Goal:** place the base and wheels fully within a permitted parking area and remain at rest.

**Preconditions/trigger:** a visible/supported parking target with the selected AP-SCENE boundaries and dwell rules; valid approach corridor.

**Flow:** approach, reduce speed, verify full containment, floor support and stationary finish. Apply the selected controller's additional footprint/inspection rules where relevant.

**Acceptance:** AP-SCENE `park` or the parking stage of the selected task. Nearness, touching a boundary, passing through the bay or a model's report is insufficient. Home/object-view arrival must not be silently graded with a different parking contract.

**Negative cases:** partially contained base/wheels, motion during dwell, ungrounded pose, incorrect bay and stale completion after leaving.

### UC-13: Return To A Charger And Recharge

**Goal:** remember an observed charger, perform the requested outward work, return and verify charging.

**Preconditions/trigger:** known charger evidence, supported dock, battery reserve and frozen departure/survey/charging criteria.

**Flow:** establish charger memory, depart, perform the outward stage, return after the required battery condition, park and remain stationary while charge increases.

**Acceptance:** AP-SCENE `recharge`: departure/survey and low-battery return requirements, correct-pad parking and charge reaching at least 90%. Remaining on the charger or completing only the survey is not full success.

**Negative cases:** wrong/unknown dock, moving or partial containment on the pad, depleted battery, no charge increase and failure of the return leg.

### UC-14: Yield To A Crossing Obstacle

**Goal:** avoid contact, wait when necessary, then continue only when a fresh route is supported.

**Preconditions/trigger:** frozen route and crossing event with feasible stopping space; unchanged safety margins and a declared observation model.

**Flow:** the local shield brakes independently of model latency; hold/wait, reacquire evidence and validate any continuation. Persistent blockage stays an explicit unfinished task.

**Acceptance:** AP-SCENE `pedestrian_crossing` distinguishes yielding, crossing and final parking. Buffered-braking tests verify a narrow no-contact response at specified speeds; a full real-model task also needs renewed route selection and final parking. Physics freezes during idle inference/Stop in this simulator; do not claim real-time pedestrian prediction from those pauses.

**Negative cases:** stale observations, abrupt intrusion, contact, person moving into a stopped robot, old obstacle evidence and resuming the old route without revalidation.

### UC-15: Pick Up And Place Supported Objects

**Goal:** genuinely grasp, lift, move and release a supported object at its designated destination.

**Preconditions/trigger:** supported shape/mass, reachable pose and arm configuration, empty compatible gripper and the AP-SCENE target zone. Autonomous selection needs separate UC-07 evidence.

**Flow:** approach/grasp, verify retention/lift, transfer, release and verify settled full containment. Two-object sorting retains both independent subgoal states.

**Acceptance:** AP-SCENE `tidy`, `sort` or `workshop`; pushing-only, hovering or wrong-color placement cannot pass. Disclose grasp assistance and known-geometry commands. These stationary manipulation fixtures do not qualify carrying a load while navigating.

**Negative cases:** failed grasp, dropped object, collision, unsupported object, incorrect destination, partial containment and claiming success before release/settling.

### UC-16: Redirect A Running Task

**Goal:** apply the operator's newest instruction without losing relevant history or allowing an old reply to overwrite it.

**Preconditions/trigger:** active compatible session and a new instruction carrying current episode/session identity.

**Flow:** acknowledge the change, establish new instruction authority, retain applicable memory and consumed budget, revoke incompatible motion/replies, then replan. Exact Stop commands are handled locally.

**Acceptance:** existing instruction/API tests; declare the controller mode. Generic-route mode may preserve compatible bounded motion; legacy modes can stop before replanning. Test newest-instruction precedence and draft retention without promising rolling redirection in every mode.

**Negative cases:** wrong session, old model reply, concurrent Start, reset during redirect, budget reset and silently retaining incompatible old constraints or goals.

### UC-17: Sustain A Real-Model Mission For Five Minutes

**Goal:** useful supervised motion over a five-minute mission from several starting positions, without stopping merely because review is pending.

**Preconditions/trigger:** a separately approved, frozen real-model profile: task, map, starts/headings, repetitions, budgets, latency/workload and independent scorer. This profile is **Pending profile**, not a change to AP-HF1.

**Acceptance target:** 300 seconds of bounded mission execution with useful progress and correct safety/authority stops, no stale-command execution and no contacts. Record independently verified early task completion separately; it does not demonstrate five-minute endurance. Numeric travel/coverage, permitted stationary share and required repetition matrix must be fixed before qualification. Existing request/token limits are not automatically raised to make a trial last longer.

**Negative cases:** renewal starvation, sustained no progress, repeated fresh model requests after the same unresolved timing fault, budget exhaustion without completion and combining separate short runs into a five-minute claim.

### UC-18: Inspect And Explain Robot Behavior

**Goal:** the operator can understand what was sensed, proposed, executed and stopped without confusing model narration with controller evidence.

**Preconditions/trigger:** a live or archived run; diagnostic fields may be absent in old recordings.

**Flow:** inspect compact activity, expand exact inputs/results, view current policy/objective and buffer histories, filter recorded failures, and copy evidence. Camera/minimap default to thumbnails; instructions/settings can collapse without losing drafts or hiding Stop. Movement zones are opt-in operator geometry, not motion permission.

**Acceptance:** AP-UI plus telemetry tests: separate live versus historical sensor ages, model versus controller reasons, objective versus buffer authority, missing versus zero values and independent task outcome. Show unknown space distinctly and stale/foreign-pose zones as unavailable. Diagnostic viewing must not issue motion/model calls, expose spectator data to the model or add a duplicate sensor poll.

**Negative cases:** stale green clearance, mismatched scene/frame, overwritten stop evidence, misleading clocks, copied secrets, hidden controls, clipped mobile text and model prose counted as a recorded controller failure.

### UC-19: Future Task Extensions

These retain the curriculum's intent but have **Pending profile** status. They are not enabled or qualified by this document.

| Variant | Required contract before implementation/qualification |
| --- | --- |
| `UC-19/figure-eight` | Two visually selected objects, opposite-winding lobes, safe crossover, order/laps/clearance, independent scorer and held-out placements. Use the generic route interface; a successful legacy circle is not evidence for this task. |
| `UC-19/follow-person` | Explicitly selected/cooperative target, maintained identity, distance band, allowed area, duration, target-loss stop and an independent tracking scorer. No silent person substitution or hidden actor IDs supplied to the policy. |
| `UC-19/find-person` | Designated target, permitted identifying evidence, ambiguity handling, search budget, safe approach and independent identity/arrival checks. Relationships are not inferred from appearance. |
| `UC-19/transport-and-tidy` | UC-15 grasp/placement plus carrying footprint, load retention during UC-03 navigation, declared objects/destinations and independent delivery checks. Cube sorting alone is insufficient. |
| `UC-19/recreational-driving` | Bounded area, speed/energy profile, stopping envelope and interruption checks. No higher speed is authorized by naming this use case. |

## Evidence Snapshot: 2026-09-16

This is a navigation aid to existing results, not a new evaluation or promotion decision. Read the linked report for exact code, scene, setup and evidence type.

| Use cases | Evidence available | Remaining qualification gap |
| --- | --- | --- |
| UC-02 to UC-06, UC-09 | Latest fixed series: each executable capability 3/3, 18 passes total. | Fixed nearby starts and scripted executive; not general autonomy. |
| UC-03 doorway | Three arrivals plus Stop/closed-door cases passed; one arrival recovered from buffer expiry. | Repeatability under the original model/publication workload and varied layouts. |
| UC-08 named rooms | Three frozen-series prerequisite blocks. | Genuine room annotations and separate semantic verification; no substitute checkpoints. |
| UC-10 | Six scripted delayed-review variants; deterministic policy boundaries. | Natural-cadence real-model continuation, including unexplained worker stalls. |
| UC-11 to UC-15 | Scripted circuits, parking, charge return, manipulation and pedestrian braking pass in supported fixtures. | Actual model target selection and full task completion on the evaluated current version. Historical task-specific model passes retain their original scope. |
| UC-07/UC-08/UC-17 end-to-end | Earlier real-Luna session recorded about 0.355 m objective-accounted travel, repeated buffer-expiry failures, 106,186 reported tokens and no full mission receipt. | Latest scripted successes do not erase this failure or establish the five-minute target. |
| UC-01/UC-16/UC-18 | Focused UI/lifecycle/diagnostic tests exist; policy unit gate passes. | Full UI suite is not wholly passing; new zone telemetry still needs the current backend loaded. |
| UC-19 variants | Roadmap requirements and some generic-route experiments. | Frozen evaluators and independently verified completion remain missing. |

Source records: [capability ledger](DESIGN_PERFORMANCE.md), [fixed-series report](../.runtime/performance/capability-review-v1-20260916/summary.md), [control-boundary tests](../.runtime/capability-review-v1/control-boundaries.xml), [mechanical tests](../.runtime/capability-review-v1/mechanical-skills.xml), [policy unit gate](../.runtime/policy-unit-tests-v1/final-gate.xml). Local `.runtime` artifacts may not accompany a repository clone; missing artifacts mean evidence unavailable, not a pass. Preserve or export linked evidence before deleting runtime directories.

## Change And Review Process

1. Reference the applicable UC/INV IDs before a design or bug fix. State the proposed behavioral change and its falsifiable acceptance check.
2. Reuse an existing acceptance profile when the contract is unchanged. For a new goal, scene, map or scoring rule, declare a new profile/version and get agreement before qualification; do not edit a failing benchmark to make it pass.
3. Run decision-level tests first, then relevant physical/UI gates. Navigation/control changes require the complete matched AP-HF1 series. Reserve real-model calls for a separately agreed budget and label them distinctly.
4. Record every outcome and evidence type in the ledger, including intermediate recoveries, blocked prerequisites and invalid recordings. Claim improvement only from compatible frozen comparisons; never pool unrelated capabilities into one autonomy percentage.
5. Update this catalog only when intent or acceptance changes. Append a specification revision note; never recycle UC IDs. Preserve replaced profiles and old result meanings. Update the dated evidence snapshot deliberately, not every numeric requirement after each test run.

### Run Record Template

Use this in an issue/report alongside the existing machine-readable evaluator artifacts; it does not change their schema. Replace `REQUIRED` before execution. Null result fields mean not measured.

```json
{
  "specification": "milo-use-cases-v1.0",
  "use_cases": ["UC-03"],
  "invariants": ["INV-01", "INV-02", "INV-03", "INV-04", "INV-11"],
  "acceptance_profile": "AP-HF1",
  "case_key": "navigate-r1",
  "design_label": "REQUIRED",
  "evidence_class": "Scripted controller evaluation",
  "inputs": {
    "suite_map_fixture_hashes": "REQUIRED",
    "source_build_hash": "REQUIRED",
    "start_pose_and_seed": "REQUIRED",
    "controller_renderer_model_settings": "REQUIRED",
    "budget_and_repetitions": "REQUIRED"
  },
  "results": {
    "task_outcome": "Not run",
    "safety_response": null,
    "actual_distance_m": null,
    "added_observed_area_m2": null,
    "wall_setup_control_and_simulation_times": null,
    "moving_holding_and_inference_wait_times": null,
    "contacts_and_sampling_basis": null,
    "independent_identity_arrival_dwell": null,
    "failure_codes_and_renewal_history": null,
    "sensor_ages_and_clock_labels": null,
    "model_calls_tokens_and_missing_usage": null,
    "assistance_and_recording_completeness": null
  },
  "artifacts": [],
  "limitations": []
}
```

### Revision History

| Version | Date | Change |
| --- | --- | --- |
| `milo-use-cases-v1.0` | 2026-09-16 | Consolidated existing behavior, curriculum targets and frozen benchmark bindings. Added stable references and explicit pending profiles; no runtime or scorer changes. |