# Household Robot Task Curriculum

Version: `household-curriculum-v1`, authorized 2026-09-13. This is an evaluation curriculum for the existing wheeled PyBullet robot, not a claim of implemented household autonomy or a model-training recipe. The [performance ledger](DESIGN_PERFORMANCE.md) records measured outcomes separately.

The consolidated [use-case specification](USE_CASE_SPECIFICATION.md) provides stable `UC-*` references, acceptance-profile bindings and explicit qualification gaps. Use it for new requirements and test discussions; this curriculum retains historical progression plans and their original scope. It does not establish current implementation or performance status.

## User Requirements

Confirmed 2026-09-14. The user requires varied natural-language tasks without a dedicated motion program for each task, smooth continuous movement, and new instructions while another task is running. Examples are circling an observed table, following a selected person and driving a figure eight around two observed objects. These examples are acceptance targets, not claims of current qualification.

The selected architecture is **AI-generated routes plus a generic deterministic safety/controller layer**. The AI identifies targets, chooses route waypoints, tracks task progress and revises its plan from current permitted sensors/images. Generic interpolation, footprint checks, speed/acceleration limits and braking remain programmed. No figure-eight or person-following motion routine is added to implement these requirements. Existing circle/exploration skills remain available as legacy comparison modes, and their successful tests cannot establish the generic mode's task competence.

| Requirement | Acceptance Evidence | Current Status |
| --- | --- | --- |
| General AI-supervised tasks | Same generic interface completes circle, two-object figure eight and selected-person following without a task-specific planner or private scene/actor coordinates; test unseen placements separately. | Interface implemented, capability gate **not met**. Nine recorded development attempts did not establish full task success. Latest table trial: 15/360 degrees; figure-eight/following reported limitations. Fast identity tracking and independent figure-eight/follow evaluators remain missing. |
| Smooth continuous movement | In an observed clear route, model inference and compatible path replacements preserve the motion runtime/velocity; measure stationary share, speed, path renewals and contacts. Curves must be continuous; loop endpoints must not cause premature arrival. | Local spline/path-progress contracts and recorded scripted replacements pass. End-to-end generic tasks remain frequently stationary; their smooth-motion gate is **not met**. Safety/unknown-space/lease stops remain required; arbitrary constant speed is not promised. |
| Live command updates | New commands retain session, episode, memory, token counters and original budget; newest command revision alone can install motion. A delayed old reply never overwrites a new task. Compatible replacement keeps rolling; Stop wins immediately. | Implemented for generic-route mode; two recorded real-physics scripted cases passed, including Stop racing a delayed reply. Legacy modes retain halt/restart behavior. |

Select **Run settings > AI-generated routes (experimental)** with continuous local control, enter the Robot goal, then use **New run instruction** during execution. The setting is explicit rather than silently replacing previously qualified legacy behavior. A compatible already-authorized route may continue to its bounded endpoint during replanning; the AI is not a hard-real-time controller. Sharp reversals, target loss, stale observations and insufficient known space require stopping. No successful-following claim is allowed from a single visual detection or an unrelated scenario score.

Custom-task evaluator runs use `--ai-generated-routes --goal "..."` with one scene. They are explicitly unscored unless a matching independent evaluator exists; original scenario success never qualifies a different goal. Record every attempt, preserve failures and distinguish scripted safety tests, model attempts, operator intervention and independent task verification. See [generic route evidence](DESIGN_PERFORMANCE.md#generic-ai-routes).

## Shared Contract

Each trial fixes the task, target description, layout version, starting pose, scene seed, timeout, controller/model settings and evidence type before execution. A user chooses an observed or discoverable target, not hidden simulator coordinates. Model input remains the current AgentObservation and its permitted paired camera/history images. Privileged geometry and identities are evaluator-only.

Every motion task retains the validated worker as wheel-command owner. Stop, cancellation, takeover, reset, disconnect, stale sensing and command expiry invalidate pending motion. Safety monitoring operates during all tasks and never waits for a cloud-model answer. No controller is credited for a task merely because it reports completion.

Report independent physics success, recording completeness, operator assistance, contacts, false completion claims, completion time, travel, sensing age, recovery events and inference cost/latency. Preserve missing, failed, blocked and timed-out attempts in the denominator. Unknown values remain unknown. A safe abort is desirable safety behavior but is not successful task completion.

Search must distinguish successful inspection, ambiguous target and budget exhausted without a find. An unsuccessful search does not prove absence from the entire home. A target is never silently replaced with a similar object or another person.

## Capability Contracts

| Task | Initial Parameters | Success | Failure Or Safe Response | Existing Starting Point |
| --- | --- | --- | --- | --- |
| Circle object | Specific object description; clockwise/counterclockwise; one lap | Correct object, physical lap closure, permitted clearance, final stop; no contact | Ambiguous target requests clarification; unreachable circuit or expired sensing stops; bounded recovery retains target | `furniture_circuit`, four named furniture types and two directions; table/clockwise remains the frozen anchor |
| Figure eight | Two visually selected objects; order, clearance and lap count | Opposite-winding lobes around the correct two objects, a safe crossover, complete pattern and final stop | Stop on ambiguous identity, obstructed crossover, loss of observed support or expired plan | Generic AI waypoint route interface; no dedicated figure-eight routine or qualified independent task result yet |
| Find object | Object description; search budget | Correct object nearby, visible and inspected while stationary | Decoy rejected; ambiguous instance clarified; report not found within the tested budget | `apartment` currently searches for a yellow cube with a red decoy, not arbitrary object recognition |
| Find room | Room type or known room name; search budget | Relevant visible fixtures support identification; whole base enters the destination and stops | A single ambiguous fixture is insufficient; blocked doorway leads to safe replan or abort | `kitchen_bathroom` and `flat_kitchen`; geometry-based arrival alone does not independently measure semantic recognition |
| Return to charger | Observed charging station or remembered station; battery reserve | Return, park within dock tolerance, remain stationary and verify simulated charge increase | Unknown/unreachable dock or insufficient energy is reported; no hidden homing coordinates | `recharge` already provides a cyan pad, return requirement and simulated battery |
| Yield to obstacle | Route, crossing event and feasible stopping distance | No contact; stop before intrusion, wait for clearance, reobserve before continuing | Stale sensing stops; persistent obstruction remains blocked; unavoidable injections are labelled separately | `pedestrian_crossing`; extend to low obstacles and different approach speeds before household safety claims |
| Follow person | Explicitly selected person; distance band; allowed area; duration | Maintain target identity and distance band through the route | Stop on lost/uncertain target or unsafe clearance; never switch people silently | New tracking and benchmark work required |
| Find person | Explicitly designated person; search budget; safe approach distance | Correct target located and confirmed without unsafe approach | Ambiguity triggers clarification; a search timeout does not prove absence | New benchmark work; start with visually distinguishable simulated people, not hidden actor IDs supplied to policy |
| Recreational driving | Bounded test area; duration; speed profile; battery reserve | Sustained controllable movement within permitted area and verified stopping envelope | Stop/cancel always authoritative; blind corners and obstacles reduce speed | Existing wheel control is a foundation, not a validated high-speed household mode |
| Lift object | Graspable object, reachable pose, lift/hold criteria | Object is genuinely retained and lifted; no pushing-only success | Failed grasp or unsafe contact ends/retries within a fixed budget | Existing `tidy` mechanics; independently compare SmolVLA with simpler grasp control |
| Transport object | Pickup target, destination, load constraints | Retain object through navigation, release at destination and verify stable placement | Grasp loss, payload clearance violation or unreachable route causes safe stop | Future composition; carrying footprint and arm configuration must enter motion constraints |
| Tidy floor objects | Bounded set of supported objects and destinations | Each object picked up, delivered and released at its designated place | Unsupported, ambiguous or unsafe objects are left with an explicit report | Existing `tidy`/`sort` fixtures are bounded starting points, not general tidying |

Person identity must be explicitly configured and consented for real-world work. Family relationships are not inferred from appearance. Initial experiments use chosen simulated targets or cooperative visual identifiers. Child- and pet-sized obstacles are safety geometry cases; stopping must not depend on reliable demographic or species classification.

## Shared Apartment V1

Reuse the existing room/furniture assets in one static apartment with living area, kitchen, bedroom, bathroom, hall/doorways and a visibly marked charging station. Separate tasks select goals in the same environment. They do not require the robot to complete every objective during every run.

Implemented first task set under `shared-apartment-v1`: table circuit in either direction, yellow-cube search with a red decoy, kitchen search and recharge. A fixed 16 x 12 m floor contains a spacious living area, central hall, kitchen, bathroom and bedroom. Identical geometry/materials and the same visible charger/survey pads are used for all tasks. Tasks start independently: circuit and searches use the same living-room start; recharge begins on its cyan pad. The survey is just across the living-room doorway in the hall, a deliberately short first return task with adequate battery reserve. Existing scorers and all motion limits are reused; the standalone circuit's geometry and scorer thresholds are unchanged.

Select **Environment > Shared Apartment V1** in the UI or supply `environment: "shared_apartment_v1"` to `/api/challenges/load` with one of `furniture_circuit`, `apartment`, `flat_kitchen`, `recharge`. Reset and page reload preserve the active selection; choosing an option alone does not load an episode. Unsupported task/object combinations are rejected without altering the active episode. The evaluator accepts `--environment shared_apartment_v1`, prefixes case IDs and persists the environment alongside challenge hashes. Saved results display it and use the correct scene reconstruction when a recorded scene is missing.

Mechanical evidence: five scripted, known-geometry, real-wheel cases passed (two circle directions plus three complete task routes). The circle fixture begins on the annulus with evaluator-set orientation; it does not test camera-based approach. Search routes begin at their prescribed starts. Negative scoring checks cover missing/occluded/distant target observations, motion during inspection and staying on the charger without departure. Kitchen arrival remains a physical containment/dwell check, not an independent semantic-recognition metric. In the 2026-09-14 [real-Luna breadth pilot](DESIGN_PERFORMANCE.md#navigation-breadth-pilot), one attempt each at object search, kitchen and recharge produced no verified completions: two timeouts and one false kitchen completion claim. Recharge completed the survey only. All three were fully recorded, unassisted and had zero sampled contacts. Apartment circling and generalization remain unmeasured; individual qualification gates are not met.

Development variants change one factor at a time: starting pose, target location, direction, distractor or blocked route. Reserve separate layouts/placements for held-out evaluation. Do not tune on those results and continue describing them as held out.

First combined mission after individual qualification: find the kitchen, then return to the charger without resetting memory. Later compose object search, circling and charging. Task switching must retain relevant observations while invalidating stale motion predictions.

Moving people and low crossing obstacles follow static-task qualification. Before these trials are presented in 3D replay, record timestamped actor poses: the current initial-scene overlay cannot replay actor movement. Manipulation and recreational speed increases are later stages, not enabled by approval of the navigation groundwork.

## Progression Gates

1. Freeze and run five sequential table/clockwise repetitions of `circuit-freshness-v1`, enhanced rendering, high reasoning, adaptive navigation, camera history enabled, moving handoff disabled and 180-second control budgets. Save every outcome. Do not change source during the batch.
2. An accepted circle qualification requires five complete, unassisted, physics-verified laps without sampled contacts. Any failure remains a blocker to claiming that anchor is repeatable; inspect its trace before changing the controller. Five passes would still be a small development qualification, not a reliability guarantee.
3. Build and mechanically validate Shared Apartment V1 independently of model success. Its individual task evaluations must remain separate from the standalone circuit batch and use their own frozen design label.
4. Broaden individual tasks with fixed variations, then test combined missions. Report results per capability/difficulty; do not pool easy tasks with safety cases into one score.
5. Add dynamic obstacles, person tracking and manipulation only through their respective safety and measurement gates. Simulation success does not certify operation around real people or animals.

Independent CPU-only checks may run concurrently in isolated worlds. Measured model/renderer runs start sequentially on the detected RTX 3080 10 GB; record workload and distinguish throughput experiments from latency comparisons. The intended RTX 5080 16 GB is not the installed device. Recheck hardware before later claims.

## Execution Status

- Capability contracts: documented; acceptance thresholds for the initial circle are fixed above.
- Five-run circle batch: completed with 5/5 verified passes, complete recordings, zero sampled contacts and no source drift. Completion times 97.969-104.375 s. The fixed-task development gate passed; other objects, layouts and general reliability remain unqualified. See the ledger for artifacts and sensing limitations.
- Shared apartment: implemented and mechanically validated, with responsive UI and evaluator/archive integration. The first three real-Luna task attempts completed no full tasks; recharge reached survey-only progress, and kitchen made a false completion claim. Individual task qualification remains open. Seeded variants, held-out layouts and combined missions remain future work.
- Person tracking, high-speed operation and future manipulation: deferred; no new model training authorized or performed by this curriculum setup.