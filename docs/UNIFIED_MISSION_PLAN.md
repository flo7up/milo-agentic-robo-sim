# Unified Map-Aware Mission Plan

Date: 2026-09-15. Status: proposed implementation plan, not implemented or performance-qualified.
Design family: `unified-map-mission-v1`. Scope: one local FastAPI/PyBullet system with Luna semantic supervision and Built-in navigation as the initial supported default.

## Outcome

The user selects a scene, enters a goal and presses Start. A single mission automatically coordinates exploration, navigation, object/room inspection, recovery, optional return and completion. Stop applies to the entire mission. The user does not switch between Task / Luna and Explore / local to finish an instruction.

Luna receives the current head image plus a bounded view of sensor-built map evidence. It interprets the goal, evaluates visual hypotheses and prioritizes offered destinations. Local navigation runs independently of inference within existing authority limits. One worker remains the only owner of actuators and safety enforcement.

Local-only operation, map-disabled supervision and alternative planners remain diagnostic configurations of the same mission pipeline. They are not separate product workflows.

## Starting Point

| Existing surface | Reuse or address |
| --- | --- |
| [AgentController](../backend/agent.py) | Keep session lifecycle, serialized inference, recording, instruction revision and Stop handling. |
| [Continuous supervisor](../backend/continuous_supervisor.py) | Reuse semantic actions; replace whole-operation mapped waits with bounded progress/decision events. |
| [HomeMission](../backend/home_mission.py) | Reuse localization, named destinations, frontiers, route failure history and mapped execution. It must become a capability subordinate to the mission, not a competing session owner. |
| [SimulationWorker](../backend/worker.py) | Retain sole actuator authority, freshness, leases, footprint checks and final simulator clearance shield. |
| [ObservedMap](../backend/spatial.py) and [HomeMap](../backend/home_mapping.py) | Keep distinct local-safety and persistent-memory responsibilities, with explicit transforms; avoid building a third mapper. |
| [Object goals](../backend/object_navigation.py) | Reuse persistent approach identity, standoff, facing, fresh image verification and stopped dwell. |
| [Navigation backend contracts](../backend/navigation_backends.py) | Reuse typed proposals and capability rejection; extend only where necessary. No silent planner fallback. |
| [Run controls](../frontend/src/LunaNavigationControl.tsx) and [preferences](../frontend/src/Preferences.tsx) | Reuse one goal form, run inspector and persistence; migrate the old top-level mode selection deliberately. |

Current gaps are behavioral: only explicit local exploration enables its rolling continuation path; Luna's mapped capability waits for the full operation; persistent map updates are tied to mapping/exploration stages; the model's compact spatial state excludes the occupancy grid. Removing two buttons alone resolves none of these gaps.

Existing object/Home completion and settings persistence are implemented but not grounds for general reliability claims. The retained object candidate scored 16 passed, two failed and three blocked in the full household series, versus an earlier matching 18/0/3 baseline. One separate table/Home real-model diagnostic passed. See the [object review](../.runtime/object-approach-v1-review/summary.md) and [preference review](../.runtime/preferences-v1-review/summary.md).

## Ownership And Invariants

1. AgentController owns one session, top-level instruction revision, inference budget and terminal outcome. A mission executive owns phase/subgoal transitions inside that session. HomeMission and other skills own only bounded local operations.
2. All actuator changes go through SimulationWorker. There is at most one active motion owner and one outstanding model request. The mission never runs a second wheel-control loop.
3. Every operation and model proposal carries run/episode, mission, instruction/Stop revisions and source observation/map identity. Reject incompatible replies; an advanced map revision alone may be revalidated against current geometry, but a changed frame, target or authority cannot be silently remapped.
4. Stop, takeover, reset and disconnect revoke pending operations and model replies immediately. Resume permits sensing or a new explicitly started mission; it does not resurrect a cancelled mission.
5. Unknown space is not traversable. Stale sensing or lost localization cannot be repaired by model reassurance. Existing speed, clearance, lease, retry and scorer limits remain unchanged.
6. A semantic label is a hypothesis with evidence, not ground truth. Current physical arrival, semantic acceptance and whole-instruction completion are separate states. The evaluator remains independent.
7. A saved map is optional. Skipping reuse cannot delete or overwrite it. Restoring UI settings or restarting the server never restores motion authority.
8. Preserve the local OpenAI Responses architecture. No hosted-agent scaffold, cloud provisioning, model training or ROS migration is included.

## Stage 0: Freeze The Comparison And Contracts

Design label: `unified-mission-baseline-v1` for the pre-change capture, not a new benchmark suite.

- Snapshot current source/build, task goals, fixture versions, map inputs and settings before runtime changes. Run the complete existing household suite against `.runtime/performance/household-foundation-v1-stop-reporting-controlled-20260915`; retain the latest 16/2/3 candidate as additional historical evidence.
- Record known dwell, stale-depth, frontend and archive failures separately. If a failure is a blocker for a required workflow, isolate and repair it in its own labelled change; do not attribute it to consolidation without evidence.
- Define the first supported end-to-end tasks: explore-only; find/verify a visible object; explore to find an object; approach then return Home; find/inspect a known room then return. Unknown-room workflows retain explicit map/annotation prerequisites and cannot be qualified using colocated test labels.
- Declare proposed mission state and sensor-derived map input contracts before prompts or UI change. Document the addition to the AgentObservation/head-image boundary in repository instructions: the map image may only render validated AgentObservation map data, never spectator or evaluator state.

Exit: frozen inputs, a recorded full-series result, an explicit list of blockers and agreed success criteria. No new capability claim.

## Stage 1: One Mission Owner

Design label: `unified-mission-core-v1`.

- Introduce a small mission state machine, preferably as a dedicated module used by AgentController, reusing existing progress/memory classes. Avoid merging every controller into one large class.
- State includes mission ID, instruction revision, requested subgoals and return policy, phase, current operation ID, selected target, original deadline, bounded recovery counts and completion receipts.
- Phases: preparing, interpreting, exploring, navigating, inspecting, returning, recovering and terminal completed/blocked/failed/cancelled. Phase changes describe actual execution, not just a model proposal.
- Normalize existing local-only and Luna entrypoints into this state machine through compatibility adapters. Diagnostic local-only missions use a typed exploration objective and no semantic calls.
- Keep one top-level budget across all phases. Skill sub-budgets can only consume remaining time. Subgoal changes, replans and model delays must not reset the mission deadline or extend worker leases.
- Allow goal corrections through the existing instruction-update path; revoke incompatible operations, retain measured history and establish a new revision before further movement.

Exit: scripted transition tests demonstrate one session/owner/deadline and cancellation during preparation, motion, recovery, inference and completion verification. Compatibility entrypoints cannot create competing missions.

## Stage 2: Shared Observed State And Local Execution

Design label: `unified-observed-state-v1`.

- Keep the rolling local occupancy map for current safety and the persistent sensor map for accumulated spatial memory. Define a mission-facing snapshot over them, not an unvalidated fusion or duplicate map.
- With no reused map, establish a fresh odometry-relative draft and Home reference from measured start pose. With reuse, require supported sensor-based localization before map-frame goals become available; on ambiguity, stop/request help or explicitly start a separate local draft without changing the stored map.
- Maintain observed coverage, visited path and obstacle updates during authorized mission navigation as well as exploration. Separate in-memory observation updates from saving: frozen maps remain read-only, and durable map changes follow the existing explicit save policy.
- Convert frames only through a current validated transform. Do not overlay a saved map on the local crop when localization is absent or invalid. Show such memory as unavailable/unlocalized instead.
- Extract common frontier selection, bounded route execution, compatible continuation and failure accounting from the entrypoint-specific path. Both local-only and semantic missions invoke the same implementation; continuation must not depend on which UI button started it.
- Preserve footprint corridors, planned-path checks and original travel/deadline bounds through continuations. Model inference is not a dependency of the local control tick.

Exit: identical scripted exploration commands follow the same local path under both diagnostic configurations; fresh/reused/read-only maps, disconnected components, localization loss and cancellation are covered. Complete household comparison runs before promotion.

## Stage 3: Observed Map Context For Luna

Design label: `observed-map-context-v1`. Implement and test first in shadow mode without changing decisions.

- Add a typed, versioned map observation in [contracts](../backend/contracts.py), produced from immutable worker-owned sensor-map snapshots. Attach map ID, revision, run/epoch, source sensor sequence/time, coordinate frame, origin, scale, crop bounds, pose/head direction, localization status and transform validity.
- Start with one bounded local crop. Include observed free/occupied/unknown cells, measured trail, separately styled planned route, offered frontier/place IDs, selected target and timestamped tentative sightings. Do not draw unseen room extents, true obstacle meshes, scorer target locations or successful outcomes from simulator truth.
- If reducing resolution, let any occupied contributor stay occupied and retain unknown contributors as unknown unless occupied; never average them into free space. Keep the planner's full-resolution map unchanged.
- Render the map image deterministically from that typed snapshot using existing image tooling. The numeric IDs and raster must refer to the same snapshot and have a documented pixel-to-frame mapping. Prefer stable short IDs and a compact legend to dense prose or full-grid JSON in every prompt.
- Use a wider overview on explicit request or a meaningful topology/progress change. Cache by geometry/annotation revision, crop and rendering version; stamp pose and age correctly rather than making cached geometry appear newly observed.
- Keep the current head image mandatory. Budget the map and historical images explicitly with the existing request/context settings; do not silently add an unbounded image or drop current camera evidence. Initial crop size and overhead targets are provisional until measured.
- Trace the exact map image, snapshot hash, included IDs and image roles. Record assembly time, bytes and model token/latency cost. If snapshot production fails or exceeds its budget, expose map-unavailable and preserve local safety behavior; do not delay Stop or actuator updates.

Exit: fixture tests cover rotation/origin transforms, crops at map edges, unknown-space preservation, stale/unlocalized maps and deterministic raster/ID agreement. Privacy tests vary hidden scene truth while keeping sensor evidence fixed and verify identical model input. Trace replay reproduces the sent map.

## Stage 4: Event-Driven Luna Supervision

Design label: `unified-map-supervision-v1`.

- Replace the full mapped-operation wait with bounded status/decision events. The mission controller chooses when to request semantic review, without invoking inference inside the worker thread.
- Trigger reviews on initial interpretation, meaningful new coverage/junctions, arrival, repeated local blockage, uncertainty and subgoal completion. Add a bounded time/distance observation checkpoint so exploration does not pass an unseen target indefinitely just because no special event was detected.
- Debounce repeated events and retain only the latest compatible decision request. Keep inference serialized; queue/cancel logic must obey instruction and Stop revisions. No repeated paid model calls while local sensing is stale.
- Luna selects offered frontier/place/approach IDs, reports an evidence-backed target hypothesis or requests a stopped observation. The mission dispatches the bounded skill; Luna does not issue arbitrary map coordinates or raw wheel commands.
- While waiting, allow only the existing authorized compatible corridor to execute. Stop for sharp changes, inspection, target verification or expired authority. Validate every returned proposal against the current pose/map/mission; reject stale pixel selections and reacquire after motion.
- Use the existing persistent object-goal and room-observation receipts. Evaluate requested shape/support/context as well as color when accepting a target. Failed identity checks return to search or a bounded changed-view inspection; do not mark tentative labels confirmed or repeat the same rejected observation indefinitely.
- Completion requires every requested subgoal receipt, current stopped safety verification and, when requested, verified return. Preserve the distinction between an agent completion report and independent evaluator success.
- On model/network failure, continue only an already-authorized bounded operation if safe, then hold with an actionable error. A semantic task must not silently become unlimited exploration or report completion.

Exit: scripted real-camera/physics missions execute explore -> identify -> approach -> inspect -> return without a mode switch; slow/malformed/late model replies and all interruption phases are tested. Independent localization/pose/dwell checks pass where the scenario has the required prerequisites.

## Stage 5: One Product Workflow

Design label: `unified-mission-ui-v1`. Land after runtime integration, not before it.

- Replace Task / Luna and Explore / local with one goal form and Start/Stop. Show the actual mission phase, current target, remaining budget, progress and blocked reason; keep manual takeover and Stop easy to reach.
- Move local-only execution, map context off and alternative planner settings to a Diagnostics disclosure. Use the same mission API, recorder and skill pipeline; do not maintain a second exploration form/loop as a hidden parallel product.
- Make Built-in the initial standard preset because it supports the current object/Home capabilities. Keep alternative planners explicit, with capability checks and unsupported-task errors; never silently substitute one mid-mission.
- Preserve user preferences. Version the preference schema and migrate old control-mode/planner selections into diagnostic preferences without enabling diagnostic mode or starting a mission automatically. Keep goal drafts, scene choice, inspector selection and connection-apply behavior.
- Show the exact map context used by Luna in the trace, separately from the operator's live/spectator views, including age and localization limitations.
- Prevent the prior mixed-version preview failure: expose a backend capability/build handshake and reject unsupported requests clearly. Capture sanitized exception type/location and a correlation ID without secrets; do not blame model configuration for every internal error. Restart/update previews only after preserving active runs and drafts.

Exit: desktop/mobile browser tests verify one normal start path, automatic phase progression, persistent selections, unsupported-capability feedback, safe reload/reconnect and Stop during every phase. Old supported preference data still loads.

## Stage 6: Qualification And Promotion

Design label: `unified-map-mission-v1`, with frozen configuration labels for each diagnostic comparison.

- Run focused backend safety/contract/physics tests using `./.runtime/env/python.exe -m pytest -q`, reusing [home mapping tests](../tests/test_home_mapping.py), [navigation tests](../tests/test_navigation.py), [object tests](../tests/test_object_navigation.py) and [API tests](../tests/test_api.py). Add isolated mission/map-context tests only where existing files are not a suitable home.
- Run `npm --prefix frontend run build`, then `npm --prefix frontend test`. Record baseline failures explicitly and fix changes attributable to this implementation. Any failing critical end-to-end workflow blocks standard-mode promotion.
- For each behavior-changing candidate, run the complete unchanged household series with `scripts.benchmark_household --stage run --baseline <existing-run> --design <candidate-label> --output <new-directory>`. Keep all failures/blocked cases; no selective retries or modified suite/map/scoring inputs. Treat source/runtime/load mismatch as a comparison limitation.
- Establish a separate, predeclared mission evaluation series where the household suite lacks semantic tasks. Freeze the initial map, object/room evidence prerequisites, goals, allowed assistance, budgets, scenarios and repetitions before the first run. The recolored Inspection fixture is not comparable to the former amber-case fixture without treating appearance as an experimental variable.
- First run scripted real-physics workflows and privacy/failure injections. Only after they pass, run a bounded real-Luna pilot using the existing configuration. Confirm its total inference budget and repeat count before execution; this planning task authorizes no model calls.
- Compare map enabled versus disabled on the same frozen unified controller, task fixtures, model settings, image policy and runtime conditions. Report the additional map input cost and what other imagery is retained. Keep old-controller versus unified-controller comparisons separate; do not bundle both effects into one improvement claim.
- Measure independently verified full-task success, wrong-target selections, contacts, Stop latency, stale-reply rejection, return/dwell error, useful coverage, repeated travel, recovery outcomes, wall/control time and token cost. Record map assembly overhead and source/map stability. Whole-device load snapshots are context, not causal proof.
- Publish a failure-first review with retained traces, exact input images/maps and frozen fingerprints in [the design ledger](DESIGN_PERFORMANCE.md). Local-only, scripted, operator-assisted and real-model evidence remain separate.

Exit: all critical safety/privacy gates pass; the existing 21-case household suite meets its executable baseline without unexplained new regressions, with the three room prerequisites still labelled blocked until genuinely met; repeated predeclared mission trials meet their declared criteria. A single successful demonstration is insufficient. Keep the feature diagnostic/experimental if these conditions are not met.

## Delivery Order And Non-Goals

Stages 0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6, with a focused executable check immediately after each small change and complete-series gates for control changes. Map rendering may be developed in shadow after Stage 1, but its runtime wiring depends on validated Stage 2 state. Treat each stage as a separately reviewable change with a version label and rollback to the last compatible runtime, not a collection of permanent product modes.

First integrated milestone: one object-search/approach/return mission, using one executive, one observed-state contract, one serialized Luna stream and the same worker-owned execution as local diagnostics. Room workflows follow only with genuine map/localization/semantic evidence; do not imply automatic room mapping is already solved.

Deferred: live NavDP/NoMaD/SmolVLA consolidation, full SLAM/loop closure, arbitrary moving-object tracking, new cloud infrastructure and automatic restoration of physical robot state. Resolve current dwell/stale-depth blockers based on measured evidence, not relaxed limits. The shelf-case visual correction and settings persistence are already delivered and remain separate historical designs.