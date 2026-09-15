# Household Foundation Benchmark V1

This is the standardized baseline suite for the current persistent-map navigation stack. It is distinct from software unit tests, the earlier recorder demonstration, and real-Luna scenario trials. The suite executes a scripted mission executive with real PyBullet sensors and physics; no inference or policy training occurs.

## Frozen Protocol

Implementation: [benchmark_household.py](../scripts/benchmark_household.py). Suite ID `household-foundation-v1`. The machine-readable suite, its SHA-256, fixture hashes, exact original and derived map documents, source/build hashes and runtime metadata are saved in each run directory before execution.

- Environment/scenario: Shared Apartment V1 / Find the Kitchen. The scenario's kitchen scorer remains separate from benchmark scoring.
- Controller: `builtin_mapped_navigation`. Renderer: enhanced. Original speed, clearance, lease and sensor-freshness limits are unchanged. No automatic Nav2 or learned-policy substitution.
- Three repeats at fixed initial x offsets 0, +0.10 and -0.10 m, yaw unchanged. Start placement belongs to fixture setup, not an in-run teleport. Every attempt uses a fresh worker and an isolated copy of the same map. Resources are reused between sequential attempts.
- Frozen source map: `6d764d3e-8eeb-4810-b575-110fc115f0ce`, revision 1, document hash `b9e33c46d2062f5524994b48ead331226b7e47892b25d6e65401a4a6246b902d`.
- The derived benchmark copy adds exactly one explicit destination at map pose `[0.7, 0, 0]`, checked against the original map's observed clearance. This is a fixture annotation, not a newly discovered room. Occupancy, visits, map identity and revision stay unchanged. The original home database is opened read-only and never overwritten.
- Localization receives the saved Home annotation as an approximate hint. It does not receive the actual starting offset or simulator pose. This measures hinted relocalization, not global kidnapped-robot localization.
- Safe arm-fold and depth preparation, and prerequisite localization for motion tasks, have a separate 60-second setup budget. Setup failures remain failed planned attempts. Task timing includes sensor validation, route planning, motion and completion checks; setup time is reported separately.
- Run all tasks in fixed order for repeat one, then repeats two and three. No operator assistance, rerunning failures, control tuning or threshold changes during a baseline. A new source change aborts the run and makes the partial evidence ineligible as a frozen baseline.

## Tasks And Pass Criteria

| Task | Wall budget | Independent checks |
| --- | --- | --- |
| Localize after restart | 20 s | Reported localization; evaluator pose error <=0.15 m and heading error <=0.15 rad; <=0.03 m wheel travel |
| Navigate to observed checkpoint | 60 s | Controller arrival plus evaluator endpoint error <=0.15 m and >=0.5 simulated seconds stable/grounded at the target |
| Checkpoint then return Home | 90 s total | Both legs meet arrival/dwell checks, under one total budget |
| Reject occupied destination | 20 s | A fixed 0.3x0.3x0.8 m moved object covers the checkpoint. Fresh sensors must explicitly reject it as unreachable; <=0.05 m travel, no contact |
| Expand partial map | 30 s | >=1 m2 new observed free area and >=0.5 m actual travel; buffers stopped at task end; original map unchanged. Up to 1 s final acknowledgement allowance, no additional exploration budget |
| Stop during motion | 30 s | Inject Stop after >=0.10 m actual travel; acknowledgement <=0.5 wall seconds; ticks/buffers frozen for 0.75 s; reject old-revision command after Resume and no restarted motion |
| Navigate between named rooms | 90 s total | Requires two existing saved room annotations >=2 m apart, then independently verifies both arrivals. Missing annotations are **Blocked prerequisite**, retained in the denominator |

All passes also require no sampled contact episodes, no manual relocation, complete spatial/trajectory recording, unchanged source and unchanged original map. Runtime timestamps and evaluator actual world pose/velocity/contact data are recorded separately from the robot's estimated map pose. Privileged ground truth is used only by fixture/scorer code, never for robot planning or map construction. The existing simulator collision shield remains a privileged safety backstop, so this is not a hardware safety certification.

The occupied-destination task is a **safe rejection test**, not a closed-door traversal or dynamic-person avoidance benchmark. The source map currently contains only Home, so room-to-room trials are expected to be blocked. Do not substitute the local checkpoint and claim room navigation. A future room-capability suite needs a deliberately expanded sensor-built map and named rooms, with a new suite/map fingerprint.

## Outcomes And Archive

Every planned attempt receives Passed, Failed, Blocked prerequisite, Invalid evidence, or Not run. Timeouts, sensor faults and partial returns remain failures. Missing/corrupt recordings or source drift invalidate evidence; invalid attempts are never counted as passes. No aggregate autonomy percentage is produced across heterogeneous tasks. Show per-task counts and successful-run timing separately; three nearby starts are a development baseline, not a statistical reliability estimate or held-out-layout generalization result.

Each attempted robot run retains synchronized head camera, depth, observed-map snapshots, mission events and evaluator trajectory. The Test Archive displays a **Standardized benchmark baseline** table, criteria, failure reason, setup time and fingerprints. Benchmark passes are separate from scenario success and real-model results. Blocked prerequisites have reports but no fabricated motion/replay.

## Reproduce

```powershell
./.runtime/env/python.exe -m scripts.benchmark_household --stage plan
./.runtime/env/python.exe -m scripts.benchmark_household --stage preflight --output .runtime/performance/NEW_PREFLIGHT
./.runtime/env/python.exe -m scripts.benchmark_household --stage run --output .runtime/performance/NEW_BASELINE
./.runtime/env/python.exe -m scripts.benchmark_household --stage compare --baseline .runtime/performance/BASELINE --candidate .runtime/performance/CANDIDATE
```

Output directories must be new. `--map-store` can select a preserved SQLite store containing the exact frozen source document; a changed source-map hash is rejected. Each run also saves the source map JSON for recovery of the original benchmark input without remapping. Comparison requires identical suite, derived map, fixture hashes and evidence type, distinct completed run IDs, and no source drift. Runtime differences are flagged. Do not modify thresholds or omit a task while calling the result the same suite.

## Baseline Status

The original baseline below is preserved. The matched [motion-refresh candidate comparison](../.runtime/mapped-motion-refresh-review/summary.md) now reports navigation, Home return, useful exploration and actual Stop 3/3 each, with localization/rejection3/3 retained and room prerequisites still blocked. Suite, map, fixtures, budgets and scoring are unchanged. This is fixed scripted-physics evidence, not a real-Luna track.

The setup/localization [preflight](../.runtime/performance/household-foundation-v1-preflight-20260914/summary.md) passed and remains separate. The first full [baseline](../.runtime/performance/household-foundation-v1-baseline-20260914/summary.md) completed on 2026-09-14: all 21 planned outcomes retained, 18 physical attempts with complete recordings, 3,115 samples, zero dropped records and zero sampled contact episodes. All 91 tracked source/build files and the original map remained unchanged.

| Capability | Baseline outcome |
| --- | --- |
| Hinted localization | 3/3 passed; median 0.391 s excluding setup |
| Local mapped checkpoint | 0/3 passed |
| Checkpoint then Home | 0/3 passed; second leg never qualified |
| Occupied-destination rejection | 3/3 passed the narrow safe-rejection criteria |
| Useful exploration | 0/3 passed the fixed 1 m2 / 0.5 m requirements |
| Stop after actual travel | 0/3 passed; motion stopped before reaching the injection point |
| Room to room | 3 blocked prerequisites; two room annotations are absent |

No actual Stop injection occurred in these baseline cancellation attempts, so this does not establish a Stop defect or replace the earlier isolated Stop tests. Occupied-destination rejection does not identify whether rejection was due only to the obstacle; the negative starting offset also lacks current-footprint clearance on the clear-route tasks. The narrow safety property passed, not obstacle attribution or rerouting.

Navigation failures include exhausted bounded retries, recorded motion-buffer expiry, stale depth, and insufficient mapped footprint support. Maximum task travel in any measured attempt was 0.0611 m. Exploration's observed-free **net change** was -0.15 m2, +0.11 m2 and not measured for the three attempts; it does not prove added free-space accuracy. Original map occupancy stayed unchanged; expansion modified only an isolated in-memory draft.

This is a reproducible negative development baseline, not a demonstration of improved autonomy. Shared-machine GPU observations were 3,798 MiB used / 11% utilization before and 3,866 MiB / 16% after; these are instants, not controlled or continuous load measurements. Recorded metric overhead is part of this protocol. The next matched candidate should address observed-map footprint support and motion/sensing cadence without weakening safety thresholds. See the [review](../.runtime/performance/household-foundation-v1-baseline-20260914/review.md) and [maintained ledger](DESIGN_PERFORMANCE.md#standardized-household-baseline).

Luna language interpretation, object recognition, manipulation, real closed-door rerouting, other start headings, randomized layouts and long-distance navigation are outside v1. A later real-Luna track should use separately labelled fixed instructions, the same source/map/task protocol and independent scoring. Do not pool that evidence with this deterministic control baseline.