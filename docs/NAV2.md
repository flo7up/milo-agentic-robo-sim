# Nav2 Navigation And Built-In Backup

Status: bounded local integration verified, full-task capability unqualified. Nav2 is the primary selectable destination planner/controller when started with `-Nav2`. The original stack is retained as **Built-in (backup)**. No automatic controller switching occurs after a failure or safety stop.

## Start

```powershell
./start.ps1 -Nav2 -Port 8013
```

Docker Desktop's Linux engine must be running. The launcher uses `milo-nav2:jazzy`, building it from the ROS Dockerfile if missing, and starts an owned container with an isolated ROS domain and a read-only ROS source mount. It enables the local ROS API only in this backend process. Closing the launcher removes its container. Normal startup without `-Nav2` retains the built-in path. No model inference or motion starts automatically.

In Run settings, choose **Navigation stack > Nav2 (primary)**. Start remains disabled until bridge heartbeat and Nav2 lifecycle readiness are available. Choose **Built-in (backup)** while stopped to use the previous implementation; existing composer, rolling exploration and furniture-circuit behavior remain there. The selection is saved locally and recorded per session/architecture. Switching requires a new run; an expired or stopped Nav2 command cannot act on the backup session.

## Scope

Luna retains semantic decisions and observed destination selection. Nav2 receives a bounded odometry-frame destination and runs its planner, regulated pure-pursuit controller, velocity smoother and collision monitor. The worker remains the sole wheel-command owner and checks sensor age, command sequence, goal identity, episode, task revision, Stop, speed and whole-body clearance. Nav2 result success is checked against local odometry tolerance and never establishes mission success by itself.

Nav2 mode currently executes one destination at a time, including `explore` as a bounded destination. It does not implement built-in rolling continuation, composed motion paths or furniture circuits. Look, turn, scan and stationary task-verification actions remain available. Select Built-in backup for the unsupported path primitives; this is explicit, never silent fallback.

Existing camera-derived candidate generation and marked-floor `approach_target` remain in the supervisor. This replaces the selected destination planner/tracker, not the entire semantic navigation or task-specific approach layer. Local/global Nav2 costmaps are odometry-referenced rolling maps; no SLAM loop closure or persistent global apartment map is enabled.

The bridge publishes simulated laser, paired head RGB/depth, footprint and wheel odometry, not privileged object identities or scene geometry. The model continues receiving only its existing permitted observations/images. Nav2's laser is an additional simulated sensor for the navigation stack, so any future performance comparison must disclose the changed sensor input rather than claim an otherwise identical algorithm comparison.

The bridge persists across sequential goals and drains cancellation/results before accepting another. An episode change exits the ROS launch, and the owned container restarts with empty costmaps. Readiness is false until a new current-episode heartbeat and active navigator/planner/controller are present. A stopped command stream is bounded by the existing 0.5-second watchdog and local motion limits; host/ROS clock synchronization is required.

Independent wheel-odometry acquisition drives `/odom` and `/clock`; paired camera frames retain their exact acquisition-time transforms. Four callback groups separate odometry, cameras, command delivery and action/lifecycle handling. The worker runs ROS physics on a paced 20 Hz cadence without forcing a physics tick for every API request. Cancellation bypasses that schedule and is also checked on queue wakeup. Lightweight odometry cannot authorize motion: each velocity still cites a fresh paired sensor sequence. The 0.35 m/s and 0.5 rad/s caps, 0.5 s command watchdog, whole-body guards and observation-relative model lease are unchanged.

## Validation

The final [38-test scoped gate](../.runtime/nav2-scoped-verified-20260914.xml) covers ROS/Nav2, shared spatial sensing and provenance. Two scripted supervisor cases dispatch `navigate` and `explore` to agent-owned Nav2 sessions and report a blocked result without claiming mission success. After a concurrent API source change, the required integration was reapplied and the [four-case API contract gate](../.runtime/nav2-api-options-final-20260914.xml) passed, including status, odometry, architecture metadata and Stop/reset/takeover/disconnect. The primary/backup browser test uses scripted readiness and captures requests without inference.

The [final frozen-source Jazzy integration](../.runtime/nav2-current-frozen-20260914/report.json) passed two sequential operator-selected goals (0.7 m and then 0.35 m relative): 0.661 m and 0.311 m actual travel, both within the unchanged 0.04 m arrival tolerance, no contacts in 60 sampled states. A third goal was interrupted after motion began: cancelled, physics frozen, old velocity rejected with HTTP 409 both after Stop and resume. Enhanced rendering, standalone parking fixture, no model calls. This is a bounded integration check, not the parking mission or a benchmark success rate.

After that motion run, the metadata endpoint was extended to accept the existing `none` reasoning option; the final API gate above covers it. No motion code changed. The current preview was restarted and checked without another motion trial; its API source hash consequently differs from the motion report.

To reproduce on an idle, dedicated instance with a fresh parking episode:

```powershell
node scripts/check_nav2.mjs http://127.0.0.1:8013 .runtime/nav2-check-unique
```

The harness refuses an active controller, writes to a new directory, captures source hashes before/after and stops on exit. It intentionally moves the selected instance. Keep failure reports; do not pool development retries with frozen evaluation.

Production frontend build passed. Full browser gate: 46 passed, one skipped, three archive failures, all three reproduced in isolation. Full backend gate before the final cancellation repair: 686 passed, 10 failed; the two ROS failures were repaired and rechecked, while eight non-Nav2 failures remain outside this change. The broader affected-surface rerun passed 76 and reproduced built-in pedestrian parking failure. These are not full-green suite claims. Earlier watchdog/clock/clearance failures and the API source-change failure are retained with details in the [design ledger](DESIGN_PERFORMANCE.md#nav2-primary-integration).

No real-model capability, full apartment completion, autonomous fallback or deployment to a physical robot is established by this integration. Existing safety constraints remain required; simulation watchdog tests do not certify physical-robot safety.

## Comparison Attempt: 2026-09-14

**No improvement demonstrated.** The requested follow-up attempted both real-model task evaluation and a separately labelled local-controller comparison. Neither supports a performance-gain claim.

The [real-model pilot](../.runtime/performance/nav2-task-pilot-v1-20260914/experiment.json) planned Built-in/Nav2 pairs on parking, kitchen navigation and object search with identical task definitions, Luna/high, enhanced rendering and 180-second time-only budgets. Only Built-in parking ran: its first inference failed with Foundry HTTP 403, zero input/output tokens and no model response. A concurrent worker source change triggered cancellation of the other five cases. This is blocked infrastructure, not a task-success measurement. No inference retry or authentication-policy bypass was attempted.

The [separate scripted local probes](../.runtime/performance/nav2-local-pilot-v1-20260914/experiment.json) used fresh parking fixtures, observed-floor targets, 0.7 m and 1.2 m distances, two repeats each and reversed backend order on the second repeat. All eight recordings are complete: 373 samples, zero drops and zero sampled contacts.

| Probe | Built-in | Nav2 |
| --- | --- | --- |
| 0.7 m, repeat 1 | Arrived, 16.000 s | Command watchdog, no travel |
| 1.2 m, repeat 1 | Blocked by stale depth | Goal rejected, no travel |
| 0.7 m, repeat 2 | Arrived, 7.375 s | Goal rejected, no travel |
| 1.2 m, repeat 2 | Arrived, 8.500 s | Goal rejected, no travel |

These are diagnostic outcomes, **not a clean 3/4 versus 0/4 benchmark**. Source changes were detected by the final check; the container log also records HTTP timeouts and lifecycle restarts. The runner initially checked Nav2 readiness before the head scan, which could invalidate it. The [runner](../scripts/evaluate_nav2.py) now requires a new ready heartbeat after scanning and fails explicitly on final-trial source drift. The [post-scan preflight](../.runtime/performance/nav2-postscan-preflight-20260914/experiment.json) reached readiness without inference or a wheel command, but another source change invalidated its frozen-source gate. No replacement motion results were collected after that correction.

Nav2 retains its additional laser, 0.35 m/s cap and final-heading check; Built-in retains its 0.5 m/s cap. Both use the same 0.04 m position criterion and bridge polling overhead. No speedup, task-success gain or causal regression is inferred. Restore model access, finish concurrent controller edits, resolve the Nav2 readiness/delivery failures, then run a new labelled frozen comparison. All failed attempts remain preserved; the port-8013 preview was not reset or driven.