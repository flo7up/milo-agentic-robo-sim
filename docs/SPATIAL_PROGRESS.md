# Spatial Progress Recording And Replay

Design label: `spatial-replay-v1`. This is the first progress-capture milestone: automatic spatial workflow recording, synchronized replay and a per-attempt overview. It does not establish improved robot competence, replace the existing scenario scorer, or implement the proposed matched capability benchmark.

## Use

Open **Test Archive** and choose **Operator spatial workflow** for manual home sessions, **Scripted test** for automated fixtures, or **Real model** for Luna sessions. The default remains Real model. Search by run/design/scenario; the replay demonstration is `spatial-replay-v1-20260914` under Scripted test.

The **Spatial capability evidence** table lists each attempt's map identity/revision, evidence category, controller outcome, observed-free-area change, visited-cell change and recording completeness. It does not pool evidence categories, calculate a new success rate or claim that additional observed area was physically visited. Three rows are shown initially, then Show more/View all.

Select a trial to use **Synchronized replay**:

- A single wall-time slider controls head RGB, depth, observed occupancy, live obstacles, estimated map-frame pose and planned route.
- Play/pause, rewind and 1x/4x/10x speed are available. Mission-event buttons jump to localization transitions, route progress, failures, cancellation, contacts and recording gaps.
- The coverage chart shows recorded observed-free area over time, with hover/focus values and selectable points. It does not show whole-house coverage, whose denominator is unknown.
- Frame timestamps remain visible. Images are the latest recorded past samples, never future frames or generated interpolation. Depth and RGB may have different capture times; both are displayed. This is sampled replay, not full-rate video.
- The observed map is separate from the archive's existing evaluator-only world/scene view. Estimated pose trails break on map/transform/localization changes and manual relocations. They are not ground-truth position traces.
- Missing frames, corrupt snapshots, recording gaps and capped/incomplete data are explicit. Old recordings can replay their existing camera samples even when no persistent map was captured. No map history is invented for old trials.

The standalone archive makes only read requests. When opened inside the connected cockpit, Stop remains available for independent home motion as well as Luna motion.

## Capture Ownership

[recording.py](../backend/recording.py) remains the common recorder. It samples existing worker state; it does not call inference, advance simulation, change map geometry or modify robot observations. Its bounded queue is flushed outside the physics thread. Map copying and compact telemetry capture occur at most twice per second or at meaningful transitions; compression and disk I/O run in the writer. Current maximum spatial-capture cost is reported in new scorecards as `capture_max_s`, not a hard real-time guarantee.

[session_recording.py](../backend/session_recording.py) adds automatic operator sessions around initial mapping, map loading, localization, named navigation and expansion requests. Save/load/localization finalize their corresponding workflows; navigation/expansion finalize when terminal. Stop, takeover, reset, disconnect, shutdown, rejection and Luna handoff also finalize. Source hashes, scenario hash, request parameters, timestamps, evidence type, renderer, map snapshots and recording errors are retained.

Guided mapping continues recording while the operator drives and reviews. The initial UI arm-fold/head-scan preparation precedes the first home request and is therefore outside that mapping recording. A load or localization attempt is a separate record. The 30-minute recording limit ends capture; it does not restart or authorize motion. In-flight Stop/task revisions are checked after recorder startup so delayed I/O cannot authorize motion after Stop/resume.

An existing Luna session keeps its normal recorder, which now captures map telemetry too. Operator-session creation cannot replace an active recorder. Writer failures stop the robot and publish incomplete evidence with the failure retained. A rejected navigation request can have a complete recording; rejection and data loss are different outcomes.

Operator spatial workflows are never counted as independently verified scenario passes. The scripted browser fixture explicitly records its own evidence category. Manual command/relocation assistance is retained, including prior assistance in the episode. A controller's named-arrival result is shown as a controller outcome, separately from physical scenario completion.

## Data Contract

The existing trajectory stream retains movement samples, monotonic elapsed wall seconds, simulation time, camera/depth references, operator assistance and Stop revision. New `home` fields add:

- Map UUID, revision and snapshot reference; coverage and scan count.
- Estimated map-frame pose and map-from-odometry transform.
- Localization status/scan quality, lidar/depth age at capture, stage and task state.
- References to separately sampled planned-route/live-obstacle telemetry.
- Bounded transition events including task status, reason, segment count and retries.

Map snapshots contain a 400x400 signed-byte occupancy grid, bit-packed visited cells, map frame/origin/resolution, labels/graph, revision and SHA-256 of the grid/visit payload. They are immutable, compressed JSON assets. The previous full snapshot remains explicitly labelled when limits prevent a fresh one; it is not silently presented as current geometry.

Limits: 512 queued samples by default; 2,048 map snapshots, 2,000 detailed telemetry samples and 2,000 captured mission transitions per recording. Surviving samples carry their referenced media even if older queue entries were dropped. Dropped or omitted data invalidates spatial completeness. These limits bound capture; long records can still be too large for interactive replay.

[saved_results.py](../backend/saved_results.py) exposes a read-only replay timeline plus referenced-media endpoints. It checks recording root containment, rejects links/path traversal and unreferenced media, bounds file sizes and decompression, verifies snapshot checksums, and rejects invalid timestamps/cell values. Interactive replay allows up to 50,000 source samples, 6,000 keyframes and 64 MB trajectory input. Oversized recordings remain on disk and are reported unavailable for interactive replay rather than silently truncated.

Snapshot hashes prove saved display-payload integrity; they are not the full environment/model/configuration fingerprint. Existing experiment manifests retain source/settings provenance. Recorder data contains privileged evaluation measurements and must never be supplied to Luna or a policy as sensor input.

## Verified Evidence

- Focused checks cover immutable maps, dropped queue entries, existing recording behavior, referenced-media security, corrupt payloads, contact/gap events, interrupted operator sessions, writer failures and Stop during recorder startup.
- Browser fixtures check exact past-frame selection at timeline boundaries, failure-event jumps, play/pause, map pixel changes, responsive desktop/phone layouts and zero robot writes from the standalone archive.
- The real-sensor scripted [demonstration](../.runtime/performance/spatial-replay-v1-20260914/replay-check.json) uses an isolated SQLite copy of the saved shared map. It records 414 samples, 90 original replay keyframes, 11 captured mission events and three map snapshots; derived diagnostic events may add to the displayed count. Zero dropped records and zero sampled contact episodes. Total recorded distance 0.418 m over 51.031 wall seconds; this includes scripted manual setup and is not an autonomy result.
- That demonstration's mapped Home task reported controller completion after two segments/two retries. The subsequent eight-second expansion ended `limited`. Observed free cells changed from 70.59 to 70.81 m2; known cells from 7,441 to 7,560; visited-cell count remained two. The original saved-map document was verified unchanged. No model calls or existing session resets occurred.
- The demonstration is a development integration artifact. Later capture-limit, diagnostic-event and UI fixes were software-tested but were not retroactively applied to its source fingerprint. No matched A/B performance claim is made.

## Next Milestone

The [Household Foundation V1 benchmark](HOUSEHOLD_BENCHMARK.md) now provides the first fixed seven-task, three-repeat control baseline with per-task denominators, independent pose/arrival checks and preserved replays. It records missing room annotations as blocked prerequisites and does not claim closed-door or real-Luna qualification. Statistical comparisons and automatic "improved/regressed" claims are not included in the replay milestone; future candidates must retain the frozen suite and map inputs.