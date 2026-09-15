# Persistent Home Mapping

Initial design: `persistent-home-map-v1`. Current mapped controller: `guided-waypoints-v1` / `0.4.0`. Guided safety and sensor-processing changes are tested; safe Kitchen/report/Home demonstration remains unmet. [Current evidence and blockers](../.runtime/guided-waypoints-review/summary.md). Prior measured motion design: [mapped-motion-refresh-v1 / 0.2.0](DESIGN_PERFORMANCE.md#mapped-motion-refresh).

## Room Memory

Luna now receives the compact named graph, current clearance of each recorded connection path, up to eight reachable frontiers and recent room observations. A blocked recorded path does not establish that every alternate route is blocked. Old graph links without a stored path report unknown clearance.

`observe_room` records the original paired head image, measured map pose, head orientation, run/episode/frame, evidence text and reported confidence. A new room or doorway label is tentative. Existing coordinate-only names are operator-named but unreviewed. Only the connected operator's **Room observations > Confirm** action can confirm a label; Luna has no review capability. Save the map to persist the reviewed place metadata. Room images and review records have separate storage, bounded to 1,000 images per map; the most recent 100 are available for review.

After `navigate_place`, a room match or mismatch requires a new image captured after stable physical arrival at that same destination. The report expires after 15 seconds or a changed base/head view, localization, task, episode or Stop authority. It is always an annotation, never independent semantic proof. The matching report stays in historical room observations after returning Home, explicitly separate from current verification.

For a known-room round trip, Luna supplies `navigate_place(place_id=ROOM_ID, return_place_id=HOME_ID, time_budget=...)`. The worker reaches the room and waits for a fresh `observe_room` match/mismatch. A match starts the guarded Home return; mismatch fails. One original deadline covers both legs and the inference wait. `room_workflow` reports the intermediate and terminal states; completion means the requested physical sequence plus a reported visual match, not independently verified room identity. The manual panel's Navigate action remains single-destination; room-report selection is through Luna's validated capability.

For an unknown room, Luna can choose a supplied `frontier_id` using `explore_frontier` and a 1-300 second budget. The worker revalidates that frontier; blockage stops the selected task without silently choosing another destination. General `explore_map` retains bounded geometric frontier reselection. Both expand only an explicitly writable draft.

**Continue mapping** resumes guided sensing on the loaded map identity. For experiments, set the non-secret `MILO_HOME_MAP_STORE` process variable to a separate working SQLite database before starting the application. The default remains the original runtime store. Never point an expanded-map evaluation at the frozen v1 database.

The separate [room-v2 runner](../scripts/benchmark_rooms.py) copies a store with SQLite backup, retains all camera/physics records and checks original-map/source fingerprints. Its scripted tasks are Kitchen/report/Home, blocked recorded doorway, and unknown-ID rejection. Missing camera-backed room prerequisites remain blocked, not substituted with nearby checkpoints. Scripted room-match reports exercise the contract only and cannot measure recognition accuracy. `--stage survey` performs bounded sensor-only expansion without room labels; `--stage run --map-store PATH` evaluates the separate working map. A saved survey does not imply complete-home coverage.

Current results: two scripted surveys stopped on stale depth before 0.28 m. The published room-v2 baseline has one unknown-ID pass and two missing-room prerequisites. One real-Luna180s attempt travelled8.127m but selected ordinary local actions, recorded no room observations and did not complete the round trip. Its original map remained unchanged. [Reports, provenance and limitations](DESIGN_PERFORMANCE.md#room-memory).

## Scope And Choice

Keep PyBullet, FastAPI, the calibrated head camera, wheel odometry, and the validated physics worker. Add a persistent observed map and a small mission executive beside the existing rolling depth map. No simulator replacement, neural-policy training, cloud provisioning, or mandatory ROS installation.

| Existing component | Inspection finding | Reuse |
| --- | --- | --- |
| [simulation.py](../backend/simulation.py) | Wheel-integrated odometry; paired 160x120 metric RGB-D; measured pan/tilt; current robot-link footprint; eight short-range beams | Sensor source and robot geometry only |
| [ros_navigation.py](../backend/ros_navigation.py) | Simulated 720-ray, 8 m planar lidar already exists | Extracted `capture_laser`; no ROS process required to sample it |
| [spatial.py](../backend/spatial.py) | Rolling 8x8 m depth map, camera extrinsics, observed-floor support, clearance, short SciPy routes; episode-local history | Keep local sensing, calibrated projection and object-surface measurement |
| [worker.py](../backend/worker.py) and [continuous_navigation.py](../backend/continuous_navigation.py) | Sole motion owner; short buffered paths; freshness, collision, expiry, Stop and ownership gates | Execute bounded route segments independently of Luna |
| [continuous_supervisor.py](../backend/continuous_supervisor.py) | Luna selects observed local objectives; no durable home coordinates | Add grounded place selection and compact spatial feedback |
| Recording and NavDP/NoMaD adapters | Recordings contain evaluator state; learned backends have separate observation/goal contracts | Never use recordings or private scene geometry as mapping inputs; keep learned backends separate |

[SLAM Toolbox](https://github.com/SteveMacenski/slam_toolbox) fits the available lidar/odometry and provides pose-graph serialization, loop closure and continued mapping. [AMCL](https://github.com/ros-navigation/navigation2/tree/main/nav2_amcl) fits localization against a saved 2D map. Together with the existing Nav2 bridge, these are the stronger next integration for larger environments. The current bridge uses rolling odometry-frame costmaps, not this saved map. Connecting lifecycle, map server, map-to-odom transforms and cross-scenario localization remains additional work, not a feature enabled here.

[RTAB-Map](https://github.com/introlab/rtabmap_ros) supports RGB-D, stereo and lidar, with appearance-based loop closure and persistent databases. Milo has RGB-D but no stereo pair or implemented visual odometry. Moving-head transforms are available; synchronized visual registration and the native/ROS C++ dependency integration would be additional work. It is not required for the initial planar workflow.

The initial implementation reuses installed NumPy/SciPy for occupancy processing, distance transforms, connected components, scan matching and graph routing. It is an odometry-anchored mapper with scan-consistency monitoring, **not a replacement for full SLAM Toolbox or RTAB-Map loop closure**. The optional Nav2 selector continues to apply to the existing local navigation workflow; saved-home missions explicitly use `builtin_mapped_navigation`. There is no silent Nav2 fallback.

## Operator Workflow

1. Load the intended environment. For reusable household scenarios, select `shared_apartment_v1`.
2. Open **Map > Home map**, then **Start guided mapping**. This folds the arms through the existing validated scan, enables depth, and anchors a new map at the robot's measured starting pose. Select a position on the observed map and choose **Survey to selected point**. The worker plans through footprint-clear space while continuing draft mapping, within the selected task budget. Unknown or blocked points are rejected. The rotation buttons also require fresh sensors and whole-robot clearance. Existing maps use **Continue mapping** after localization.
3. Select **Review map**. Add rooms, doorways and destinations at the current robot position or a selected map position. A name does not authorize an unknown coordinate: annotations must have footprint clearance. Explicit graph connections must have an observed connected route. Room nodes are representative navigable coordinates, not inferred room polygons.
4. Name and **Save map**. A transactional SQLite store permits one initial map per environment. Starting a second initial map is rejected. Existing map revisions are updated with optimistic revision checks; stale writers cannot silently overwrite newer work.
5. Scenario changes automatically discover the same map identity and saved revision. A new worker starts **unlocalized**; no simulator starting coordinates are copied into its map pose. Select a known approximate place, select an approximate map position and heading, or use scan-only search, then **Localize**. Ambiguous or mismatched scans are rejected.
6. Select a reachable named destination and **Navigate**. The executive follows short routes using the saved geometry and current obstacles. Arrival requires position within 0.15 m and a 0.5 simulated-second stopped check. This does not imply the scenario objective is complete.
7. **Expand map** explicitly explores reachable free/unknown boundaries within a 1-300 second wall-clock budget. It modifies an in-memory draft of the same map, not a second map identity. Review and save to publish the next revision and retain expanded coverage/visits/frontier attempts. Closed, blocked and unknown regions remain unexplored. Stop or timeout does not claim complete-home coverage.

An existing map can be loaded after a server restart from the panel. The store uses a local SQLite database under the runtime maps directory; no credential values are stored. Standalone scenarios have distinct map identities by scenario ID. Shared-apartment scenarios reuse `shared_apartment_v1`. If an environment's physical layout is intentionally changed, give that environment a new version instead of pretending it is the same house.

## Ownership And Representations

[home_mapping.py](../backend/home_mapping.py) owns the map, coordinate transforms, named graph, scan matching, frontier bookkeeping and transactional storage. [home_mission.py](../backend/home_mission.py) owns mapping stages and mission state. The physics worker owns sensor sampling and every actuation.

| Representation | Contents | Lifetime |
| --- | --- | --- |
| Geometric map | 0.1 m cells: unknown, observed free, occupied; ray evidence and measured depth obstacles | Saved, versioned, read-only during ordinary navigation |
| Named graph | Operator labels, room/doorway/destination type, map-frame pose, explicit validated links | Saved with map |
| Live obstacle layer | Recent lidar hits and head-depth obstacles, free-ray clearing, 2 s expiry | Per worker; never silently persisted as scenario geometry |
| Coverage/history | Observed area, visited cells, frontier attempts and knowledge count at each attempt | Retained in draft; persisted on Save |
| Object observations | Measured visible-surface position, label, timestamp, reported confidence, run/frame provenance and original PNG | Separate SQLite records and images; geometry unchanged |

Frames: `map`, `wheel_odometry`, `robot_base`, `laser`, calibrated optical camera. Coordinates are metres, yaw/pan/tilt radians. The map-to-odometry transform is established from scans after restart. The lidar is fixed to the base; depth projection uses the **corresponding frame's** measured head pan/tilt and odometry. Persistence uses Unix timestamps; control leases and task deadlines use monotonic time. Every map has an environment identity, UUID and revision.

Luna receives only `AgentObservation.spatial`, a compact observation-derived summary with reachable places, pose quality, progress and failures. It does not receive the occupancy array, spectator geometry, scorer state or hidden object coordinates. New actions are `navigate_place`, `explore_map`, `spatial_state`, `cancel_task`, and `remember_object`. Place IDs must come from the supplied summary. Delayed moving selections expire after 15 seconds. The executive runs without further inference while executing an accepted task.

Object labels and confidence are supplied annotations, not verified recognition. `remember_object` projects a normalized box in the paired camera image onto visible measured surfaces, never the simulator's object center. The original image is retained. `currently_observed` requires the same current run, episode and sensor sequence plus age <=2 seconds; otherwise the UI says **Last seen**. No automatic detector, identity association, object-motion tracker or recognition-accuracy claim is included. `ObjectMemory` is an extensible storage protocol.

## Safety And Limits

- Planning inflates occupied **and unknown** space by the current robot footprint plus one grid cell. The initial robot support disk is an explicit flat-floor/proprioceptive assumption, matching the existing local planner's bootstrap concept; it is not an observed corridor.
- Live depth is required before and during motion; lidar/pose data must be fresh. Pose jumps, insufficient scan returns, ambiguous localization, mismatched scans and missing sensors stop or reject motion. Scan residuals are diagnostics, not calibrated covariance or proof against all perceptual aliasing.
- Enhanced mapped sensing renders immutable paired RGB-D snapshots outside the physics thread. Accepted depth can update live obstacle checks before rolling-map processing completes; each retains its original timestamp and measured camera pose. Stop, episode and sensing-generation checks discard revoked in-flight frames. The processed rolling map is never retimestamped to pretend it is newer.
- Raw mapped capture may continue during a pending map job; at most one capture and one map job are in flight. Extra raw frames update only live sensing until the map processor becomes available. Batched depth projection and lidar cell-ID deduplication preserve the exact original cell-update rules.
- In mapping mode, raw manual drive/rotation is guarded before motion and every 50 simulated milliseconds by current localization, depth, observed predicted footprint clearance and the existing whole-robot shield. The UI uses planned `guided_to` waypoints for translation. Non-mapping manual drive remains the original direct-control path, not a sensor-qualified autonomous controller.
- The existing whole-body simulator clearance shield remains a final, privileged safety check. It is not used to construct the map or select destinations. Thus these tests are not evidence of a hardware-ready sensor-only safety system.
- One local segment is at most 1 m, existing speed/acceleration caps remain, and mapped motion leases expire after 0.5 seconds without refresh. At most two stopped replans share the original mission deadline. Stale required depth and localization failure remain terminal.
- Arrival waits for a full 0.5 simulated seconds of measured stable odometry after settling, within the task deadline and a two-second settling bound. Safe braking time does not count as an already stable dwell. Far-object AABB filtering reduces expensive exact collision queries without changing nearby checks or margins.
- Stop, takeover, reset, disconnect and changed task revisions invalidate pending motion. Resume can resume sensing but cannot resurrect a cancelled mission. Manual relocation invalidates localization.
- The current map is bounded to 40x40 m, assumes a flat floor and has no loop closure, map deformation, multi-floor support or calibrated drift covariance. Long guided paths can accumulate odometry error. Re-localize from a stopped known area when needed. Larger-home reliability remains unmeasured.
- A planar scan can miss low/high objects. Head depth and the existing safety shield supplement it. A moving item observed during initial mapping may become a conservative stored obstruction; ordinary scenario navigation does not erase that obstruction.
- Frontier selection is geometric, with at most 20 candidate groups. Two attempts without 25 additional known cells suppress a repeated frontier. A named room region currently means within 3 m of that room's annotated coordinate, not an inferred room boundary. Expansion is bounded and experimental, not guaranteed coverage or shortest exploration time.
- Unsaved mapping/annotation/expansion edits remain drafts and are not crash-recovered. Saved maps, visits and attempt history survive restarts. Whole-home coverage percentage is intentionally unknown because no hidden floor plan supplies a denominator.

## APIs And Scenario Tests

`GET /api/home` returns operator map state. `POST /api/home` accepts `run_id`, `episode_epoch`, `action`, and bounded action fields. Actions: `start_mapping`, `review`, `save_map`, `load_map`, `localize`, `add_place`, `navigate_to`, `explore`, `get_spatial_state`, `remember_object`, `cancel_task`. Mutations require the connected operator and exclusive control. Existing same-origin protection applies. Worker calls additionally bind Stop/task revisions and reject mismatched map identity.

The existing scenario evaluator accepts an explicit saved map without remapping:

```powershell
./.runtime/env/python.exe -m scripts.evaluate_supervised --mode luna_continuous --stage challenges --environment shared_apartment_v1 --challenges apartment flat_kitchen recharge --home-map-id MAP_UUID --localization-place-id HOME_PLACE_UUID --time-limit-s 180 --design persistent-home-map-v1 --output .runtime/performance/NEW_RUN_DIRECTORY
```

Omit the localization place for scan-only search, or use `--localization-pose X_M Y_M YAW_RAD` for an approximate operator hint. Never obtain that hint from evaluator/simulator object coordinates. Localization setup time and seed source are recorded separately. Each trial records map UUID, revision, document SHA-256 and localization quality. The evaluator disables expansion so the saved map remains a matched, frozen input. A failed localization ends the setup; it does not silently create a fresh map. This command uses configured real inference and was **not run** as part of the software implementation.

## Evidence

- Focused checks live in [test_home_mapping.py](../tests/test_home_mapping.py): ray occlusion/unknown space, transactional save-once/reload, footprint/live barriers, transforms, named graph and frontier persistence, scan matching, real scripted guided drive/save/fresh-worker localization/named arrival/Stop, cross-scenario API reuse, evaluator options and image-backed object records.
- [home-mapping.spec.ts](../frontend/tests/home-mapping.spec.ts) covers map controls, annotations, save conflicts, loading/localization, unavailable destinations and responsive canvas rendering. The home-state responses are scripted UI fixtures; the arm/head preparation uses real physics. This is not autonomous mapping evidence.
- Early development checks exposed a list/array bounds bug, excessive sensing work causing safe lease expiry, a destination too close to parking posts for circular clearance, and a native slider margin causing mobile overflow. Fixes and fixture changes do not establish comparative robot-performance gains.
- The bounded shared-apartment expansion probe initially failed with stale depth in TinyRenderer mode. On the default enhanced renderer, newly observed blockage of a frontier exposed missing reselection; bounded reselection then passed with real sensor-driven displacement and unchanged saved geometry. These are development checks with different renderers/code revisions, not a matched performance comparison.
- No real-Luna task trial, large-home loop closure, semantic recognition benchmark, learned-backend comparison or hardware validation was performed for this design. Final gate results are recorded in [DESIGN_PERFORMANCE.md](DESIGN_PERFORMANCE.md#persistent-home-map).