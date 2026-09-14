# Persistent Home Mapping

Design: `persistent-home-map-v1`, native mapped controller `0.1.0`.

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
2. Open **Map > Home map**, then **Start guided mapping**. This folds the arms through the existing validated scan, enables depth, and anchors a new map at the robot's measured starting pose. Use the panel's bounded drive/turn controls or the existing manual controls to visit accessible space.
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
- The existing whole-body simulator clearance shield remains a final, privileged safety check. It is not used to construct the map or select destinations. Thus these tests are not evidence of a hardware-ready sensor-only safety system.
- One local segment is at most 1 m, existing speed/acceleration caps remain, and mapped motion leases expire after 0.5 seconds without refresh. At most two stopped replans share the original mission deadline. Stale required depth and localization failure remain terminal.
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