import asyncio
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
from uuid import uuid4
from typing import Literal

from pydantic import Field

from backend.contracts import StrictModel
from backend.home_mapping import HomeMap
from backend.spatial_memory import MemoryObservation, environment_identity


class MemoryRequest(StrictModel):
    run_id: str
    episode_epoch: int
    context_id: str
    action: Literal["select_profile", "fresh_profile", "checkpoint", "fork_checkpoint", "lookup_room", "find_object_sightings", "get_search_history", "get_exploration_summary", "record_observation"]
    profile_id: str | None = None
    snapshot_id: str | None = None
    name: str = Field(default="New knowledge", min_length=1, max_length=80)
    query: str = Field(default="", max_length=160)
    kind: Literal["room", "object", "search", "entrance"] = "room"
    label: str = Field(default="", max_length=80)
    description: str = Field(default="", max_length=500)
    result: Literal["seen", "not_seen", "inconclusive"] = "inconclusive"
    visibility_limits: str = Field(default="", max_length=500)
    inspection_scope: Literal["viewpoint", "surface", "entrance"] = "viewpoint"
    object_bounds: list[float] | None = Field(default=None, min_length=4, max_length=4)
    entity_id: str | None = Field(default=None, min_length=1, max_length=80)


class MemorySession:
    def __init__(self, store, scope):
        self.store, self.scope = store, scope
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spatial-memory")
        self.pending = deque()
        self.active = True
        self.observation_lock = asyncio.Lock()
        self.closing = None
        self.error = None
        self.summary = {"scope": scope.model_dump(), "rooms": [], "objects": [], "connections": [],
            "search_history": [], "entrances": [], "counts": {}, "motion_authorized": False}

    def submit(self, function, *arguments, **keywords):
        while self.pending and self.pending[0].done():
            completed = self.pending.popleft()
            if not completed.cancelled() and completed.exception() and not isinstance(completed.exception(), ValueError):
                self.error = str(completed.exception())
        if not self.active or len(self.pending) >= 128:
            raise ValueError("MEMORY_WRITE_UNAVAILABLE: context closed or persistence queue full")
        def operation():
            if not self.active:
                raise ValueError("STALE_MEMORY_CONTEXT: pending observation was invalidated")
            result = function(*arguments, **keywords)
            self.summary = self.store.memory_summary(self.scope)
            return result
        future = self.executor.submit(operation)
        self.pending.append(future)
        return future

    def observe(self, observation, image=None, *, allow_new_entity=False, event_id=None, validate=None):
        return self.submit(self.store.ingest, self.scope, observation, image, allow_new_entity=allow_new_entity, event_id=event_id,
            validate=validate)

    def observe_record(self, kind, record, image):
        belief = next((item for item in self.summary.get("rooms", []) if item.get("place_id") == record.get("place_id")), None) if kind == "room" else None
        item = MemoryObservation(kind=kind, source_key=(record.get("place_id") or json_key(record.get("object_bounds", []))) + ":" + str(record.get("room_matches")),
            source="operator_camera" if record.get("source") == "operator_annotation" else "luna_camera",
            sequence=record["spatial_sequence"], frame_ref=record.get("frame_ref", str(record["spatial_sequence"])),
            observed_unix_s=record["observed_unix_s"], pose_m_rad=record.get("pose_m_rad", [0., 0., 0.]),
            head_rad=record.get("head_rad"), label=record["label"], description=record.get("evidence", ""),
            confidence=record["confidence"], entity_id=belief["entity_id"] if belief else record.get("place_id"), place_id=record.get("place_id"),
            position_m=record.get("position_m"), room_matches=record.get("room_matches"))
        return self.observe(item, image, allow_new_entity=kind == "room", event_id=record["observation_id"])

    async def refresh(self):
        self.summary = await asyncio.wrap_future(self.submit(self.store.memory_summary, self.scope))
        return self.summary

    async def flush(self):
        futures = list(self.pending)
        if futures:
            results = await asyncio.gather(*(asyncio.wrap_future(future) for future in futures), return_exceptions=True)
            failures = [result for result in results if isinstance(result, BaseException)]
            if failures:
                self.error = str(failures[0])
                raise ValueError("MEMORY_SAVE_FAILED: " + self.error)
        if self.error:
            raise ValueError("MEMORY_SAVE_FAILED: " + self.error)

    async def persist(self, home):
        copied = deepcopy(home)
        await asyncio.wrap_future(self.submit(self.store.save, copied, copied.name, scope=self.scope))
        return copied.revision

    async def close(self):
        if self.closing is None:
            self.active = False
            async def finish():
                future = self.executor.submit(self.store.close_context, self.scope)
                try:
                    await asyncio.wrap_future(future)
                finally:
                    await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)
            self.closing = asyncio.create_task(finish())
        try:
            await asyncio.shield(self.closing)
        except asyncio.CancelledError:
            await self.closing
            raise


def json_key(value):
    import json
    return json.dumps(value, separators=(",", ":"))


async def configure_memory(worker, *, instance_id=None, profile_id=None, fresh=False, name="Default knowledge", reuse=True):
    await worker.home_state()
    home = worker.home_mission
    if worker.latest.get("busy") or home.active:
        raise ValueError("MEMORY_BUSY: finish or stop the current task before switching knowledge")
    previous = worker.memory
    expected = worker.stop_revision, worker.task_revision
    if previous:
        await persist_memory(worker)
        await previous.flush()
    descriptor = environment_identity(worker.challenge, instance_id or worker.memory_instance_id, worker.scene)
    store = home.store
    await asyncio.to_thread(store.ensure_environment, **descriptor)
    if fresh:
        profile = await asyncio.to_thread(store.create_profile, descriptor["environment_id"], descriptor["revision"], name)
    elif profile_id:
        profiles = await asyncio.to_thread(store.profiles, descriptor["environment_id"], descriptor["revision"])
        profile = next((item for item in profiles if item["profile_id"] == profile_id), None)
        if profile is None:
            raise ValueError("MEMORY_SCOPE_MISMATCH: profile belongs to another environment or revision")
    else:
        profile = await asyncio.to_thread(store.selected_profile, descriptor["environment_id"], descriptor["revision"])
    catalog = await asyncio.to_thread(store.catalog)
    saved = next((item for item in catalog if item["profile_id"] == profile["profile_id"] and item["environment_revision"] == descriptor["revision"]), None)
    selected = None
    if saved and reuse:
        selected = await asyncio.to_thread(store.load, saved["map_id"], home.environment_id,
            profile_id=profile["profile_id"], environment_revision=descriptor["revision"])
    imported = []
    if selected is None and not fresh and not profile_id and reuse and home.home and home.home.environment_revision == "legacy" and getattr(home.home, "legacy_import_approved", False):
        selected = await worker.call(lambda sim: deepcopy(home.home))
        for kind in ("room", "object"):
            records = await asyncio.to_thread(store.room_observations if kind == "room" else store.object_observations, selected.identity)
            for record in records:
                image = await asyncio.to_thread(store.room_image if kind == "room" else store.object_image, selected.identity, record["observation_id"])
                imported.append((kind, record, image))
        selected.identity, selected.revision, selected.saved = str(uuid4()), 0, False
        for place in selected.places:
            place.update(map_id=selected.identity, requires_revalidation=True)
        selected.profile_id, selected.environment_revision = profile["profile_id"], descriptor["revision"]
        selected.memory_environment_id = descriptor["environment_id"]
        await asyncio.to_thread(store.save, selected, selected.name)
    seed = selected or HomeMap(home.environment_id)
    seed.profile_id, seed.environment_revision = profile["profile_id"], descriptor["revision"]
    seed.memory_environment_id = descriptor["environment_id"]
    scope = await asyncio.to_thread(store.open_context, seed, worker.latest["run_id"], worker.epoch)
    session = MemorySession(store, scope)
    try:
        for kind, record, image in imported:
            item = MemoryObservation(kind=kind, source_key="legacy:" + record["observation_id"], source="operator_camera",
                sequence=record.get("spatial_sequence", 0), frame_ref=record.get("frame_ref", record["observation_id"]),
                observed_unix_s=max(1., record["observed_unix_s"]), pose_m_rad=record.get("pose_m_rad", [0., 0., 0.]),
                position_m=record.get("position_m"), label=record["label"], description=record.get("evidence", ""),
                confidence=record.get("confidence", .5), entity_id=record.get("place_id"), place_id=record.get("place_id"), requires_revalidation=True)
            await asyncio.wrap_future(session.observe(item, image, allow_new_entity=kind == "room"))
        await session.refresh()
        rooms = await asyncio.to_thread(store.room_observations, seed.identity)
        objects = await asyncio.to_thread(store.object_observations, seed.identity)
        def install(sim):
            if (worker.closed or worker.memory is not previous or expected != (worker.stop_revision, worker.task_revision)
                    or home.active or worker.inference_owner):
                raise ValueError("STALE_MEMORY_CONTEXT: authority changed during profile selection")
            home.invalidate("Knowledge profile changed; fresh localization required")
            home.home, home.task, home.workflow = selected, None, None
            home.room_records, home.object_records = rooms, objects
            home.stage = "loaded" if selected else "empty"
            home.last_odometry = home.last_mapped_pose = None
            from backend.spatial import ObservedMap
            worker.spatial_generation += 1
            worker.mapped_depth_sensor = None
            worker.spatial_map = ObservedMap(sim.run_id, sim.epoch) if worker.spatial_enabled else None
            worker.spatial_frames.clear()
            worker.map_history = worker.camera_history = None
            worker.spatial_error = None
            worker.object_goal = None
            home.memory_path = []
            home.memory_origin = None
            worker.memory, worker.memory_instance_id = session, descriptor["environment_id"]
            worker.memory_descriptor, worker.memory_profile = descriptor, profile
            worker.memory_catalog = [item for item in catalog if item["profile_id"] == profile["profile_id"]]
        await worker.call(install)
        if previous:
            await previous.close()
        await asyncio.to_thread(store.select_profile, descriptor["environment_id"], descriptor["revision"], profile["profile_id"])
        return await memory_state(worker)
    except BaseException:
        await session.close()
        raise


async def persist_memory(worker):
    session = worker.memory
    if session is None:
        return
    await session.flush()
    home = await worker.call(lambda sim: deepcopy(worker.home_mission.home))
    if home is None or not home.scan_count or not worker.home_mission.allow_expansion:
        return
    home.places[:] = [place for place in home.places if not place.get("mission_id")]
    revision = await session.persist(home)
    catalog = await asyncio.to_thread(session.store.catalog)
    def apply(sim):
        current = worker.home_mission.home
        if worker.memory is session and current and current.identity == home.identity and current.revision == home.revision:
            current.revision = revision
            worker.memory_catalog = [item for item in catalog if item["profile_id"] == session.scope.profile_id]
    await worker.call(apply)


async def memory_state(worker):
    session = worker.memory
    if session is None:
        return {"enabled": False}
    scope, store = session.scope, session.store
    profiles = await asyncio.to_thread(store.profiles, scope.environment_id, scope.environment_revision)
    snapshots = await asyncio.to_thread(store.snapshots, scope.environment_id, scope.environment_revision, scope.profile_id)
    if worker.memory is not session or not session.active:
        raise ValueError("STALE_MEMORY_CONTEXT: knowledge selection changed")
    return {"enabled": True, **deepcopy(session.summary), "profile": worker.memory_profile, "profiles": profiles,
        "snapshots": snapshots, "pending_writes": sum(not future.done() for future in session.pending), "error": session.error,
        "generation": worker.memory_descriptor["generation"], "scenario_id": worker.memory_descriptor["scenario_id"]}


async def record_observation(worker, **arguments):
    session = worker.memory
    if session is None:
        raise ValueError("STALE_MEMORY_CONTEXT: no active knowledge profile")
    async with session.observation_lock:
        return await _record_observation(worker, **arguments)


async def _record_observation(worker, *, context_id, kind, label="", description="", target="", result="inconclusive",
                             visibility="unknown", visibility_limits="", inspection_scope="viewpoint", bounds=None,
                             entity_id=None, selected_evidence=None, confidence=.5, place_id=None, source="luna_camera",
                             model_observation=None, mission=None):
    from backend.home_mapping import transform_pose
    from backend.spatial import measure_visible_region
    session = worker.memory
    if session is None or session.scope.context_id != context_id:
        raise ValueError("STALE_MEMORY_CONTEXT: select the current knowledge context")
    if kind not in {"room", "object", "search", "entrance"}:
        raise ValueError("Only the worker can report executed pathway evidence")
    expected = worker.stop_revision, worker.task_revision
    frame_transform = tuple(worker.home_mission.transform) if worker.home_mission.transform is not None else None
    def capture(sim):
        if (worker.memory is not session or not session.active or sim.cancel.is_set() or expected != (worker.stop_revision, worker.task_revision)
                or (worker.home_mission and worker.home_mission.active) or (worker.continuous and worker.continuous.active)):
            raise ValueError("STALE_MEMORY_CONTEXT: stopped observation authority changed")
        if selected_evidence is None:
            worker._sample_spatial(force=True)
            worker.home_mission.sample(force=True)
        paired = selected_evidence or (next(reversed(worker.spatial_frames.values()))[:2] if worker.spatial_frames else None)
        if paired is None:
            raise ValueError("MEMORY_EVIDENCE_REQUIRED: capture paired camera/depth first")
        sensor, image = paired
        if model_observation is not None:
            if (model_observation.run_id != sensor.run_id or model_observation.episode_epoch != sensor.episode_epoch
                    or list(model_observation.odometry_m_rad) != list(sensor.odometry_m_rad)
                    or list(model_observation.head_rad) != list(sensor.head_rad)
                    or model_observation.simulated_time_s != sensor.simulated_time_s
                    or sim.frames.get(model_observation.frame_ref) != image):
                raise ValueError("STALE_OBSERVATION: report requires its original paired camera image")
        if mission is not None:
            worker._mission_read_guard(sim, worker._mission_read_authority(mission))
            mission.check((sim.run_id, sim.epoch, *expected))
        if (sensor.run_id != sim.run_id or sensor.episode_epoch != sim.epoch or not 0 <= time.monotonic()-sensor.captured_at <= 15.
                or math.dist(sensor.odometry_m_rad[:2], sim.odometry[:2]) > .05):
            raise ValueError("STALE_OBSERVATION: memory requires the paired current viewpoint")
        current = sim.observe(render=False)
        if abs(sensor.odometry_m_rad[2]-sim.odometry[2]) > .05 or max(abs(actual-observed) for actual, observed in zip(current.head_rad, sensor.head_rad)) > .05:
            raise ValueError("STALE_OBSERVATION: camera viewpoint changed")
        home = worker.home_mission
        localized = home.home is not None and home.transform is not None and home.localization["status"] == "localized"
        if localized:
            home.require_localized()
        pose = transform_pose(sensor.odometry_m_rad, home.transform) if localized else list(sensor.odometry_m_rad)
        position = None
        if kind == "object" and localized and bounds is not None:
            try:
                measured = measure_visible_region(sensor, bounds)["surface_median_xyz_m"]
                position = transform_pose([*measured[:2], 0.], home.transform)[:2] + [measured[2]]
            except ValueError:
                pass
        source_key = json_key([kind, bounds, target, label if kind == "room" else None, inspection_scope])
        previous = next((item for item in session.summary.get("rooms" if kind == "room" else "objects", [])
            if item.get("source_key") == source_key and item.get("last_source_sequence") == sensor.sequence
            and item.get("last_run_id") == session.scope.run_id), None)
        linked = place_id or (previous.get("place_id") if previous else None)
        associated = entity_id or (previous.get("entity_id") if previous else None)
        if linked and localized:
            existing_place = next((item for item in home.home.places if item["place_id"] == linked), None)
            if existing_place and math.dist(existing_place["pose_m_rad"][:2], pose[:2]) > .2:
                linked = None
        if localized and kind == "room" and not linked:
            room = next((item for item in session.summary.get("rooms", []) if item["label"] == label
                and not item["requires_revalidation"] and math.dist(item["pose_m_rad"][:2], pose[:2]) <= .2), None)
            if room:
                linked, associated = room.get("place_id"), room["entity_id"]
        staged_place = None
        if localized and kind in {"room", "object"} and not linked:
            try:
                draft = copy(home.home)
                draft.places, draft.edges = list(home.home.places), list(home.home.edges)
                staged_place = draft.add_place(label[:60] + " " + str(uuid4())[:8], "room" if kind == "room" else "destination", pose, sim.robot_footprint()["radius_m"])
                staged_place.update(source="camera_room_hypothesis" if kind == "room" else "object_observation_viewpoint", identity_status="tentative")
                linked = staged_place["place_id"]
            except ValueError:
                pass
        if linked and staged_place is None and (home.home is None or not any(place["place_id"] == linked for place in home.home.places)):
            raise ValueError("UNKNOWN_PLACE: sighting viewpoint belongs to another map")
        item = MemoryObservation(kind=kind, source_key=source_key,
            source=source, sequence=sensor.sequence, frame_ref=str(sensor.sequence),
            observed_unix_s=time.time()-(time.monotonic()-sensor.captured_at), pose_m_rad=pose,
            pose_frame="map" if localized else "wheel_odometry", head_rad=sensor.head_rad, label=label,
            description=description, confidence=confidence, entity_id=associated, place_id=linked,
            position_m=position, metric_valid=localized, target=target, result=result,
            visibility=visibility, visibility_limits=visibility_limits, inspection_scope=inspection_scope)
        def still_current():
            return (worker.memory is session and session.active and not worker.closed and not sim.cancel.is_set()
                and expected == (worker.stop_revision, worker.task_revision)
                and frame_transform == (tuple(home.transform) if home.transform is not None else None))
        return session.observe(item, image, validate=still_current), staged_place
    future, staged_place = await worker.call(capture)
    record = await asyncio.wrap_future(future)
    def publish(sim):
        if (worker.memory is not session or not session.active or worker.closed or sim.cancel.is_set()
                or expected != (worker.stop_revision, worker.task_revision)):
            raise ValueError("STALE_MEMORY_CONTEXT: memory write completed for an obsolete context")
        home = worker.home_mission.home
        if frame_transform != (tuple(worker.home_mission.transform) if worker.home_mission.transform is not None else None):
            raise ValueError("STALE_MEMORY_CONTEXT: localization changed before publication")
        if staged_place and home and home.identity == session.scope.map_id and record["place_id"] == staged_place["place_id"]:
            if not any(place["place_id"] == staged_place["place_id"] for place in home.places):
                staged_place.update(memory_entity_id=record["entity_id"], evidence_id=record["observation_id"])
                home.places.append(staged_place)
                home.annotations_dirty = True
        if kind == "room" and record not in worker.home_mission.room_records:
            worker.home_mission.room_records.insert(0, record)
            del worker.home_mission.room_records[100:]
        if kind == "object" and record not in worker.home_mission.object_records:
            worker.home_mission.object_records.insert(0, record)
            del worker.home_mission.object_records[64:]
        if home and record.get("place_id") and not record["requires_revalidation"]:
            for place in home.places:
                if place["place_id"] == record["place_id"] and math.dist(place["pose_m_rad"][:2], record["pose_m_rad"][:2]) <= .2:
                    place.update(requires_revalidation=False, memory_entity_id=record["entity_id"])
    await worker.call(publish)
    return record


async def lookup_memory(worker, kind, query):
    session = worker.memory
    if session is None:
        return {"status": "unavailable", "records": [], "motion_authorized": False}
    pose = worker.home_mission.pose
    if kind == "lookup_room":
        records = await asyncio.wrap_future(session.submit(session.store.lookup_room, session.scope, query))
    elif kind == "find_object_sightings":
        records = await asyncio.wrap_future(session.submit(session.store.find_object_sightings, session.scope, query, pose))
    elif kind == "get_search_history":
        records = await asyncio.wrap_future(session.submit(session.store.search_history, session.scope, query))
    else:
        summary = deepcopy(session.summary)
        summary["scope"] = {key: summary["scope"][key] for key in ("profile_id", "map_id", "frame_revision")}
        summary["connections"] = [{key: value for key, value in item.items() if key != "path_m"} for item in summary["connections"]]
        return {"status": "memory_summary", "summary": summary, "motion_authorized": False}
    if worker.memory is not session or not session.active:
        raise ValueError("STALE_MEMORY_CONTEXT: memory read belongs to an obsolete selection")
    now = time.time()
    for item in records:
        observed = item.get("last_observed_unix_s", item.get("observed_unix_s", now))
        item["age_s"] = now-observed if now >= observed else None
    return {"status": kind, "records": records, "motion_authorized": False}


async def memory_command(worker, request):
    session = worker.memory
    if (session is None or not session.active or request.context_id != session.scope.context_id
            or request.run_id != worker.latest["run_id"] or request.episode_epoch != worker.epoch):
        raise ValueError("STALE_MEMORY_CONTEXT: refresh the active environment and knowledge profile")
    if request.action in {"lookup_room", "find_object_sightings", "get_search_history", "get_exploration_summary"}:
        return await lookup_memory(worker, request.action, request.query)
    if request.action == "record_observation":
        return await record_observation(worker, context_id=request.context_id, kind=request.kind,
            label=request.label, description=request.description, target=request.query, result=request.result,
            inspection_scope=request.inspection_scope, visibility="partial", visibility_limits=request.visibility_limits,
            bounds=request.object_bounds, source="operator_camera", entity_id=request.entity_id)
    if worker.latest.get("busy") or worker.home_mission.active or worker.inference_owner:
        raise ValueError("MEMORY_BUSY: stop the active task before changing knowledge")
    if request.action == "checkpoint":
        async with session.observation_lock:
            await persist_memory(worker)
            copied = await worker.call(lambda sim: deepcopy(worker.home_mission.home))
            if copied is None:
                raise ValueError("MAP_REQUIRED: observe a map before saving a checkpoint")
            copied.places[:] = [place for place in copied.places if not place.get("mission_id")]
            await asyncio.wrap_future(session.submit(session.store.checkpoint, session.scope, copied, request.name))
        return await memory_state(worker)
    if request.action == "fork_checkpoint":
        fork = await asyncio.to_thread(session.store.fork_snapshot, session.scope.environment_id,
            session.scope.environment_revision, request.snapshot_id, request.name)
        return await configure_memory(worker, profile_id=fork["profile_id"])
    if request.action == "select_profile" and not request.profile_id:
        raise ValueError("PROFILE_REQUIRED: select an existing knowledge profile")
    return await configure_memory(worker, profile_id=request.profile_id if request.action == "select_profile" else None,
        fresh=request.action == "fresh_profile", name=request.name)


async def rotate_memory_context(worker):
    session = worker.memory
    if session is None:
        return
    await session.flush()
    home = await worker.call(lambda sim: deepcopy(worker.home_mission.home))
    await session.close()
    scope = await asyncio.to_thread(session.store.open_context, home, worker.latest["run_id"], worker.epoch,
        frame_revision=session.scope.frame_revision)
    replacement = MemorySession(session.store, scope)
    await replacement.refresh()
    worker.memory = replacement


def track_memory_travel(home):
    worker, session = home.worker, home.worker.memory
    if not session or not home.pose or home.localization["status"] != "localized":
        return
    if (worker.sim.cancel.is_set() or home.captured_at is None or not 0 <= time.monotonic()-home.captured_at <= 1.
            or worker.latest.get("proximity", {}).get("collisions")):
        home.memory_path, home.memory_origin = [], None
        return
    pose = home.pose
    if home.memory_path and math.dist(pose[:2], home.memory_path[-1]) > .5:
        home.memory_path, home.memory_origin = [], None
    if not home.memory_path or math.dist(pose[:2], home.memory_path[-1]) >= .04:
        home.memory_path.append(list(pose[:2]))
    if len(home.memory_path) > 1024:
        home.memory_path, home.memory_origin = [list(pose[:2])], None
    if home.home.saved:
        column, row = home.home.indices(pose[:2])
        if home.home.inside([column, row]):
            home.home.visits[row, column] = max(1, home.home.visits[row, column])
    rooms = session.summary.get("rooms", [])
    current = next((room for room in rooms if not room["requires_revalidation"] and math.dist(pose[:2], room["pose_m_rad"][:2]) <= .2), None)
    if not current:
        return
    if (home.memory_origin and home.memory_origin["entity_id"] != current["entity_id"] and len(home.memory_path) >= 2
            and math.dist(home.memory_origin["pose_m_rad"][:2], current["pose_m_rad"][:2]) >= .5):
        source = home.memory_origin
        observation = MemoryObservation(kind="traversal", source_key=str(uuid4()), source="worker_odometry",
            sequence=worker.spatial_sequence, frame_ref=str(worker.spatial_sequence), observed_unix_s=time.time(),
            pose_m_rad=list(pose), from_entity_id=source["entity_id"], to_entity_id=current["entity_id"],
            path_m=home.memory_path, duration_s=max(0., time.monotonic()-home.memory_departed_at))
        try:
            session.observe(observation)
        except ValueError as error:
            session.error = str(error)
    if not home.memory_origin or home.memory_origin["entity_id"] != current["entity_id"]:
        home.memory_origin, home.memory_path = current, [list(pose[:2])]
        home.memory_departed_at = time.monotonic()