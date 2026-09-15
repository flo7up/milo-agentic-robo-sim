import asyncio
import json
import math
import time

import pytest

from backend.mission import Mission, MissionDecision, MissionPlan
from tests.test_agent import ScriptedModel, controller_for, model_response, start_settings


@pytest.mark.parametrize("field,value", [("objective_duration_s", 0), ("objective_duration_s", 61),
    ("objective_duration_s", float("nan")), ("objective_travel_m", 0),
    ("objective_travel_m", 6.1), ("objective_travel_m", float("inf"))])
def test_objective_schema_has_finite_conservative_bounds(field, value):
    with pytest.raises(ValueError):
        MissionDecision(action="explore", **{field: value})


def test_objective_renewal_preserves_operation_travel_and_original_deadline():
    now = [10.]
    authority = ("run", 0, 2, 3)
    mission = Mission(*authority, deadline=70., clock=lambda: now[0])
    mission.configure(MissionPlan(kind="room", target="Bathroom"))
    operation = mission.begin("exploring", authority)
    decision = MissionDecision(action="explore", objective_duration_s=20., objective_travel_m=2.)
    mission.authorize_objective(operation, decision, authority, [0., 0., 0.])
    mission.check_objective(operation, authority, [1., 0., 0.])
    now[0] = 25.
    mission.authorize_objective(operation, MissionDecision(action="explore"), authority, [.5, 0., 0.], renew=True)
    assert mission.operation == operation and mission.objective.travel_m == 1.5
    assert mission.objective.travel_limit_m == 7.5 and mission.objective.expires_at == mission.deadline == 70.
    assert mission.objective.revision == 1 and not mission.receipts


@pytest.mark.parametrize("fault", ["duration", "travel", "stop", "task", "operation", "cancel"])
def test_objective_cannot_renew_or_resurrect_after_expiry_or_revocation(fault):
    now = [0.]
    authority = ("run", 0, 2, 3)
    mission = Mission(*authority, deadline=100., clock=lambda: now[0])
    mission.configure(MissionPlan(kind="room", target="Bathroom"))
    operation = mission.begin("exploring", authority)
    decision = MissionDecision(action="explore", objective_duration_s=10., objective_travel_m=1.)
    mission.authorize_objective(operation, decision, authority, [0., 0., 0.])
    pose, changed = [0., 0., 0.], list(authority)
    if fault == "duration":
        now[0] = 10.
    elif fault == "travel":
        pose[0] = 1.
    elif fault in {"stop", "task"}:
        changed[2 if fault == "stop" else 3] += 1
    elif fault == "operation":
        operation = "late"
    else:
        mission.finish("cancelled", "Stop")
    with pytest.raises(ValueError):
        mission.check_objective(operation, changed, pose)
    with pytest.raises(ValueError):
        mission.authorize_objective(mission.operation, decision, authority, [0., 0., 0.], renew=True)
    assert mission.objective.status == "ended" and not mission.receipts


class DelayedReviewModel(ScriptedModel):
    def __init__(self, duration=60.):
        replies = [{"action": "plan", "plan": {"kind": "room", "target": "Bathroom"}},
            {"action": "explore", "objective_duration_s": duration}, {"action": "explore"},
            {"action": "look", "pitch_rad": .2},
            {"action": "report_observation", "evidence_text": "Current visible fixtures remain unverified."}]
        super().__init__([model_response("guide_mission", json.dumps(reply), call_id=f"async-review-{index}")
            for index, reply in enumerate(replies)])
        self.review_entered = asyncio.Event()
        self.review_release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.inflight = self.max_inflight = 0

    async def respond(self, *args):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            response = await super().respond(*args)
            if len(self.inputs) == 3:
                self.review_entered.set()
                try:
                    await self.review_release.wait()
                except asyncio.CancelledError:
                    self.cancelled.set()
                    await self.review_release.wait()
            return response
        finally:
            self.inflight -= 1


@pytest.mark.parametrize("ending", ["continue", "stop", "expiry", "stale_sensor", "blocked_path"])
async def test_delayed_review_keeps_one_enhanced_physics_task_and_revokes_late_motion(
        tmp_path, monkeypatch, record_property, ending):
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from backend.mission_supervisor import MappedReviewOperation
    from backend.worker import SimulationWorker
    worker = SimulationWorker(challenge=get_challenge("kitchen_bathroom"), rendering="enhanced", pace=True)
    model = DelayedReviewModel()
    controller = controller_for(model)
    samples = []
    pauses = []
    original_review = MappedReviewOperation.review

    async def review_during_travel(mapped):
        initial_travel = mapped.mission.objective.travel_m
        async with asyncio.timeout(15.):
            while mapped.mission.objective.travel_m - initial_travel < .15:
                result = await mapped.poll()
                assert result["task"]["status"] == "running", result
                await asyncio.sleep(.05)
        mapped.reviewed_at = time.monotonic() - 15.
        return await original_review(mapped)

    monkeypatch.setattr(MappedReviewOperation, "review", review_during_travel)
    try:
        await asyncio.wrap_future(worker.ready)
        home = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        worker.home_mission = home
        original_tick, original_fail = home.tick, home.fail

        def tracked_tick():
            original_tick()
            samples.append({"time": time.monotonic(), "pose": worker.sim.odometry.tolist(),
                "task": (home.task or {}).get("task_id"), "operation": controller.mission.operation if controller.mission else None,
                "status": (home.task or {}).get("status"), "contacts": list(worker.sim.proximity_sensors().collisions)})

        def tracked_fail(reason, status="failed"):
            if home.active:
                pauses.append({"reason": reason, "status": status, "time": time.monotonic()})
            return original_fail(reason, status)

        monkeypatch.setattr(home, "tick", tracked_tick)
        monkeypatch.setattr(home, "fail", tracked_fail)
        controller.start(worker, start_settings(worker, execution_mode="luna_continuous", unified_mission=True,
            map_context=True, reasoning="low", images_per_request=2, feedback_interval_s=15.,
            mission_budget_s=90., max_turns=5 if ending == "continue" else 3,
            max_model_requests=5 if ending == "continue" else 3))
        await asyncio.wait_for(model.review_entered.wait(), 40.)
        started = time.monotonic()
        mission, operation, task_id = controller.mission, controller.mission.operation, home.task["task_id"]
        original_deadline = mission.deadline
        assert operation and home.active, {"task": home.task, "state": controller.state}
        payload = json.loads(model.inputs[2][-1]["content"][0]["text"])
        assert payload["last_execution"]["task"]["status"] == "running"
        assert payload["last_execution"]["task"]["task_id"] == task_id
        assert payload["mission"]["operation_id"] == operation
        if payload["map_context"]["status"] == "available":
            assert payload["observation"]["observed_map"]["geometry_source"] == "accumulated_sensor_map"
        else:
            assert payload["map_context"]["status"] == "unavailable" and payload["map_context"]["reason"]
            assert payload["observation"]["observed_map"] is None
        assert not pauses, pauses
        async with asyncio.timeout(12.):
            while True:
                during = [sample for sample in samples if sample["time"] >= started]
                distance = sum(math.dist(first["pose"][:2], second["pose"][:2])
                    for first, second in zip(during, during[1:]))
                if time.monotonic() - started >= 3. and distance >= .15:
                    break
                assert home.active, {"task": home.task, "pauses": pauses}
                await asyncio.sleep(.05)
        assert all(sample["task"] == task_id and sample["operation"] == operation and sample["status"] == "running"
            for sample in during)
        assert not pauses and len(model.inputs) == 3 and model.inflight == 1
        if ending == "stop":
            worker.stop()
            await asyncio.wait_for(model.cancelled.wait(), 3.)
            frozen = await worker.call(lambda sim: (sim.odometry.copy(), sim.ticks))
            await asyncio.sleep(.25)
            after = await worker.call(lambda sim: (sim.odometry.copy(), sim.ticks))
            assert math.dist(frozen[0], after[0]) == 0. and frozen[1] == after[1]
        elif ending in {"expiry", "stale_sensor", "blocked_path"}:
            if ending == "expiry":
                await worker.call(lambda sim: setattr(mission.objective, "expires_at", time.monotonic()))
            elif ending == "stale_sensor":
                monkeypatch.setattr(HomeMission, "captured_at", property(lambda self: started - 2.))
            elif ending == "blocked_path":
                monkeypatch.setattr(home, "path_valid", lambda *args, **kwargs: False)
            async with asyncio.timeout(12.):
                while home.active:
                    await asyncio.sleep(.05)
            terminal = await worker.mission_objective(mission, operation)
            assert terminal["task"]["status"] != "running"
            assert not model.review_release.is_set() and not controller.task.done()
            assert mission.objective.status == "ended" and mission.objective.revision == 0
            if ending == "stale_sensor":
                assert "SPATIAL_STALE" in home.task["reason"], home.task
            elif ending == "blocked_path":
                assert "OBSERVED_PATH_BLOCKED" in home.task["reason"], home.task
            await asyncio.sleep(.5)
            stopped = await worker.call(lambda sim: sim.odometry.copy())
            await asyncio.sleep(.5)
            after = await worker.call(lambda sim: sim.odometry.copy())
            assert math.dist(stopped[:2], after[:2]) < .005
        model.review_release.set()
        await asyncio.wait_for(controller.task, 45.)
        events = controller.trace()["events"]
        renewals = [event for event in events if event["title"] == "Mission objective renewed"]
        if ending == "continue":
            assert len(renewals) == 1 and renewals[0]["payload"]["task"]["task_id"] == task_id
            assert renewals[0]["payload"]["mission"]["operation_id"] == operation
            deferred = [event for event in events if event["title"] == "Moving mission decision deferred"]
            assert len(deferred) == 1 and deferred[0]["payload"]["discarded_action"] == "look"
            assert deferred[0]["payload"]["fresh_spatial_sequence"] > deferred[0]["payload"]["source_spatial_sequence"]
            reports = [event["payload"]["result"] for event in events if event["title"] == "Mission capability feedback"
                and event["payload"]["result"].get("status") == "observation_reported"]
            assert len(reports) == 1 and not reports[0]["arrival_verified"] and not reports[0]["identity_verified"]
            assert reports[0]["source_frame"] == "wheel_odometry" and reports[0]["source_run_id"] == mission.authority[0]
            assert controller.state["error"] is None, controller.state
        else:
            assert not renewals and home.task["task_id"] == task_id and len(model.inputs) == 3
        assert mission.deadline == original_deadline and "target" not in mission.receipts
        assert model.max_inflight == 1 and model.inflight == 0 and not home.active
        assert mission.operation is None and getattr(worker, "_mission_objective_binding", None) is None
        assert worker.latest["stopped"] and not any(sample["contacts"] for sample in samples)
        record_property("async_review", json.dumps({"evidence": "scripted_enhanced_physics", "ending": ending,
            "review_trigger": "test_checkpoint_after_measured_translation",
            "pending_translation_m": distance, "pending_duration_s": during[-1]["time"] - started,
            "task_id": task_id, "operation_id": operation, "mission_id": mission.identity,
            "max_inflight": model.max_inflight, "requests": len(model.inputs), "renewals": len(renewals),
            "contacts": sum(len(sample["contacts"]) for sample in samples), "pauses": pauses}))
    finally:
        record_property("review_diagnostic", json.dumps({"ending": ending, "requests": len(model.inputs),
            "mission": controller.mission.state() if controller.mission else None,
            "task": worker.home_mission.task if worker.home_mission else None, "pauses": pauses,
            "error": controller.state.get("error"),
            "events": [{"title": event["title"], "payload": event["payload"]}
                for event in controller.trace()["events"] if event["kind"] in {"policy", "error"}],
            "recent_samples": samples[-20:]}, default=str))
        model.review_release.set()
        await controller.halt()
        await worker.close()


@pytest.fixture
def objective_worker(monkeypatch):
    import threading
    from types import SimpleNamespace
    import numpy as np
    from backend.worker import SimulationWorker
    worker = object.__new__(SimulationWorker)
    worker.sim = SimpleNamespace(run_id="run", epoch=0, odometry=np.zeros(3), cancel=threading.Event(), on_tick=lambda: None)
    worker.epoch = worker.stop_revision = worker.task_revision = 0
    worker.closed, worker.powered = False, True
    worker.inference_owner = "controller"
    worker.renderer = worker.skill = worker.ros_navigation = worker.continuous = worker.navigation = None
    worker.spatial_enabled, worker.spatial_error, worker.spatial_map = True, None, object()
    sensor = SimpleNamespace(run_id="run", episode_epoch=0, captured_at=time.monotonic(),
        odometry_m_rad=[0., 0., 0.], head_rad=[0., .2], simulated_time_s=1.)
    worker.spatial_frames = {1: (sensor, b"image", None)}
    worker._sample_spatial = lambda **kwargs: None
    mission = Mission("run", 0, 0, 0, deadline=time.monotonic() + 90.)
    mission.configure(MissionPlan(kind="room", target="Bathroom"))
    operation = mission.begin("exploring", mission.authority)

    class Home:
        task = None
        mission_owner = mission.identity

        @property
        def active(self):
            return bool(self.task and self.task["status"] == "running")

        def command(self, request, *args):
            self.task = {"status": "running", "task_id": "task", "mission_id": self.mission_owner,
                "reason": "Exploring", "deadline": time.monotonic() + request.time_budget}

        def fail(self, reason, status="failed"):
            if self.active:
                self.task.update(reason=reason, status=status)

        def state(self, compact=True):
            return {"task": self.task.copy() if self.task else None}

        def require_localized(self):
            return None

    worker.home_mission = Home()

    async def call(operation):
        return operation(worker.sim)

    worker.call = call
    return worker, mission, operation, sensor


@pytest.mark.parametrize("fault", ["age", "future", "run", "epoch", "pose", "heading", "stop_revision",
    "task_revision", "owner", "inference_owner", "stop", "closed", "power", "foreign_navigation"])
def test_moving_feedback_rejects_stale_frames_and_foreign_authority(objective_worker, fault):
    from types import SimpleNamespace
    worker, mission, _, sensor = objective_worker
    expected = worker._mission_read_authority(mission)
    if fault == "age":
        sensor.captured_at -= 1.1
    elif fault == "future":
        sensor.captured_at += 2.
    elif fault in {"run", "epoch"}:
        setattr(sensor, "run_id" if fault == "run" else "episode_epoch", "foreign" if fault == "run" else 1)
    elif fault in {"pose", "heading"}:
        sensor.odometry_m_rad[0 if fault == "pose" else 2] = .4
    elif fault in {"stop_revision", "task_revision"}:
        setattr(worker, fault, 1)
    elif fault == "owner":
        worker.home_mission.mission_owner = "replacement"
    elif fault == "inference_owner":
        worker.inference_owner = "replacement"
    elif fault == "stop":
        worker.sim.cancel.set()
    elif fault == "closed":
        worker.closed = True
    elif fault == "power":
        worker.powered = False
    else:
        worker.navigation = SimpleNamespace(status="running")
    with pytest.raises(ValueError):
        worker._mission_read_guard(worker.sim, expected, sensor)


async def test_objective_worker_renewal_never_changes_trajectory_lease(objective_worker):
    from backend.navigation import NavigationRuntime
    worker, mission, operation, sensor = objective_worker
    await worker.mission_objective(mission, operation, decision=MissionDecision(action="explore", objective_duration_s=1.))
    task = worker.home_mission.task
    runtime = worker.navigation = NavigationRuntime()
    runtime.expires_at = time.monotonic() - 1.
    runtime.skill_deadline = time.monotonic() - 2.
    original = (runtime.expires_at, runtime.skill_deadline, runtime.revision, list(runtime.buffer))
    sensor.odometry_m_rad = [.1, 0., 0.]
    worker.sim.odometry[0] = .1
    await worker.mission_objective(mission, operation, decision=MissionDecision(action="explore"), renew=True)
    assert worker.home_mission.task is task and mission.objective.revision == 1
    assert original == (runtime.expires_at, runtime.skill_deadline, runtime.revision, list(runtime.buffer))
    assert mission.objective.travel_m == pytest.approx(.1)
    assert task["deadline"] == mission.objective.expires_at <= mission.deadline
    await worker.mission_objective(mission, operation, end_reason="Test ended")
    assert not worker.home_mission.active and worker._mission_objective_binding is None


@pytest.mark.parametrize("fault", ["stop", "task", "deadline", "exception"])
async def test_objective_start_interruption_brakes_only_new_task(objective_worker, monkeypatch, fault):
    worker, mission, operation, _ = objective_worker
    home = worker.home_mission
    original = home.command

    def interrupted(*args):
        original(*args)
        if fault == "stop":
            worker.sim.cancel.set()
        elif fault == "task":
            worker.task_revision += 1
        elif fault == "deadline":
            mission.objective.expires_at = time.monotonic() - 1.
        else:
            raise RuntimeError("start failed after task creation")

    monkeypatch.setattr(home, "command", interrupted)
    with pytest.raises((ValueError, RuntimeError)):
        await worker.mission_objective(mission, operation, decision=MissionDecision(action="explore"))
    assert home.task["status"] == "cancelled" and mission.objective.status == "ended"
    assert getattr(worker, "_mission_objective_binding", None) is None


@pytest.mark.parametrize("fault", ["travel", "deadline", "task", "failed_path", "replaced"])
async def test_worker_tick_revokes_objective_without_inference_poll(objective_worker, fault):
    worker, mission, operation, _ = objective_worker
    original_tick = worker.sim.on_tick
    await worker.mission_objective(mission, operation, decision=MissionDecision(action="explore"))
    if fault == "travel":
        worker.sim.odometry[0] = 6.
    elif fault == "deadline":
        mission.objective.expires_at = time.monotonic() - 1.
    elif fault == "task":
        worker.task_revision += 1
    elif fault == "failed_path":
        worker.home_mission.fail("OBSERVED_PATH_BLOCKED")
    else:
        worker.home_mission.task = {"task_id": "replacement", "status": "running", "reason": "New task"}
    worker.sim.on_tick()
    assert mission.objective.status == "ended"
    assert worker.home_mission.active == (fault == "replaced")
    with pytest.raises(ValueError):
        await worker.mission_objective(mission, operation, decision=MissionDecision(action="explore"), renew=True)
    await worker.mission_objective(mission, operation, end_reason="Discard old reply")
    assert worker.sim.on_tick is original_tick and worker.home_mission.active == (fault == "replaced")


async def test_moving_feedback_preserves_sensor_pair_and_original_timestamps(objective_worker):
    from types import SimpleNamespace
    worker, mission, _, sensor = objective_worker
    captured_at = sensor.captured_at
    worker.sim.odometry[0] = .1
    worker.home_mission.sample = lambda: None
    worker._feedback = lambda sim, image: (SimpleNamespace(model_copy=lambda update:
        SimpleNamespace(run_id="run", episode_epoch=0, seq=7, wall_timestamp=12., **update)), image)
    returned_sensor, image, observation = await worker.mission_feedback(mission)
    assert returned_sensor is sensor and image == b"image" and sensor.captured_at == captured_at
    assert observation.odometry_m_rad == sensor.odometry_m_rad != list(worker.sim.odometry)
    assert observation.head_rad == sensor.head_rad and observation.simulated_time_s == sensor.simulated_time_s
    assert observation.wall_timestamp == 12. and observation.seq == 7
    observation.odometry_m_rad = [.1, 0., 0.]
    with pytest.raises(ValueError, match="paired observation"):
        await worker.mission_map(sensor, observation, mission=mission)


async def test_review_wait_observes_feedback_interval_changes(monkeypatch):
    from types import SimpleNamespace
    import backend.mission_supervisor as supervisor
    events = []
    controller = SimpleNamespace(state={"feedback_interval_s": 30.},
        mission=SimpleNamespace(objective=SimpleNamespace(travel_m=0.), state=lambda: {}),
        _trace=lambda *args: events.append(args))
    mapped = supervisor.MappedReviewOperation(controller, None, None)
    entered = asyncio.Event()

    async def poll():
        entered.set()
        return {"task": {"status": "running"}}

    mapped.poll = poll
    monkeypatch.setattr(supervisor, "time", SimpleNamespace(monotonic=lambda: 16.))
    pending = asyncio.create_task(mapped.review())
    try:
        await asyncio.wait_for(entered.wait(), .5)
        assert not pending.done()
        controller.state["feedback_interval_s"] = 2.
        result = await asyncio.wait_for(pending, .5)
        assert result["task"]["status"] == "running"
        assert len(events) == 1 and events[0][2]["paused_for_review"] is False
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


async def test_cancelled_supervisor_start_drains_bound_local_operation(objective_worker, monkeypatch):
    from types import SimpleNamespace
    from backend.mission_supervisor import MappedReviewOperation
    worker, mission, _, _ = objective_worker
    mission.operation = None
    events = []
    controller = SimpleNamespace(mission=mission, _trace=lambda *args: events.append(args))
    mapped = MappedReviewOperation(controller, worker, None)
    entered = asyncio.Event()
    never = asyncio.Event()
    original = worker.mission_objective

    async def interrupted(*args, **kwargs):
        result = await original(*args, **kwargs)
        if kwargs.get("decision") is not None:
            entered.set()
            await never.wait()
        return result

    async def home_state(**kwargs):
        return worker.home_mission.state()

    monkeypatch.setattr(worker, "mission_objective", interrupted)
    monkeypatch.setattr(worker, "home_state", home_state)
    pending = asyncio.create_task(mapped.start(MissionDecision(action="explore")))
    try:
        await asyncio.wait_for(entered.wait(), .5)
        assert worker.home_mission.active and worker._mission_objective_binding
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, .5)
        assert mapped.identity is None and mission.operation is None
        assert not worker.home_mission.active and worker._mission_objective_binding is None
        assert mission.objective.status == "ended" and len(events) == 1
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)