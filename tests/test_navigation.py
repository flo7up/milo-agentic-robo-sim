import asyncio
from uuid import uuid4

import pytest
@pytest.mark.parametrize("recover_clearance", [False, True])
@pytest.mark.parametrize("speed", [.15, .35, .5])
def test_crossing_pedestrian_stops_buffered_navigation_before_contact(recover_clearance, speed):
    import pybullet as bullet
    from backend.challenges import get_challenge
    from backend.simulation import BulletSimulation
    from backend.navigation import NavigationRuntime
    from scripts.navigation_policy import apply
    sim = BulletSimulation(challenge=get_challenge("pedestrian_crossing"), width=160, height=120)
    runtime = NavigationRuntime()
    runtime.clock = lambda: sim.ticks / 240
    runtime.recover_clearance = recover_clearance
    runtime.continuous_motion = speed > .15
    try:
        pedestrian = [item for item in sim.objects if item.get("pedestrian")]
        apply(runtime, sim, "begin_local_subgoal", {"goal": "Approach crossing with live clearance checks"})
        contact = False
        for _ in range(240):
            if runtime.status in {"failed", "cancelled"} or "CLEARANCE_STOP" in runtime.reason:
                break
            if sum(entry[1] for entry in runtime.buffer) < 120:
                from backend.continuous_navigation import ContinuousMotionBuffer
                observation = sim.observe(render=False)
                runtime.observe(sim, observation)
                runtime.apply(sim, "replace_motion_buffer", ContinuousMotionBuffer(expected_revision=runtime.revision,
                    segments=[{"kind": "drive", "linear_mps": speed, "angular_radps": 0., "duration_s": 1.}] * 2), observation.seq)
            runtime.tick(sim)
            contact |= any(bullet.getContactPoints(sim.robot, item["id"], physicsClientId=sim.client) for item in pedestrian)
        assert sim.pedestrian_crossing["started_at"] is not None
        assert "CLEARANCE_STOP" in runtime.reason, runtime.reason
        assert not runtime.buffer and not runtime.tickets
        stopped_revision = runtime.revision
        runtime.tick(sim)
        assert runtime.revision == stopped_revision and not runtime.buffer
        assert not contact
        assert .30 < sim.odometry[0] < .75
        sim.cancel.clear()
        sim.hold_current()
        sim._ticks(4 * 240)
        assert abs(bullet.getBaseVelocity(sim.robot, physicsClientId=sim.client)[0][0]) < .01
        assert not sim.pedestrian_crossing["contact"]
        assert sim.pedestrian_crossing["yielded"]
        assert sim.challenge_status()["completed_objectives"] == 2
        assert sim.challenge_status()["status"] == "in_progress"
        assert not any(bullet.getContactPoints(sim.robot, item["id"], physicsClientId=sim.client) for item in pedestrian)
        assert any(bullet.getBasePositionAndOrientation(item["id"], physicsClientId=sim.client)[0][1] != item["position"][1] for item in pedestrian)
        from tests.test_challenges import drive_to
        preset = get_challenge("pedestrian_crossing")
        assert preset.floor_size_m == [12, 10]
        assert preset.objectives[-1].center[0] == 4.5
        drive_to(sim, preset.objectives[-1].center[:2])
        assert sim.challenge_status()["status"] == "completed"
        assert not sim.pedestrian_crossing["contact"]
        sim.stop()
        from backend.simulation import MotionError
        with pytest.raises(MotionError, match="Navigation was stopped"):
            apply(runtime, sim, "begin_local_subgoal", {"goal": "Cannot restart after Stop"})
    finally:
        sim.close()


def test_supervisor_clearance_pause_still_brakes_and_invalidates_pending_motion():
    from backend.challenges import get_challenge
    from backend.simulation import BulletSimulation, MotionError
    from backend.navigation import NavigationRuntime
    from scripts.navigation_policy import apply
    sim = BulletSimulation(challenge=get_challenge("park"), width=160, height=120)
    runtime = NavigationRuntime()
    try:
        apply(runtime, sim, "begin_local_subgoal", {"goal": "Inspect clear space"})
        apply(runtime, sim, "replace_motion_buffer", {"segments": [{"kind": "drive", "linear_mps": .15, "angular_radps": 0, "duration_s": 1}]})
        previous_revision = runtime.revision
        runtime.recover_clearance = True
        def blocked(*args):
            raise MotionError("CLEARANCE_STOP", "Predicted obstruction")
        runtime.check_clearance = blocked
        runtime.tick(sim)
        assert runtime.status == "awaiting_feedback" and not runtime.buffer
        assert runtime.revision > previous_revision and not runtime.tickets
        assert sim.ticks == 0 and not sim.cancel.is_set()
        sim.stop()
        with pytest.raises(MotionError):
            apply(runtime, sim, "begin_local_subgoal", {"goal": "Cannot override operator Stop"})
    finally:
        sim.close()


@pytest.mark.parametrize("status", ["cancelled", "failed"])
def test_navigation_tick_preserves_revocation_inside_physics_callback(status):
    from backend.challenges import get_challenge
    from backend.navigation import NavigationRuntime
    from backend.simulation import BulletSimulation, MotionError
    from scripts.navigation_policy import apply
    sim = BulletSimulation(challenge=get_challenge("park"), width=160, height=120, rendering="tiny")
    runtime = NavigationRuntime(clock=lambda: sim.ticks / 240)
    try:
        apply(runtime, sim, "begin_local_subgoal", {"goal": "Exercise callback revocation"})
        apply(runtime, sim, "replace_motion_buffer", {"segments": [
            {"kind": "drive", "linear_mps": .15, "angular_radps": 0., "duration_s": 1.}]})
        revision = runtime.revision
        reason = "Objective authorization expired during physics publication"
        sim.ticks = 6
        sim.on_tick = lambda: getattr(runtime, "cancel" if status == "cancelled" else "fail")(sim, reason)
        runtime.tick(sim)
        assert sim.ticks == 12
        assert runtime.status == status and runtime.reason == reason
        assert runtime.revision == revision + 1 and not runtime.buffer and not runtime.tickets
        assert not runtime.velocity.any() and sim.cancel.is_set() == (status == "failed")
        runtime.tick(sim)
        assert sim.ticks == 12 and runtime.revision == revision + 1
        with pytest.raises(MotionError):
            runtime.validate_ticket(sim, 0, revision)
    finally:
        sim.close()


@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.parametrize("phase", ["before_tick", "during_tick", "stop"])
def test_motion_lease_expiry_brakes_without_reviving_authority(recover, phase):
    from backend.challenges import get_challenge
    from backend.contracts import NavigationFeedback
    from backend.navigation import NavigationRuntime
    from backend.simulation import BulletSimulation, MotionError
    from scripts.navigation_policy import apply
    sim = BulletSimulation(challenge=get_challenge("park"), width=160, height=120)
    runtime = NavigationRuntime(clock=lambda: sim.ticks / 240)
    runtime.recover_deadline = recover
    try:
        apply(runtime, sim, "begin_local_subgoal", {"goal": "Exercise expired motion authority"})
        apply(runtime, sim, "replace_motion_buffer", {"segments": [{"kind": "drive", "linear_mps": .15, "angular_radps": 0., "duration_s": 1.}]})
        runtime.skill_deadline = runtime.clock() + (2 / 240 if phase == "during_tick" else 0.)
        if phase == "stop":
            sim.stop()
        revision = runtime.revision
        runtime.tick(sim)
        assert NavigationFeedback.model_validate(runtime.state()).deadline_recovery_enabled == recover
        assert not runtime.buffer and not runtime.tickets and runtime.revision > revision
        assert not runtime.velocity.any()
        if phase == "stop":
            assert runtime.status == "cancelled" and sim.cancel.is_set()
            assert "Stop" in runtime.reason
        elif recover:
            assert runtime.status == "cancelled" and not sim.cancel.is_set()
            assert runtime.reason.startswith("MOTION_LEASE_EXPIRED:")
            assert runtime.state()["skill_deadline_in_s"] == 0
            apply(runtime, sim, "begin_local_subgoal", {"goal": "A new observed plan has separate authority"})
            assert not runtime.buffer and runtime.skill_deadline > runtime.clock()
        else:
            assert runtime.status == "failed" and sim.cancel.is_set()
            assert "60-second" not in runtime.reason and "deadline expired" in runtime.reason
        if sim.cancel.is_set():
            with pytest.raises(MotionError):
                apply(runtime, sim, "begin_local_subgoal", {"goal": "Cannot override Stop"})
    finally:
        sim.close()


def test_supervisor_floor_projection_uses_only_measured_camera_and_odometry():
    import math
    import numpy as np
    from backend.navigation_supervisor import floor_target, intention_to_target
    from backend.simulation import BulletSimulation
    sim = BulletSimulation(width=320, height=240)
    try:
        for yaw, pitch in [(0., .45), (.7, .9), (-.8, .65)]:
            import pybullet as bullet
            bullet.resetBasePositionAndOrientation(sim.robot, [0, 0, .155], [0, 0, 0, 1], physicsClientId=sim.client)
            bullet.resetJointState(sim.robot, sim.joints["head_yaw"], yaw, physicsClientId=sim.client)
            bullet.resetJointState(sim.robot, sim.joints["head_pitch"], pitch, physicsClientId=sim.client)
            observation = sim.observe(render=False)
            eye, rotation = sim.camera_pose()
            ray = rotation @ np.array([1., 0., -.2 * math.tan(math.radians(65) / 2)])
            expected = eye[:2] - eye[2] / ray[2] * ray[:2]
            assert floor_target(observation, [.5, .6]) == pytest.approx(expected, abs=.005)
        with pytest.raises(ValueError):
            floor_target(observation.model_copy(update={"head_rad": [0., 0.]}), [.5, 0.])
        target = observation.odometry_m_rad[:2]
        assert intention_to_target(observation, target)[2]
    finally:
        sim.close()


def test_visible_floor_center_falls_back_for_plain_floor_without_privileged_state():
    from io import BytesIO
    from PIL import Image
    from backend.navigation_supervisor import visible_floor_center, floor_target
    from tests.test_agent import observation
    current = observation().model_copy(update={"head_rad": [0., .8]})
    buffer = BytesIO()
    Image.new("RGB", (320, 240), (150, 150, 150)).save(buffer, format="PNG")
    assert visible_floor_center(current, buffer.getvalue(), [.5, .6]) == floor_target(current, [.5, .6])


def test_relative_subgoal_is_robot_relative_bounded_and_can_backtrack():
    import math
    from backend.navigation_supervisor import relative_target, intention_to_target, GuideNavigation
    from tests.test_agent import observation
    current = observation().model_copy(update={"odometry_m_rad": [1., 2., math.pi / 2]})
    assert relative_target(current, [.4, .2]) == pytest.approx([.8, 2.4])
    assert intention_to_target(current, [1., 1.6])[0] == "backward"
    with pytest.raises(ValueError):
        relative_target(current, [1.5, 1.5])
    with pytest.raises(ValueError):
        GuideNavigation(motion="relative_waypoint", steps=1, look_yaw_rad=0, look_pitch_rad=.45,
            status="continue", reason="Need a measured waypoint")


def test_waypoint_policy_instruction_uses_measured_relative_goal_not_world_state():
    import math
    from backend.navigation_supervisor import waypoint_instruction
    from tests.test_agent import observation
    current = observation().model_copy(update={"odometry_m_rad": [1., 2., math.pi / 2]})
    instruction = waypoint_instruction(current, [.8, 2.4])
    assert "forward +0.40 m, left +0.20 m" in instruction
    assert "Stop within 0.05 m" in instruction
    for target in ([float("nan"), 0], [1., 4.], [1.]):
        with pytest.raises(ValueError):
            waypoint_instruction(current, target)
    shifted = current.model_copy(update={"odometry_m_rad": [11., 12., math.pi / 2]})
    assert waypoint_instruction(shifted, [10.8, 12.4]) == instruction


def test_supervisor_target_feedback_does_not_confuse_turning_with_arrival():
    import math
    from backend.navigation_supervisor import target_feedback
    from tests.test_agent import observation
    target = [0., .65]
    turned = observation().model_copy(update={"odometry_m_rad": [0., 0., math.pi / 2]})
    feedback = target_feedback(turned, target)
    assert feedback["remaining_distance_m"] == .65
    assert not feedback["arrived"]
    assert feedback["next_local_intention"] == "forward"
    assert "continue_target" in feedback["next_decision"]
    translated = turned.model_copy(update={"odometry_m_rad": [0., .62, math.pi / 2]})
    assert target_feedback(translated, target)["arrived"]


def test_marker_target_uses_visible_colored_patch_not_challenge_metadata():
    from io import BytesIO
    from PIL import Image, ImageDraw
    from backend.navigation_supervisor import marker_target
    from tests.test_agent import observation
    current = observation().model_copy(update={"head_rad": [0., .8]})
    image = Image.new("RGB", (320, 240), (150, 150, 150))
    ImageDraw.Draw(image).rectangle((100, 100, 220, 190), fill=(20, 180, 50))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    target = marker_target(current, buffer.getvalue(), "green")
    assert 0 < target[0] < 1.8 and abs(target[1]) < .1
    with pytest.raises(ValueError):
        marker_target(current, buffer.getvalue(), "orange")


async def test_luna_navigation_api_requires_guide_tool_without_changing_other_modes():
    from types import SimpleNamespace
    from backend.agent import FoundryModel, ModelProfile
    payloads = []

    async def create(**kwargs):
        payloads.append(kwargs)
        return None

    model = FoundryModel.__new__(FoundryModel)
    model.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    model.execution_mode = "luna_navigation"
    await model.respond(ModelProfile(id="luna", label="Luna", deployment="scripted"), "high", "Navigate", [])
    assert payloads[-1]["tool_choice"] == {"type": "function", "name": "guide_navigation"}
    assert not payloads[-1]["parallel_tool_calls"] and not payloads[-1]["store"]
    model.execution_mode = "single_step"
    await model.respond(ModelProfile(id="luna", label="Luna", deployment="scripted"), "low", "Navigate", [])
    assert "tool_choice" not in payloads[-1]


def test_supervisor_contract_cannot_supply_velocities_or_unbounded_intentions():
    from backend.navigation_supervisor import GuideNavigation, POLICY_TASKS, navigation_supervisor_tools
    valid = {"motion": "backward", "steps": 2, "look_yaw_rad": 0, "look_pitch_rad": .45,
             "status": "continue", "reason": "Rear is clear; create turning space."}
    assert GuideNavigation(**valid).motion == "backward"
    assert set(POLICY_TASKS) == {"forward", "backward", "left", "right", "hold"}
    assert [tool["name"] for tool in navigation_supervisor_tools()] == ["guide_navigation"]
    for update in ({"linear_mps": .15}, {"code": "move()"}, {"steps": 99}, {"motion": "teleport"}, {"look_yaw_rad": 4}):
        with pytest.raises(ValueError):
            GuideNavigation(**{**valid, **update})


@pytest.mark.parametrize("invalid_first", [False, True])
async def test_luna_supervises_without_direct_wheel_actions_or_evaluator_inputs(monkeypatch, invalid_first):
    import asyncio
    import json
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from backend.contracts import AgentObservation
    from backend.worker import SimulationWorker
    from tests.test_agent import model_response
    instructions = []
    from backend import navigation_supervisor
    original_run = navigation_supervisor.run_supervised_navigation
    failures = []

    async def capture(*arguments):
        try:
            return await original_run(*arguments)
        except Exception as error:
            failures.append(error)
            raise

    monkeypatch.setattr(navigation_supervisor, "run_supervised_navigation", capture)

    class Policy:
        @staticmethod
        def check_available():
            pass

        async def start(self):
            pass

        async def predict(self, observation, image):
            instructions.append(self.instruction)
            assert set(observation.model_dump()) == set(AgentObservation.model_fields)
            return {"action": [.1, 0.] if "forward" in self.instruction else [0., 0.]}

        async def close(self):
            pass

    class Supervisor:
        count = 0

        async def respond(self, profile, reasoning, goal, inputs):
            assert profile.id == "luna"
            assert set(json.loads(inputs[-1]["content"][0]["text"])) == set(AgentObservation.model_fields)
            self.count += 1
            if invalid_first and self.count == 1:
                return model_response("guide_navigation", json.dumps({"motion": "forward", "steps": 8,
                    "look_yaw_rad": 0, "look_pitch_rad": .45, "status": "continue",
                    "reason": "This invalid reply must not move the robot.", "memory": "oversized" * 100}), call_id="invalid-guide")
            if invalid_first and self.count == 2:
                assert "guide_rejected" in json.dumps(inputs)
                assert instructions == ["Remain stationary."]
            if self.count == 2 + invalid_first:
                assert '"place_sightings": [{"id":' in json.dumps(inputs).replace('\\"', '"')
                assert "unverified" in json.dumps(inputs) and "Scripted sink and stove" in json.dumps(inputs)
            return model_response("guide_navigation", json.dumps({"motion": "forward", "steps": 2,
                "look_yaw_rad": 0, "look_pitch_rad": .45, "status": "continue" if self.count == 1 + invalid_first else "blocked",
                "place_sighting": {"label": "Kitchen", "evidence": "Scripted sink and stove"},
                "reason": "Current front view shows an open path."}), call_id=f"guide-{self.count}")

        async def close(self):
            pass

    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"),
        model_factory=lambda config: Supervisor(), local_navigation_factory=Policy)
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, AgentStart(run_id=worker.latest["run_id"], episode_epoch=0,
            goal=get_challenge("park").goal, execution_mode="luna_navigation", max_turns=3, feedback_interval_s=.25))
        await asyncio.wait_for(controller.task, 10)
        if failures:
            raise failures[0]
        assert controller.state["error"] is None, controller.state["error"]
        assert controller.state["local_model"]["requests_completed"] == 2
        assert controller.state["turns"] == 2 + invalid_first
        assert "Drive forward slowly while keeping your heading." in instructions
        assert worker.latest["stopped"] and worker.latest["snapshot"]["simulated_time_s"] > 2
    finally:
        await controller.halt()
        await worker.close()


async def test_supervisor_target_alias_cannot_retry_blocked_physical_intention(monkeypatch):
    import json
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from backend.navigation import NavigationRuntime
    from backend.simulation import MotionError
    from backend.worker import SimulationWorker
    from tests.test_agent import model_response
    predictions = []

    def blocked(*arguments):
        raise MotionError("CLEARANCE_STOP", "Predicted arm obstruction")

    monkeypatch.setattr(NavigationRuntime, "check_clearance", blocked)

    class Policy:
        @staticmethod
        def check_available():
            pass

        async def start(self):
            pass

        async def predict(self, observation, image):
            predictions.append(self.instruction)
            return {"action": [.1, 0.]}

        async def close(self):
            pass

    class Supervisor:
        count = 0

        async def respond(self, profile, reasoning, goal, inputs):
            self.count += 1
            if self.count == 3:
                assert "same clearance-blocked primitive" in json.dumps(inputs)
            return model_response("guide_navigation", json.dumps({
                "motion": "forward" if self.count == 1 else "relative_waypoint",
                "relative_target_m": [.3, 0.], "steps": 1, "look_yaw_rad": 0,
                "look_pitch_rad": .45, "status": "blocked" if self.count == 3 else "continue",
                "reason": "Scripted attempt to retry the same physical motion."}), call_id=f"alias-{self.count}")

        async def close(self):
            pass

    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"),
        model_factory=lambda config: Supervisor(), local_navigation_factory=Policy)
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, AgentStart(run_id=worker.latest["run_id"], episode_epoch=0,
            goal="Approach a visible point", execution_mode="luna_navigation", max_turns=3, feedback_interval_s=.25))
        await asyncio.wait_for(controller.task, 10)
        assert controller.state["error"] is None
        assert len(predictions) == 2
        assert controller.state["local_model"]["requests_completed"] == 0
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("phase", ["supervisor", "policy"])
@pytest.mark.parametrize("interruption", ["stop", "takeover", "reset", "disconnect"])
@pytest.mark.parametrize("time_only", [False, True])
async def test_supervised_navigation_cancels_late_model_replies(phase, interruption, time_only):
    import json
    from backend.agent import AgentController, AgentStart, FoundryConfig, NavigationEvaluationBudget
    from backend.challenges import get_challenge
    from backend.worker import SimulationWorker
    from tests.test_agent import model_response
    entered = asyncio.Event()
    closed = []

    class Policy:
        @staticmethod
        def check_available():
            pass

        async def start(self):
            pass

        async def predict(self, observation, image):
            if self.instruction != "Remain stationary." and phase == "policy":
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    return {"action": [.15, 0.]}
            return {"action": [0., 0.]}

        async def close(self):
            closed.append("policy")

    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            if phase == "supervisor":
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
            return model_response("guide_navigation", json.dumps({"motion": "forward", "steps": 8,
                "look_yaw_rad": 0, "look_pitch_rad": .45, "status": "continue", "reason": "Scripted delayed supervisor response.",
                "place_sighting": {"label": "Late room", "evidence": "Scripted delayed visual evidence"}}))

        async def close(self):
            closed.append("supervisor")

    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"),
        model_factory=lambda config: Supervisor(), local_navigation_factory=Policy,
        evaluation_budget=NavigationEvaluationBudget(timeout_s=60) if time_only else None)
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, AgentStart(run_id=worker.latest["run_id"], episode_epoch=0,
            goal=get_challenge("park").goal, execution_mode="luna_navigation"))
        await asyncio.wait_for(entered.wait(), 5)
        ticks = worker.latest["snapshot"]["simulated_time_s"]
        await asyncio.wait_for(controller.halt(interruption), 3)
        assert not controller.active and worker.latest["stopped"]
        assert worker.latest["snapshot"]["simulated_time_s"] == ticks
        assert worker.navigation is None or not worker.navigation.buffer
        assert sorted(closed) == ["policy", "supervisor"]
        if phase == "supervisor":
            assert await worker.place_sightings() == []
    finally:
        await controller.halt()
        await worker.close()


def test_map_history_is_bounded_expires_and_uses_observation_viewpoints():
    from types import SimpleNamespace
    from backend.spatial import MapHistory
    history = MapHistory("episode", 2)
    polygon = [[-.2, -.3], [.2, -.3], [.2, .3], [-.2, .3]]
    history.record_pose([1, 2, 0], polygon, now=10)
    history.record_pose([1, 2, 0], polygon, now=20)
    assert len(history.trail) == 1
    assert history.public(now=20)["footprints"][0]["polygon_m"][0] == [.8, 1.7]
    assert history.public(now=310)["footprints"] == []
    observation = SimpleNamespace(run_id="episode", episode_epoch=2, odometry_m_rad=[1, 2, 0], seq=4, frame_ref="camera-4")
    identifier = history.record_label(observation, "Kitchen", "Sink and stove visible", now=10)
    assert history.record_label(observation, "Kitchen", "Sink visible again", now=20) == identifier
    label = history.public(now=150)["labels"][0]
    assert label["position_m"] == [1, 2] and label["anchor"] == "observation_viewpoint"
    assert label["verification"] == "unverified" and label["stale"]
    assert history.public(now=1820)["labels"] == []
    observation.episode_epoch = 3
    with pytest.raises(ValueError, match="another episode"):
        history.record_label(observation, "Bathroom", "Tub visible", now=30)
    for index in range(600):
        history.record_pose([index * .2, 0, 0], polygon, now=1000 + index)
    assert len(history.trail) == 512


async def test_worker_map_history_records_motion_and_rejects_late_labels():
    from backend.challenges import get_challenge
    from backend.spatial import PlaceSighting
    from backend.simulation import MotionError
    from backend.contracts import Command, ManualPlacement
    from backend.worker import SimulationWorker
    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        observation, _ = await worker.feedback()
        sighting = PlaceSighting(label="Kitchen", evidence="Scripted fixture identification")
        await worker.record_place_sighting(observation, sighting, worker.stop_revision)
        result = await worker.execute(Command(run_id=observation.run_id, episode_epoch=0, observation_seq=observation.seq,
            action_id="history-drive", tool="drive_base", arguments={"linear_mps": .1, "angular_radps": 0., "duration_s": 2.}), assisted=False)
        assert result.status == "ok"
        state = await worker.call(lambda sim: worker.spatial_state())
        assert len(state["history"]["footprints"]) >= 2
        assert state["history"]["labels"][0]["observation_seq"] == observation.seq
        assert "history" not in result.observation.model_dump()
        await worker.reposition(ManualPlacement(run_id=observation.run_id, episode_epoch=0,
            observation_seq=result.observation.seq, xy_m=[-.2, 0]))
        assert await worker.place_sightings() == []
        revision = worker.stop_revision
        worker.stop()
        with pytest.raises(MotionError, match="stopped"):
            await worker.record_place_sighting(observation, sighting, revision)
    finally:
        await worker.close()


async def test_recovery_demonstrations_are_physical_replayable_and_model_input_only(tmp_path):
    from scripts.recovery_policy import challenge_for, record_episode, load_episode, replay_episode
    from backend.worker import SimulationWorker
    worker = SimulationWorker(challenge=challenge_for(0), pace=False)
    try:
        metadata = await worker.call(lambda sim: record_episode(sim, 0, tmp_path / "episode-000"))
        assert metadata["scripted_demonstration"] and not metadata["task_success_claimed"]
    finally:
        await worker.close()
    _, frames = load_episode(tmp_path / "episode-000")
    assert any(frame["action"][0] < 0 for frame in frames)
    assert len({frame["task"] for frame in frames}) == 5
    worker = SimulationWorker(challenge=challenge_for(0), pace=False)
    try:
        replay = await worker.call(lambda sim: replay_episode(sim, frames))
        assert replay["verified"] and replay["max_state_error"] < 1e-5
    finally:
        await worker.close()

import pybullet as bullet
from pydantic import ValidationError

from backend.navigation import CompleteNavigationSkill, NavigationPlan, NavigationRuntime, ReplaceMotionBuffer
from backend.simulation import BulletSimulation, MotionError
from backend.contracts import Command, ManualPlacement
from backend.worker import SimulationWorker


def ticket(runtime, sim):
    observation = sim.observe()
    runtime.observe(sim, observation)
    return observation.seq, runtime.revision


def plan(runtime, sim):
    sequence, revision = ticket(runtime, sim)
    runtime.apply(sim, "set_navigation_plan", NavigationPlan(expected_revision=revision, steps=[
        {"skill": skill, "goal": goal} for skill, goal in [
            ("inspect_room", "Identify visible room fixtures"), ("locate_doorway", "Find a visible opening"),
            ("approach", "Approach the observed opening"), ("cross", "Cross and inspect the next room")]]), sequence)


def scan(runtime, sim):
    sequence, revision = ticket(runtime, sim)
    runtime.apply(sim, "replace_motion_buffer", ReplaceMotionBuffer(expected_revision=revision, segments=[
        {"kind": "head", "yaw_rad": -.6, "pitch_rad": .2, "duration_s": 1},
        {"kind": "head", "yaw_rad": .6, "pitch_rad": .2, "duration_s": 1}]), sequence)
    while runtime.buffer:
        runtime.tick(sim)


def complete(runtime, sim):
    sequence, revision = ticket(runtime, sim)
    runtime.apply(sim, "complete_navigation_skill", CompleteNavigationSkill(
        expected_revision=revision, evidence="Fixture observation reviewed in the camera"), sequence)


def test_navigation_recording_and_replay_use_real_wheels_without_private_model_features(tmp_path):
    from scripts.navigation_policy import CASES, STATE_NAMES, challenge_for, load_episode, record_episode, replay_episode
    directory = tmp_path / "episode-000"
    sim = BulletSimulation(challenge=challenge_for(CASES[0]), width=160, height=120)
    try:
        result = record_episode(sim, CASES[0], directory)
        assert result["success"] and result["travel_m"] > .5
    finally:
        sim.close()
    import json
    path = directory / "frames.json"
    original = path.read_text()
    for key, value in (("timestamp", .5), ("action", [9, 0]), ("observation.state", [float("nan")] * 20),
                       ("private_target", [1, 1]), ("image_sha256", "incorrect")):
        corrupted = json.loads(original)
        corrupted[0][key] = value
        path.write_text(json.dumps(corrupted))
        with pytest.raises(ValueError):
            load_episode(directory)
    path.write_text(original)
    metadata, frames = load_episode(directory)
    assert len(STATE_NAMES) == len(frames[0]["observation.state"]) == 20
    assert not {"target", "case", "position", "geometry"} & frames[0].keys()
    sim = BulletSimulation(challenge=challenge_for(CASES[0]), width=160, height=120)
    try:
        result = replay_episode(sim, metadata, frames)
        assert result["success"] and result["max_state_error"] < 1e-5
    finally:
        sim.close()


def test_stopping_evaluation_cases_are_disjoint_from_training_and_validation():
    from scripts.navigation_policy import CASES, EVALUATION_CASES
    fixture = lambda case: (*case["target"], case["doorway"])
    assert len(CASES) == 16 and sum(case["split"] == "train" for case in CASES) == 12
    assert len(EVALUATION_CASES) == 4
    assert {fixture(case) for case in CASES}.isdisjoint({fixture(case) for case in EVALUATION_CASES})
    assert all(case["split"] == "evaluation_only" for case in EVALUATION_CASES)
    assert [case["id"] for case in CASES + EVALUATION_CASES] == list(range(20))


async def test_navigation_comparison_matches_seeds_and_retains_premature_stop_failures(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import navigation_policy
    calls = []
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"

    async def fake_evaluate(options):
        calls.append(options)
        failed = options.checkpoint == candidate and options.case == 16
        return {"weights_sha256": options.checkpoint.name, "success": not failed,
                "status": "stopped_outside_bay" if failed else "parked", "saturated_requests": int(failed),
                "requests": [{"index": 0, "measured": {"parked": not failed}}], "final": {"parked": not failed}}

    monkeypatch.setattr(navigation_policy, "evaluate", fake_evaluate)
    output = tmp_path / "results"
    await navigation_policy.compare(SimpleNamespace(output=output, baseline=baseline, checkpoint=candidate, seed=800, requests=30))
    assert len(calls) == 8
    for index in range(0, len(calls), 2):
        first, second = calls[index:index + 2]
        assert first.case == second.case == 16 + index // 2
        assert first.seed == second.seed == 800 + first.case
        assert first.requests == second.requests == 30
        assert first.checkpoint == baseline and second.checkpoint == candidate
    summary = json.loads((output / "comparison.json").read_text())
    assert summary["complete"] and not summary["controller_changed"]
    assert summary["totals"]["baseline"]["successes"] == 4
    assert summary["totals"]["candidate"] == {"trials": 4, "successes": 3, "premature_stops": 1,
                                               "requests": 4, "saturated_requests": 1}
    assert "does not isolate" in summary["qualification"]


def test_task_waypoint_demonstration_is_replayable_and_rejects_private_features(tmp_path):
    import json
    from scripts.task_navigation import case_for, record_episode, replay_episode, load_episode, teacher_action
    from tests.test_agent import observation
    current = observation().model_copy(update={"odometry_m_rad": [0., 0., 0.]})
    assert teacher_action(current, [.5, 0])[0] > 0
    assert teacher_action(current, [-.3, 0])[0] < 0
    assert teacher_action(current, [0, .5])[1] > 0
    assert teacher_action(current, [0, 0]) == [0., 0.]
    challenge, route = case_for(0)
    assert case_for(16)[0].initial_xy != challenge.initial_xy
    directory = tmp_path / "episode-000"
    sim = BulletSimulation(challenge=challenge, width=320, height=240)
    try:
        result = record_episode(sim, 0, directory)
        assert result["success"] and result["scripted_demonstration"]
    finally:
        sim.close()
    metadata, frames = load_episode(directory)
    assert metadata["split"] == "train" and metadata["episode"] == 0
    assert all(frame["task"].startswith("Reach the local waypoint:") for frame in frames)
    sim = BulletSimulation(challenge=challenge, width=320, height=240)
    try:
        assert replay_episode(sim, frames)["verified"]
    finally:
        sim.close()
    frames[0]["private_target"] = route[-1][0]
    (directory / "frames.json").write_text(json.dumps(frames))
    with pytest.raises(ValueError, match="private fields"):
        load_episode(directory)


@pytest.mark.parametrize("premature", [False, True])
async def test_task_waypoint_benchmark_checks_completion_and_cleans_up(tmp_path, monkeypatch, premature):
    import json
    import re
    from types import SimpleNamespace
    from backend import local_navigation
    from scripts.task_navigation import evaluate, teacher_action
    lifecycle = []

    class ScriptedPolicy:
        def __init__(self, checkpoint):
            self.instruction = ""

        async def start(self):
            lifecycle.append("start")

        async def reset(self):
            lifecycle.append("reset")

        async def predict(self, observation, image):
            coordinates = re.search(r"forward ([+-]\d+\.\d+) m, left ([+-]\d+\.\d+) m", self.instruction)
            assert coordinates and image.startswith(b"\x89PNG")
            relative = observation.model_copy(update={"odometry_m_rad": [0., 0., 0.]})
            action = [0., 0.] if premature else teacher_action(relative, [float(value) for value in coordinates.groups()])
            return {"action": action, "raw_action": action, "saturated_axes": []}

        async def close(self):
            lifecycle.append("close")

    monkeypatch.setattr(local_navigation, "LocalNavigationClient", ScriptedPolicy)
    output = tmp_path / "evaluation"
    await evaluate(SimpleNamespace(output=output, checkpoint=tmp_path / "scripted-policy", cases=[0]))
    report = json.loads((output / "case-000" / "report.json").read_text())
    assert lifecycle == ["start", "reset", "close"]
    assert report["teacher_waypoints"] and not report["autonomous_task_success"]
    assert report["success"] == (not premature)
    assert report["full_task_physics_success"] == (not premature)
    if premature:
        assert "PREMATURE_STOP" in report["error"] and report["waypoints_completed"] == 0
    else:
        assert report["error"] is None and report["waypoints_completed"] == 2


@pytest.mark.parametrize("external_stop", [False, True])
async def test_live_instruction_preserves_episode_and_stop_wins(monkeypatch, external_stop):
    from backend.agent import AgentController, AgentStart, FoundryConfig, RunInstruction
    from backend.challenges import get_challenge
    import time
    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"))
    started, draining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    goals = []

    async def run(worker, settings, profile, stop_revision):
        goals.append(settings.goal)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            draining.set()
            await release.wait()

    monkeypatch.setattr(controller, "_run", run)
    try:
        await asyncio.wrap_future(worker.ready)
        settings = AgentStart(run_id=worker.sim.run_id, episode_epoch=0, execution_mode="luna_continuous", goal="Original goal")
        controller.start(worker, settings)
        await started.wait()
        original_session = controller.state["session_id"]
        deadline = controller.session_deadline
        instruction = RunInstruction(run_id=worker.sim.run_id, episode_epoch=0, session_id=original_session, message="Go to the doorway")
        pending = asyncio.create_task(controller.redirect(worker, instruction))
        await draining.wait()
        assert worker.sim.cancel.is_set()
        if external_stop:
            controller.interrupt("Operator Stop")
        release.set()
        if external_stop:
            with pytest.raises(RuntimeError):
                await pending
            assert goals == ["Original goal"] and not controller.active
        else:
            await pending
            await asyncio.sleep(0)
            assert goals == ["Original goal", "Go to the doorway"]
            assert controller.session_deadline == deadline and time.monotonic() < deadline
            assert controller.state["session_id"] != original_session
            assert controller.run_messages[-2]["status"] == "applied"
            with pytest.raises(RuntimeError):
                await controller.redirect(worker, instruction)
        assert worker.sim.ticks == 0 and worker.sim.run_id == settings.run_id
    finally:
        release.set()
        await controller.halt()
        await worker.close()


def test_navigation_memory_tracks_executed_turns_and_clears_between_episodes():
    import math
    from types import SimpleNamespace
    from backend.navigation_memory import NavigationMemory
    memory = NavigationMemory()
    def observation(yaw, run="run"):
        return SimpleNamespace(run_id=run, episode_epoch=0, odometry_m_rad=[0., 0., yaw], head_rad=[0., .4], simulated_time_s=yaw)
    before = observation(0.)
    for step in range(4):
        after = observation((step + 1) * math.pi / 2)
        memory.remember(before, after, {"action": "turn", "turn_rad": math.pi / 2}, {"status": "ok"})
        before = after
    summary = memory.observe(before)
    assert summary["rotation_without_translation_rad"] == pytest.approx(2 * math.pi, abs=.01)
    assert len(summary["recent_actions"]) == 4 and len(summary["inspected_heading_sectors_here"]) == 4
    memory.bind("run", 0)
    assert memory.summary()["recent_actions"]
    memory.bind("run", 0, reset=True)
    assert not memory.summary()["recent_actions"]
    memory.remember(observation(0.), observation(.5), {"action": "turn"}, {"status": "ok"})
    memory.observe(observation(0., "new-run"))
    assert not memory.summary()["recent_actions"]
    assert "geometry" not in str(summary) and "world" not in summary["frame"]


def test_navigation_memory_expires_views_and_bounds_no_progress_recovery():
    import math
    from backend.navigation_memory import NavigationMemory
    from tests.test_agent import observation
    memory = NavigationMemory()
    initial = observation().model_copy(update={"wall_timestamp": 100., "odometry_m_rad": [0., 0., 0.], "head_rad": [0., .15]})
    memory.observe(initial)
    assert memory.recent_sectors(initial, now=101.) == {0}
    assert not memory.recent_sectors(initial, now=191.)
    assert not memory.recent_sectors(initial, now=99.)
    assert memory.next_scan_turn(initial, now=101.) == pytest.approx(math.pi / 2)
    for heading in [math.pi / 2, math.pi, -math.pi / 2]:
        memory.observe(initial.model_copy(update={"odometry_m_rad": [0., 0., heading]}))
    assert memory.next_scan_turn(initial, now=101.) is None
    for _ in range(4):
        memory.remember(initial, initial, {"action": "navigate"}, {"status": "blocked"})
    assert memory.progress()["recovery_needed"]
    memory.recovery_attempts = 2
    moved = initial.model_copy(update={"odometry_m_rad": [1., 0., 0.]})
    memory.observe(moved)
    memory.remember(initial, moved, {"action": "navigate"}, {"status": "arrived"})
    assert memory.progress() == {"stagnant_actions": 0, "recovery_needed": False, "recovery_attempts": 0, "recovery_limit": 2}
    assert memory.routes[-1]["end_m"] == [1., 0.]
    assert memory.routes[-1]["clearance"] == "must_revalidate"
    memory.bind("another-episode", 2)
    assert not memory.routes and not memory.views


def test_navigation_memory_route_sightings_expire_and_reject_cross_episode(monkeypatch):
    from backend.navigation_memory import NavigationMemory
    from tests.test_agent import observation
    monkeypatch.setattr("backend.navigation_memory.time.time", lambda: 101.)
    initial = observation().model_copy(update={"odometry_m_rad": [0., 0., 0.], "wall_timestamp": 100.})
    reached = initial.model_copy(update={"odometry_m_rad": [1., 0., 0.]})
    memory = NavigationMemory()
    memory.remember(initial, reached, {"action": "navigate", "source": "controller", "reason": "Recovery"}, {"status": "arrived"})
    routes = memory.sighting_routes([{"id": "kitchen-view", "position_m": [0., 0.]}, {"id": "bath-view", "position_m": [1., 0.]}])
    assert routes[0]["from_sighting_ids"] == ["kitchen-view"]
    assert routes[0]["to_sighting_ids"] == ["bath-view"]
    assert routes[0]["clearance"] == "must_revalidate"
    assert memory.assessments[-1]["source"] == "controller_recovery"
    monkeypatch.setattr("backend.navigation_memory.time.time", lambda: 1901.)
    assert memory.sighting_routes([]) == [] and memory.summary()["observed_routes"] == []
    changed = reached.model_copy(update={"episode_epoch": reached.episode_epoch + 1})
    with pytest.raises(ValueError, match="across episodes"):
        memory.remember(initial, changed, {}, {})
    memory.observe(changed)
    assert not memory.routes and not memory.reached_positions


def test_adaptive_recovery_uses_current_candidates_and_excludes_failed_targets():
    import time
    from backend.navigation_memory import NavigationMemory
    from backend.continuous_supervisor import recovery_choice
    from tests.test_agent import observation
    memory = NavigationMemory()
    current = observation().model_copy(update={"wall_timestamp": time.time()})
    candidates = [{"id": 1, "target_m": [1., 0.], "distance_m": 1., "distance_to_visited_m": 1.},
        {"id": 2, "target_m": [0., 1.], "distance_m": 1., "distance_to_visited_m": .8}]
    memory.remember(current, current, {"action": "navigate", "target_m": [1., 0.]}, {"status": "blocked"})
    assert recovery_choice(memory, current, candidates).candidate_id == 2
    assert recovery_choice(memory, current, []).action == "turn"
    memory.next_scan_turn = lambda observation: None
    memory.routes.append({"start_m": [0., 1.], "end_m": [0., 0.]})
    candidates[1]["distance_to_visited_m"] = 0.
    assert "backtrack" in recovery_choice(memory, current, candidates).reason
    assert recovery_choice(memory, current, [candidates[0]]) is None
    memory.recovery_attempts = 2
    assert recovery_choice(memory, current, candidates) is None


async def test_adaptive_scan_does_not_move_after_stop_during_feedback():
    from types import SimpleNamespace
    from backend.continuous_supervisor import adaptive_scan
    from backend.navigation_memory import NavigationMemory
    from tests.test_agent import observation
    stopped = False
    async def feedback():
        nonlocal stopped
        stopped = True
        return observation(), b"unused"
    def check_live(*args):
        if stopped:
            raise asyncio.CancelledError
    controller = SimpleNamespace(_check_live=check_live, navigation_memory=NavigationMemory())
    with pytest.raises(asyncio.CancelledError):
        await adaptive_scan(controller, SimpleNamespace(feedback=feedback), SimpleNamespace(), 0)


def test_sensor_wait_conditions_are_allowlisted_and_reject_missing_readings():
    from types import SimpleNamespace
    from backend.continuous_supervisor import SensorCondition
    from backend.contracts import BatterySensor, ProximitySensors, DistanceReading
    observation = SimpleNamespace(battery=BatterySensor(charge_pct=54., low=False, charging=True), proximity=None)
    condition = SensorCondition(sensor="battery.charge_pct", comparison="gte", value=90.)
    assert not condition.matches(observation)
    observation.battery.charge_pct = 90.
    assert condition.matches(observation)
    assert SensorCondition(sensor="battery.charging", comparison="eq", value=True).matches(observation)
    for sensor, comparison, value in [("hidden.task_success", "eq", True), ("battery.charge_pct", "gte", 101.),
            ("battery.charge_pct", "gte", float("nan")), ("battery.charge_pct", "gte", True),
            ("battery.charging", "gte", True), ("battery.charging", "eq", 1.), ("proximity.front_m", "gte", 2.1)]:
        with pytest.raises(ValueError):
            SensorCondition(sensor=sensor, comparison=comparison, value=value)
    observation.battery = None
    with pytest.raises(ValueError, match="battery reading"):
        condition.matches(observation)
    distance = SensorCondition(sensor="proximity.front_m", comparison="gte", value=1.)
    with pytest.raises(ValueError, match="distance reading"):
        distance.matches(observation)
    observation.proximity = ProximitySensors(simulated_time_s=0., distances=[
        DistanceReading(direction="front", bearing_rad=0., distance_m=2., status="clear")], collisions=[])
    assert distance.matches(observation)
    observation.proximity.distances[0].status = "occluded"
    with pytest.raises(ValueError, match="distance reading"):
        distance.matches(observation)


@pytest.mark.parametrize("stop_when", [None, "feedback", "tick"])
async def test_continuous_stationary_wait_advances_physics_and_honors_stop(monkeypatch, stop_when):
    from backend.continuous_supervisor import GuideContinuous, wait_stationary
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    worker = SimulationWorker(challenge=get_challenge("pedestrian_crossing"), pace=False)
    controller = AgentController(FoundryConfig())
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.resume_manual(expected_stop_revision=worker.stop_revision)
        settings = AgentStart(run_id=worker.sim.run_id, episode_epoch=worker.epoch, execution_mode="luna_continuous",
            goal="Wait here for the route to clear")
        guide = GuideContinuous(action="wait", duration_s=4., reason="Hold position for the crossing to clear")
        with pytest.raises(ValueError):
            GuideContinuous(action="wait", duration_s=4.1, reason=guide.reason)
        before = worker.sim.ticks
        original_feedback = worker.feedback
        if stop_when == "feedback":
            async def stopped_feedback():
                feedback = await original_feedback()
                worker.stop()
                return feedback
            monkeypatch.setattr(worker, "feedback", stopped_feedback)
        elif stop_when == "tick":
            original_tick = worker.sim.on_tick
            def stopped_tick():
                original_tick()
                if worker.sim.ticks > before + 24:
                    worker.stop()
            await worker.call(lambda sim: setattr(sim, "on_tick", stopped_tick))
        if stop_when:
            with pytest.raises(asyncio.CancelledError):
                await wait_stationary(controller, worker, settings, guide.duration_s)
            assert worker.sim.cancel.is_set()
            assert worker.sim.ticks - before < 240
        else:
            await wait_stationary(controller, worker, settings, guide.duration_s)
            assert worker.sim.ticks - before == 4 * 240
        assert abs(worker.sim.odometry[0]) < .01
        assert abs(worker.sim.odometry[1]) < .01
    finally:
        await worker.close()


@pytest.mark.parametrize("fault", [None, "timeout", "guard", "missing", "stale", "map", "clearance", "floor", "surface", "contact", "pose", "stop"])
async def test_sensor_wait_is_bounded_stationary_and_interruptible(monkeypatch, fault):
    import math
    import time
    from backend.continuous_supervisor import SensorCondition, wait_for_sensor
    from backend.continuous_navigation import ContinuousScan
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from backend.contracts import BatterySensor
    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    controller = AgentController(FoundryConfig())
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.resume_manual(expected_stop_revision=worker.stop_revision)
        settings = AgentStart(run_id=worker.sim.run_id, episode_epoch=worker.epoch, execution_mode="luna_continuous",
            goal="Scripted bounded sensor dwell")
        await worker.scan_continuous(ContinuousScan(run_id=settings.run_id, episode_epoch=settings.episode_epoch))
        before = worker.sim.ticks
        initial_pose = list(worker.sim.odometry)
        feedback = worker.feedback

        async def sensor_feedback():
            observation, image = await feedback()
            elapsed = (worker.sim.ticks - before) / 240
            observation = observation.model_copy(deep=True)
            observation.battery = BatterySensor(charge_pct=80. if fault == "timeout" else min(100., 80. + elapsed * 10),
                low=False, charging=not (fault == "guard" and elapsed >= .5))
            if fault == "missing":
                observation.battery = None
            if elapsed >= .5:
                if fault == "stale":
                    observation.wall_timestamp = time.time() - 2
                elif fault == "map":
                    await worker.call(lambda sim: setattr(worker, "spatial_enabled", False))
                elif fault == "contact":
                    observation.bumpers = ["front"]
                elif fault == "pose":
                    observation.odometry_m_rad[0] += .1
                elif fault == "stop":
                    worker.stop()
            return observation, image

        async def parking(target=None, *, stationary_request=None):
            assert stationary_request is settings
            if fault == "map" and worker.sim.ticks > before:
                raise ValueError("Injected unavailable stationary map")
            return {"parking_margin_ok": not (fault == "clearance" and worker.sim.ticks > before),
                "floor_rgb": [100, 100, 100], "inside_floor_region": fault != "floor"}

        monkeypatch.setattr(worker, "feedback", sensor_feedback)
        monkeypatch.setattr(worker, "parking_clearance", parking)
        condition = SensorCondition(sensor="battery.charge_pct", comparison="gte", value=90.)
        guard = SensorCondition(sensor="battery.charging", comparison="eq", value=True)
        from backend import continuous_supervisor
        from types import SimpleNamespace
        monkeypatch.setattr(continuous_supervisor, "time", SimpleNamespace(
            monotonic=lambda: (worker.sim.ticks - before) / 480, time=time.time))
        if fault == "stop":
            with pytest.raises(asyncio.CancelledError):
                await wait_for_sensor(controller, worker, settings, condition, 1.5, guard)
        else:
            result = await wait_for_sensor(controller, worker, settings, condition, 1.5, guard,
                floor_target={"id": "selected-floor"} if fault == "floor" else None,
                destination_floor=[10, 150, 160] if fault == "surface" else None)
            assert result["status"] == ("condition_met" if fault is None else "timed_out" if fault == "timeout" else "blocked"), result
            assert not result["mission_success_verified"]
            if fault is None:
                assert result["elapsed_simulated_s"] == 1.
            elif fault == "timeout":
                assert result["elapsed_simulated_s"] == 1.5
        assert worker.sim.ticks - before <= 1.5 * 240
        assert math.dist(worker.sim.odometry[:2], initial_pose[:2]) < .02
    finally:
        await worker.close()


@pytest.mark.parametrize("fault", [None, "refresh_failure", "refresh_stop"])
async def test_sensor_wait_refreshes_real_depth_and_charging_physics(monkeypatch, fault):
    import math
    from uuid import uuid4
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from backend.contracts import Command
    from backend.continuous_navigation import ContinuousScan
    from backend.continuous_supervisor import SensorCondition, rotate, wait_for_sensor
    worker = SimulationWorker(challenge=get_challenge("recharge"), pace=False)
    controller = AgentController(FoundryConfig())
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.resume_manual(expected_stop_revision=worker.stop_revision)
        settings = AgentStart(run_id=worker.sim.run_id, episode_epoch=worker.epoch,
            execution_mode="luna_continuous", goal="Scripted observed charging condition, not mission qualification")
        await worker.scan_continuous(ContinuousScan(run_id=settings.run_id, episode_epoch=settings.episode_epoch, compact_arms=True))
        for speed in (-.2, -.2, .2, .2):
            current, _ = await worker.feedback()
            reply = await worker.execute(Command(run_id=settings.run_id, episode_epoch=settings.episode_epoch,
                observation_seq=current.seq, action_id=str(uuid4()), tool="drive_base",
                arguments={"linear_mps": speed, "angular_radps": 0., "duration_s": 2.}), assisted=False)
            assert reply.status == "ok", reply.message
            await worker.call(lambda sim: worker._sample_spatial(force=True))
        await rotate(controller, worker, settings, worker.stop_revision, 2 * math.pi, panoramic=True, pitch=.85)
        await worker.call(lambda sim: setattr(sim.challenge_progress.battery, "charge_pct", 85.))
        before = worker.sim.ticks
        pose = list(worker.sim.odometry)
        sequence = worker.spatial_sequence
        sample = worker._sample_spatial

        def sample_with_fault(force=False):
            if fault == "refresh_failure":
                worker.spatial_error = "Injected depth processing failure"
            elif fault == "refresh_stop":
                worker.stop()
            else:
                sample(force)

        monkeypatch.setattr(worker, "_sample_spatial", sample_with_fault)
        condition = SensorCondition(sensor="battery.charge_pct", comparison="gte", value=86.)
        if fault == "refresh_stop":
            with pytest.raises(asyncio.CancelledError):
                await wait_for_sensor(controller, worker, settings, condition, 30.)
        else:
            result = await wait_for_sensor(controller, worker, settings, condition, 30.)
            assert result["status"] == ("blocked" if fault else "condition_met"), result
            if fault is None:
                assert worker.spatial_sequence > sequence
                assert worker.sim.battery_sensor().charge_pct >= 86
                assert not worker.sim.challenge_status()["status"] == "completed"
        if fault:
            assert worker.sim.ticks == before
        assert math.dist(pose[:2], worker.sim.odometry[:2]) < .02
    finally:
        await worker.close()


async def test_sensor_wait_wall_timeout_and_session_deadline_authorize_no_action(monkeypatch):
    from types import SimpleNamespace
    import time
    from backend import continuous_supervisor
    from backend.contracts import BatterySensor
    clock = [0.]
    observation = SimpleNamespace(run_id="run", episode_epoch=0, frame_ref="frame", wall_timestamp=time.time(),
        simulated_time_s=0., odometry_m_rad=[0., 0., 0.], bumpers=[], proximity=None, navigation=None, skill=None,
        battery=BatterySensor(charge_pct=90., low=False, charging=True))

    async def feedback():
        return observation, b"paired"

    async def parking(*args, **kwargs):
        clock[0] = 2.
        return {"parking_margin_ok": True, "floor_rgb": [100, 100, 100]}

    async def unexpected_execution(*args, **kwargs):
        raise AssertionError("Expired wait must not execute any command")

    controller = SimpleNamespace(_check_live=lambda *args: None)
    worker = SimpleNamespace(feedback=feedback, parking_clearance=parking, execute=unexpected_execution, continuous=None)
    settings = SimpleNamespace(run_id="run", episode_epoch=0)
    monkeypatch.setattr(continuous_supervisor, "time", SimpleNamespace(monotonic=lambda: clock[0], time=time.time))
    condition = continuous_supervisor.SensorCondition(sensor="battery.charge_pct", comparison="gte", value=90.)
    result = await continuous_supervisor.wait_for_sensor(controller, worker, settings, condition, 1.)
    assert result["status"] == "timed_out" and not result["mission_success_verified"]

    def expired_session(*args):
        raise TimeoutError("Original session deadline expired")

    controller._check_live = expired_session
    with pytest.raises(TimeoutError, match="session deadline"):
        await continuous_supervisor.wait_for_sensor(controller, worker, settings, condition, 60.)


async def test_continuous_sensor_wait_reports_condition_without_claiming_mission(monkeypatch):
    import json
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from backend.continuous_supervisor import GuideContinuous
    from tests.test_agent import model_response
    calls = []
    for timeout in (0., 60.1, float("inf")):
        with pytest.raises(ValueError):
            GuideContinuous(action="wait_until", wait_timeout_s=timeout, reason="Reject an unbounded sensor wait request")

    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            data = next(json.loads(part["text"]) for part in inputs[-1]["content"]
                if part["type"] == "input_text" and part["text"].startswith('{"reachable_floor_candidates"'))
            calls.append(data)
            if len(calls) == 1:
                decision = {"action": "wait_until", "wait_timeout_s": 30.,
                    "wait_condition": {"sensor": "battery.charge_pct", "comparison": "gte", "value": 90.},
                    "reason": "Check the current battery condition without marking the mission successful"}
            else:
                assert data["local_execution"]["status"] == "condition_met"
                assert data["local_execution"]["mission_success_verified"] is False
                decision = {"action": "wait", "duration_s": .5, "reason": "A fresh scripted decision follows the condition result"}
            return model_response("guide_continuous", json.dumps(decision), call_id=f"sensor-wait-{len(calls)}")

        async def close(self):
            pass

    async def fixture_clearance(target=None, *, stationary_request=None):
        return {"parking_margin_ok": True, "floor_rgb": [100, 100, 100]}

    worker = SimulationWorker(challenge=get_challenge("recharge"), pace=False)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor())
    try:
        await asyncio.wrap_future(worker.ready)
        monkeypatch.setattr(worker, "parking_clearance", fixture_clearance)
        controller.start(worker, AgentStart(run_id=worker.latest["run_id"], episode_epoch=worker.epoch,
            execution_mode="luna_continuous", goal="Scripted sensor-action plumbing, not recharge qualification", max_turns=2))
        await asyncio.wait_for(asyncio.shield(controller.task), 60)
        assert controller.state["error"] is None, controller.state["error"]
        assert len(calls) == 2 and controller.state["outcome"]["kind"] == "limited"
        assert worker.sim.challenge_status()["status"] != "completed" and worker.latest["stopped"]
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("action", ["navigate", "explore"])
async def test_nav2_supervisor_dispatches_observed_goal_and_reports_blocked_result(monkeypatch, action):
    import json
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from tests.test_agent import model_response
    calls, destinations = [], []
    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            data = next(json.loads(part["text"]) for part in inputs[-1]["content"]
                if part["type"] == "input_text" and part["text"].startswith('{"reachable_floor_candidates"'))
            calls.append(data)
            if len(calls) == 1:
                choice = min(data["reachable_floor_candidates"], key=lambda candidate: candidate["target_m"][0] ** 2 + candidate["target_m"][1] ** 2)
                decision = {"action": action, "candidate_id": choice["id"], "reason": "Scripted Nav2 dispatch check"}
            else:
                assert data["local_execution"]["status"] == "blocked"
                assert data["local_execution"]["controller"] == "nav2_external"
                assert data["local_execution"]["mission_success_verified"] is False
                decision = {"action": "wait", "duration_s": .5, "reason": "Review the blocked Nav2 result"}
            return model_response("guide_continuous", json.dumps(decision), call_id=f"nav2-{len(calls)}")

        async def close(self):
            pass

    worker = SimulationWorker(challenge=get_challenge("park"), pace=False, rendering="tiny")
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor())
    try:
        await asyncio.wrap_future(worker.ready)
        monkeypatch.setattr(worker, "nav2_status", lambda: {"ready": True, "message": "Scripted bridge readiness"})
        original = worker.start_nav2_target
        async def scripted_nav2(sensor, target, stop_revision, task_revision):
            result = await original(sensor, target, stop_revision, task_revision)
            destinations.append(result)
            def finish(sim):
                worker.ros_navigation.finish(sim, "blocked", "Scripted Nav2 blocked result")
                worker._publish_navigation()
            await worker.call(finish)
            return worker.ros_navigation.state()
        monkeypatch.setattr(worker, "start_nav2_target", scripted_nav2)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=0, execution_mode="luna_continuous",
            navigation_backend="nav2", goal="Scripted Nav2 plumbing, not model or Nav2 qualification", max_turns=2))
        await asyncio.wait_for(asyncio.shield(controller.task), 45)
        assert controller.state["error"] is None, controller.state["error"]
        assert len(destinations) == 1 and destinations[0]["owner"] == "agent"
        assert len(calls) == 2 and controller.state["outcome"]["kind"] == "limited"
        assert worker.latest["stopped"] and not worker.navigation.buffer
    finally:
        await controller.halt()
        await worker.close()


async def test_adaptive_supervisor_bounds_scripted_no_progress(monkeypatch):
    import json
    from backend import continuous_supervisor
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from tests.test_agent import model_response
    calls = 0
    class Supervisor:
        async def respond(self, *args):
            nonlocal calls
            calls += 1
            return model_response("guide_continuous", json.dumps({"action": "look", "pitch_rad": .15,
                "reason": "Scripted repeated view without useful translation"}), call_id=f"repeat-{calls}")
        async def close(self):
            pass
    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor())
    try:
        await asyncio.wrap_future(worker.ready)
        original = worker.continuous_candidates
        async def no_candidates(visited=()):
            sensor, image, _, observation = await original(visited)
            return sensor, image, [], observation
        async def empty_scan(*args):
            return []
        monkeypatch.setattr(worker, "continuous_candidates", no_candidates)
        monkeypatch.setattr(continuous_supervisor, "adaptive_scan", empty_scan)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=0, execution_mode="luna_continuous",
            goal="Inspect the next reachable room", max_turns=20))
        await asyncio.wait_for(asyncio.shield(controller.task), 45)
        assert controller.state["error"] is None
        assert controller.state["outcome"]["kind"] == "unachievable"
        recoveries = [event["payload"] for event in controller.trace()["events"] if event["title"] == "No-progress recovery"]
        assert len(recoveries) == 3 and recoveries[-1]["replacement"] is None
        assert controller.navigation_memory.recovery_attempts == 2
        assert calls <= 15
        assert abs(worker.sim.odometry[0]) < .03 and abs(worker.sim.odometry[1]) < .03
        assert any(event["title"] == "Exploration preparation" and event["payload"]["full_startup_rotations"] == 0
            for event in controller.trace()["events"])
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("stop_during_reply", [False, True])
async def test_continuous_crossing_pause_returns_to_supervisor_for_stationary_yield(stop_during_reply):
    import json
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from tests.test_agent import model_response
    calls = []
    stopped_ticks = []
    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            content = inputs[-1]["content"]
            data = next(json.loads(item["text"]) for item in content
                if item["type"] == "input_text" and item["text"].startswith('{"reachable_floor_candidates"'))
            assert "pedestrian_crossing" not in json.dumps(data)
            calls.append(data)
            if len(calls) == 1:
                candidates = [item for item in data["reachable_floor_candidates"]
                    if item["target_m"][0] > .8 and abs(item["target_m"][1]) < .15]
                choice = max(candidates, key=lambda item: item["target_m"][0])
                decision = {"action": "navigate", "candidate_id": choice["id"],
                    "reason": "Approach the visible bay along observed clear floor"}
            else:
                assert data["local_execution"]["status"] == "blocked"
                assert "Moving obstacle" in data["local_execution"]["reason"]
                if stop_during_reply:
                    stopped_ticks.append(worker.sim.ticks)
                    worker.stop()
                decision = {"action": "wait", "duration_s": 4., "reason": "Hold while the visible crossing obstacle passes"}
            return model_response("guide_continuous", json.dumps(decision), call_id=f"crossing-{len(calls)}")
        async def close(self):
            pass
    worker = SimulationWorker(challenge=get_challenge("pedestrian_crossing"), pace=False)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=worker.epoch,
            execution_mode="luna_continuous", goal=get_challenge("pedestrian_crossing").goal, max_turns=2))
        await asyncio.wait_for(asyncio.shield(controller.task), 60)
        assert len(calls) == 2
        assert controller.state["error"] is None, (controller.state["error"], calls[-1].get("local_execution"))
        if stop_during_reply:
            assert worker.sim.ticks == stopped_ticks[0]
            assert worker.sim.challenge_status()["completed_objectives"] == 0
        else:
            assert worker.sim.challenge_status()["completed_objectives"] == 2
        assert not worker.sim.pedestrian_crossing["contact"]
        assert .3 < worker.sim.odometry[0] < .75
        assert not worker.navigation.buffer and not worker.navigation.tickets
        assert worker.latest["stopped"]
    finally:
        await controller.halt()
        await worker.close()


async def test_continuous_floor_target_generates_bounded_camera_derived_approaches():
    import numpy as np
    from backend.challenges import get_challenge
    from backend.continuous_navigation import ContinuousScan
    from backend.spatial import FloorRegionTracker
    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.scan_continuous(ContinuousScan(run_id=worker.sim.run_id, episode_epoch=worker.epoch, compact_arms=True))
        sensor, image, _, _ = await worker.continuous_candidates()
        tracker = FloorRegionTracker(worker.sim.run_id, worker.epoch)
        regions = tracker.update(sensor, image)
        target = next(region for region in regions if region["color"] == "green")
        _, _, candidates, observation = await worker.continuous_candidates(floor_target=target)
        forward = [candidate for candidate in candidates if candidate.get("floor_target_id") == target["id"]]
        assert forward and all(candidate["path_length_m"] <= 1.8 for candidate in forward)
        assert min(candidate["goal_distance_m"] for candidate in forward) < .1
        assert all(candidate["goal_progress_m"] > 0 for candidate in forward)
        assert not (await worker.parking_clearance(target))["inside_floor_region"]
        worker.stop()
        assert np.linalg.norm(observation.odometry_m_rad[:2]) < .01
    finally:
        await worker.close()


@pytest.mark.parametrize("stop_during_reply", [False, True])
@pytest.mark.parametrize("challenge_id", ["park", "pedestrian_crossing"])
async def test_continuous_floor_region_parks_without_spinning_and_rejects_late_motion(stop_during_reply, monkeypatch, challenge_id):
    import json
    from backend import continuous_supervisor
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from tests.test_agent import model_response
    calls = []
    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            data = next(json.loads(part["text"]) for part in inputs[-1]["content"]
                if part["type"] == "input_text" and part["text"].startswith('{"reachable_floor_candidates"'))
            calls.append(data)
            assert "objects" not in data and "geometry" not in data
            region = next((region for region in data["floor_regions"] if region["color"] == "green"), None)
            if stop_during_reply:
                worker.stop()
            if "Moving obstacle" in (data.get("local_execution") or {}).get("reason", ""):
                decision = {"action": "wait", "duration_s": 4, "reason": "Yield to the visible moving obstacle"}
            elif region:
                decision = {"action": "finish" if data["current_parking"].get("inside_floor_region") else "approach_target",
                    "floor_target_id": region["id"], "entry_confirmed": True, "reason": "Use only the measured green floor region and fresh clearance"}
            else:
                choice = max(data["reachable_floor_candidates"], key=lambda candidate: candidate["target_m"][0])
                decision = {"action": "navigate", "candidate_id": choice["id"], "reason": "Advance along the visible forward floor"}
            return model_response("guide_continuous", json.dumps(decision), call_id=f"floor-{len(calls)}")
        async def close(self):
            pass
    async def unexpected_rotation(*args, **kwargs):
        raise AssertionError("Observed bay parking must not require a full rotation")
    monkeypatch.setattr(continuous_supervisor, "rotate", unexpected_rotation)
    worker = SimulationWorker(challenge=get_challenge(challenge_id), pace=False)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=worker.epoch,
            execution_mode="luna_continuous", goal="Park fully in the green floor region, yielding to crossing obstacles", max_turns=12))
        await asyncio.wait_for(asyncio.shield(controller.task), 180)
        assert controller.state["error"] is None, controller.state["error"]
        if stop_during_reply:
            assert len(calls) == 1 and worker.sim.odometry[0] < .01
        else:
            assert worker.sim.challenge_status()["status"] == "completed", (controller.state["message"],
                [(item["local_execution"], item["floor_regions"]) for item in calls])
            assert controller.state["outcome"]["kind"] == "completed"
            assert len(calls) >= 3
        assert (worker.navigation is None or not worker.navigation.buffer) and worker.latest["stopped"]
    finally:
        await controller.halt()
        await worker.close()


def test_task_progress_retains_sensor_evidence_and_bounds_unchanged_inspection():
    from types import SimpleNamespace
    from backend.contracts import BatterySensor
    from backend.navigation_memory import TaskProgress
    progress = TaskProgress()
    observation = SimpleNamespace(run_id="episode", episode_epoch=1, frame_ref="before.png", simulated_time_s=26.65,
        odometry_m_rad=[2.905, .104, 0.], head_rad=[0., .3],
        battery=BatterySensor(charge_pct=86.452, low=False, charging=False))
    assert progress.observe(observation)["sensor_changes"] == []
    observation.frame_ref = "after.png"
    observation.simulated_time_s = 27.65
    observation.battery = BatterySensor(charge_pct=26.351, low=True, charging=False)
    measured = progress.observe(observation)
    change = measured["sensor_changes"][0]
    assert change["charge_delta_pct"] == pytest.approx(-60.101)
    assert change["from_frame"] == "before.png" and change["to_frame"] == "after.png"
    for attempt in range(3):
        observation.frame_ref = f"inspection-{attempt}.png"
        progress.observe(observation)
        result = progress.record_inspection("not_contained")
        assert result["inspection_exhausted"] == (attempt == 2)
    assert len(progress.summary()["sensor_changes"]) == 1
    transition = progress.advance("Review the next user-requested subgoal", event_id=change["id"])
    assert transition["source"] == "model_assessment_not_verified" and not transition["motion_authorized"]
    assert progress.summary()["inspection_attempts_here"] == 0
    with pytest.raises(ValueError, match="Repeated"):
        progress.advance("Another subgoal", event_id=change["id"])
    with pytest.raises(ValueError, match="current paired frame"):
        progress.advance("Visual subgoal", frame_ref="before.png")
    observation.episode_epoch = 2
    assert progress.observe(observation)["sensor_changes"] == []
    assert progress.summary()["current_subgoal"] is None
    with pytest.raises(ValueError, match="unavailable"):
        progress.advance("Old episode evidence", event_id=change["id"])


def test_task_progress_bounds_transitions_and_requires_one_available_evidence_source():
    from types import SimpleNamespace
    from backend.navigation_memory import TaskProgress
    progress = TaskProgress()
    observation = SimpleNamespace(run_id="episode", episode_epoch=1, frame_ref="current.png", simulated_time_s=0.,
        odometry_m_rad=[0., 0., 0.], head_rad=[0., .3], battery=None)
    progress.observe(observation)
    with pytest.raises(ValueError, match="Cite one"):
        progress.advance("Missing evidence")
    with pytest.raises(ValueError, match="Cite one"):
        progress.advance("Ambiguous evidence", event_id="unknown", frame_ref="current.png")
    with pytest.raises(ValueError, match="unavailable"):
        progress.advance("Unknown event", event_id="unknown")
    for index in range(progress.transition_limit):
        observation.frame_ref = f"frame-{index}.png"
        progress.observe(observation)
        progress.advance(f"Subgoal {index}", frame_ref=observation.frame_ref)
    observation.frame_ref = "another.png"
    progress.observe(observation)
    with pytest.raises(ValueError, match="exhausted"):
        progress.advance("Excess transition", frame_ref=observation.frame_ref)
    assert progress.summary()["transitions_remaining"] == 0


def test_task_progress_requires_changed_pose_or_view_not_new_frames():
    from types import SimpleNamespace
    from backend.navigation_memory import TaskProgress
    progress = TaskProgress()
    observation = SimpleNamespace(run_id="episode", episode_epoch=1, frame_ref="initial.png", simulated_time_s=0.,
        odometry_m_rad=[0., 0., 0.], head_rad=[0., .3], battery=None)
    progress.observe(observation)
    progress.record_inspection("not_contained")
    observation.frame_ref = "new.png"
    observation.simulated_time_s = 10.
    assert progress.observe(observation)["inspection_attempts_here"] == 1
    observation.head_rad = [.3, .3]
    assert progress.observe(observation)["inspection_attempts_here"] == 0
    progress.record_inspection("not_contained")
    observation.odometry_m_rad = [.2, 0., 0.]
    assert progress.observe(observation)["inspection_attempts_here"] == 0


@pytest.mark.parametrize("transition,stop_during_reply", [(False, False), (True, False), (True, True)])
async def test_continuous_bounds_inspection_and_subgoal_transition_preserves_stop(monkeypatch, transition, stop_during_reply):
    import json
    import math
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from tests.test_agent import model_response
    calls = []

    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            data = next(json.loads(part["text"]) for part in inputs[-1]["content"]
                if part["type"] == "input_text" and part["text"].startswith('{"reachable_floor_candidates"'))
            calls.append(data)
            assert "completed_objectives" not in json.dumps(data["task_progress"])
            if transition and len(calls) == 2:
                if stop_during_reply:
                    worker.stop()
                decision = {"action": "advance_subgoal", "next_subgoal": "Review another requested objective",
                    "evidence_frame_ref": data["task_progress"]["current_frame_ref"],
                    "reason": "Scripted assessment of the current paired view; not verified success"}
            elif transition and len(calls) == 3:
                assert data["selected_floor_target"] is None and data["destination_floor_reference"] is None
                assert not data["arrival_inspected"] and not data["visual_inspection_checked"]
                assert data["task_progress"]["inspection_attempts_here"] == 0
                assert data["task_progress"]["current_subgoal"] == "Review another requested objective"
                assert data["local_execution"]["motion_authorized"] is False
                decision = {"action": "wait", "duration_s": .5, "reason": "Wait on the next fresh decision without authorizing navigation"}
            else:
                region = next(region for region in data["floor_regions"] if region["color"] == "green")
                decision = {"action": "inspect_arrival", "floor_target_id": region["id"],
                    "reason": "Repeat a scripted rejected marked-floor inspection at the same pose"}
            return model_response("guide_continuous", json.dumps(decision), call_id=f"progress-{len(calls)}")

        async def close(self):
            pass

    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor())
    try:
        await asyncio.wrap_future(worker.ready)
        original_clearance = worker.parking_clearance

        async def rejected_clearance(target=None):
            return {**await original_clearance(target), "inside_floor_region": False}

        monkeypatch.setattr(worker, "parking_clearance", rejected_clearance)
        controller.start(worker, AgentStart(run_id=worker.latest["run_id"], episode_epoch=worker.epoch,
            execution_mode="luna_continuous", goal="Scripted inspection and planning-state checks", max_turns=3 if transition else 8))
        await asyncio.wait_for(asyncio.shield(controller.task), 60)
        assert controller.state["error"] is None, controller.state["error"]
        assert math.hypot(*worker.sim.odometry[:2]) < .02
        advanced = [event for event in controller.trace()["events"] if event["title"] == "Subgoal advanced"]
        assert len(advanced) == int(transition and not stop_during_reply)
        assert len(calls) == (2 if stop_during_reply else 3)
        if not transition:
            assert controller.state["outcome"]["kind"] == "unachievable"
        assert worker.latest["stopped"] and not controller.active
    finally:
        await controller.halt()
        await worker.close()


def test_continuous_recovery_keeps_observed_floor_goal():
    from types import SimpleNamespace
    from backend.continuous_supervisor import recovery_choice
    memory = SimpleNamespace(recovery_attempts=0, actions=[])
    observation = SimpleNamespace(odometry_m_rad=[0., 0., 0.])
    target = {"id": "floor-1", "center_m": [2., 0.]}
    candidates = [{"id": 1, "target_m": [.8, 0.], "distance_m": .8},
        {"id": 2, "target_m": [-1., 0.], "distance_m": 1., "distance_to_visited_m": 3.}]
    choice = recovery_choice(memory, observation, candidates, target)
    assert choice.candidate_id == 1 and choice.floor_target_id == "floor-1"
    choice = recovery_choice(memory, observation, candidates[1:], target)
    assert choice.action == "look" and choice.floor_target_id == "floor-1"


def test_continuous_surface_consistency_uses_sampled_colors_not_room_labels():
    from backend.continuous_supervisor import matching_floor
    assert matching_floor([110, 150, 160], [88, 120, 128])
    assert matching_floor([150, 150, 150], [100, 100, 100])
    assert not matching_floor([110, 150, 160], [150, 150, 150])
    assert not matching_floor(None, [150, 150, 150])


async def test_continuous_visual_inspection_requires_fresh_dwell_without_parking_rotation(monkeypatch):
    import json
    from backend import continuous_supervisor
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from tests.test_agent import model_response
    captured = []
    class Supervisor:
        async def respond(self, *args):
            data = next(json.loads(item["text"]) for item in args[-1][-1]["content"]
                if item["type"] == "input_text" and item["text"].startswith('{"reachable_floor_candidates"'))
            assert data["visual_inspection_checked"] == bool(captured)
            captured.append(await worker.call(lambda sim: (sim.ticks, sim.odometry.copy(), sim.observe(render=False).head_rad)))
            return model_response("guide_continuous", json.dumps({"action": "finish", "completion_mode": "visual_inspection",
                "entry_confirmed": True, "reason": "Scripted inspection claim for workflow validation only"}), call_id=f"inspect-{len(captured)}")
        async def close(self):
            pass
    async def unexpected_rotation(*args, **kwargs):
        raise AssertionError("Visual inspection must preserve its view rather than run a parking panorama")
    monkeypatch.setattr(continuous_supervisor, "rotate", unexpected_rotation)
    worker = SimulationWorker(challenge=get_challenge("apartment"), pace=False)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=worker.epoch,
            execution_mode="luna_continuous", goal=get_challenge("apartment").goal, max_turns=2))
        await asyncio.wait_for(asyncio.shield(controller.task), 45)
        assert controller.state["error"] is None, controller.state["error"]
        assert len(captured) == 2
        assert captured[1][0] - captured[0][0] == 240
        assert captured[1][1] == pytest.approx(captured[0][1], abs=.01)
        assert captured[1][2] == pytest.approx(captured[0][2], abs=.01)
        assert controller.state["outcome"]["source"] == "agent"
        assert worker.sim.challenge_status()["status"] != "completed"
        assert worker.latest["stopped"]
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("history_enabled", [True, False])
async def test_luna_motion_history_original_requires_fresh_view_and_stop_discards_late_reply(history_enabled):
    import base64
    import json
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from tests.test_agent import model_response
    received = []
    entered = asyncio.Event()
    chosen = None
    class Supervisor:
        async def respond(self, *args):
            nonlocal chosen
            content = args[-1][-1]["content"]
            data = next(json.loads(part["text"]) for part in content if part["type"] == "input_text"
                and part["text"].startswith('{"reachable_floor_candidates"'))
            images = [base64.b64decode(part["image_url"].split(",", 1)[1]) for part in content if part["type"] == "input_image"]
            received.append((data, await worker.call(lambda sim: sim.ticks), images))
            assert 1 <= len(images) <= 3
            assert "objects" not in data["camera_history"] and data["camera_history"]["source"] == "paired_head_camera"
            if not history_enabled:
                assert len(images) == 1 and not data["camera_history"]["enabled"]
                assert not data["camera_history"]["frames"] and not data["camera_history"]["available_frames"]
                if len(received) == 1:
                    return model_response("guide_continuous", json.dumps({"action": "inspect_history", "history_frame_id": "unavailable",
                        "reason": "Scripted baseline cannot access original images"}), call_id="disabled-history")
                entered.set()
                await asyncio.Event().wait()
            if len(received) == 1:
                assert data["camera_history"]["frames"] and len(images) == 2
                chosen = data["camera_history"]["frames"][0]["frame_id"]
                decision = {"action": "inspect_history", "history_frame_id": chosen}
            elif len(received) == 2:
                assert data["historical_original"]["frame_id"] == chosen and data["fresh_view_required_before_motion"]
                assert images[-1] == (await worker.inspect_camera_history(chosen, worker.stop_revision))[1]
                decision = {"action": "navigate", "candidate_id": 0,
                    "place_sighting": {"label": "Historical target", "evidence": "This old image must not label the current viewpoint"}}
            elif len(received) == 3:
                assert data["historical_original"] is None and not data["fresh_view_required_before_motion"]
                decision = {"action": "look", "yaw_rad": .4, "pitch_rad": .3}
            else:
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    decision = {"action": "circle", "object_label": "table", "object_bounds": [.2, .2, .7, .7]}
            return model_response("guide_continuous", json.dumps({**decision, "reason": "Scripted historical sighting inspection and safety regression"}), call_id=f"history-{len(received)}")
        async def close(self):
            pass
    worker = SimulationWorker(challenge=get_challenge("park"), pace=False)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor())
    controller.evaluation_camera_history = history_enabled
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=0, execution_mode="luna_continuous",
            goal="Inspect brief object sightings, then look again before moving", max_turns=8))
        waiting = asyncio.create_task(entered.wait())
        await asyncio.wait([waiting, controller.task], timeout=75, return_when=asyncio.FIRST_COMPLETED)
        assert entered.is_set(), controller.state
        if not history_enabled:
            assert received[0][1] == received[1][1] and worker.continuous is None
            assert any(event["title"] == "Historical frame inspection" and event["payload"]["status"] == "unavailable"
                       for event in controller.trace()["events"])
            return
        assert [entry[1] for entry in received[:3]] == [received[0][1]] * 3
        assert received[3][1] > received[2][1]
        assert worker.continuous is None
        assert not await worker.place_sightings()
        traces = controller.trace()["events"]
        assert any(event["title"] == "Historical motion selection deferred" for event in traces)
        assert any(event["title"] == "Historical frame inspection" for event in traces)
        ticks = worker.sim.ticks
        await controller.halt("Stop after historical review")
        assert worker.sim.ticks == ticks and worker.latest["stopped"] and worker.continuous is None
    finally:
        if 'waiting' in locals() and not waiting.done():
            waiting.cancel()
        await controller.halt()
        await worker.close()


def test_continuous_room_search_instructions_follow_requested_destination():
    from backend.continuous_supervisor import INSTRUCTIONS
    assert "fixture sightings for the requested destination" in INSTRUCTIONS
    assert "Follow the destination in the user goal" in INSTRUCTIONS
    assert "must retain bathroom fixture sightings" not in INSTRUCTIONS
    assert "When a bathroom is sighted" not in INSTRUCTIONS


def test_continuous_panorama_expires_on_translation_and_episode_change():
    import time
    from types import SimpleNamespace
    from backend.continuous_supervisor import nearby_panorama
    original = SimpleNamespace(run_id="run", episode_epoch=1, odometry_m_rad=[0., 0., 0.])
    rotated = SimpleNamespace(run_id="run", episode_epoch=1, odometry_m_rad=[.01, 0., 3.14])
    moved = SimpleNamespace(run_id="run", episode_epoch=1, odometry_m_rad=[.3, 0., 0.])
    reset = SimpleNamespace(run_id="other", episode_epoch=2, odometry_m_rad=[0., 0., 0.])
    frames = [(original, b"paired-image")]
    assert nearby_panorama(frames, rotated) == frames
    assert not nearby_panorama(frames, moved)
    assert not nearby_panorama(frames, reset)
    original.wall_timestamp = time.time() - 91
    assert not nearby_panorama(frames, rotated)
    original.wall_timestamp = time.time() + 10
    assert not nearby_panorama(frames, rotated)


@pytest.mark.parametrize("inspect_first", [False, True])
async def test_luna_continuous_uses_observed_candidates_and_preserves_stop(monkeypatch, inspect_first):
    import json
    from backend.agent import AgentController, AgentStart, FoundryConfig, robot_tools
    from backend.challenges import get_challenge
    from tests.test_agent import model_response
    inputs_seen = []
    entered = asyncio.Event()

    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            inputs_seen.append(inputs)
            if inspect_first and len(inputs_seen) == 1:
                return model_response("guide_continuous", json.dumps({"action": "inspect_arrival",
                    "place_sighting": {"label": "Kitchen", "evidence": "Scripted stove fixture sighting"},
                    "reason": "Inspect even without a destination-surface match"}), call_id="inspect-first")
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return model_response("guide_continuous", json.dumps({"action": "navigate", "candidate_id": 0,
                    "place_sighting": {"label": "Cancelled room", "evidence": "Scripted late sighting"},
                    "reason": "This cancelled candidate must not execute"}))

        async def close(self):
            pass

    worker = SimulationWorker(challenge=get_challenge("kitchen_bathroom"), pace=False)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor())
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.call(lambda sim: (setattr(sim, "width", 320), setattr(sim, "height", 240)))
        controller.start(worker, AgentStart(run_id=worker.latest["run_id"], episode_epoch=0,
            execution_mode="luna_continuous", goal=get_challenge("kitchen_bathroom").goal))
        arrived = asyncio.create_task(entered.wait())
        await asyncio.wait([arrived, controller.task], timeout=75, return_when=asyncio.FIRST_COMPLETED)
        assert entered.is_set(), {"error": controller.state["error"], "navigation": worker.latest["navigation"], "pose": worker.sim.odometry.tolist()}
        await arrived
        assert worker.spatial_enabled and controller.state["local_model"] is None
        assert {tool["name"] for tool in robot_tools("luna_continuous")} == {"guide_continuous"}
        payload = json.dumps(inputs_seen)
        assert "reachable_floor_candidates" in payload and "bathroom_floor" not in payload and "geometry" not in payload
        if inspect_first:
            assert any(entry["title"] == "Room entry verification" and entry["payload"]["status"] == "inspection_required"
                for entry in controller.trace()["events"])
        ticks = worker.sim.ticks
        await controller.halt("Stop during continuous supervisor")
        assert worker.sim.ticks == ticks and not controller.active
        assert worker.continuous is None and worker.latest["stopped"]
        labels = await worker.place_sightings()
        assert [item["label"] for item in labels] == (["Kitchen"] if inspect_first else [])
        if inspect_first:
            assert "Scripted stove fixture sighting" in json.dumps(inputs_seen[-1])
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("stop_during_reply", [False, True])
@pytest.mark.parametrize("challenge_id", ["park", "flat_kitchen"])
async def test_supervisor_plans_during_motion_and_discards_late_reply(monkeypatch, stop_during_reply, challenge_id):
    import json
    from backend import continuous_supervisor
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from tests.test_agent import model_response
    entered, release = asyncio.Event(), asyncio.Event()
    received = []

    async def no_panorama(*args, **kwargs):
        return []

    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            assert goal == get_challenge(challenge_id).goal
            payload = next(json.loads(part["text"]) for part in inputs[-1]["content"]
                if part["type"] == "input_text" and part["text"].startswith('{"reachable_floor_candidates"'))
            received.append(payload)
            candidates = payload["reachable_floor_candidates"]
            review = payload["camera_history"]
            assert all(frame["sequence"] < review["through_sequence"] for frame in review["frames"])
            assert review["through_sequence"] <= worker.camera_history.last_sequence
            if len(received) == 1:
                choice = min(candidates, key=lambda item: abs(item["target_m"][1]) + abs(item["distance_m"] - .8))
            else:
                assert payload["planning_while_moving"]
                choice = min(candidates, key=lambda item: abs(item["target_m"][1]) + abs(item["distance_m"] - 1.4))
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            return model_response("guide_continuous", json.dumps({"action": "navigate", "candidate_id": choice["id"],
                "reason": "Scripted timing test of observed moving continuation"}), call_id=f"moving-{len(received)}")

        async def close(self):
            pass

    monkeypatch.setattr(continuous_supervisor, "rotate", no_panorama)
    worker = SimulationWorker(challenge=get_challenge(challenge_id), pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=0, execution_mode="luna_continuous",
            continuous_handoff=True, goal=get_challenge(challenge_id).goal, max_turns=2))
        await asyncio.wait_for(entered.wait(), 25)
        before = worker.sim.odometry[0]
        deadline = asyncio.get_running_loop().time() + 3
        while worker.sim.odometry[0] - before < .04:
            assert asyncio.get_running_loop().time() < deadline, {"before": before,
                "odometry": worker.sim.odometry.tolist(), "error": controller.state["error"],
                "continuous": worker.latest.get("continuous_navigation"), "navigation": worker.latest.get("navigation")}
            await asyncio.sleep(.02)
        assert controller.state["phase"] == "thinking" and worker.continuous.active
        if stop_during_reply:
            worker.stop()
            await controller.halt("Stop while preparing a moving continuation")
            assert worker.continuous.handoffs == 0 and not worker.navigation.buffer
        else:
            release.set()
            await asyncio.wait_for(controller.task, 10)
            assert worker.continuous.handoffs == 1, controller.state["error"]
            assert worker.continuous.buffer_stops == 0
        assert not controller.active
    finally:
        release.set()
        await controller.halt()
        await worker.close()


def test_navigation_backend_boundary_rejects_stale_camera_and_unsupported_goals():
    from backend.navigation_backends import NavigationCapabilities, NavigationFrame, NavigationGoal, NavigationProposal
    goal = NavigationGoal(kind="point", evidence="observed_map", point_m=[1., 0.])
    frame = NavigationFrame(run_id="test", episode_epoch=0, sequence=1, captured_at=10., stop_revision=0,
        goal_id=goal.goal_id, odometry_m_rad=[0., 0., 0.], head_rad=[0., .3])
    capability = NavigationCapabilities(name="candidate", goals=["point"], sensors=["rgb", "depth"], fixed_head=True)
    proposal = NavigationProposal(ticket=frame, backend="candidate", path_m=[[0., 0.], [1., 0.]])
    assert proposal.checked_path(frame, goal, capability, 10.5, lambda path: True) == proposal.path_m
    for change, message in [({"stop_revision": 1}, "revision"), ({"goal_id": "changed"}, "revision"),
                            ({"head_rad": [.3, .3]}, "head"), ({"odometry_m_rad": [.2, 0., 0.]}, "moved")]:
        with pytest.raises(MotionError, match=message):
            proposal.checked_path(frame.model_copy(update=change), goal, capability, 10.5, lambda path: True)
    with pytest.raises(MotionError, match="fresh"):
        proposal.checked_path(frame, goal, capability, 11.01, lambda path: True)
    with pytest.raises(MotionError, match="unknown"):
        proposal.checked_path(frame, goal, capability, 10.5, lambda path: False)
    with pytest.raises(MotionError, match="representation"):
        proposal.checked_path(frame, goal, capability.model_copy(update={"goals": ["image", "explore"]}), 10.5, lambda path: True)
    with pytest.raises(ValueError, match="NONHOLONOMIC_PATH"):
        NavigationProposal(ticket=frame, backend="candidate", path_m=[[0., 0.], [0., 1.]],
            poses_m_rad=[[0., 0., 0.], [0., 1., 0.]])
    with pytest.raises(ValueError, match="retained goal view"):
        NavigationGoal(kind="image", evidence="remembered_view", point_m=[1., 0.])


def test_navigation_images_reset_and_navdp_metric_depth_encoding():
    from io import BytesIO
    from types import SimpleNamespace
    import math
    import numpy as np
    from PIL import Image
    from backend.navigation_backends import NavigationImages, NavigationFrame, navdp_depth_png, relative_proposal
    frame = NavigationFrame(run_id="test", episode_epoch=0, sequence=1, captured_at=10., stop_revision=0,
        goal_id="goal", odometry_m_rad=[1., 2., math.pi / 2], head_rad=[0., .3])
    history = NavigationImages(4)
    image = BytesIO()
    Image.new("RGB", (4, 2)).save(image, format="PNG")
    image = image.getvalue()
    for index in range(4):
        history.append(frame.model_copy(update={"sequence": index + 1, "captured_at": 10 + index * .25}), image)
    assert len(history.frames) == 4
    view = history.remember(frame, image)
    assert history.goal_image(view, frame, 11.) == image
    second_view = history.remember(frame.model_copy(update={"sequence": 2, "captured_at": 11.}), image)
    history.connect(view, second_view, status="arrived", length_m=.5)
    assert history.connections[-1]["requires_fresh_clearance"]
    with pytest.raises(ValueError, match="successfully traversed"):
        history.connect(view, second_view, status="blocked", length_m=.5)
    history.append(frame.model_copy(update={"sequence": 5, "captured_at": 11., "head_rad": [0., .5]}), image)
    assert len(history.frames) == 1
    with pytest.raises(MotionError, match="different head"):
        history.goal_image(view, history.frames[-1][0], 11.)
    with pytest.raises(MotionError, match="another episode"):
        history.goal_image(view, frame.model_copy(update={"episode_epoch": 1}), 11.)
    sensor = SimpleNamespace(calibration=SimpleNamespace(width=4, height=2), depth_m=[None, .05, .1, 1.2345, 5., 5.1, float("nan"), float("inf")])
    with Image.open(BytesIO(navdp_depth_png(sensor, image))) as depth:
        assert np.asarray(depth).tolist() == [[0, 0, 1000, 12345], [50000, 0, 0, 0]]
    proposal = relative_proposal(frame, "nomad", [[.5, 0.], [.7, .1]])
    np.testing.assert_allclose(proposal.path_m, [[1., 2.], [1., 2.5], [.9, 2.7]])
    from backend.navigation_backends import navdp_camera_proposal
    from backend.spatial import camera_in_odometry
    eye, optical = camera_in_odometry(frame.head_rad, frame.odometry_m_rad)
    navdp = navdp_camera_proposal(frame, [[.5, 0., .4], [.7, .1, .5]])
    np.testing.assert_allclose(navdp.path_m[1], (eye + .5 * optical[:, 2])[:2])
    np.testing.assert_allclose(navdp.path_m[2], (eye + .7 * optical[:, 2] - .1 * optical[:, 0])[:2])
    with pytest.raises(MotionError, match="no translational"):
        navdp_camera_proposal(frame, [[0., 0., .5], [0., 0., 1.]])


@pytest.mark.parametrize("fault", ["look", "expiry", "limit", "stop", "deadline"])
async def test_continuous_control_recovery_retains_task_and_authority(fault):
    import json
    import time
    from backend.agent import AgentController, AgentStart, FoundryConfig, NavigationEvaluationBudget
    from backend.challenges import get_challenge
    from tests.test_agent import model_response
    received = []
    deadlines = []
    session_ids = []

    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            context = next(json.loads(part["text"]) for part in inputs[-1]["content"]
                if part["type"] == "input_text" and part["text"].startswith('{"reachable_floor_candidates"'))
            received.append(context)
            deadlines.append(controller.session_deadline)
            session_ids.append(controller.state["session_id"])
            request = len(received)
            if request == 1 or (fault == "limit" and request % 2 == 1):
                if request > 1:
                    assert not context["planning_while_moving"]
                    assert context["local_execution"]["motion_authorized"] is False
                choice = min(context["reachable_floor_candidates"], key=lambda item: abs(item["bearing_from_base_rad"]))
                decision = {"action": "explore", "candidate_id": choice["id"], "reason": "Scripted observed exploration"}
            elif request == 2 or fault == "limit":
                assert context["planning_while_moving"] and worker.continuous.active
                if fault == "stop":
                    worker.stop()
                elif fault == "deadline":
                    controller.session_deadline = time.monotonic() - 1
                else:
                    def change_deadline(sim):
                        worker.navigation.skill_deadline = time.monotonic() + (2. if fault == "look" else -1.)
                        if fault != "look":
                            worker.navigation.tick(sim)
                            worker.continuous.status, worker.continuous.reason = "blocked", worker.navigation.reason
                            worker._publish_navigation()
                        return worker.navigation.revision
                    await worker.call(change_deadline)
                decision = {"action": "look", "yaw_rad": .9, "reason": "Scripted inspection requested near lease expiry"}
            else:
                assert not context["planning_while_moving"] and not worker.continuous.active
                assert not worker.navigation.buffer and not worker.sim.cancel.is_set()
                result = context["local_execution"]
                assert result["motion_authorized"] is False
                assert result["status"] == ("stopped_for_review" if fault == "look" else "blocked")
                assert context["measured_episode_memory"]["revision"] > 0
                decision = {"action": "wait", "duration_s": .5, "reason": "Use new stationary feedback without claiming success"}
            return model_response("guide_continuous", json.dumps(decision), call_id=f"lease-{request}")

        async def close(self):
            pass

    worker = SimulationWorker(challenge=get_challenge("park"), pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor(),
        evaluation_budget=NavigationEvaluationBudget(timeout_s=180))
    if fault == "limit":
        progress = controller.navigation_memory.progress
        controller.navigation_memory.progress = lambda: {**progress(), "recovery_needed": False}
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=0, execution_mode="luna_continuous",
            goal="Retain this task across recoverable local expiry", max_turns=10 if fault == "limit" else 3))
        controller.turn_indices = lambda settings: range(20 if fault == "limit" else 3)
        await asyncio.wait_for(controller.task, 90)
        events = controller.trace()["events"]
        recoveries = [entry for entry in events if entry["title"] == "Motion lease recovery"]
        discarded = [entry for entry in events if entry["title"] == "Expired motion decision discarded"]
        assert len(set(session_ids)) == 1 and len(set(deadlines)) == 1
        assert not worker.navigation.buffer and not controller.active
        if fault in {"look", "expiry"}:
            assert len(received) == 3 and controller.state["error"] is None
            assert len(recoveries) == int(fault == "expiry") and len(discarded) == int(fault == "expiry")
            assert received[1]["task_progress"]["current_frame_ref"] != received[2]["task_progress"]["current_frame_ref"]
            assert not any(entry["payload"].get("action") == "look" for entry in events if entry["title"] == "Continuous goal selected")
            assert any(entry["payload"].get("action") == "wait" for entry in events if entry["title"] == "Continuous execution feedback")
        elif fault == "limit":
            assert len(recoveries) == 4 and len(discarded) == 3 and len(received) <= 20, controller.state["error"]
            assert "3 recoveries" in controller.state["error"]
        else:
            assert len(received) == 2 and not recoveries
            if fault == "stop":
                assert worker.latest["stopped"]
            else:
                assert "Evaluation time budget exhausted" in controller.state["outcome"]["message"]
    finally:
        await controller.halt()
        await worker.close()


def test_continuous_decision_prose_cannot_override_executable_validation():
    import json
    from backend.continuous_supervisor import parse_continuous_decision
    arguments = {"action": "wait", "duration_s": .5, "reason": "r" * 559,
        "memory": "m" * 1600, "next_subgoal": "unused" * 100}
    original = json.dumps(arguments)
    guide, changes = parse_continuous_decision(original)
    assert guide.action == "wait" and guide.duration_s == .5
    assert len(guide.reason) == 500 and len(guide.memory) == 1400 and guide.next_subgoal is None
    assert {item["field"] for item in changes} == {"reason", "memory", "next_subgoal"}
    assert json.loads(original) == arguments
    assert parse_continuous_decision('{"action":"wait"}')[0].reason
    for invalid in ({**arguments, "duration_s": 999}, {**arguments, "action": "drive"},
                    {**arguments, "action": "advance_subgoal"}, {**arguments, "floor_target_id": "x" * 81},
                    {**arguments, "skill_plan": {"components": [{"kind": "cruise", "length_m": .1}]}},
                    {**arguments, "unexpected": True}):
        with pytest.raises(ValueError):
            parse_continuous_decision(json.dumps(invalid))
    with pytest.raises(ValueError):
        parse_continuous_decision('{"action":')


@pytest.mark.parametrize("outcome", ["recovered", "exhausted", "stop", "moving", "deadline", "prose"])
async def test_continuous_schema_recovery_is_bounded(monkeypatch, outcome):
    import json
    import time
    from backend.agent import AgentController, AgentStart, FoundryConfig, NavigationEvaluationBudget
    from backend.challenges import get_challenge
    from tests.test_agent import model_response
    calls = []
    correction = asyncio.Event()

    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            calls.append(inputs)
            if outcome == "moving" and len(calls) == 1:
                context = next(json.loads(part["text"]) for message in inputs for part in message["content"]
                    if part["type"] == "input_text" and part["text"].startswith('{"reachable_floor_candidates"'))
                choice = min(context["reachable_floor_candidates"], key=lambda candidate: abs(candidate["bearing_from_base_rad"]))
                return model_response("guide_continuous", json.dumps({"action": "explore", "candidate_id": choice["id"],
                    "reason": "Exercise schema rejection while rolling"}), call_id="schema-moving")
            if outcome == "moving" and len(calls) == 2:
                assert worker.continuous.active and worker.navigation.buffer
            if outcome == "moving" and len(calls) == 3:
                assert not worker.continuous.active and not worker.navigation.buffer
            if len(calls) > 1:
                correction.set()
                if outcome == "deadline":
                    controller.session_deadline = time.monotonic() - 1.
                if outcome == "stop":
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        pass
            invalid = outcome != "prose" and (len(calls) == (2 if outcome == "moving" else 1) or outcome == "exhausted")
            decision = {"action": "wait", "duration_s": 999 if invalid else .5, "reason": "Hold for fresh evidence"}
            if outcome == "prose":
                decision.update(reason="r" * 559, memory="m" * 1600, next_subgoal="unused" * 100)
            return model_response("guide_continuous", json.dumps(decision), call_id=f"schema-{len(calls)}")

        async def close(self):
            pass

    worker = SimulationWorker(challenge=get_challenge("park"), pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor(),
        evaluation_budget=NavigationEvaluationBudget(timeout_s=180) if outcome == "deadline" else None)
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=0, execution_mode="luna_continuous",
            goal="Exercise bounded response correction", max_turns=5 if outcome == "exhausted" else 3 if outcome == "moving" else 2))
        if outcome == "stop":
            async with asyncio.timeout(45):
                while not correction.is_set():
                    assert not controller.task.done(), controller.state["error"]
                    await asyncio.sleep(.02)
            await controller.halt("Stop during schema correction")
            ticks = worker.sim.ticks
            await worker.hold_stopped()
            assert worker.sim.ticks == ticks and worker.latest["stopped"]
        else:
            await asyncio.wait_for(controller.task, 60)
        events = controller.trace()["events"]
        rejected = [entry for entry in events if entry["title"] == "Continuous decision rejected"]
        results = [entry for entry in events if entry["title"] == "Continuous execution feedback"]
        assert len(calls) == (3 if outcome in {"exhausted", "moving"} else 2), controller.state["error"]
        if outcome == "prose":
            assert not rejected and len(results) == 2 and controller.state["error"] is None
            assert len([entry for entry in events if entry["title"] == "Decision commentary normalized"]) == 2
            response = next(entry for entry in events if entry["kind"] == "response")
            assert len(json.loads(response["payload"]["calls"][0]["arguments"])["reason"]) == 559
            assert not worker.continuous and controller.navigation_memory.instructions[-1] == "Exercise bounded response correction"
            return
        assert len(rejected) == (3 if outcome == "exhausted" else 1)
        assert all(entry["payload"]["motion_authorized"] is False for entry in rejected)
        correction_inputs = calls[2 if outcome == "moving" else 1]
        context = next(json.loads(part["text"]) for message in correction_inputs for part in message["content"]
            if part["type"] == "input_text" and part["text"].startswith('{"reachable_floor_candidates"'))
        assert context["decision_validation"]["status"] == "schema_rejected"
        assert context["decision_validation"]["errors"][0]["field"] == "duration_s"
        assert context["decision_validation"]["corrections_remaining"] == 2
        assert context["planning_while_moving"] is False
        assert "x" * 559 not in json.dumps(calls[1])
        assert len(results) == (2 if outcome == "moving" else 1 if outcome == "recovered" else 0)
        assert not worker.continuous or not worker.continuous.active
        if outcome == "exhausted":
            assert "2 correction attempts" in controller.state["error"]
        elif outcome == "deadline":
            assert "Evaluation time budget exhausted" in controller.state["outcome"]["message"]
        elif outcome in {"recovered", "moving"}:
            assert controller.state["error"] is None
    finally:
        await controller.halt()
        await worker.close()


@pytest.mark.parametrize("fault", [None, "stop", "revision", "task", "episode"])
async def test_continuous_schema_pause_cancels_only_matching_motion(fault):
    from types import SimpleNamespace
    from backend.continuous_navigation import ContinuousNavigation, ContinuousScan
    sim = SimpleNamespace(run_id="schema", epoch=int(fault == "episode"), cancel=asyncio.Event())
    if fault == "stop":
        sim.cancel.set()
    runtime = SimpleNamespace(status="running", buffer=["authorized motion"])

    def cancel(current, reason):
        runtime.status = "cancelled"
        runtime.buffer.clear()

    runtime.cancel = cancel
    async def call(operation):
        return operation(sim)
    control = ContinuousNavigation([[0., 0.], [1., 0.]])
    worker = SimpleNamespace(call=call, continuous=control, navigation=runtime,
        stop_revision=int(fault == "revision"), task_revision=int(fault == "task"), _publish_navigation=lambda: None)
    request = ContinuousScan(run_id="schema", episode_epoch=0)
    if fault:
        with pytest.raises(MotionError):
            await SimulationWorker.pause_continuous(worker, request, 0, 0, "Rejected decision")
        assert control.active and runtime.buffer and runtime.status == "running"
    else:
        await SimulationWorker.pause_continuous(worker, request, 0, 0, "Rejected decision")
        assert not control.active and not runtime.buffer and runtime.status == "cancelled"


@pytest.mark.parametrize("stop_phase", [None, "reply", "motion", "inspection", "budget"])
async def test_motion_skill_controller_uses_real_depth_and_preserves_stop(monkeypatch, stop_phase):
    import json
    from backend import continuous_supervisor
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from tests.test_agent import model_response
    entered = asyncio.Event()
    calls = []
    wait = continuous_supervisor.wait_stationary

    async def inspect_wait(*args, **kwargs):
        if stop_phase == "inspection":
            entered.set()
            await asyncio.Event().wait()
        return await wait(*args, **kwargs)

    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            calls.append(inputs)
            if len(calls) == 1:
                if stop_phase == "reply":
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        pass
                return model_response("guide_continuous", json.dumps({"action": "compose",
                    "skill_plan": {"components": [{"kind": "cruise", "length_m": .4},
                        {"kind": "curve", "length_m": .7, "angle_rad": .25}, {"kind": "inspect"}]},
                    "completion_mode": "visual_inspection", "reason": "Scripted observed motion and inspection test"}), call_id="skills-first")
            if stop_phase == "budget":
                context = next(json.loads(part["text"]) for message in inputs for part in message["content"]
                    if part["type"] == "input_text" and part["text"].startswith('{"reachable_floor_candidates"'))
                result = context["local_execution"]
                assert result["admission"] == "no_feasible_plan" and result["error_code"] == "PLAN_BUDGET"
                assert not context["planning_while_moving"] and result["motion_authorized"] is False
                assert result["skill_timing"]["shorter_component_options"] == {}
                assert result["skill_timing"]["validation_phase"] == "before_depth"
                assert result["skill_timing"]["throughput"]["sample_windows"] == 1
            return model_response("guide_continuous", json.dumps({"action": "wait", "duration_s": .5,
                "reason": "Review the new inspection evidence without claiming success"}), call_id="skills-review")

        async def close(self):
            pass

    monkeypatch.setattr(continuous_supervisor, "wait_stationary", inspect_wait)
    worker = SimulationWorker(challenge=get_challenge("park"), pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor())
    try:
        await asyncio.wrap_future(worker.ready)
        if stop_phase == "budget":
            await worker.call(lambda sim: worker.motion_throughput.windows.append(10.))
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=0, execution_mode="luna_continuous",
            skill_composer=True, goal="Scripted observed curve and inspection", max_turns=2))
        if stop_phase == "budget":
            await asyncio.wait_for(controller.task, 75)
            events = controller.trace()["events"]
            results = [entry["payload"] for entry in events if entry["title"] == "Continuous execution feedback"]
            assert controller.state["error"] is None and len(calls) == 2
            assert results[0]["admission"] == "no_feasible_plan" and not results[0]["inspection_ready"]
            assert results[1]["action"] == "wait" and results[1]["status"] == "ok"
            assert worker.continuous is None and abs(worker.sim.odometry[0]) < .01
            feedback = [entry["payload"]["observation"]["frame_ref"] for entry in events if entry["kind"] == "feedback"]
            assert len(feedback) == 2 and feedback[0] != feedback[1]
        elif stop_phase is None:
            await asyncio.wait_for(controller.task, 75)
            results = [entry["payload"] for entry in controller.trace()["events"] if entry["title"] == "Continuous execution feedback"]
            assert results and results[0].get("inspection_ready"), {"results": results, "error": controller.state["error"]}
            assert results[0]["mission_success_verified"] is False
            timing = results[0]["skill_timing"]
            assert timing["estimated_motion_s"] < timing["remaining_lease_s"]
            assert timing["terminal_slowdown_sim_s"] > 0 and timing["throughput"]["fallback_wall_ratio"] == 2.
            assert worker.motion_throughput.state()["sample_windows"] > 0
            assert worker.sim.odometry[0] > .8 and len(calls) == 2
            assert not worker.sim.proximity_sensors().collisions
        else:
            async with asyncio.timeout(75):
                while not entered.is_set():
                    if stop_phase == "motion" and worker.continuous and worker.continuous.active and worker.continuous.updates > 3:
                        break
                    assert not controller.task.done(), controller.state
                    await asyncio.sleep(.02)
            await controller.halt("Scripted composer Stop")
            ticks = worker.sim.ticks
            await worker.hold_stopped()
            assert worker.sim.ticks == ticks and worker.latest["stopped"] and not controller.active
            assert not worker.navigation or not worker.navigation.buffer
            if stop_phase == "reply":
                assert worker.continuous is None
        payload = json.dumps(calls)
        assert "reachable_floor_candidates" in payload and "geometry" not in payload and "challenge_status" not in payload
    finally:
        await controller.halt()
        await worker.close()


def test_motion_skill_tools_are_opt_in_and_versioned():
    from backend.agent import AgentStart, controller_instructions, robot_tools
    from scripts.evaluate_supervised import argument_parser
    from backend.continuous_supervisor import GuideContinuous
    from backend.experiment_variants import variant_snapshot
    from types import SimpleNamespace
    settings = AgentStart(run_id="skills", episode_epoch=0, goal="Inspect a visible target", execution_mode="luna_continuous")
    baseline = robot_tools("luna_continuous")[0]["parameters"]
    candidate = robot_tools("luna_continuous", True)[0]["parameters"]
    assert not settings.skill_composer and "skill_plan" not in baseline["properties"]
    assert argument_parser().parse_args(["--output", "unused", "--skill-composer"]).skill_composer
    with pytest.raises(ValueError, match="continuous control"):
        AgentStart(run_id="skills", episode_epoch=0, goal="Wrong execution mode", skill_composer=True)
    assert "compose" not in baseline["properties"]["action"]["enum"]
    assert "compose" in candidate["properties"]["action"]["enum"]
    assert "Motion skill composition" not in controller_instructions("luna_continuous")
    assert "Motion skill composition" in controller_instructions("luna_continuous", True)
    with pytest.raises(ValueError, match="skill_plan"):
        GuideContinuous(action="compose", reason="Select a bounded observed path")
    profile = SimpleNamespace(provider="foundry", id="luna", deployment="test")
    before = variant_snapshot(settings, profile)
    after = variant_snapshot(settings.model_copy(update={"skill_composer": True}), profile)
    assert before["architecture"]["id"] == "observed-continuous"
    assert after["architecture"]["id"] == "motion-skills"
    assert before["code_sha256"] == after["code_sha256"]
    assert before["model_variant"]["revision"] != after["model_variant"]["revision"]


@pytest.mark.parametrize("fault", [None, "stop", "task", "episode", "expired", "moved", "unknown", "stale", "stop_refresh", "budget", "budget_refresh"])
async def test_motion_skill_worker_rejects_superseded_or_unobserved_plans(fault):
    import time
    import numpy as np
    from types import SimpleNamespace
    from backend.worker import MotionThroughput
    from backend.continuous_navigation import MotionSkillPlan, ContinuousNavigation
    sim = SimpleNamespace(cancel=asyncio.Event(), run_id="skills", epoch=int(fault == "episode"),
        odometry=np.array([.2 if fault == "moved" else 0., 0., 0.]), held=None,
        robot_footprint=lambda: {"radius_m": .3})
    if fault == "stop":
        sim.cancel.set()
    async def call(operation):
        return operation(sim)
    worker = SimpleNamespace(call=call, stop_revision=0, task_revision=2 if fault == "task" else 1,
        renderer=None, skill=None, navigation=None, continuous=None, spatial_error=None,
        motion_throughput=MotionThroughput(),
        spatial_map=SimpleNamespace(public=lambda: {"stale": fault == "stale"},
            traversable=lambda *args: True, contains_path=lambda *args: fault != "unknown"),
        _publish_navigation=lambda: None)
    worker._continuous_guard = lambda current, request, revision: SimulationWorker._continuous_guard(worker, current, request, revision)
    def refresh(force=False):
        if fault == "stop_refresh":
            worker.stop_revision += 1
        if fault == "budget_refresh":
            sensor.captured_at -= 14.
    worker._sample_spatial = refresh
    def start(path, **kwargs):
        worker.continuous = ContinuousNavigation(path)
        worker.navigation = SimpleNamespace(skill_deadline=time.monotonic() + 60)
    worker._start_navigation_path = start
    sensor = SimpleNamespace(run_id="skills", episode_epoch=0, captured_at=time.monotonic() - (16 if fault == "expired" else 14 if fault == "budget" else 0),
        odometry_m_rad=[0., 0., 0.])
    plan = MotionSkillPlan(components=[{"kind": "cruise", "length_m": 1.5 if fault in {"budget", "budget_refresh"} else .6}, {"kind": "inspect"}])
    if fault:
        with pytest.raises((MotionError, ValueError)) as rejected:
            await SimulationWorker.start_motion_skills(worker, sensor, plan, [], 0, 1)
        assert worker.continuous is None
        if fault in {"budget", "budget_refresh"}:
            assert rejected.value.timing["validation_phase"] == ("after_depth" if fault == "budget_refresh" else "before_depth")
            assert rejected.value.timing["throughput"]["fallback_wall_ratio"] == 2.
    else:
        result = await SimulationWorker.start_motion_skills(worker, sensor, plan, [], 0, 1)
        assert result["status"] == "running" and len(result["skill_components"]) == 1
        assert worker.navigation.skill_deadline == sensor.captured_at + 20.


def test_motion_skill_lease_admission_preserves_selected_geometry():
    from backend.continuous_navigation import MotionSkillPlan, compile_motion_skills, motion_skill_timing
    plan = MotionSkillPlan(components=[{"kind": "approach", "candidate_id": 1}, {"kind": "inspect"}])
    path, components = compile_motion_skills([0., 0., 0.], plan, [{"id": 1, "target_m": [1.58, 0.]}])
    with pytest.raises(MotionError, match="Composed motion needs an estimated"):
        motion_skill_timing(path, 0., 14.328)
    assert path[-1].tolist() == [1.58, 0.] and len(plan.components) == 2
    fresh = motion_skill_timing(path, 0., 1.)
    assert fresh["estimated_motion_s"] < fresh["remaining_lease_s"]
    assert fresh["planning_speed_mps"] <= .5 and fresh["estimate_only"]
    short, _ = compile_motion_skills([0., 0., 0.], MotionSkillPlan(components=[{"kind": "cruise", "length_m": .3}]), [])
    assert motion_skill_timing(short, 0., 14.328)["remaining_lease_s"] < 6.


@pytest.mark.parametrize("remaining_s, expected", [(2., "no_feasible_plan"), (7., "shorter_plan_required"), (10., "shorter_plan_required")])
def test_motion_skill_budget_rejection_reports_legal_options(remaining_s, expected):
    import json
    from backend.continuous_navigation import MotionSkillPlan, compile_motion_skills, motion_skill_timing
    path, _ = compile_motion_skills([0., 0., 0.], MotionSkillPlan(components=[{"kind": "cruise", "length_m": 1.48}]), [])
    with pytest.raises(MotionError) as rejected:
        motion_skill_timing(path, 0., 20. - remaining_s, wall_seconds_per_sim_second=2.)
    timing = rejected.value.timing
    assert timing["admission"] == expected
    assert timing["remaining_lease_s"] == pytest.approx(remaining_s)
    assert timing["estimated_motion_s"] > remaining_s and timing["estimate_only"]
    assert timing["scope"] == "rejected_path_timing_estimate"
    assert timing["motion_authorized"] is False
    for kind, limits in timing["shorter_component_options"].items():
        assert limits["minimum_length_m"] == (.15 if kind == "approach" else .2)
        assert limits["minimum_length_m"] <= limits["maximum_length_m"] <= timing["maximum_length_m"]
    if expected == "no_feasible_plan":
        assert timing["shorter_component_options"] == {}
        assert "Select a shorter" not in str(rejected.value)
        assert timing["next_actions"] == ["look", "scan", "wait"]
        assert "Do not request sub-minimum components" in str(rejected.value)
    else:
        assert timing["shorter_component_options"]
        assert set(timing["shorter_component_options"]) == ({"approach"} if remaining_s == 7. else {"cruise", "curve", "approach"})
    json.dumps(timing, allow_nan=False)


def test_motion_skill_throughput_excludes_inference_and_retains_slow_windows():
    from backend.worker import MotionThroughput
    timing = MotionThroughput()
    assert timing.state()["wall_seconds_per_sim_second"] == 2.
    timing.observe(0., 0., True)
    timing.observe(.25, .5, True)
    assert timing.state()["sample_windows"] == 0
    timing.observe(.5, 1., True)
    assert timing.state()["wall_seconds_per_sim_second"] == pytest.approx(2.4)
    timing.observe(.5, 2., False)
    timing.observe(.5, 100., False)
    timing.observe(.5, 200., True)
    timing.observe(1., 200.5, True)
    assert timing.state()["sample_windows"] == 2
    assert timing.state()["wall_seconds_per_sim_second"] == pytest.approx(2.4)
    timing.observe(1.5, 202., True)
    assert timing.state()["wall_seconds_per_sim_second"] == pytest.approx(3.6)
    for index in range(9):
        timing.observe(2. + index * .5, 202.5 + index * .5, True)
    assert timing.state()["sample_windows"] == 8
    assert timing.state()["wall_seconds_per_sim_second"] == 2.


def test_motion_skill_timing_accounts_for_terminal_slowdown_and_wall_clock():
    import math
    from backend.continuous_navigation import MotionSkillPlan, compile_motion_skills, motion_skill_timing
    path, _ = compile_motion_skills([0., 0., 0.], MotionSkillPlan(components=[{"kind": "cruise", "length_m": 1.48}]), [])
    ideal_tail_s = math.log((.5 / .7) / .04) / .7
    estimate = motion_skill_timing(path, 0., 0., wall_seconds_per_sim_second=1.)
    assert estimate["terminal_slowdown_sim_s"] == pytest.approx(ideal_tail_s)
    assert estimate["estimated_motion_s"] >= ideal_tail_s + (1.48 - .5 / .7) / .5
    slower = motion_skill_timing(path, 0., 0., wall_seconds_per_sim_second=1.8)
    assert slower["estimated_motion_s"] > estimate["estimated_motion_s"]
    assert slower["wall_seconds_per_sim_second"] == 1.8
    with pytest.raises(MotionError, match="Composed motion needs an estimated"):
        motion_skill_timing(path, 0., 9.406, wall_seconds_per_sim_second=1.8)
    short, _ = compile_motion_skills([0., 0., 0.], MotionSkillPlan(components=[{"kind": "cruise", "length_m": .3}]), [])
    assert motion_skill_timing(short, 0., 0., wall_seconds_per_sim_second=1.8)["estimated_motion_s"] < slower["estimated_motion_s"]


@pytest.mark.parametrize("components", [
    [{"kind": "cruise", "length_m": 1.48}],
    [{"kind": "approach", "candidate_id": 0}],
    [{"kind": "cruise", "length_m": .4}, {"kind": "curve", "length_m": .8, "angle_rad": .4}],
])
def test_motion_skill_timing_covers_slow_clock_physics_arrival(components):
    import numpy as np
    from types import SimpleNamespace
    from backend.challenges import get_challenge
    from backend.continuous_navigation import MotionSkillPlan, compile_motion_skills, motion_skill_timing, ContinuousNavigation
    sim = BulletSimulation(challenge=get_challenge("park"), width=160, height=120)
    sim.on_tick = None
    ratio = 2.
    clock = lambda: sim.ticks / 240 * ratio
    runtime = NavigationRuntime(clock=clock)
    path, boundaries = compile_motion_skills(sim.odometry, MotionSkillPlan(components=components),
        [{"id": 0, "target_m": [1.4, .3]}])
    started = clock()
    estimate = motion_skill_timing(path, started, started, wall_seconds_per_sim_second=ratio)
    control = ContinuousNavigation(path, clock=clock)
    control.ai_route = True
    control.skill_components = boundaries
    try:
        control.start(sim, runtime)
        runtime.authorized_travel_m = 1.8
        runtime.skill_deadline = started + 20.
        for _ in range(600):
            control.update(sim, runtime, SimpleNamespace(captured_at=clock(), max_frame_age_s=1.), lambda *args: True)
            if not control.active:
                break
            runtime.tick(sim)
            assert not sim.proximity_sensors().collisions
        assert control.status == "arrived", {"status": control.status, "reason": runtime.reason,
            "remaining_m": control.distance_m, "actual_wall_s": clock() - started, "estimate": estimate}
        assert clock() - started < estimate["estimated_motion_s"], {
            "actual_wall_s": clock() - started, "estimate": estimate, "remaining_m": control.distance_m}
        assert np.linalg.norm(sim.odometry[:2] - path[-1]) < .04
        assert control.buffer_stops == 0 and not runtime.buffer
    finally:
        sim.close()


def test_motion_skill_physics_preserves_velocity_across_curve_join():
    import numpy as np
    from types import SimpleNamespace
    from backend.challenges import get_challenge
    from backend.continuous_navigation import MotionSkillPlan, compile_motion_skills, ContinuousNavigation
    sim = BulletSimulation(challenge=get_challenge("park"), width=160, height=120)
    sim.on_tick = None
    clock = lambda: sim.ticks / 240
    runtime = NavigationRuntime(clock=clock)
    plan = MotionSkillPlan(components=[{"kind": "cruise", "length_m": .45},
        {"kind": "curve", "length_m": .8, "angle_rad": .4}, {"kind": "inspect"}])
    path, components = compile_motion_skills(sim.odometry, plan, [])
    control = ContinuousNavigation(path, clock=clock)
    control.ai_route = True
    control.skill_components = components
    join_speeds = []
    try:
        control.start(sim, runtime)
        runtime.authorized_travel_m = 1.4
        for _ in range(600):
            control.update(sim, runtime, SimpleNamespace(captured_at=clock(), max_frame_age_s=1.), lambda *args: True)
            if not control.active:
                break
            previous = runtime.velocity.copy()
            runtime.tick(sim)
            assert np.max(np.abs(runtime.velocity - previous) / np.array([.3, 1.])) <= .05 + 1e-9
            assert not sim.proximity_sensors().collisions
            if .3 < runtime.travel < .7:
                join_speeds.append(runtime.velocity[0])
        assert control.status == "arrived", control.state()
        assert join_speeds and min(join_speeds) > .15
        assert control.buffer_stops == 0 and not runtime.buffer
        assert np.linalg.norm(sim.odometry[:2] - path[-1]) < .05
    finally:
        sim.close()


def test_motion_skill_compiler_blends_geometry_and_binds_observed_approaches():
    import numpy as np
    from backend.continuous_navigation import MotionSkillPlan, compile_motion_skills
    plan = MotionSkillPlan(components=[{"kind": "cruise", "length_m": .4},
        {"kind": "curve", "length_m": .8, "angle_rad": .5}, {"kind": "inspect"}])
    path, boundaries = compile_motion_skills([0., 0., 0.], plan, [])
    np.testing.assert_allclose(path[0], [0., 0.])
    headings = np.unwrap(np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0])))
    assert abs(headings[0]) < .001 and abs(headings[-1] - .5) < .001
    join = boundaries[0]["end_index"]
    assert abs(headings[join] - headings[join - 1]) < .001
    assert np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)) <= .021
    assert boundaries[-1]["distance_m"] == pytest.approx(1.2, abs=.001)
    approach = MotionSkillPlan(components=[{"kind": "approach", "candidate_id": 3}, {"kind": "inspect"}])
    path, _ = compile_motion_skills([0., 0., 0.], approach, [{"id": 3, "target_m": [1., .2]}])
    np.testing.assert_allclose(path[-1], [1., .2])
    with pytest.raises(ValueError, match="current observed"):
        compile_motion_skills([0., 0., 0.], approach, [])
    with pytest.raises(ValueError, match="reorientation"):
        compile_motion_skills([0., 0., 0.], approach, [{"id": 3, "target_m": [0., 1.]}])


@pytest.mark.parametrize("components", [[], [{"kind": "inspect"}],
    [{"kind": "cruise", "length_m": .3}] * 4,
    [{"kind": "cruise", "length_m": float("nan")}],
    [{"kind": "cruise", "length_m": .5, "linear_mps": 10}],
    [{"kind": "inspect"}, {"kind": "cruise", "length_m": .5}],
    [{"kind": "approach", "candidate_id": 0}, {"kind": "cruise", "length_m": .5}],
    [{"kind": "approach", "candidate_id": True}]])
def test_motion_skill_contract_rejects_unbounded_or_ambiguous_plans(components):
    from backend.continuous_navigation import MotionSkillPlan
    with pytest.raises(ValueError):
        MotionSkillPlan(components=components)


def test_motion_skill_compiler_rejects_excess_distance_and_curvature():
    from backend.continuous_navigation import MotionSkillPlan, compile_motion_skills
    for components in ([{"kind": "cruise", "length_m": 1.}, {"kind": "cruise", "length_m": 1.}],
                       [{"kind": "curve", "length_m": .2, "angle_rad": 1.2}]):
        with pytest.raises(ValueError):
            compile_motion_skills([0., 0., 0.], MotionSkillPlan(components=components), [])


def test_ai_route_smoothing_is_task_independent_bounded_and_heading_continuous():
    import numpy as np
    from backend.continuous_navigation import smooth_observed_route
    path = smooth_observed_route([0., 0., 0.], [[1., .2], [2., -.3], [3., 0.]], moving=True)
    np.testing.assert_allclose(path[0], [0., 0.])
    np.testing.assert_allclose(path[-1], [3., 0.])
    assert len(path) > 100 and np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)) < .04
    assert abs(np.arctan2(*(path[1] - path[0])[::-1])) < .02
    assert np.linalg.norm(np.diff(path, axis=0), axis=1).sum() <= 6.
    for invalid in ([], [[0., 0.]], [[7., 0.]], [[float("nan"), 0.]], [[1., 0.], [1., 0.]]):
        with pytest.raises(ValueError):
            smooth_observed_route([0., 0., 0.], invalid)
    with pytest.raises(ValueError, match="sharp"):
        smooth_observed_route([0., 0., 0.], [[0., 1.]], moving=True)


def test_custom_ai_goal_cannot_inherit_unrelated_scenario_success():
    from scripts.evaluate_supervised import summarize_benchmark
    result = {"physics_success": True, "verification_eligible": False,
        "recording_scorecard": {"complete_recording": True, "completion_time_s": 10.}}
    assert summarize_benchmark([result])["successes"] == 0


def test_ai_loop_route_does_not_arrive_at_its_start():
    import numpy as np
    from types import SimpleNamespace
    from backend.continuous_navigation import ContinuousNavigation, smooth_observed_route
    sim = BulletSimulation(width=160, height=120)
    sim.on_tick = None
    runtime = NavigationRuntime()
    control = ContinuousNavigation(smooth_observed_route([0., 0., 0.], [[.8, 0.], [1., .8], [.2, 1.], [0., 0.]]))
    control.ai_route = True
    try:
        control.start(sim, runtime)
        control.update(sim, runtime, SimpleNamespace(captured_at=runtime.clock(), max_frame_age_s=1.), lambda *args: True)
        assert control.active and runtime.buffer and control.distance_m > 2.
        assert runtime.buffer[0][0].linear_mps > .2
        np.testing.assert_equal(control.path[0], control.path[-1])
    finally:
        sim.close()


@pytest.mark.parametrize("fault", [None, "stop", "command", "episode", "expired", "lease", "unknown", "stale", "posture"])
async def test_ai_route_replacement_preserves_motion_and_rejects_revoked_authority(fault):
    import time
    import numpy as np
    from types import SimpleNamespace
    from backend.continuous_navigation import ContinuousNavigation
    control = ContinuousNavigation([[0., 0.], [3., 0.]])
    control.ai_route = True
    control.radius = .3
    cancelled = asyncio.Event()
    sim = SimpleNamespace(cancel=cancelled, run_id="route", epoch=int(fault == "episode"), odometry=np.array([.6, 0., 0.]),
        held=None, robot_footprint=lambda: {"radius_m": .5 if fault == "posture" else .3})
    runtime = SimpleNamespace(status="running", buffer=[object()], travel=.6, velocity=np.array([.4, 0.]),
        expires_at=time.monotonic() + (-1 if fault == "lease" else 2), skill_deadline=time.monotonic() + 20,
        authorized_travel_m=3.15, check_clearance=lambda *args: None)
    ticket = {"run_id": "route", "epoch": 0, "stop_revision": 0, "task_revision": 1,
        "captured_at": time.monotonic() - (16 if fault == "expired" else 1), "position": np.array([0., 0., 0.]),
        "identity": control.identity, "travel": 0.}
    if fault == "stop":
        cancelled.set()
    async def call(operation):
        return operation(sim)
    worker = SimpleNamespace(call=call, continuous=control, navigation=runtime, stop_revision=0,
        task_revision=2 if fault == "command" else 1, spatial_error=None, _publish_navigation=lambda: None,
        spatial_map=SimpleNamespace(public=lambda: {"stale": fault == "stale"}, traversable=lambda *args: True,
            contains_path=lambda *args: fault != "unknown"))
    original_path = control.path.copy()
    if fault is None:
        await SimulationWorker.apply_ai_route(worker, ticket, [[1., 0.], [2., .1], [3., .3]])
        assert control.handoffs == 1 and control.active and runtime.buffer
        np.testing.assert_allclose(control.path[0], sim.odometry[:2])
        assert runtime.authorized_travel_m <= runtime.travel + 6.15
        assert runtime.skill_deadline == ticket["captured_at"] + 20.
    else:
        with pytest.raises((MotionError, ValueError)):
            await SimulationWorker.apply_ai_route(worker, ticket, [[1., 0.], [2., .1], [3., .3]])
        np.testing.assert_equal(control.path, original_path)
        assert control.handoffs == 0
    np.testing.assert_equal(runtime.velocity, [.4, 0.])
    assert runtime.travel == .6


@pytest.mark.parametrize("refresh_succeeds", [False, True])
async def test_ai_route_refreshes_stopped_map_after_model_latency(refresh_succeeds):
    import time
    import numpy as np
    from types import SimpleNamespace
    from backend.continuous_navigation import ContinuousNavigation
    stale = [True]
    calls = []
    sim = SimpleNamespace(cancel=asyncio.Event(), run_id="stopped-route", epoch=0, odometry=np.zeros(3),
        held=None, robot_footprint=lambda: {"radius_m": .3})
    async def call(operation):
        return operation(sim)
    def refresh(force=False):
        calls.append(force)
        stale[0] = not refresh_succeeds
    worker = SimpleNamespace(call=call, continuous=None, navigation=None, stop_revision=0, task_revision=1,
        spatial_error=None, _sample_spatial=refresh, _publish_navigation=lambda: None,
        spatial_map=SimpleNamespace(public=lambda: {"stale": stale[0]}, traversable=lambda *args: True,
            contains_path=lambda *args: True))
    def start(path, **kwargs):
        worker.continuous = ContinuousNavigation(path)
        worker.continuous.ai_route = kwargs["ai_route"]
        worker.navigation = SimpleNamespace(travel=0.)
    worker._start_navigation_path = start
    ticket = {"run_id": sim.run_id, "epoch": 0, "stop_revision": 0, "task_revision": 1,
        "captured_at": time.monotonic() - 5, "position": np.zeros(3), "identity": None, "travel": 0.}
    if refresh_succeeds:
        await SimulationWorker.apply_ai_route(worker, ticket, [[1., 0.], [2., 0.]])
        assert worker.continuous.active and worker.continuous.ai_route
    else:
        with pytest.raises(MotionError, match="fresh"):
            await SimulationWorker.apply_ai_route(worker, ticket, [[1., 0.], [2., 0.]])
        assert worker.continuous is None
    assert calls == [True]


async def test_ai_route_task_update_keeps_session_budget_and_invalidates_old_command():
    import time
    from types import SimpleNamespace
    from backend.agent import AgentController, FoundryConfig, RunInstruction
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"))
    worker = SimpleNamespace(latest={"run_id": "live"}, epoch=0, task_revision=1)
    controller.worker = worker
    controller.active = True
    controller.command_revision = 1
    controller.session_deadline = time.monotonic() + 100
    deadline = controller.session_deadline
    controller.state.update(execution_mode="luna_continuous", ai_generated_routes=True, session_id="same-session",
        goal="First task", turns=4, max_turns=20, input_tokens=123, output_tokens=45)
    for goal in ("Drive a figure eight around the two observed objects", "Follow the selected person"):
        await controller.redirect(worker, RunInstruction(run_id="live", episode_epoch=0, session_id="same-session", message=goal))
        assert controller.state["goal"] == goal and controller.state["session_id"] == "same-session"
        assert controller.session_deadline == deadline and controller.active
        assert controller.state["turns"] == 4 and controller.state["input_tokens"] == 123
        assert worker.task_revision == controller.command_revision and controller.command_changed.is_set()
    assert controller.command_revision == 3


@pytest.mark.parametrize("stop_during_update", [False, True])
async def test_generic_ai_route_live_update_is_recorded(stop_during_update):
    import hashlib
    import json
    import sys
    import time
    from backend.agent import AgentController, AgentStart, FoundryConfig, RunInstruction
    from backend.challenges import get_challenge
    from backend.recording import RunRecorder
    from scripts.record_scenario_tests import PhysicsTestBatch
    from tests.test_agent import model_response
    pending, cancelled, release, updated, done = (asyncio.Event() for _ in range(5))
    goals = []
    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            goals.append(goal)
            payload = next(json.loads(part["text"]) for part in inputs[-1]["content"]
                if part["type"] == "input_text" and part["text"].startswith('{"command_revision"'))
            observation = next(json.loads(part["text"]) for part in inputs[-1]["content"]
                if part["type"] == "input_text" and '"odometry_m_rad"' in part["text"])
            pose = observation["odometry_m_rad"]
            if len(goals) == 2:
                pending.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    await release.wait()
            elif len(goals) >= 4:
                updated.set()
                await asyncio.Event().wait()
            decision = {"action": "route", "waypoints_m": [[pose[0] + 1., pose[1]],
                [pose[0] + 2.8, pose[1] + (.2 if goal == "Curve left along clear floor" else 0.)]],
                "reason": "Scripted generic observed route for live command revision validation"}
            assert "observed_free_floor_xy_m" in payload and "challenge" not in observation
            return model_response("guide_route", json.dumps(decision), call_id=f"generic-{len(goals)}")
        async def close(self):
            pass
    challenge = get_challenge("furniture_circuit")
    challenge.objects = [item for item in challenge.objects if item["name"].endswith("wall") or item["name"] == "floor"]
    worker = SimulationWorker(challenge=challenge, pace=True, rendering="enhanced")
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor())
    batch = PhysicsTestBatch(label="generic-ai-routes-v1-scripted")
    case_id = f"generic-live-update-stop-{stop_during_update}"
    batch.manifest.update(mode="scripted_continuous", note="Scripted generic waypoint proposals on an explicit open-floor fixture; actual depth, physics and live command handling. Not AI task recognition.")
    batch.manifest["cases"].append({"case_id": case_id, "challenge": challenge.id, "environment": challenge.environment,
        "challenge_sha256": hashlib.sha256(json.dumps(challenge.model_dump(), sort_keys=True).encode()).hexdigest()})
    batch.write()
    recorder = RunRecorder(batch.directory / case_id / "recording", context=lambda: controller.state)
    async def flush():
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), .5)
            except TimeoutError:
                await asyncio.to_thread(recorder.flush)
    writer = asyncio.create_task(flush())
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.call(lambda sim: setattr(worker, "recorder", recorder))
        controller.start(worker, AgentStart(run_id=worker.latest["run_id"], episode_epoch=0, execution_mode="luna_continuous",
            ai_generated_routes=True, goal="Drive along clear floor", max_turns=8))
        await asyncio.wait_for(pending.wait(), 35)
        deadline = time.monotonic() + 8
        while worker.navigation.travel < .7:
            assert worker.continuous.active, worker.continuous.state()
            assert time.monotonic() < deadline
            await asyncio.sleep(.05)
        session, route_identity = controller.state["session_id"], worker.continuous.identity
        budget = controller.session_deadline
        revision = worker.stop_revision
        await controller.redirect(worker, RunInstruction(run_id=worker.latest["run_id"], episode_epoch=0,
            session_id=session, message="Curve left along clear floor"))
        await asyncio.wait_for(cancelled.wait(), 3)
        assert len(goals) == 2 and worker.continuous.active and worker.navigation.velocity[0] > .2
        assert worker.stop_revision == revision and controller.session_deadline == budget
        assert controller.state["session_id"] == session
        if stop_during_update:
            controller.interrupt("Stop during task update")
            release.set()
            await controller.halt()
            assert len(goals) == 2
        else:
            release.set()
            await asyncio.wait_for(updated.wait(), 12)
            assert goals[2:] == ["Curve left along clear floor"] * 2
            assert worker.continuous.identity == route_identity and worker.continuous.handoffs == 1
            assert worker.continuous.active and worker.navigation.velocity[0] > .2
            await controller.halt()
        assert not worker.navigation.buffer and not worker.navigation.velocity.any()
        ticks = await worker.call(lambda sim: sim.ticks)
        await asyncio.sleep(.15)
        assert await worker.call(lambda sim: sim.ticks) == ticks
    finally:
        failure = sys.exc_info()[0]
        release.set()
        await controller.halt()
        worker.stop()
        done.set()
        await writer
        def detach(sim):
            recorder.capture(worker)
            worker.recorder = None
            return worker.latest["challenge"], next(reversed(worker.camera_frames.values()))
        status, image = await worker.call(detach)
        score = await asyncio.to_thread(recorder.finish, {"real_model": False, "evidence": "scripted_test",
            "test_passed": failure is None, "setup": batch.manifest["note"]})
        report = {"case_id": case_id, "challenge": challenge.id, "environment": challenge.environment,
            "challenge_status": status, "evidence": "scripted_test", "physics_success": False,
            "test_passed": failure is None, "error": failure.__name__ if failure else None,
            "termination_reason": "test_failed" if failure else "test_finished", "rendering": "enhanced",
            "evaluation_elapsed_s": score["elapsed_wall_s"], "recording_scorecard": score}
        (batch.directory / case_id / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (batch.directory / case_id / "trace.json").write_text(json.dumps(controller.trace(), indent=2), encoding="utf-8")
        (batch.directory / case_id / "terminal.png").write_bytes(image)
        batch.results.append(report)
        batch.finish()
        await worker.close()


def test_exploration_paths_and_review_authority_are_bounded():
    import math
    import numpy as np
    from backend.continuous_navigation import ObservedExploration
    intent = ObservedExploration(.3, clock=lambda: 10.)
    paths = list(intent.paths([0., 0., 0.]))
    assert len(paths) == 5
    for path in paths:
        assert path.shape == (13, 2) and np.isfinite(path).all()
        assert np.linalg.norm(np.diff(path, axis=0), axis=1).sum() == pytest.approx(1.4)
        assert np.all(np.diff(path[:, 0]) > 0)
        headings = np.arctan2(np.diff(path[:, 1]), np.diff(path[:, 0]))
        assert np.max(np.abs(np.diff(headings))) <= .06
    assert not list(intent.paths([0., 0., math.pi]))
    intent.renew(.2, captured_at=14., captured_travel=1., now=20., travel=3.)
    assert intent.expires_at == 34. and intent.travel_limit == 7. and intent.revision == 1
    for now, travel in ((34., 3.), (21., 7.), (30., 3.)):
        with pytest.raises(ValueError, match="authority"):
            intent.renew(.4, captured_at=14., captured_travel=1., now=now, travel=travel)
    intent.ending = True
    with pytest.raises(ValueError, match="authority"):
        intent.renew(.4, captured_at=20., captured_travel=3., now=21., travel=3.)
    assert intent.heading == .2 and intent.revision == 1


@pytest.mark.parametrize("fault", [None, "stale", "unknown", "stop", "revision", "deadline", "lease", "distance", "ending", "posture"])
def test_exploration_worker_renewal_preserves_velocity_and_stops_at_authority_limits(fault):
    import time
    import numpy as np
    from types import SimpleNamespace
    from backend.continuous_navigation import ContinuousNavigation, ObservedExploration
    sim = BulletSimulation(width=160, height=120)
    sim.on_tick = None
    runtime = NavigationRuntime()
    control = ContinuousNavigation([[0., 0.], [.8, 0.]])
    try:
        control.start(sim, runtime)
        runtime.status = "running"
        runtime.buffer.append(object())
        runtime.expires_at = time.monotonic() + 2
        runtime.velocity[:] = [.4, .1]
        runtime.travel = .4
        control.exploration = ObservedExploration(0.)
        control.exploration_stop_revision = 0
        control.exploration_run_id = sim.run_id
        control.exploration_epoch = sim.epoch
        control.radius = sim.robot_footprint()["radius_m"] - (.1 if fault == "posture" else 0.)
        control.start_position = sim.odometry[:2].copy()
        if fault == "deadline":
            control.exploration.expires_at = time.monotonic() - 1
        if fault == "distance":
            control.exploration.travel_limit = runtime.travel
        if fault == "lease":
            runtime.expires_at = time.monotonic() - 1
        if fault == "stop":
            sim.stop()
        control.exploration.ending = fault == "ending"
        deadline = runtime.skill_deadline
        mapped = SimpleNamespace(public=lambda: {"stale": fault == "stale"}, traversable=lambda *args: True,
            contains_path=lambda *args: fault != "unknown")
        worker = SimpleNamespace(sim=sim, continuous=control, navigation=runtime, spatial_map=mapped,
            spatial_error=None, stop_revision=int(fault == "revision"))
        SimulationWorker._continue_exploration(worker)
        assert runtime.skill_deadline == deadline
        if fault is None:
            assert control.handoffs == 1 and control.active and runtime.buffer
            np.testing.assert_equal(runtime.velocity, [.4, .1])
            assert runtime.travel == .4 and runtime.authorized_travel_m == pytest.approx(1.95)
        else:
            assert control.handoffs == 0 and runtime.authorized_travel_m is None
            if fault in {"stop", "revision", "deadline", "distance", "posture"}:
                assert not control.active and not runtime.buffer and not runtime.velocity.any()
    finally:
        sim.close()


@pytest.mark.parametrize("fault", [None, "stale", "moved", "missing"])
async def test_continuous_selected_sensor_survives_cache_eviction_without_bypassing_freshness(fault):
    import time
    import numpy as np
    from types import SimpleNamespace
    from backend.continuous_navigation import ContinuousTarget
    sensor = SimpleNamespace(run_id="selected", episode_epoch=0, captured_at=time.monotonic() - (46 if fault == "stale" else 2),
        odometry_m_rad=[0., 0., 0.])
    sim = SimpleNamespace(run_id=sensor.run_id, epoch=0, odometry=np.array([.06 if fault == "moved" else 0., 0., 0.]),
        robot_footprint=lambda: {"radius_m": .3})
    started = []
    async def call(operation):
        return operation(sim)
    mapped = SimpleNamespace(sequence=9, captured_at=time.monotonic(), observation=SimpleNamespace(head_rad=[0., .45]),
        plan=lambda origin, target, radius: [list(origin), target], traversable=lambda *args: True,
        contains_path=lambda *args: True)
    worker = SimpleNamespace(call=call, stop_revision=0, spatial_frames={}, spatial_enabled=True, spatial_map=mapped,
        spatial_error=None, _sample_spatial=lambda force=False: None, _continuous_guard=lambda *args: None,
        _start_navigation_path=lambda path, *args, **kwargs: started.append(path))
    request = ContinuousTarget(run_id=sensor.run_id, episode_epoch=0, spatial_sequence=1, pixel=[.5, .5])
    if fault is None:
        await SimulationWorker.start_continuous(worker, request, selected_sensor=sensor, selected_target=[1., 0.])
        assert started == [[[0., 0.], [1., 0.]]]
    else:
        with pytest.raises(MotionError):
            await SimulationWorker.start_continuous(worker, request,
                selected_sensor=None if fault == "missing" else sensor, selected_target=[1., 0.])
        assert not started


@pytest.mark.parametrize("fault", [None, "stop", "revision", "episode", "route", "late", "lease", "stale", "sharp", "ending"])
async def test_exploration_reviews_revalidate_authority_without_resetting_motion(fault):
    import time
    import numpy as np
    from types import SimpleNamespace
    from backend.continuous_navigation import ContinuousNavigation, ObservedExploration
    control = ContinuousNavigation([[0., 0.], [1., 0.]])
    control.exploration = ObservedExploration(0.)
    control.revision = 9
    cancelled = asyncio.Event()
    sim = SimpleNamespace(cancel=cancelled, run_id="rolling", epoch=0, odometry=np.array([1., 0., 0.]))
    runtime = SimpleNamespace(status="running", buffer=[object()], expires_at=time.monotonic() + 2.,
        skill_deadline=control.exploration.expires_at, travel=1., velocity=np.array([.4, .1]))
    proposal = {"run_id": sim.run_id, "epoch": 0, "stop_revision": 0, "identity": control.identity,
        "revision": 0, "intent_revision": 0, "position": np.array([0., 0., 0.]),
        "captured_at": time.monotonic() - 3., "captured_travel_m": .1,
        "candidates": [{"id": 1, "target_m": [0., 1.] if fault == "sharp" else [1., .1]}]}
    async def call(operation):
        return operation(sim)
    worker = SimpleNamespace(call=call, continuous=control, navigation=runtime, stop_revision=int(fault == "revision"),
        spatial_map=SimpleNamespace(public=lambda: {"stale": fault == "stale"}), spatial_error=None,
        _publish_navigation=lambda: None)
    if fault == "stop":
        cancelled.set()
    if fault == "episode":
        sim.epoch = 1
    if fault == "route":
        control.identity = "replacement"
    if fault == "late":
        proposal["captured_at"] -= 20.
    if fault == "lease":
        runtime.expires_at = time.monotonic() - 1.
    if fault in {None, "ending"}:
        await SimulationWorker.review_exploration(worker, proposal, 1, ending=fault == "ending")
        assert control.exploration.revision == 1 and control.active
        assert control.exploration.ending == (fault == "ending")
        if fault is None:
            assert control.exploration.expires_at == pytest.approx(proposal["captured_at"] + 20.)
            assert control.exploration.travel_limit == pytest.approx(6.1)
    else:
        with pytest.raises(ValueError):
            await SimulationWorker.review_exploration(worker, proposal, 1)
        assert control.exploration.revision == 0
    np.testing.assert_equal(runtime.velocity, [.4, .1])
    assert runtime.travel == 1. and runtime.buffer and control.revision == 9


@pytest.mark.parametrize("fault", [None, "stop", "stale", "lease", "inspect"])
async def test_rolling_exploration_during_slow_supervisor_is_recorded(fault, monkeypatch):
    import hashlib
    import json
    import sys
    import time
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.challenges import get_challenge
    from backend.recording import RunRecorder
    from scripts.record_scenario_tests import PhysicsTestBatch
    from tests.test_agent import model_response
    entered, release, reviewed, done = (asyncio.Event() for _ in range(4))
    received = []
    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            payload = next(json.loads(part["text"]) for part in inputs[-1]["content"]
                if part["type"] == "input_text" and part["text"].startswith('{"reachable_floor_candidates"'))
            received.append(payload)
            if len(received) == 1:
                candidates = [item for item in payload["reachable_floor_candidates"] if abs(item["bearing_from_base_rad"]) < .15]
                choice = max(candidates, key=lambda item: item["distance_m"])
                guide = {"action": "explore", "candidate_id": choice["id"], "reason": "Scripted observed forward direction for rolling control test"}
            elif len(received) == 3 and fault == "inspect":
                assert not payload["planning_while_moving"] and not payload["rolling_exploration"]
                assert not worker.continuous.active and not worker.navigation.buffer
                reviewed.set()
                await asyncio.Event().wait()
            else:
                if len(received) == 2:
                    assert payload["rolling_exploration"] and payload["planning_while_moving"]
                    entered.set()
                else:
                    reviewed.set()
                try:
                    if len(received) == 2:
                        await release.wait()
                    else:
                        await asyncio.Event().wait()
                except asyncio.CancelledError:
                    pass
                guide = {"action": "look" if fault == "inspect" else "continue",
                    "reason": "Scripted delayed review of the observed exploration direction"}
            return model_response("guide_continuous", json.dumps(guide), call_id=f"rolling-{len(received)}")
        async def close(self):
            pass
    challenge = get_challenge("furniture_circuit")
    challenge.objects = [item for item in challenge.objects if item["name"].endswith("wall") or item["name"] == "floor"]
    worker = SimulationWorker(challenge=challenge, pace=True, rendering="enhanced")
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Supervisor())
    batch = PhysicsTestBatch(label="rolling-exploration-v1-scripted")
    case_id = f"rolling-review-{fault or 'slow'}"
    batch.manifest.update(mode="scripted_continuous", note="Open-floor fixture with furniture removed; scripted observed-candidate decisions, real enhanced depth and physics. Not a furniture circuit or model autonomy test.")
    batch.manifest["cases"].append({"case_id": case_id, "challenge": challenge.id, "environment": challenge.environment,
        "challenge_sha256": hashlib.sha256(json.dumps(challenge.model_dump(), sort_keys=True).encode()).hexdigest()})
    batch.write()
    recorder = RunRecorder(batch.directory / case_id / "recording", context=lambda: controller.state)
    async def flush():
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), .5)
            except TimeoutError:
                await asyncio.to_thread(recorder.flush)
    writer = asyncio.create_task(flush())
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.call(lambda sim: setattr(worker, "recorder", recorder))
        controller.start(worker, AgentStart(run_id=worker.latest["run_id"], episode_epoch=0, execution_mode="luna_continuous",
            goal="Explore the observed open floor while reviewing progress.", max_turns=4))
        await asyncio.wait_for(entered.wait(), 35)
        if fault == "stop":
            worker.stop()
        elif fault == "stale":
            monkeypatch.setattr(worker, "_sample_spatial", lambda force=False: None)
        elif fault == "lease":
            await worker.call(lambda sim: setattr(worker.continuous.exploration, "expires_at", time.monotonic() + .3))
        deadline = time.monotonic() + 15
        if fault in {None, "inspect"}:
            while worker.navigation.travel < 3.:
                assert worker.continuous.active, worker.continuous.state()
                assert time.monotonic() < deadline, worker.continuous.state()
                await asyncio.sleep(.05)
            assert worker.continuous.handoffs > 3 and worker.continuous.buffer_stops == 0
            assert worker.navigation.velocity[0] > .35
            release.set()
            await asyncio.wait_for(reviewed.wait(), 15)
            assert worker.continuous.exploration.revision == (0 if fault == "inspect" else 1)
            assert worker.continuous.active == (fault is None)
        else:
            while worker.continuous.active:
                assert time.monotonic() < deadline
                await asyncio.sleep(.05)
            assert not worker.navigation.buffer and not worker.navigation.velocity.any()
            if fault == "stale":
                assert "SPATIAL_STALE" in worker.continuous.reason
            if fault == "lease":
                assert "EXPLORATION_REVIEW_EXPIRED" in worker.continuous.reason
        await controller.halt("Recorded rolling-exploration safety check")
        release.set()
        ticks = await worker.call(lambda sim: sim.ticks)
        await asyncio.sleep(.15)
        assert await worker.call(lambda sim: sim.ticks) == ticks
        assert not worker.navigation.buffer and not worker.navigation.velocity.any()
    finally:
        failure = sys.exc_info()[0]
        await controller.halt()
        release.set()
        worker.stop()
        done.set()
        await writer
        def detach(sim):
            recorder.capture(worker)
            worker.recorder = None
            return worker.latest["challenge"], next(reversed(worker.camera_frames.values()))
        status, image = await worker.call(detach)
        score = await asyncio.to_thread(recorder.finish, {"real_model": False, "evidence": "scripted_test",
            "test_passed": failure is None, "fault": fault, "setup": batch.manifest["note"]})
        report = {"case_id": case_id, "challenge": challenge.id, "environment": challenge.environment,
            "challenge_status": status, "evidence": "scripted_test", "physics_success": False,
            "test_passed": failure is None, "error": failure.__name__ if failure else None,
            "termination_reason": "test_failed" if failure else "test_finished", "rendering": "enhanced",
            "evaluation_elapsed_s": score["elapsed_wall_s"], "recording_scorecard": score}
        (batch.directory / case_id / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (batch.directory / case_id / "trace.json").write_text(json.dumps(controller.trace(), indent=2), encoding="utf-8")
        (batch.directory / case_id / "terminal.png").write_bytes(image)
        batch.results.append(report)
        batch.finish()
        await worker.close()


def test_continuous_circuit_path_replacement_preserves_motion_and_rejects_invalid_paths():
    import numpy as np
    from backend.continuous_navigation import ContinuousNavigation
    controller = ContinuousNavigation([[0., 0.], [1., 0.]])
    identity, started = controller.identity, controller.started
    controller.updates = 15
    controller.last_update = 4.
    controller.progress_at = 3.
    controller.continue_path([[.3, 0.], [.8, .05], [1.4, .15]])
    assert controller.active and controller.identity == identity and controller.started == started
    assert controller.updates == 15 and controller.last_update == 4. and controller.progress_at == 3.
    assert controller.handoffs == 1 and controller.revision == 1 and controller.buffer_stops == 0
    valid = controller.path.copy()
    for invalid in ([[0., 0.]], [[0., 0.], [float("nan"), 0.]], [[0., 0.], [2., 0.]]):
        with pytest.raises(ValueError, match="bounded"):
            controller.continue_path(invalid)
        np.testing.assert_equal(controller.path, valid)
    controller.status = "cancelled"
    with pytest.raises(ValueError, match="active"):
        controller.continue_path([[.3, 0.], [1., 0.]])


@pytest.mark.parametrize("fault", [None, "stale", "unknown", "stop", "revision", "deadline", "lease", "distance", "final"])
def test_circuit_renewal_preserves_velocity_and_bounds_authority(fault):
    import time
    import numpy as np
    from types import SimpleNamespace
    from backend.continuous_navigation import ContinuousNavigation
    sim = BulletSimulation(width=160, height=120)
    sim.on_tick = None
    now = time.monotonic()
    runtime = NavigationRuntime()
    control = ContinuousNavigation([[0., 0.], [.8, 0.]])
    try:
        control.start(sim, runtime)
        runtime.status = "running"
        runtime.buffer.append(object())
        runtime.expires_at = now + 2
        runtime.velocity[:] = [.35, .1]
        runtime.travel = .4
        deadline = runtime.skill_deadline
        orbit = SimpleNamespace(run_id=sim.run_id, episode_epoch=sim.epoch, started=now-1, swept=.3, radius=1.7,
            label="table", direction="clockwise", center=np.array([1.7, 0.]),
            path=lambda pose: ([[*pose[:2]], [.6, .03], [1.1, .12]], False))
        control.observed_orbit = orbit
        control.orbit_stop_revision = 0
        control.orbit_final_arc = fault == "final"
        control.orbit_last_attempt = -np.inf
        control.orbit_travel_limit = .5 if fault == "distance" else 12.
        control.orbit_travel_start = 0.
        control.radius = sim.robot_footprint()["radius_m"]
        control.start_position = sim.odometry[:2].copy()
        mapped = SimpleNamespace(public=lambda: {"stale": fault == "stale"}, traversable=lambda *args: True,
            contains_path=lambda *args: fault != "unknown")
        worker = SimpleNamespace(sim=sim, continuous=control, navigation=runtime, spatial_map=mapped,
            spatial_error=None, stop_revision=1 if fault == "revision" else 0)
        if fault == "stop":
            sim.stop()
        if fault == "deadline":
            orbit.started = now - 181
        if fault == "lease":
            runtime.expires_at = now - 1
        SimulationWorker._continue_observed_orbit(worker)
        if fault is None:
            assert control.handoffs == 1 and control.active and runtime.buffer
            np.testing.assert_equal(runtime.velocity, [.35, .1])
            assert runtime.travel == .4 and runtime.authorized_travel_m <= runtime.travel + 1.8 + .15
            assert runtime.skill_deadline <= orbit.started + 180
        else:
            assert control.handoffs == 0 and runtime.authorized_travel_m is None
            assert runtime.skill_deadline == deadline
            if fault in {"stop", "revision", "deadline"}:
                assert not control.active and not runtime.buffer and not runtime.velocity.any()
    finally:
        sim.close()


def test_circuit_lookahead_does_not_chase_close_radial_waypoint_after_recovery():
    import numpy as np
    from types import SimpleNamespace
    from backend.continuous_navigation import ContinuousNavigation
    sim = BulletSimulation(width=160, height=120)
    sim.on_tick = None
    runtime = NavigationRuntime()
    control = ContinuousNavigation([[0., 0.], [.05, .08], [.15, .09], [.3, .08], [.6, .02], [1., -.1]])
    control.circuit_following = True
    try:
        control.start(sim, runtime)
        control.update(sim, runtime, SimpleNamespace(captured_at=runtime.clock(), max_frame_age_s=1.), lambda *args: True)
        segment = runtime.buffer[0][0]
        assert control.index >= 4
        assert segment.linear_mps > .4
        assert abs(segment.angular_radps) < .2
        np.testing.assert_equal(runtime.velocity, [0., 0.])
    finally:
        sim.close()


def test_observed_orbit_uses_depth_and_preserves_direction_and_local_budget():
    from backend.continuous_navigation import ObservedOrbit
    from backend.contracts import SpatialObservation
    from backend.spatial import calibration
    import math
    import numpy as np
    sensor = SpatialObservation(run_id="orbit-test", episode_epoch=0, sequence=1, captured_at=0., simulated_time_s=0.,
        calibration=calibration(160, 120), head_rad=[0., .1], odometry_m_rad=[0., 0., 0.], depth_m=[2.] * (160 * 120))
    for direction, sign in (("clockwise", -1), ("counterclockwise", 1)):
        orbit = ObservedOrbit(sensor, [.4, .3, .6, .6], "table", direction, .4)
        assert np.isfinite(orbit.center).all() and 1.2 <= orbit.radius <= 2.1
        start = orbit.center + [orbit.radius, 0.]
        path, complete = orbit.path([*start, 0.])
        assert not complete and (path[-1][1] - orbit.center[1]) * sign > 0
        assert np.linalg.norm(np.diff(path, axis=0), axis=1).sum() < 1.8
        for angle in np.linspace(.1, 2 * math.pi + .01, 80):
            pose = orbit.center + orbit.radius * np.array([math.cos(sign * angle), math.sin(sign * angle)])
            _, complete = orbit.path([*pose, 0.])
        assert complete
    with pytest.raises(ValueError, match="insufficient"):
        ObservedOrbit(sensor.model_copy(update={"depth_m": [None] * (160 * 120)}), [.4, .3, .6, .6], "table", "clockwise", .4)


async def test_circle_command_stop_discards_pending_arc():
    from types import SimpleNamespace
    from backend.continuous_supervisor import circle_observed_object, GuideContinuous
    released = asyncio.Event()
    stopped = False
    calls = []
    class Controller:
        def _check_live(self, worker, settings):
            if stopped:
                raise asyncio.CancelledError
        def _trace(self, *args):
            calls.append(args)
    class Worker:
        continuous = SimpleNamespace(active=True)
        async def locate_observed_orbit(self, *args):
            return object()
        async def feedback(self):
            return SimpleNamespace(seq=1), b""
        async def execute(self, command, **kwargs):
            return SimpleNamespace(status="ok")
        async def start_observed_orbit_step(self, *args):
            released.set()
            return {"complete": False}
    task = asyncio.create_task(circle_observed_object(Controller(), Worker(), SimpleNamespace(run_id="test", episode_epoch=0),
        None, GuideContinuous(action="circle", object_label="table", object_bounds=[.2, .2, .6, .6], reason="Visible flat tabletop and legs"), 0))
    await asyncio.wait_for(released.wait(), 1)
    stopped = True
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 1)
    assert len(calls) == 1


async def test_circle_unknown_floor_retries_are_bounded_without_replacing_target(monkeypatch):
    from types import SimpleNamespace
    from backend.continuous_supervisor import circle_observed_object, GuideContinuous
    import math
    scans = []
    targets = []
    rotations = []
    orbit = SimpleNamespace(path=lambda pose: ([[0., 0.], [0., 1.]], False))
    async def rotate(*args):
        rotations.append(args[-1])
    monkeypatch.setattr("backend.continuous_supervisor.rotate", rotate)
    class Controller:
        def _check_live(self, *args):
            pass
        def _trace(self, *args):
            pass
    class Worker:
        async def locate_observed_orbit(self, *args):
            return orbit
        async def feedback(self):
            return SimpleNamespace(seq=1, odometry_m_rad=[0., 0., 0.]), b""
        async def execute(self, *args, **kwargs):
            return SimpleNamespace(status="ok")
        async def start_observed_orbit_step(self, value, revision):
            targets.append(value)
            raise MotionError("OBSERVED_PATH_BLOCKED", "Missing floor")
        async def scan_continuous(self, request):
            scans.append(request)
    result = await circle_observed_object(Controller(), Worker(), SimpleNamespace(run_id="test", episode_epoch=0), None,
        GuideContinuous(action="circle", object_label="table", object_bounds=[.2, .2, .6, .6], reason="Visible tabletop and supporting legs"), 0)
    assert result["status"] == "blocked" and len(scans) == 2 and targets == [orbit] * 3
    assert rotations == pytest.approx([math.pi / 2] * 2)


@pytest.mark.parametrize("fault", ["start", "moving", "persistent", "other", "stop"])
async def test_circle_stale_recovery_preserves_orbit_and_is_bounded(monkeypatch, fault):
    import time
    from types import SimpleNamespace
    from backend import continuous_supervisor
    from backend.continuous_supervisor import circle_observed_object, GuideContinuous
    targets, events = [], []
    orbit = SimpleNamespace(center=[2., 0.], radius=1.7, swept=.6, started=time.monotonic())
    original = vars(orbit).copy()
    stopped = False
    async def pause(duration):
        nonlocal stopped
        assert duration == .1
        if fault == "stop":
            stopped = True
    async def settled(*args):
        pass
    monkeypatch.setattr(continuous_supervisor.asyncio, "sleep", pause)
    monkeypatch.setattr(continuous_supervisor, "wait_stationary", settled)
    class Controller:
        def _check_live(self, *args):
            if stopped:
                raise asyncio.CancelledError
        def _trace(self, kind, title, payload):
            events.append((title, payload))
    class Worker:
        continuous = SimpleNamespace(active=False, status="blocked", reason="SPATIAL_STALE: Fresh depth is required")
        async def locate_observed_orbit(self, *args):
            assert not targets
            return orbit
        async def feedback(self):
            return SimpleNamespace(seq=1), b""
        async def execute(self, *args, **kwargs):
            return SimpleNamespace(status="ok")
        async def start_observed_orbit_step(self, value, revision):
            targets.append(value)
            assert value is orbit and vars(value) == original and revision == 7
            if fault == "other":
                raise MotionError("NO_PROGRESS", "No progress")
            if fault == "persistent" or (len(targets) == 1 and fault != "moving"):
                raise MotionError("SPATIAL_STALE", "Delayed depth")
            return {"complete": len(targets) > 1, "swept_degrees": 35.}
    action = circle_observed_object(Controller(), Worker(), SimpleNamespace(run_id="test", episode_epoch=0), None,
        GuideContinuous(action="circle", object_label="table", object_bounds=[.2, .2, .6, .6],
            reason="Scripted table selection for recovery test"), 7)
    if fault == "stop":
        with pytest.raises(asyncio.CancelledError):
            await action
        assert len(targets) == 1
    else:
        result = await action
        assert result["status"] == ("arrived" if fault in {"start", "moving"} else "blocked")
        assert len(targets) == {"start": 2, "moving": 2, "persistent": 3, "other": 1}[fault]
    recoveries = [payload for title, payload in events if title == "Circuit sensing recovery"]
    assert len(recoveries) == {"persistent": 2, "other": 0}.get(fault, 1)
    assert vars(orbit) == original


@pytest.mark.parametrize("stop_after_first_arc", [True, False])
@pytest.mark.parametrize("stale_interruption", [False, True])
async def test_circle_depth_grounded_lap_and_stop(stop_after_first_arc, stale_interruption, monkeypatch):
    import time
    import sys
    import json
    import hashlib
    from types import SimpleNamespace
    from backend.challenges import get_challenge
    from backend.continuous_navigation import ContinuousScan
    from backend.continuous_supervisor import circle_observed_object, GuideContinuous
    from backend.recording import RunRecorder
    from scripts.record_scenario_tests import PhysicsTestBatch
    events = []
    interrupted = False
    withheld = False
    recovered = False
    stop_requested_at = None
    worker = SimulationWorker(challenge=get_challenge("furniture_circuit"), pace=True, rendering="enhanced")
    batch = PhysicsTestBatch(label="rolling-circuit-v1-scripted")
    recorder = None
    writer = None
    done = asyncio.Event()
    original_sample = worker._sample_spatial
    def sample(force=False):
        nonlocal withheld, recovered
        if stale_interruption and not recovered and worker.continuous and worker.continuous.active and worker.navigation.travel > .2:
            withheld = True
            return
        if withheld:
            recovered = True
        return original_sample(force=force)
    monkeypatch.setattr(worker, "_sample_spatial", sample)
    class Controller:
        def _check_live(self, *args):
            if interrupted:
                raise asyncio.CancelledError
        def _trace(self, kind, title, payload):
            nonlocal interrupted, stop_requested_at
            events.append((title, payload))
            if title == "Circuit sensing recovery":
                assert not worker.navigation.buffer and not worker.navigation.velocity.any()
            if stop_after_first_arc and title == "Observed furniture circuit" and payload["swept_degrees"] > 20:
                interrupted = True
                stop_requested_at = time.monotonic()
                worker.stop()
    try:
        await asyncio.wrap_future(worker.ready)
        case_id = f"circle-stop-{stop_after_first_arc}-stale-{stale_interruption}"
        challenge = worker.challenge
        batch.manifest["cases"].append({"case_id": case_id, "challenge": challenge.id, "environment": challenge.environment,
            "challenge_sha256": hashlib.sha256(json.dumps(challenge.model_dump(), sort_keys=True).encode()).hexdigest()})
        batch.manifest["mode"] = "scripted_continuous"
        batch.manifest["note"] = "Scripted image bounding box; actual depth, observed map, controller and physics. No model inference or hidden geometry steering."
        batch.write()
        recorder = RunRecorder(batch.directory / case_id / "recording", context=lambda: {"phase": "scripted_test"})
        await worker.call(lambda sim: setattr(worker, "recorder", recorder))
        async def flush():
            while not done.is_set():
                try:
                    await asyncio.wait_for(done.wait(), .5)
                except TimeoutError:
                    await asyncio.to_thread(recorder.flush)
        writer = asyncio.create_task(flush())
        settings = SimpleNamespace(run_id=worker.latest["run_id"], episode_epoch=0)
        await worker.scan_continuous(ContinuousScan(**vars(settings), compact_arms=True))
        sensor, _, _, _ = await worker.continuous_candidates()
        guide = GuideContinuous(action="circle", object_label="table", object_bounds=[.32, 0., .68, .58],
            reason="Scripted current-image box for controller integration, not model recognition")
        action = circle_observed_object(Controller(), worker, settings, sensor, guide, worker.stop_revision)
        if stop_after_first_arc:
            with pytest.raises(asyncio.CancelledError):
                result = await asyncio.wait_for(action, 100)
                pytest.fail(f"Circuit ended before the first arc completed: {result}; events={events}")
            assert interrupted and worker.latest["challenge"]["status"] != "completed"
        else:
            result = await asyncio.wait_for(action, 190)
            assert result["status"] == "arrived", {"result": result, "events": events}
            assert worker.latest["challenge"]["status"] == "completed", worker.latest["challenge"]
            assert any(payload.get("handoffs", 0) > 2 for title, payload in events if title == "Observed furniture circuit")
        assert any(title == "Circuit floor rescan" for title, _ in events)
        if stale_interruption:
            assert withheld and recovered
            assert any(title == "Circuit sensing recovery" for title, _ in events)
        progress = [payload for title, payload in events if title == "Observed furniture circuit"]
        assert all(item["observed_center_m"] == progress[0]["observed_center_m"] and item["radius_m"] == progress[0]["radius_m"]
            for item in progress)
        assert all(after["swept_degrees"] >= before["swept_degrees"] for before, after in zip(progress, progress[1:]))
        ticks = await worker.call(lambda sim: sim.ticks)
        stop_acknowledged_s = time.monotonic() - stop_requested_at if stop_requested_at is not None else None
        await asyncio.sleep(.15)
        assert await worker.call(lambda sim: sim.ticks) == ticks
        assert not await worker.call(lambda sim: bool(sim.proximity_sensors().collisions))
        spatial = await worker.call(lambda sim: worker.spatial_state())
        print({"source": "scripted_bbox_real_physics", "stale_interruption": stale_interruption,
            "events": events, "ticks": ticks, "stop_acknowledged_s": stop_acknowledged_s,
            "maximum_processing_age_s": spatial["maximum_processing_age_s"], "timing": spatial["timing"]})
    finally:
        failure = sys.exc_info()[0]
        worker.stop()
        done.set()
        if writer:
            await writer
        if recorder:
            def detach(sim):
                recorder.capture(worker)
                worker.recorder = None
                return worker.latest["challenge"], next(reversed(worker.camera_frames.values()))
            status, image = await worker.call(detach)
            score = await asyncio.to_thread(recorder.finish, {"real_model": False, "evidence": "scripted_test",
                "test_passed": failure is None, "stop_injection": stop_after_first_arc, "stale_injection": stale_interruption})
            report = {"case_id": case_id, "challenge": challenge.id, "challenge_status": status, "evidence": "scripted_test",
                "physics_success": score["final_physics_success"], "test_passed": failure is None,
                "error": failure.__name__ if failure else None, "termination_reason": "test_failed" if failure else "test_finished",
                "rendering": "enhanced", "evaluation_elapsed_s": score["elapsed_wall_s"], "recording_scorecard": score}
            (batch.directory / case_id / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            (batch.directory / case_id / "trace.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
            (batch.directory / case_id / "terminal.png").write_bytes(image)
            batch.results.append(report)
            batch.finish()
        await worker.close()


def test_observed_path_planner_rejects_unknown_and_inflates_obstacles():
    import numpy as np
    from backend.spatial import ObservedMap
    observed = ObservedMap("test", 0)
    with pytest.raises(ValueError):
        observed.plan([0, 0], [1.2, 0], .25)
    observed.cells[60:105, 65:115] = 0
    observed.cells[78:83, 91:94] = 100
    route = observed.plan([0, 0], [1.3, .1], .12)
    assert len(route) > 2
    traversable = observed.traversable([0, 0], .12)
    for start, end in zip(route, route[1:]):
        assert observed.contains_path(traversable, np.linspace(start, end, 100))
    assert not observed.contains_path(traversable, [[.58, 0]])
    with pytest.raises(ValueError):
        observed.plan([0, 0], [3, 3], .25)


@pytest.mark.parametrize("fault", ["stale", "blocked", "lease", "stop", "deadline", "no_progress"])
def test_continuous_goal_brakes_on_invalid_authorization_or_sensors(fault):
    from types import SimpleNamespace
    from backend.challenges import get_challenge
    from backend.continuous_navigation import ContinuousNavigation
    sim = BulletSimulation(challenge=get_challenge("park"), width=160, height=120)
    sim.on_tick = None
    now = [0.]
    clock = lambda: now[0]
    runtime = NavigationRuntime(clock=clock)
    controller = ContinuousNavigation([[0, 0], [1.2, 0]], clock=clock)
    observed = SimpleNamespace(captured_at=0., max_frame_age_s=1.)
    try:
        controller.start(sim, runtime)
        controller.update(sim, runtime, observed, lambda *args: True)
        assert runtime.buffer
        now[0] = .2
        if fault == "stale":
            observed.captured_at = -2.
        elif fault == "lease":
            runtime.expires_at = 0.
        elif fault == "stop":
            sim.stop()
        elif fault == "deadline":
            runtime.skill_deadline = 0.
        elif fault == "no_progress":
            controller.progress_at = -4.
        before = sim.ticks
        controller.update(sim, runtime, observed, lambda *args: fault != "blocked")
        assert not controller.active and not runtime.buffer and not runtime.velocity.any()
        assert sim.ticks == before
        controller.update(sim, runtime, observed, lambda *args: True)
        assert not runtime.buffer
    finally:
        sim.close()


@pytest.mark.parametrize("spent_distance", [.3, 1.])
def test_continuous_replan_preserves_deadline_travel_and_stop_authority(spent_distance):
    from types import SimpleNamespace
    from backend.challenges import get_challenge
    from backend.continuous_navigation import ContinuousNavigation
    sim = BulletSimulation(challenge=get_challenge("park"), width=160, height=120)
    sim.on_tick = None
    now = [0.]
    runtime = NavigationRuntime(clock=lambda: now[0])
    controller = ContinuousNavigation([[0, 0], [1.2, 0]], clock=lambda: now[0])
    observed = SimpleNamespace(captured_at=0., max_frame_age_s=1.)
    try:
        controller.start(sim, runtime)
        controller.update(sim, runtime, observed, lambda *args: True)
        runtime.travel = spent_distance
        deadline = runtime.skill_deadline
        now[0] = .1
        controller.update(sim, runtime, observed, lambda *args: False, lambda start, target: [start, [.2, .3], target])
        if spent_distance == 1.:
            assert not controller.active and not runtime.buffer
            assert runtime.skill_deadline == deadline and runtime.travel == spent_distance
            return
        assert controller.active and controller.replans == 1 and not runtime.buffer
        assert runtime.skill_deadline == deadline and runtime.travel == .3
        now[0] = .2
        controller.update(sim, runtime, observed, lambda *args: True)
        assert runtime.buffer and controller.buffer_stops == 0
        sim.stop()
        controller.update(sim, runtime, observed, lambda *args: True)
        assert not runtime.buffer and not controller.active
    finally:
        sim.close()


def test_continuous_goal_rotates_inside_clearance_before_entering_narrow_path():
    from types import SimpleNamespace
    from backend.challenges import get_challenge
    from backend.continuous_navigation import ContinuousNavigation
    sim = BulletSimulation(challenge=get_challenge("park"), width=160, height=120)
    sim.on_tick = None
    clock = lambda: sim.ticks / 240
    runtime = NavigationRuntime(clock=clock)
    controller = ContinuousNavigation([[0, 0], [1.2, .5]], clock=clock)
    attempts = []
    def valid(pose, linear, angular):
        attempts.append(linear)
        return linear == 0.
    try:
        controller.start(sim, runtime)
        controller.update(sim, runtime, SimpleNamespace(captured_at=clock(), max_frame_age_s=1), valid)
        assert controller.active and controller.updates == 1 and runtime.buffer
        assert len(attempts) == 3 and attempts[0] > attempts[1] > attempts[2] == 0
        runtime.tick(sim)
        assert runtime.velocity[0] == 0 and runtime.velocity[1] > 0
    finally:
        sim.close()


@pytest.mark.parametrize("direction", [-1, 1])
def test_validated_navigation_can_drive_smooth_left_and_right_curves(direction):
    from backend.challenges import get_challenge
    from scripts.navigation_policy import apply
    sim = BulletSimulation(challenge=get_challenge("park"), width=160, height=120)
    sim.on_tick = None
    runtime = NavigationRuntime(clock=lambda: sim.ticks / 240)
    samples = []
    try:
        apply(runtime, sim, "begin_local_subgoal", {"goal": "Scripted curve capability check"})
        for _ in range(40):
            apply(runtime, sim, "replace_motion_buffer", {"segments": [{"kind":"drive", "linear_mps": .12,
                "angular_radps": direction * .25, "duration_s": 1.}]})
            runtime.tick(sim)
            samples.append(runtime.velocity.copy())
        assert sim.odometry[0] > .15 and direction * sim.odometry[1] > .025
        assert direction * sim.odometry[2] > .35
        assert all(velocity[0] > .1 and direction * velocity[1] > .2 for velocity in samples[12:])
        assert not sim.proximity_sensors().collisions
        runtime.cancel(sim, "Curve test finished")
    finally:
        sim.close()


@pytest.mark.parametrize("speed", [.35, .5])
@pytest.mark.parametrize("challenge", ["park", "local_park"])
def test_continuous_goal_preserves_velocity_and_brakes_at_arrival(monkeypatch, speed, challenge):
    import pybullet as bullet
    from types import SimpleNamespace
    from backend import continuous_navigation
    from backend.challenges import get_challenge
    from backend.continuous_navigation import ContinuousNavigation
    monkeypatch.setattr(continuous_navigation, "CONTINUOUS_SPEED_MPS", speed)
    sim = BulletSimulation(challenge=get_challenge(challenge), width=160, height=120)
    sim.on_tick = None
    clock = lambda: sim.ticks / 240
    runtime = NavigationRuntime(clock=clock)
    controller = ContinuousNavigation([[0, 0], [1.2, .05]], clock=clock)
    peak_speed = 0.
    try:
        controller.start(sim, runtime)
        for _ in range(600):
            controller.update(sim, runtime, SimpleNamespace(captured_at=clock(), max_frame_age_s=1), lambda *args: True)
            if not controller.active:
                break
            previous_speed = runtime.velocity[0]
            runtime.tick(sim)
            assert runtime.status == "running", runtime.reason
            assert abs(runtime.velocity[0] - previous_speed) <= .3 * .05 + 1e-9
            assert 0 <= runtime.velocity[0] <= speed + 1e-9
            assert not sim.proximity_sensors().collisions
            peak_speed = max(peak_speed, bullet.getBaseVelocity(sim.robot, physicsClientId=sim.client)[0][0])
        assert controller.status == "arrived", controller.state()
        assert controller.distance_m < .04 and controller.updates > 30
        assert controller.buffer_stops == 0 and controller.minimum_cruise_speed > .1
        assert not runtime.buffer and list(runtime.velocity) == [0., 0.]
        assert (.3 if challenge == "park" else speed - .05) < peak_speed < speed + .03
        assert sim.ticks / 240 < 10
        print({"challenge": challenge, "speed_limit_mps": speed, "peak_speed_mps": peak_speed, "motion_simulated_s": clock(),
               "arrival_error_m": controller.distance_m})
    finally:
        sim.close()


def test_faster_buffers_are_restricted_to_continuous_controller():
    from backend.continuous_navigation import ContinuousMotionBuffer
    from backend.navigation import ReplaceMotionBuffer
    from backend.simulation import MotionError
    from scripts.navigation_policy import apply, bounded_velocity
    segments = [{"kind": "drive", "linear_mps": .5, "angular_radps": 0., "duration_s": 1.}]
    with pytest.raises(ValueError):
        ReplaceMotionBuffer(expected_revision=0, segments=segments)
    with pytest.raises(ValueError):
        ContinuousMotionBuffer(expected_revision=0, segments=[{**segments[0], "linear_mps": .51}])
    assert bounded_velocity([.5, 0.])[0] == [.15, 0.]
    sim = BulletSimulation(width=160, height=120)
    runtime = NavigationRuntime()
    try:
        apply(runtime, sim, "begin_local_subgoal", {"goal": "Reject faster unqualified policy commands"})
        observation = sim.observe(render=False)
        runtime.observe(sim, observation)
        with pytest.raises(MotionError, match="speed limit"):
            runtime.apply(sim, "replace_motion_buffer", ContinuousMotionBuffer(expected_revision=runtime.revision,
                segments=segments), observation.seq)
        assert not runtime.buffer
    finally:
        sim.close()


def test_performance_tracking_fingerprints_design_and_retains_failed_missing_attempts():
    from scripts.evaluate_supervised import design_snapshot, performance_rows
    snapshot = design_snapshot("history-candidate")
    assert {"backend/simulation.py", "backend/camera.py", "backend/visual_history.py", "scripts/render_camera.mjs", "frontend/src/style.css"} <= snapshot["code_sha256"].keys()
    assert snapshot["source_sha256"] == design_snapshot("another-label")["source_sha256"]
    manifest = {"experiment_id": "test", "design": snapshot, "evidence": "scripted_test", "cases": [
        {"case_id": name, "challenge": "park", "challenge_sha256": "fixture"} for name in ("pass", "fail", "missing", "incomplete")]}
    results = [{"case_id": "pass", "physics_success": True, "recording_scorecard": {
        "complete_recording": True, "completion_time_s": 12., "actual_distance_m": 1., "contact_episodes": 0}},
        {"case_id": "fail", "physics_success": False, "termination_reason": "time_limit", "input_tokens": 100},
        {"case_id": "incomplete", "physics_success": True, "recording_scorecard": {"completion_time_s": 1., "complete_recording": False}}]
    rows = performance_rows(manifest, results)
    assert len(rows) == 4 and sum(row["verified_success"] for row in rows) == 1
    assert rows[2]["status"] == "missing_report" and rows[2]["physics_success"] is None
    assert rows[3]["completion_s"] is None and all(row["evidence"] == "scripted_test" for row in rows)
    results[0]["manual_placements"] = 1
    assert not performance_rows(manifest, results)[0]["verified_success"]


def test_navigation_evaluation_budget_expands_turns_or_uses_only_deadline():
    from itertools import islice
    from types import SimpleNamespace
    from backend.agent import AgentController, AgentStart, NavigationEvaluationBudget
    settings = SimpleNamespace(max_turns=80, execution_mode="luna_navigation")
    controller = AgentController()
    assert len(list(controller.turn_indices(settings))) == 80
    assert controller.local_request_limit(settings) == 640
    controller.evaluation_budget = NavigationEvaluationBudget(max_turns=120, timeout_s=900)
    assert len(list(controller.turn_indices(settings))) == 120
    assert controller.local_request_limit(settings) == 960
    controller.evaluation_budget = NavigationEvaluationBudget(timeout_s=900)
    assert list(islice(controller.turn_indices(settings, 119), 3)) == [119, 120, 121]
    assert controller.local_request_limit(settings) is None
    for values in ({"timeout_s": 0}, {"timeout_s": float("nan")}, {"timeout_s": 3601}, {"timeout_s": 10, "max_turns": 0}):
        with pytest.raises(ValueError):
            NavigationEvaluationBudget(**values)
    with pytest.raises(ValueError):
        AgentStart(run_id="test", episode_epoch=0, goal="Navigate", evaluation_budget={"timeout_s": 900})


@pytest.mark.parametrize("phase", ["supervisor", "policy", "motion", "request_timeout"])
async def test_navigation_evaluation_deadline_brakes_and_rejects_late_results(phase):
    import json
    from backend.agent import AgentController, AgentStart, FoundryConfig, NavigationEvaluationBudget
    from backend.challenges import get_challenge
    from tests.test_agent import model_response
    cancelled = []
    calls = []

    class Policy:
        @staticmethod
        def check_available():
            pass

        async def start(self):
            pass

        async def predict(self, observation, image):
            calls.append(self.instruction)
            if phase == "policy" and len(calls) > 1:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.append("policy")
            return {"action": [.1, 0.] if len(calls) > 1 else [0., 0.]}

        async def close(self):
            pass

    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            if phase == "request_timeout":
                await asyncio.Event().wait()
            if phase == "supervisor":
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.append("supervisor")
            return model_response("guide_navigation", json.dumps({"motion": "forward", "steps": 8,
                "look_yaw_rad": 0, "look_pitch_rad": .45, "status": "continue", "reason": "Scripted bounded forward motion"}))

        async def close(self):
            pass

    worker = SimulationWorker(challenge=get_challenge("park"), pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"),
        model_factory=lambda config: Supervisor(), local_navigation_factory=Policy,
        request_timeout_s=.1 if phase == "request_timeout" else 10,
        evaluation_budget=NavigationEvaluationBudget(timeout_s=2))
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, AgentStart(run_id=worker.latest["run_id"], episode_epoch=0,
            goal="Test bounded navigation", execution_mode="luna_navigation", max_turns=1, feedback_interval_s=.25))
        await asyncio.wait_for(controller.task, 6)
        assert not controller.active and worker.latest["stopped"]
        assert worker.latest["navigation"]["remaining_s"] == 0
        assert controller.state["max_turns"] is None
        assert controller.state["local_model"]["request_limit"] is None
        if phase == "request_timeout":
            assert controller.state["phase"] == "error" and controller.state["outcome"]["kind"] == "error"
        else:
            assert controller.state["error"] is None and controller.state["outcome"]["kind"] == "limited"
            assert "time budget" in controller.state["message"]
        if phase in {"supervisor", "policy"}:
            assert cancelled == [phase]
            assert controller.state["local_model"]["requests_completed"] == 0
        if phase == "motion":
            assert controller.state["local_model"]["requests_completed"] >= 1
    finally:
        await controller.halt()
        await worker.close()


def test_navigation_evaluator_budget_options_are_explicit():
    from types import SimpleNamespace
    from scripts.evaluate_supervised import argument_parser, evaluation_budget
    defaults = {"mode": "luna_navigation", "turns": 120, "time_limit_s": None, "session_limit_s": None}
    assert evaluation_budget(SimpleNamespace(**defaults)).model_dump() == {"max_turns": 120, "timeout_s": 1200.}
    assert evaluation_budget(SimpleNamespace(**(defaults | {"turns": None, "time_limit_s": 1800}))).max_turns is None
    with pytest.raises(ValueError):
        evaluation_budget(SimpleNamespace(**(defaults | {"time_limit_s": 1800, "session_limit_s": 600})))
    with pytest.raises(ValueError):
        evaluation_budget(SimpleNamespace(**(defaults | {"time_limit_s": 1800})))
    parser = argument_parser()
    options = parser.parse_args(["--output", "unused-test-output"])
    assert evaluation_budget(options).model_dump() == {"max_turns": None, "timeout_s": 1800.}
    options = parser.parse_args(["--output", "unused-test-output", "--turns", "120", "--session-limit-s", "1800"])
    assert evaluation_budget(options).model_dump() == {"max_turns": 120, "timeout_s": 1800.}
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", "unused-test-output", "--turns", "120", "--time-limit-s", "1800"])


def test_mode_comparison_metrics_retain_failures_and_discarded_requests():
    from scripts.evaluate_supervised import execution_metrics
    events = [
        {"kind": "response", "title": "LLM response", "payload": {"latency_s": latency}} for latency in (2., 1., 6.)]
    events.extend([
        {"kind": "tool", "title": "Robot tool requested", "payload": {"tool": "replace_motion_buffer"}},
        {"kind": "result", "title": "Robot tool result", "payload": {"result": {"status": "error", "error": "STALE_OBSERVATION"}}},
        {"kind": "session", "title": "Navigation feedback refreshed", "payload": {}},
        {"kind": "policy", "title": "Circuit sensing recovery", "payload": {"attempt": 1}},
    ])
    metrics = execution_metrics(events)
    assert metrics["tool_requests"] == {"replace_motion_buffer": 1}
    assert metrics["tool_errors"] == {"STALE_OBSERVATION": 1}
    assert metrics["inference_median_s"] == 2. and metrics["inference_p95_s"] == 6.
    assert metrics["discarded_navigation_requests"] == 1
    assert metrics["circuit_sensing_recoveries"] == 1
    assert execution_metrics([])["inference_median_s"] is None
    assert execution_metrics([])["circuit_sensing_recoveries"] == 0


def test_shared_apartment_evaluation_cases_keep_environment_provenance():
    from scripts.evaluate_supervised import argument_parser, benchmark_cases
    options = argument_parser().parse_args(["--output", "unused", "--environment", "shared_apartment_v1",
        "--challenges", "furniture_circuit", "apartment", "flat_kitchen", "recharge"])
    cases = benchmark_cases(options.challenges, environment=options.environment)
    assert len(cases) == 4
    assert all(name.startswith("shared-apartment-v1-") for name, _ in cases)
    assert all(challenge.environment == "shared_apartment_v1" for _, challenge in cases)
    assert all(challenge.scene() == cases[0][1].scene() for _, challenge in cases)
    assert benchmark_cases(["furniture_circuit"])[0][1].environment == "standalone"
    with pytest.raises(ValueError, match="standalone"):
        benchmark_cases(["kitchen_bathroom"], "kitchen-v1", environment="shared_apartment_v1")
    with pytest.raises(ValueError, match="supports"):
        benchmark_cases(["park"], environment="shared_apartment_v1")


def test_evaluation_workload_snapshot_is_scoped_and_handles_missing_gpu(monkeypatch):
    from types import SimpleNamespace
    from scripts import evaluate_supervised
    def gpu_probe(command, **kwargs):
        assert command[0] == "nvidia-smi" and kwargs["timeout"] == 5
        return SimpleNamespace(stdout="NVIDIA Test GPU, 2048, 8192, 12\n")
    monkeypatch.setattr(evaluate_supervised.subprocess, "run", gpu_probe)
    snapshot = evaluate_supervised.workload_snapshot()
    assert snapshot["gpu"] == [{"name": "NVIDIA Test GPU", "memory_used_mib": "2048", "memory_free_mib": "8192", "utilization_pct": "12"}]
    assert snapshot["observed_at"] and "instant" in snapshot["scope"]
    def missing_gpu(*args, **kwargs):
        raise FileNotFoundError()
    monkeypatch.setattr(evaluate_supervised.subprocess, "run", missing_gpu)
    assert evaluate_supervised.workload_snapshot()["gpu"] is None


@pytest.mark.parametrize("mode", ["single_step", "navigation_plan"])
async def test_mode_comparison_scripted_smoke_never_counts_claimed_success_as_physics(tmp_path, monkeypatch, mode):
    import json
    from types import SimpleNamespace
    from backend.agent import AgentController, FoundryConfig
    from scripts import evaluate_supervised
    from tests.test_agent import model_response
    seen = []

    class StoppingModel:
        async def respond(self, profile, reasoning, goal, inputs):
            seen.append(self.execution_mode)
            return model_response("stop", json.dumps({"task_status": "completed", "reason": "Scripted unverified completion"}))

        async def close(self):
            pass

    async def no_local_start(self):
        raise AssertionError("Direct Luna comparisons must not load SmolVLA")

    monkeypatch.setattr(evaluate_supervised, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(FoundryConfig, "from_environment", classmethod(lambda cls: cls(endpoint="https://test.openai.azure.com")))
    monkeypatch.setattr(evaluate_supervised, "AgentController", lambda config, **kwargs:
        AgentController(config, model_factory=lambda config: StoppingModel(), **kwargs))
    monkeypatch.setattr(evaluate_supervised.LocalNavigationClient, "start", no_local_start)
    output = tmp_path / mode
    await evaluate_supervised.evaluate(SimpleNamespace(mode=mode, stage="challenges", checkpoint=None,
        output=output, challenges=["park"], turns=2, reasoning="high", images=2, context_tokens=8192, evidence="scripted_test",
        performance_index=tmp_path / "index.jsonl"))
    report = json.loads((output / "park" / "report.json").read_text())
    manifest = json.loads((output / "experiment.json").read_text())
    assert seen == [mode] and report["execution_mode"] == manifest["mode"] == mode
    assert not report["physics_success"] and not report["real_smolvla"]
    assert report["checkpoint"] is None and report["model_commands"] is None
    assert report["outcome"]["source"] == "agent" and report["outcome"]["kind"] == "completed"
    assert report["completion_source"] == "agent" and report["unverified_completion_claim"]
    assert report["error"] is None and report["manual_placements"] == 0
    assert report["tool_requests"] == {"stop": 1} and report["inference_responses"] == 1
    assert report["sensing"]["timing"]["expired_results"] == 0 and report["circuit_sensing_recoveries"] == 0
    assert "Whole-device" in manifest["workload_before"]["scope"] and manifest["workload_after"]["observed_at"]
    assert manifest["images_per_request"] == 2 and manifest["session_timeout_s"] == 600
    assert len(manifest["code_sha256"]["backend/agent.py"]) == 64
    assert manifest["architecture"]["id"] == {"single_step": "tool-step", "navigation_plan": "buffered-plan"}[mode]
    assert manifest["architecture"]["version"] == "0.1.0"
    assert len(manifest["architecture"]["revision"]) == 12
    model_variant = manifest["model_variant"]
    assert not model_variant["underlying_identity_verified"]
    assert model_variant["configuration"]["session_timeout_s"] == 600
    assert model_variant["configuration"]["max_turns"] == 2
    assert model_variant["configuration"]["camera_history"] == "not_applicable"
    assert manifest["variant_id"] == f"{manifest['architecture']['version_key']}/{model_variant['revision']}"
    rows = json.loads((output / "performance.json").read_text())
    assert len(rows) == 1 and rows[0]["evidence"] == "scripted_test" and not rows[0]["verified_success"]
    assert rows[0]["sensing"] == report["sensing"] and rows[0]["circuit_sensing_recoveries"] == 0
    assert report["evidence"] == "scripted_test" and not report["real_luna"]
    indexed = json.loads((tmp_path / "index.jsonl").read_text())
    assert indexed["experiment_id"] == manifest["experiment_id"] and indexed["planned"] == indexed["reported"] == 1
    assert manifest["finished_at"] and not manifest["source_changed_during_run"]


async def test_performance_comparison_flags_confounds_and_rejects_missing_rows(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import evaluate_supervised
    from scripts.evaluate_supervised import compare_performance, comparison_markdown, performance_rows
    directories = [tmp_path / "baseline", tmp_path / "candidate"]
    for index, directory in enumerate(directories):
        directory.mkdir()
        manifest = {"schema_version": 2, "experiment_id": str(index), "design": {"label": directory.name, "source_sha256": "same",
            "runtime": {"gpu": [{"name": "test"}]}}, "evidence": "scripted_test", "camera_history": "enabled" if index else "disabled",
            "code_sha256": {"backend/simulation.py": "same"}, "session_timeout_s": 150,
            "cases": [{"case_id": "park", "challenge": "park", "challenge_sha256": "same"}]}
        result = {"case_id": "park", "physics_success": bool(index), "termination_reason": "completed" if index else "time_limit",
            "input_tokens": 100, "output_tokens": 20, "rendering": "enhanced", "recording_scorecard": {
                "complete_recording": True, "completion_time_s": 20. if index else None, "contact_episodes": 0}}
        (directory / "experiment.json").write_text(json.dumps(manifest))
        (directory / "performance.json").write_text(json.dumps(performance_rows(manifest, [result])))
    report = compare_performance(*directories)
    assert not report["warnings"] and report["per_challenge"][0]["candidate"]["successes"] == 1
    assert "0/1 | 1/1" in comparison_markdown(report) and "No automatic winner" in report["interpretation"]
    assert "Evidence: scripted_test / scripted_test" in comparison_markdown(report)
    assert report["per_challenge"][0]["candidate"]["all_attempt_medians"]["input_tokens"] == 100
    def no_configuration(*args, **kwargs):
        raise AssertionError("Offline comparison must not access model configuration")
    monkeypatch.setattr(evaluate_supervised, "load_dotenv", no_configuration)
    output = tmp_path / "comparison"
    await evaluate_supervised.evaluate(SimpleNamespace(compare=directories, output=output))
    assert json.loads((output / "comparison.json").read_text())["per_challenge"] == report["per_challenge"]
    assert (output / "comparison.md").read_text().startswith("# Performance Comparison")
    manifest["session_timeout_s"] = 300
    (directories[1] / "experiment.json").write_text(json.dumps(manifest))
    assert "Mismatched session_timeout_s" in compare_performance(*directories)["warnings"]
    (directories[1] / "performance.json").write_text("[]")
    with pytest.raises(ValueError, match="planned cases"):
        compare_performance(*directories)


def test_navigation_policy_state_uses_only_head_wheel_and_range_measurements():
    from types import SimpleNamespace
    from scripts.navigation_policy import DIRECTIONS, STATE_NAMES, state_vector
    readings = [SimpleNamespace(direction=name, status="hit", distance_m=.7) for name in DIRECTIONS]
    readings[1] = SimpleNamespace(direction=DIRECTIONS[1], status="clear", distance_m=None)
    readings[2] = SimpleNamespace(direction=DIRECTIONS[2], status="occluded", distance_m=None)
    observation = SimpleNamespace(head_rad=[.1, .4], joints=[SimpleNamespace(name="left_wheel", velocity=1.),
        SimpleNamespace(name="right_wheel", velocity=2.)], proximity=SimpleNamespace(distances=readings),
        target=[99, 99], robot_position=[20, 30, 0])
    result = state_vector(observation)
    assert len(result) == len(STATE_NAMES) == 20
    assert result[:4] == pytest.approx([.1, .4, .135, .09 / .38])
    assert result[4:7] == [.7, 2., 0.]
    assert result[12:15] == [1., 0., -1.]


def test_navigation_velocity_adapter_logs_saturation_and_rejects_invalid_output():
    from scripts.navigation_policy import bounded_velocity
    assert bounded_velocity([.1, -.2]) == ([.1, -.2], [])
    assert bounded_velocity([.2, -.7]) == ([.15, -.5], ["linear_mps", "angular_radps"])
    for invalid in ([1], [1, 2, 3], [float("nan"), 0], [float("inf"), 0]):
        with pytest.raises(ValueError, match="two finite"):
            bounded_velocity(invalid)


def test_navigation_training_contract_is_separate_from_manipulation():
    from types import SimpleNamespace
    from scripts.navigation_policy import ACTION_NAMES, STATE_NAMES
    from scripts.train_milo import validate_dataset
    dataset = SimpleNamespace(fps=1, num_episodes=16, features={
        "action": {"names": ACTION_NAMES, "shape": (2,), "dtype": "float32"},
        "observation.state": {"names": STATE_NAMES, "shape": (20,), "dtype": "float32"},
        "observation.images.head": {"shape": (240, 320, 3), "dtype": "image"}})
    validate_dataset(dataset, navigation=True)
    with pytest.raises(ValueError):
        validate_dataset(dataset)
    dataset.features["action"]["names"] = ACTION_NAMES[::-1]
    with pytest.raises(ValueError, match="contract"):
        validate_dataset(dataset, navigation=True)


def test_navigation_checkpoint_rejects_arm_weights_and_tampered_weights(tmp_path):
    import hashlib
    import json
    from scripts.navigation_policy import ACTION_NAMES, EMBODIMENT, STATE_NAMES, load_checkpoint
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"navigation")
    report = {"status": "trained_and_reloaded", "embodiment": EMBODIMENT,
        "weights_sha256": hashlib.sha256(b"navigation").hexdigest(), "state_names": STATE_NAMES,
        "action_names": ACTION_NAMES, "reloaded_output": {"shape": [1, 1, 2]}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert load_checkpoint(checkpoint)[0] == checkpoint
    report["embodiment"] = "milo-left-arm-v1"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="navigation-specific"):
        load_checkpoint(checkpoint)
    report["embodiment"] = EMBODIMENT
    path.write_text(json.dumps(report))
    (checkpoint / "model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="navigation-specific"):
        load_checkpoint(checkpoint)


def test_navigation_skill_order_limits_and_feedback_checkpoints():
    with pytest.raises(ValidationError):
        NavigationPlan(expected_revision=0, steps=[{"skill": "cross", "goal": "Skip inspection"}])
    with pytest.raises(ValidationError):
        ReplaceMotionBuffer(expected_revision=0, segments=[
            {"kind": "drive", "linear_mps": .15, "angular_radps": 0, "duration_s": 1}] * 3)
    sim = BulletSimulation(width=160, height=120)
    runtime = NavigationRuntime()
    try:
        plan(runtime, sim)
        with pytest.raises(MotionError, match="Inspect more"):
            complete(runtime, sim)
        scan(runtime, sim)
        assert runtime.status == "awaiting_feedback" and runtime.scan_max - runtime.scan_min > .4
        complete(runtime, sim)
        assert runtime.index == 1 and runtime.steps[0]["status"] == "completed"
        sequence, revision = ticket(runtime, sim)
        with pytest.raises(MotionError, match="Locate the opening"):
            runtime.apply(sim, "replace_motion_buffer", ReplaceMotionBuffer(expected_revision=revision, segments=[
                {"kind": "drive", "linear_mps": .1, "angular_radps": 0, "duration_s": 1}]), sequence)
    finally:
        sim.close()


def test_navigation_expiry_and_stop_invalidate_late_replacements():
    now = [10.0]
    runtime = NavigationRuntime(clock=lambda: now[0])
    sim = BulletSimulation(width=160, height=120)
    try:
        plan(runtime, sim)
        sequence, revision = ticket(runtime, sim)
        buffer = ReplaceMotionBuffer(expected_revision=revision, segments=[
            {"kind": "head", "yaw_rad": .5, "pitch_rad": .2, "duration_s": 1}])
        runtime.apply(sim, "replace_motion_buffer", buffer, sequence)
        sequence, revision = ticket(runtime, sim)
        now[0] += 2
        runtime.tick(sim)
        assert runtime.status == "failed" and not runtime.buffer and sim.cancel.is_set()
        assert sim.ticks == 0 and runtime.revision > revision
        with pytest.raises(MotionError):
            runtime.apply(sim, "replace_motion_buffer", buffer, sequence)
    finally:
        sim.close()


async def submit(worker, tool, arguments):
    observation, _ = await worker.feedback()
    return await worker.execute_navigation(Command(run_id=observation.run_id, episode_epoch=observation.episode_epoch,
        observation_seq=observation.seq, action_id=str(uuid4()), tool=tool,
        arguments={"expected_revision": observation.navigation.revision, **arguments}))


def test_slow_simulation_expires_buffer_without_failing_plan():
    now = [10.0]
    sim = BulletSimulation(width=160, height=120)
    runtime = NavigationRuntime(clock=lambda: now[0])
    try:
        plan(runtime, sim)
        sequence, revision = ticket(runtime, sim)
        update = ReplaceMotionBuffer(expected_revision=revision, segments=[
            {"kind": "head", "yaw_rad": .6, "pitch_rad": .2, "duration_s": .5}])
        runtime.apply(sim, "replace_motion_buffer", update, sequence)
        sequence, revision = ticket(runtime, sim)
        update.expected_revision = revision
        for _ in range(5):
            now[0] += .21
            runtime.tick(sim)
        assert sim.ticks < 120
        assert runtime.status == "awaiting_feedback", runtime.reason
        assert "expired" in runtime.reason.lower()
        assert not runtime.buffer and runtime.velocity.tolist() == [0, 0]
        assert not sim.cancel.is_set()
        assert runtime.steps[0]["status"] == "running"
        with pytest.raises(MotionError, match="revision changed"):
            runtime.apply(sim, "replace_motion_buffer", update, sequence)
        sequence, update.expected_revision = ticket(runtime, sim)
        runtime.apply(sim, "replace_motion_buffer", update, sequence)
        runtime.tick(sim)
        assert runtime.status == "running" and runtime.buffer
    finally:
        sim.close()


async def test_worker_accepts_replacements_while_moving_and_stop_invalidates_them():
    worker = SimulationWorker(pace=True)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.begin_navigation(worker.stop_revision)
        result = await submit(worker, "set_navigation_plan", {"steps": [
            {"skill": "inspect_room", "goal": "Inspect fixtures"}]})
        assert result.status == "ok"
        result = await submit(worker, "replace_motion_buffer", {"segments": [
            {"kind": "head", "yaw_rad": -.6, "pitch_rad": .2, "duration_s": 1},
            {"kind": "head", "yaw_rad": .6, "pitch_rad": .2, "duration_s": 1}]})
        async with asyncio.timeout(5):
            while worker.latest["snapshot"]["simulated_time_s"] < .2:
                await asyncio.sleep(.01)
        observation, _ = await worker.feedback()
        assert observation.navigation.remaining_s > 0 and worker.latest["busy"]
        revision = observation.navigation.revision
        replacement = Command(run_id=observation.run_id, episode_epoch=observation.episode_epoch,
            observation_seq=observation.seq, action_id="replace-active", tool="replace_motion_buffer", arguments={
                "expected_revision": revision, "segments": [{"kind": "head", "yaw_rad": .8, "pitch_rad": .2, "duration_s": 1}]})
        assert (await worker.execute_navigation(replacement)).status == "ok"
        assert worker.latest["navigation"]["revision"] > revision
        worker.stop()
        replacement.action_id = "late-after-stop"
        result = await worker.execute_navigation(replacement)
        assert result.status == "error" and result.error == "CANCELLED"
        assert not worker.latest["navigation"]["remaining_s"] and worker.latest["stopped"]
        assert not worker.latest["busy"]
    finally:
        await worker.close()


async def test_controller_refills_before_empty_and_brakes_on_inference_failure():
    import json
    from backend.agent import AgentController, FoundryConfig, robot_tools
    from tests.test_agent import model_response, start_settings
    seen = []

    class Planner:
        async def respond(self, profile, reasoning, goal, inputs):
            observation = json.loads(inputs[-1]["content"][0]["text"])
            state = observation["navigation"]
            seen.append(state)
            if len(seen) == 1:
                return model_response("set_navigation_plan", json.dumps({"expected_revision": state["revision"],
                    "steps": [{"skill": "inspect_room", "goal": "Look around"}]}), call_id="plan")
            if len(seen) <= 3:
                if len(seen) == 3:
                    assert state["remaining_s"] > 0
                return model_response("replace_motion_buffer", json.dumps({"expected_revision": state["revision"],
                    "segments": [{"kind": "head", "yaw_rad": -.6 if len(seen) == 2 else .6, "pitch_rad": .2,
                                  "duration_s": 1}] * 2}), call_id=f"buffer-{len(seen)}")
            raise ValueError("Scripted inference failure")

        async def close(self):
            pass

    worker = SimulationWorker(pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Planner())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, execution_mode="navigation_plan", feedback_interval_s=.25, max_turns=8))
        await controller.task
        assert len(seen) == 4
        assert controller.state["phase"] == "error" and "Scripted inference failure" in controller.state["error"]
        assert worker.latest["navigation"]["remaining_s"] == 0 and worker.latest["stopped"]
        assert all(step["status"] == "cancelled" for step in worker.latest["navigation"]["steps"])
        assert not controller.state["auto_wake"]
        assert "replace_motion_buffer" in {tool["name"] for tool in robot_tools("navigation_plan")}
        assert "replace_motion_buffer" not in {tool["name"] for tool in robot_tools()}
    finally:
        await controller.halt()
        await worker.close()


def drive_buffer(runtime, sim, linear=.15, angular=0):
    sequence, revision = ticket(runtime, sim)
    runtime.apply(sim, "replace_motion_buffer", ReplaceMotionBuffer(expected_revision=revision, segments=[
        {"kind": "drive", "linear_mps": linear, "angular_radps": angular, "duration_s": 1}] * 2), sequence)


@pytest.mark.parametrize("expire", [False, True])
async def test_buffer_pause_cancels_outdated_inference_and_replans_from_rest(expire):
    import json
    from backend.agent import AgentController, FoundryConfig
    from tests.test_agent import model_response, start_settings
    entered, cancelled = asyncio.Event(), asyncio.Event()
    seen = []
    in_flight = 0

    class Planner:
        async def respond(self, profile, reasoning, goal, inputs):
            nonlocal in_flight
            in_flight += 1
            assert in_flight == 1
            try:
                state = json.loads(inputs[-1]["content"][0]["text"])["navigation"]
                seen.append(state)
                if len(seen) == 1:
                    return model_response("set_navigation_plan", json.dumps({"expected_revision": state["revision"],
                        "steps": [{"skill": "inspect_room", "goal": "Inspect"}]}), call_id="plan")
                if len(seen) == 3:
                    assert state["remaining_s"] > 0
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.set()
                if len(seen) == 4:
                    assert cancelled.is_set() and state["remaining_s"] == 0
                    assert state["status"] == "awaiting_feedback" and state["revision"] > seen[2]["revision"]
                if len(seen) in {2, 4}:
                    return model_response("replace_motion_buffer", json.dumps({"expected_revision": state["revision"],
                        "segments": [{"kind": "head", "yaw_rad": .5, "pitch_rad": .2, "duration_s": 1}] * 2}),
                        call_id=f"buffer-{len(seen)}")
                return model_response("stop", "{}", call_id="stop")
            finally:
                in_flight -= 1

        async def close(self):
            pass

    worker = SimulationWorker(pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: Planner())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, execution_mode="navigation_plan", feedback_interval_s=.25, max_turns=8))
        await asyncio.wait_for(entered.wait(), 5)
        if expire:
            await worker.call(lambda sim: setattr(worker.navigation, "expires_at", 0))
        await asyncio.wait_for(controller.task, 12)
        assert controller.state["phase"] == "completed" and controller.state["error"] is None
        assert len(seen) == 5 and in_flight == 0
        assert all(event["status"] == "ok" for event in controller.state["events"])
        assert any(entry["title"] == "Navigation feedback refreshed" for entry in controller.trace()["events"])
        assert worker.latest["stopped"] and not worker.latest["navigation"]["remaining_s"]
    finally:
        await controller.halt()
        await worker.close()


async def test_slow_model_waits_for_rest_before_requesting_next_buffer():
    import json
    from backend.agent import AgentController, FoundryConfig
    from tests.test_agent import model_response, start_settings
    seen = []

    class SlowPlanner:
        async def respond(self, profile, reasoning, goal, inputs):
            state = json.loads(inputs[-1]["content"][0]["text"])["navigation"]
            seen.append(state)
            if len(seen) == 1:
                return model_response("set_navigation_plan", json.dumps({"expected_revision": state["revision"],
                    "steps": [{"skill": "inspect_room", "goal": "Inspect"}]}), call_id="plan")
            if len(seen) == 2:
                await asyncio.sleep(.6)
                return model_response("replace_motion_buffer", json.dumps({"expected_revision": state["revision"],
                    "segments": [{"kind": "head", "yaw_rad": .5, "pitch_rad": .2, "duration_s": .3}]}), call_id="buffer")
            assert state["remaining_s"] == 0 and state["status"] == "awaiting_feedback"
            return model_response("stop", "{}", call_id="stop")

        async def close(self):
            pass

    worker = SimulationWorker(pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: SlowPlanner())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, execution_mode="navigation_plan", feedback_interval_s=.25))
        await asyncio.wait_for(controller.task, 8)
        assert controller.state["phase"] == "completed" and controller.state["error"] is None
        assert len(seen) == 3
    finally:
        await controller.halt()
        await worker.close()


def test_full_skill_plan_crosses_physical_doorway_and_blends_replacements():
    scene = [
        {"name": "floor", "size": [6, 6, .1], "position": [0, 0, -.05], "color": [.7, .7, .7, 1]},
        {"name": "wall_left", "size": [.1, 1, 1], "position": [.8, 1.15, .5], "color": [.4, .4, .4, 1]},
        {"name": "wall_right", "size": [.1, 1, 1], "position": [.8, -1.15, .5], "color": [.4, .4, .4, 1]}]
    sim = BulletSimulation(scene=scene, width=160, height=120)
    runtime = NavigationRuntime(clock=lambda: 10)
    try:
        plan(runtime, sim)
        scan(runtime, sim)
        complete(runtime, sim)
        scan(runtime, sim)
        complete(runtime, sim)
        assert runtime.steps[runtime.index]["skill"] == "approach"
        with pytest.raises(MotionError, match="0.1 m"):
            complete(runtime, sim)
        drive_buffer(runtime, sim)
        for _ in range(15):
            runtime.tick(sim)
        speed_before = runtime.velocity[0]
        assert speed_before == pytest.approx(.15)
        drive_buffer(runtime, sim)
        assert runtime.velocity[0] == speed_before
        runtime.tick(sim)
        assert runtime.velocity[0] == pytest.approx(speed_before)
        while runtime.buffer:
            runtime.tick(sim)
        assert runtime.status == "awaiting_feedback", runtime.reason
        complete(runtime, sim)
        for _ in range(4):
            drive_buffer(runtime, sim)
            while runtime.buffer:
                runtime.tick(sim)
            assert runtime.status == "awaiting_feedback", runtime.reason
        complete(runtime, sim)
        assert runtime.status == "completed"
        assert all(step["status"] == "completed" for step in runtime.steps)
        assert sim.odometry[0] > 1.1
        assert not sim.proximity_sensors().collisions
    finally:
        sim.close()


@pytest.mark.parametrize("fault", ["clearance", "buffer_expiry", "feedback_expiry", "skill_timeout"])
def test_navigation_independent_watchdogs_brake(fault):
    now = [10.0]
    sim = BulletSimulation(width=160, height=120)
    runtime = NavigationRuntime(clock=lambda: now[0])
    try:
        plan(runtime, sim)
        scan(runtime, sim)
        complete(runtime, sim)
        drive_buffer(runtime, sim, linear=0, angular=.4)
        if fault == "clearance":
            edge = max(bullet.getAABB(sim.robot, index, physicsClientId=sim.client)[1][1]
                       for index in [-1, *sim.joints.values()])
            sim.reposition(ManualPlacement(run_id=sim.run_id, episode_epoch=sim.epoch,
                observation_seq=sim.seq, xy_m=[0, 3 - edge - .005]))
        elif fault == "buffer_expiry":
            runtime.expires_at = now[0] - .01
        elif fault == "feedback_expiry":
            now[0] += .8
        else:
            runtime.skill_deadline = now[0] - .01
        runtime.tick(sim)
        assert not runtime.buffer
        assert runtime.velocity.tolist() == [0, 0]
        if fault == "buffer_expiry":
            assert runtime.status == "awaiting_feedback" and not sim.cancel.is_set(), runtime.reason
            assert runtime.steps[1]["status"] == "running" and runtime.steps[2]["status"] == "pending"
        else:
            assert runtime.status == "failed" and sim.cancel.is_set(), runtime.reason
            assert runtime.steps[1]["status"] == "failed" and runtime.steps[2]["status"] == "cancelled"
    finally:
        sim.close()


def test_exhausted_or_moved_observations_cannot_replenish_old_motion():
    sim = BulletSimulation(width=160, height=120)
    runtime = NavigationRuntime(clock=lambda: 10)
    try:
        plan(runtime, sim)
        sequence, revision = ticket(runtime, sim)
        update = ReplaceMotionBuffer(expected_revision=revision, segments=[
            {"kind": "head", "yaw_rad": .5, "pitch_rad": .2, "duration_s": .5}])
        runtime.apply(sim, "replace_motion_buffer", update, sequence)
        sequence, revision = ticket(runtime, sim)
        update.expected_revision = revision
        while runtime.buffer:
            runtime.tick(sim)
        with pytest.raises(MotionError, match="revision changed"):
            runtime.apply(sim, "replace_motion_buffer", update, sequence)
        sequence, revision = ticket(runtime, sim)
        sim.odometry[2] += .4
        update.expected_revision = revision
        with pytest.raises(MotionError, match="moved too far"):
            runtime.apply(sim, "replace_motion_buffer", update, sequence)
    finally:
        sim.close()


async def test_safety_failure_cancels_inference_without_waiting_for_model_timeout():
    import json
    from backend.agent import AgentController, FoundryConfig
    from tests.test_agent import model_response, start_settings
    entered, cancelled = asyncio.Event(), asyncio.Event()
    requests = []

    class SlowPlanner:
        async def respond(self, profile, reasoning, goal, inputs):
            state = json.loads(inputs[-1]["content"][0]["text"])["navigation"]
            requests.append(state)
            if len(requests) == 1:
                return model_response("set_navigation_plan", json.dumps({"expected_revision": state["revision"],
                    "steps": [{"skill": "inspect_room", "goal": "Inspect"}]}), call_id="plan")
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def close(self):
            pass

    worker = SimulationWorker(pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"), lambda config: SlowPlanner())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, start_settings(worker, execution_mode="navigation_plan", feedback_interval_s=.25))
        await asyncio.wait_for(entered.wait(), 5)
        await worker.call(lambda sim: setattr(worker.navigation, "skill_deadline", 0))
        await asyncio.wait_for(controller.task, 2)
        assert cancelled.is_set() and controller.state["phase"] == "error"
        assert "SKILL_TIMEOUT" in controller.state["error"]
        assert not worker.latest["navigation"]["remaining_s"] and worker.latest["stopped"]
    finally:
        await controller.halt()
        await worker.close()