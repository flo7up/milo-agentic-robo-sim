import pytest
from pydantic import ValidationError

from backend.policy import ACTION_NAMES, PolicyChunk, PolicyConfig, PolicyMetadata, PolicyTicket, future_actions
from backend.robot import NEUTRAL


def test_policy_pacer_compensates_oversleep_and_rebases_after_pause_or_revision():
    from backend.simulation import TIMESTEP
    from backend.worker import PolicyPacer
    now = [0.]
    waits = []

    class OversleepingEvent:
        def wait(self, timeout):
            waits.append(timeout)
            now[0] += timeout + .015

    pacer = PolicyPacer(clock=lambda: now[0])
    event = OversleepingEvent()
    for _ in range(96):
        began = now[0]
        now[0] += .001
        pacer.wait(event, 1, began)
    assert .4 <= now[0] < .43
    assert 10 < len(waits) < 30
    now[0] += 1
    began = now[0]
    now[0] += .001
    pacer.wait(event, 1, began)
    assert pacer.deadline == pytest.approx(began + TIMESTEP)
    began = now[0]
    now[0] += .001
    pacer.wait(event, 2, began)
    assert pacer.deadline == pytest.approx(began + TIMESTEP)
    pacer.reset()
    assert pacer.deadline is None and pacer.revision is None


def test_policy_pacer_wait_is_interruptible_by_stop():
    import threading
    from backend.worker import PolicyPacer
    stop = threading.Event()
    stop.set()
    pacer = PolicyPacer(clock=lambda: 10.)
    pacer.wait(stop, 1, 10.)
    assert stop.is_set()


async def test_policy_timing_probe_completes_known_valid_trajectories(tmp_path):
    from scripts.benchmark_control import benchmark_policy_timing
    report = await benchmark_policy_timing(tmp_path / "timing", repeats=3)
    assert report["model_calls"] == 0
    assert report["completed"] == 3, report["trials"]
    assert all(trial["actual_steps"] == trial["accepted_steps"] == 96 for trial in report["trials"])
    assert all(trial["lease_s"] == pytest.approx(.55, abs=.002) for trial in report["trials"])
    assert all(trial["max_joint_target_error_rad"] < .005 for trial in report["trials"])
    assert report["timing"]["cost_s"]["policy_tick"]["count"] >= 288


def test_policy_contract_rejects_unknown_axes_nonfinite_and_invalid_limits():
    ticket = PolicyTicket(run_id="episode", episode_epoch=1, revision=2, observation_seq=3)
    action = [*NEUTRAL, .1]
    chunk = PolicyChunk(ticket=ticket, actions=[action] * 30)
    assert len(future_actions(chunk, ticket, 10, now=10.2)) == 20
    with pytest.raises(ValueError, match="Stale"):
        future_actions(chunk, ticket.model_copy(update={"revision": 3}), 10, now=10.2)
    with pytest.raises(ValueError, match="Stale"):
        future_actions(chunk, ticket, 10, now=11.1)
    with pytest.raises(ValueError, match="no future"):
        future_actions(PolicyChunk(ticket=ticket, actions=[action]), ticket, 10, now=10.2)
    for invalid in ([0] * 6, [*NEUTRAL, float("nan")], [99, *NEUTRAL[1:], .1], [*NEUTRAL, .2]):
        with pytest.raises(ValidationError):
            PolicyChunk(ticket=ticket, actions=[invalid])


def test_policy_configuration_is_local_and_requires_exact_milo_embodiment():
    for endpoint in ("https://example.com", "http://localhost/path", "http://user:secret@localhost", "http://localhost?x=1"):
        with pytest.raises(ValidationError):
            PolicyConfig(endpoint=endpoint)
    assert PolicyConfig(endpoint="http://127.0.0.1:8085/").endpoint == "http://127.0.0.1:8085"
    metadata = dict(backend="smolvla", checkpoint="local-trained-checkpoint", embodiment="milo-left-arm-v1",
                    action_names=ACTION_NAMES, camera_key="observation.images.head", fps=20,
                    trained_for_milo=True, skills=["pick_place"])
    assert PolicyMetadata(**metadata).fps == 20
    with pytest.raises(ValidationError):
        PolicyMetadata(**{**metadata, "action_names": ACTION_NAMES[::-1]})
    with pytest.raises(ValidationError):
        PolicyMetadata(**{**metadata, "embodiment": "so100"})


def trained_metadata():
    return PolicyMetadata(backend="smolvla", checkpoint="scripted-test-only", embodiment="milo-left-arm-v1",
        action_names=ACTION_NAMES, camera_key="observation.images.head", fps=20, trained_for_milo=True, skills=["pick_place"])


@pytest.mark.parametrize("status", ["ready", "unavailable", "incompatible", "untrained"])
async def test_policy_readiness_is_explicit_and_never_runs_inference(status):
    from backend.policy import PolicyConnectionError, check_policy_readiness
    closed = []

    class Client:
        async def describe(self):
            if status in {"unavailable", "incompatible"}:
                raise PolicyConnectionError(status, f"Policy {status}")
            metadata = trained_metadata()
            metadata.trained_for_milo = status != "untrained"
            return metadata

        async def predict(self, *args):
            raise AssertionError("Readiness must not invoke the policy")

        async def close(self):
            closed.append(True)

    result = await check_policy_readiness(PolicyConfig(), lambda config: Client())
    assert result["ready"] == (status == "ready")
    assert result["status"] == status and closed == [True]
    assert (result["metadata"] is not None) == (status == "ready")


async def test_offline_arm_policy_explains_navigation_checkpoint_mismatch():
    import httpx
    from backend.policy import LocalPolicyClient, check_policy_readiness
    client = LocalPolicyClient(PolicyConfig())
    await client.close()

    def offline(request):
        assert request.url.path == "/policy"
        raise httpx.ConnectError("No listener", request=request)

    client.client = httpx.AsyncClient(base_url="http://127.0.0.1:8085", transport=httpx.MockTransport(offline))
    result = await check_policy_readiness(PolicyConfig(), lambda config: client)
    assert not result["ready"] and result["status"] == "unavailable"
    assert result["metadata"] is None
    assert "milo-left-arm-v1" in result["message"] and "not a navigation checkpoint" in result["message"]
    assert "CLI-only" in result["message"] and "selected LLM" in result["message"]
    assert client.client.is_closed


def test_skill_executor_tracks_bounded_real_arm_motion_and_rejects_stale_chunks():
    import numpy as np
    import pybullet as bullet
    from backend.policy import SkillRuntime, StartSkill
    from backend.simulation import BulletSimulation, MotionError
    now = [10.]
    sim = BulletSimulation(width=160, height=120)
    runtime = SkillRuntime(trained_metadata(), clock=lambda: now[0])
    try:
        runtime.apply(sim, "start_skill", StartSkill(expected_revision=0, skill="pick_place", instruction="Test motion"))
        initial = runtime.position.copy()
        ticket = runtime.ticket(sim, sim.observe())
        targets = []
        for index in range(20):
            value = initial.copy()
            value[0] += .001 * (index + 1)
            targets.append(value.tolist())
        chunk = PolicyChunk(ticket=ticket, actions=targets)
        runtime.accept(sim, chunk)
        for _ in range(120):
            now[0] += 1 / 240
            runtime.tick(sim)
        assert runtime.status == "running", runtime.reason
        assert sim.ticks == 120
        actual = bullet.getJointState(sim.robot, sim.arms["left"][0], physicsClientId=sim.client)[0]
        assert actual > initial[0] + .005
        assert np.linalg.norm(sim.odometry[:2]) < .005
        runtime.cancel(sim)
        with pytest.raises(MotionError, match="cancelled"):
            runtime.accept(sim, chunk)
        assert not runtime.buffer
    finally:
        sim.close()


def test_policy_lease_still_expires_before_any_late_tick_and_invalidates_pending_response():
    from backend.policy import SkillRuntime, StartSkill
    from backend.simulation import BulletSimulation, MotionError
    now = [10.]
    sim = BulletSimulation(width=160, height=120)
    runtime = SkillRuntime(trained_metadata(), clock=lambda: now[0])
    try:
        runtime.apply(sim, "start_skill", StartSkill(expected_revision=0, skill="pick_place", instruction="Lease test"))
        ticket = runtime.ticket(sim, sim.observe(render=False))
        chunk = PolicyChunk(ticket=ticket, actions=[runtime.position.tolist()] * 8)
        runtime.accept(sim, chunk)
        assert runtime.expires_at == pytest.approx(10.55)
        revision = runtime.motion_revision
        now[0] = 10.56
        runtime.tick(sim)
        assert sim.ticks == 0 and not runtime.buffer and runtime.pending is None
        assert runtime.status == "awaiting_policy" and "expired" in runtime.reason
        assert runtime.motion_revision > revision
        with pytest.raises(MotionError, match="cancelled or expired"):
            runtime.accept(sim, chunk)
    finally:
        sim.close()


def test_skill_rejects_base_checkpoint_and_unsafe_chunk_without_moving():
    from backend.policy import SkillRuntime, StartSkill
    from backend.simulation import BulletSimulation, MotionError
    sim = BulletSimulation(width=160, height=120)
    try:
        metadata = trained_metadata()
        metadata.trained_for_milo = False
        runtime = SkillRuntime(metadata, clock=lambda: 10.)
        start = StartSkill(expected_revision=0, skill="pick_place", instruction="Test")
        with pytest.raises(MotionError, match="Milo-trained"):
            runtime.apply(sim, "start_skill", start)
        metadata.trained_for_milo = True
        runtime.apply(sim, "start_skill", start)
        ticket = runtime.ticket(sim, sim.observe())
        action = runtime.position.copy()
        action[0] += .2
        with pytest.raises(MotionError, match="speed"):
            runtime.accept(sim, PolicyChunk(ticket=ticket, actions=[action.tolist()] * 20))
        assert sim.ticks == 0 and not runtime.buffer
    finally:
        sim.close()


async def test_policy_worker_uses_snapshot_renderer_and_rejects_late_response_after_stop():
    import asyncio
    from backend.contracts import Command
    from backend.worker import SimulationWorker
    from backend.simulation import MotionError
    worker = SimulationWorker(pace=True)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.begin_skill_mode(trained_metadata(), worker.stop_revision)
        observation, _ = await worker.feedback()
        command = Command(run_id=observation.run_id, episode_epoch=observation.episode_epoch,
            observation_seq=observation.seq, action_id="skill-start", tool="start_skill",
            arguments={"expected_revision": 0, "skill": "pick_place", "instruction": "Scripted test"})
        assert (await worker.execute_skill(command)).status == "ok"
        ticket, observation, image = await worker.policy_feedback()
        assert observation.skill.status == "awaiting_policy"
        assert image.startswith(b"\x89PNG")
        before = observation.seq
        async with asyncio.timeout(4):
            while observation.seq <= before:
                await asyncio.sleep(.05)
                _, observation, image = await worker.policy_feedback()
        assert worker.renderer.process.is_alive()
        worker.stop()
        with pytest.raises(MotionError):
            await worker.accept_policy(PolicyChunk(ticket=ticket, actions=[[*NEUTRAL, .11]] * 30))
    finally:
        renderer = worker.renderer
        await worker.close()
        if renderer:
            assert not renderer.process.is_alive()


async def test_policy_http_client_sends_only_agent_observation_and_camera():
    import base64
    import json
    import httpx
    from backend.policy import LocalPolicyClient
    from backend.simulation import BulletSimulation
    sim = BulletSimulation(width=160, height=120)
    client = LocalPolicyClient(PolicyConfig())
    await client.close()
    observation = sim.observe()
    ticket = PolicyTicket(run_id=sim.run_id, episode_epoch=0, revision=1, observation_seq=observation.seq)

    def handle(request):
        if request.url.path == "/policy":
            return httpx.Response(200, json=trained_metadata().model_dump())
        data = json.loads(request.content)
        assert set(data) == {"ticket", "observation", "image", "instruction"}
        assert not any(key in data["observation"] for key in ("objects", "geometry", "snapshot", "robot_position", "challenge"))
        assert base64.b64decode(data["image"]) == sim.frame(observation.frame_ref)
        return httpx.Response(200, json={"ticket": data["ticket"], "actions": [[*NEUTRAL, .1]] * 30})

    client.client = httpx.AsyncClient(base_url="http://127.0.0.1:8085", transport=httpx.MockTransport(handle))
    try:
        assert (await client.describe()).trained_for_milo
        chunk = await client.predict(ticket, observation, sim.frame(observation.frame_ref), "Pick and place")
        assert chunk.ticket == ticket and len(chunk.actions) == 30
    finally:
        await client.close()
        sim.close()


async def test_supervisor_skill_tools_and_policy_run_independently_and_stop_together():
    import asyncio
    import json
    from backend.agent import AgentController, AgentStart, FoundryConfig, robot_tools
    from backend.worker import SimulationWorker
    from tests.test_agent import model_response
    moving = asyncio.Event()
    policy_calls = []
    supervisor_observations = []

    class ScriptedPolicy:
        async def describe(self):
            return trained_metadata()

        async def predict(self, ticket, observation, image, instruction):
            policy_calls.append(ticket)
            values = {joint.name: joint.position for joint in observation.joints}
            current = [values[name] for name in ACTION_NAMES[:6]] + [observation.grippers["left"].aperture_m]
            current[6] = max(0, min(.11, current[6]))
            await asyncio.sleep(.03)
            return PolicyChunk(ticket=ticket, actions=[
                [current[0] + .0004 * (index + 1), *current[1:]] for index in range(50)])

        async def close(self):
            pass

    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            observation = json.loads(inputs[-1]["content"][0]["text"])
            supervisor_observations.append(observation)
            if len(supervisor_observations) == 1:
                return model_response("start_skill", json.dumps({"expected_revision": observation["skill"]["revision"],
                    "skill": "pick_place", "instruction": "Scripted hold, not a learned task"}), call_id="skill")
            async with asyncio.timeout(5):
                while worker.latest["snapshot"]["simulated_time_s"] < .2:
                    await asyncio.sleep(.02)
            moving.set()
            return model_response("complete_skill", json.dumps({"expected_revision": observation["skill"]["revision"],
                "reason": "Scripted completion assessment"}), call_id="finish")

        async def close(self):
            pass

    worker = SimulationWorker(pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"),
        lambda config: Supervisor(), policy_factory=lambda config: ScriptedPolicy())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=0, goal="Test",
            execution_mode="supervised_policy", feedback_interval_s=.25))
        await asyncio.wait_for(controller.task, 12)
        assert controller.state["phase"] == "completed", controller.state
        assert moving.is_set() and policy_calls
        import pybullet as bullet
        actual = await worker.call(lambda sim: bullet.getJointState(sim.robot, sim.arms["left"][0], physicsClientId=sim.client)[0])
        initial = next(joint["position"] for joint in supervisor_observations[0]["joints"] if joint["name"] == "left_joint_1")
        assert actual > initial + .0005
        assert worker.latest["skill"]["status"] == "completed"
        assert worker.latest["skill"]["completion_source"] == "supervisor"
        assert worker.latest["stopped"] and not worker.renderer
        assert worker.latest["observation"]["seq"] == worker.sim.seq
        assert {tool["name"] for tool in robot_tools("supervised_policy")} == {
            "start_skill", "cancel_skill", "complete_skill", "observe", "stop"}
        assert any(event["kind"] == "policy" for event in controller.trace()["events"])
    finally:
        await controller.halt()
        await worker.close()


async def test_policy_server_checks_epoch_age_and_checkpoint_gate():
    import base64
    import httpx
    from backend.smolvla_server import create_app, state_vector
    from backend.simulation import BulletSimulation
    sim = BulletSimulation(width=160, height=120)

    class Backend:
        metadata = trained_metadata()

        def predict(self, request):
            assert len(state_vector(request.observation)) == 7
            return PolicyChunk(ticket=request.ticket, actions=[[*NEUTRAL, .1]] * 30)

    backend = Backend()
    observation = sim.observe()
    ticket = PolicyTicket(run_id=sim.run_id, episode_epoch=0, revision=1, observation_seq=observation.seq)
    payload = {"ticket": ticket.model_dump(), "observation": observation.model_dump(),
               "image": base64.b64encode(sim.frame(observation.frame_ref)).decode(), "instruction": "Test"}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(backend)), base_url="http://local") as client:
            assert (await client.get("/policy")).json()["embodiment"] == "milo-left-arm-v1"
            assert (await client.post("/predict", json=payload)).status_code == 200
            assert (await client.post("/predict", json=payload, headers={"Origin": "https://elsewhere.test"})).status_code == 403
            backend.metadata.trained_for_milo = False
            assert (await client.post("/predict", json=payload)).status_code == 409
            backend.metadata.trained_for_milo = True
            payload["observation"]["wall_timestamp"] = 0
            assert (await client.post("/predict", json=payload)).status_code == 409
            payload["ticket"]["episode_epoch"] = 99
            assert (await client.post("/predict", json=payload)).status_code == 422
    finally:
        sim.close()


class BrowserPolicy:
    async def describe(self):
        return trained_metadata()

    async def predict(self, ticket, observation, image, instruction):
        import asyncio
        positions = {joint.name: joint.position for joint in observation.joints}
        current = [positions[name] for name in ACTION_NAMES[:6]] + [max(0, min(.11, observation.grippers["left"].aperture_m))]
        await asyncio.sleep(.03)
        return PolicyChunk(ticket=ticket, actions=[
            [current[0] + .0004 * (index + 1), *current[1:]] for index in range(50)])

    async def close(self):
        pass


@pytest.mark.parametrize("fault", ["stop", "timeout", "inference", "bad_chunk"])
async def test_supervised_policy_failures_cancel_pending_supervisor_and_motion(fault):
    import asyncio
    import json
    from backend.agent import AgentController, AgentStart, FoundryConfig
    from backend.worker import SimulationWorker
    from tests.test_agent import model_response
    entered, released, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    model_calls = 0

    class Policy(BrowserPolicy):
        async def predict(self, ticket, observation, image, instruction):
            entered.set()
            await released.wait()
            if fault == "inference":
                raise RuntimeError("Scripted inference failure")
            chunk = await super().predict(ticket, observation, image, instruction)
            if fault == "bad_chunk":
                for action in chunk.actions:
                    action[0] += .5
            return chunk

    class Supervisor:
        async def respond(self, profile, reasoning, goal, inputs):
            nonlocal model_calls
            model_calls += 1
            if model_calls == 1:
                state = json.loads(inputs[-1]["content"][0]["text"])["skill"]
                return model_response("start_skill", json.dumps({"expected_revision": state["revision"],
                    "skill": "pick_place", "instruction": "Scripted test"}), call_id="skill")
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        async def close(self):
            pass

    worker = SimulationWorker(pace=True)
    controller = AgentController(FoundryConfig(endpoint="https://test.openai.azure.com"),
        lambda config: Supervisor(), policy_factory=lambda config: Policy())
    try:
        await asyncio.wrap_future(worker.ready)
        controller.start(worker, AgentStart(run_id=worker.sim.run_id, episode_epoch=0, goal="Test",
            execution_mode="supervised_policy", feedback_interval_s=.25))
        await asyncio.wait_for(entered.wait(), 8)
        async with asyncio.timeout(3):
            while model_calls < 2:
                await asyncio.sleep(.02)
        if fault == "stop":
            await controller.halt("Test takeover")
        elif fault == "timeout":
            await worker.call(lambda sim: setattr(worker.skill, "deadline", 0))
        else:
            released.set()
        await asyncio.wait_for(controller.task, 5)
        assert cancelled.is_set()
        assert worker.latest["stopped"] and worker.latest["skill"]["remaining_s"] == 0
        assert not worker.renderer
        assert controller.state["phase"] == ("stopped" if fault == "stop" else "error")
    finally:
        await controller.halt()
        await worker.close()


async def test_policy_motion_never_calls_the_physics_threads_camera_renderer():
    import asyncio
    from backend.contracts import Command
    from backend.worker import SimulationWorker
    worker = SimulationWorker(pace=True)
    try:
        await asyncio.wrap_future(worker.ready)
        await worker.begin_skill_mode(trained_metadata(), worker.stop_revision)

        def forbid_capture(sim):
            def fail():
                raise AssertionError("Physics worker must not render during policy motion")
            sim.capture = fail

        await worker.call(forbid_capture)
        observation, _ = await worker.feedback()
        result = await worker.execute_skill(Command(run_id=observation.run_id, episode_epoch=0,
            observation_seq=observation.seq, action_id="no-render", tool="start_skill",
            arguments={"expected_revision": 0, "skill": "pick_place", "instruction": "Scripted test"}))
        assert result.status == "ok"
        ticket, observation, image = await worker.policy_feedback()
        positions = {joint.name: joint.position for joint in observation.joints}
        action = [positions[name] for name in ACTION_NAMES[:6]] + [.11]
        await worker.accept_policy(PolicyChunk(ticket=ticket, actions=[action] * 50))
        async with asyncio.timeout(3):
            while worker.latest["snapshot"]["simulated_time_s"] < .2:
                assert worker.latest["skill"]["status"] != "failed", worker.latest["skill"]
                await asyncio.sleep(.02)
        _, image = await worker.feedback()
        assert image.startswith(b"\x89PNG")
    finally:
        await worker.close()