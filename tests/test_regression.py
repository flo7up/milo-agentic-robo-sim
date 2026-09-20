import pytest

from backend.regression import CASES, SUITE_ID, RegressionStart, grade_progress, score_case


def test_observable_baseline_has_fixed_distinct_challenges_and_budgets():
    assert SUITE_ID == "observable-navigation-v2"
    assert [case["id"] for case in CASES] == ["parking", "parking-left", "parking-right", "parking-far",
        "table-clockwise", "table-counterclockwise", "object-search", "room-arrival"]
    assert sum(case["budget_s"] for case in CASES) == 1200
    assert len({case["challenge_id"] for case in CASES}) == 7
    default = RegressionStart(run_id="run", episode_epoch=0)
    assert (default.model_id, default.reasoning, default.task_supervisor_model_id) == ("qwen", "none", "luna")
    assert RegressionStart(run_id="run", episode_epoch=0, model_id="luna").task_supervisor_model_id is None
    assert RegressionStart(run_id="run", episode_epoch=0, task_supervisor_model_id=None).task_supervisor_model_id is None
    hybrid = RegressionStart(run_id="run", episode_epoch=0, model_id="qwen", reasoning="none", task_supervisor_model_id="luna")
    assert hybrid.task_supervisor_model_id == "luna"
    with pytest.raises(ValueError):
        RegressionStart(run_id="run", episode_epoch=0, model_id="luna", task_supervisor_model_id="luna")
    with pytest.raises(ValueError):
        RegressionStart(run_id="run", episode_epoch=0, model_id="unknown")


def test_parking_variants_keep_the_canonical_scene_goal_and_scoring():
    from backend.challenges import ChallengeLoad, get_challenge

    canonical = get_challenge("park").model_dump()
    expected_starts = {"park": [0., 0.], "park_left": [0., .30],
        "park_right": [0., -.30], "park_far": [-.65, 0.]}
    for case in CASES[:4]:
        identifier = ChallengeLoad(challenge_id=case["challenge_id"]).challenge_id
        challenge = get_challenge(identifier)
        assert challenge.initial_xy == expected_starts[identifier]
        assert case["budget_s"] == 120
        assert {key: value for key, value in challenge.model_dump().items()
            if key not in {"id", "title", "initial_xy"}} == {key: value for key, value in canonical.items()
            if key not in {"id", "title", "initial_xy"}}
        assert "initial_xy" not in challenge.public()
        challenge.initial_xy[0] += 1.
        challenge.objectives[0].size[0] += 1.
        assert get_challenge(identifier).initial_xy == expected_starts[identifier]
    assert get_challenge("park").model_dump() == canonical


@pytest.mark.parametrize("fault,expected", [(None, "passed"), ("incomplete", "invalid"), ("assisted", "invalid"),
    ("drops", "invalid"), ("contact", "failed"), ("physical", "failed"), ("claim", "failed"), ("writer", "invalid")])
def test_baseline_pass_requires_independent_complete_unassisted_recording(fault, expected):
    score = {"complete_recording": fault != "incomplete", "operator_assisted": fault == "assisted",
        "dropped_records": int(fault == "drops"), "contact_episodes": int(fault == "contact"),
        "final_physics_success": fault != "physical"}
    challenge = {"status": "in_progress" if fault == "claim" else "completed"}
    assert score_case(challenge, score, "writer failed" if fault == "writer" else None) == expected


@pytest.mark.parametrize("initial_gap,final_gap,progress,remaining", [
    (2., 1., 50., 50.), (2., 3., 0., 150.), (2., 0., 100., 0.),
    (2., 2., 0., 100.), (0., 0., 100., None), (0., 1., 0., None)])
def test_progress_grades_remaining_gap_without_claiming_success(initial_gap, final_gap, progress, remaining):
    initial = {"kind": "arrival", "remaining_m": initial_gap}
    final = {"kind": "arrival", "remaining_m": final_gap, "physics_complete": False}
    grade = grade_progress(initial, final)
    assert grade["progress_pct"] == pytest.approx(progress)
    assert grade["remaining_pct"] == remaining
    assert grade["remaining_m"] == final_gap and not grade["physics_complete"]
    assert grade["evaluation_only"] and "progress_pct" not in final


@pytest.mark.parametrize("degrees,expected", [(0., 0.), (90., 25.), (270., 75.), (400., 100.)])
def test_circuit_progress_uses_valid_sweep_not_distance_to_start(degrees, expected):
    grade = grade_progress({"kind": "circuit"}, {"kind": "circuit", "valid_degrees": degrees,
        "return_error_m": 0., "physics_complete": False})
    assert grade["progress_pct"] == pytest.approx(expected)
    assert grade["remaining_degrees"] == pytest.approx(360. - min(360., degrees))
    assert not grade["physics_complete"]
    assert grade_progress(None, None) is None


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_progress_snapshots_read_real_physics_without_mutating_state(case):
    import copy
    import json
    from backend.challenges import furniture_circuit, get_challenge
    from backend.contracts import AgentObservation
    from backend.regression import evaluation_snapshot
    from backend.simulation import BulletSimulation

    challenge = (furniture_circuit(case["orbit_target"], case["orbit_direction"])
        if case.get("orbit_target") else get_challenge(case["challenge_id"]))
    sim = BulletSimulation(challenge=challenge, width=160, height=120, rendering="tiny")
    try:
        before, status, sequence, ticks = sim.snapshot(), copy.deepcopy(sim.challenge_status()), sim.seq, sim.ticks
        initial = evaluation_snapshot(sim)
        final = evaluation_snapshot(sim)
        grade = grade_progress(initial, final)
        assert json.loads(json.dumps(grade, allow_nan=False)) == grade
        assert grade["progress_pct"] == 0. and not grade["physics_complete"]
        assert grade["target_xy_m"] and grade["checks"]
        if case["id"].startswith("parking"):
            import pybullet as bullet
            assert grade["position_xy_m"] == pytest.approx(challenge.initial_xy, abs=.002)
            floor = next(item["id"] for item in sim.objects if item["name"] == "floor")
            assert not [contact for contact in bullet.getContactPoints(sim.robot, physicsClientId=sim.client)
                if contact[2] != floor and contact[9] > .01]
        assert sim.snapshot() == before and sim.challenge_status() == status
        assert (sim.seq, sim.ticks) == (sequence, ticks)
        assert not {"evaluation", "target_xy_m", "remaining_m", "valid_degrees"} & AgentObservation.model_fields.keys()
    finally:
        sim.close()


def test_arrival_grade_measures_full_footprint_gap_not_target_centre():
    import pybullet as bullet
    from backend.challenges import get_challenge
    from backend.regression import evaluation_snapshot
    from backend.simulation import BulletSimulation

    sim = BulletSimulation(challenge=get_challenge("park"), width=160, height=120, rendering="tiny")
    try:
        initial = evaluation_snapshot(sim)
        position, orientation = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)
        centre = initial["target_xy_m"]
        bullet.resetBasePositionAndOrientation(sim.robot, [centre[0] - .03, centre[1], position[2]], orientation,
            physicsClientId=sim.client)
        final = evaluation_snapshot(sim)
        assert final["remaining_m"] == 0. and final["center_distance_m"] == pytest.approx(.03)
        assert final["checks"][0]["complete"]
        assert grade_progress(initial, final)["progress_pct"] == 100.
        assert not final["physics_complete"]
    finally:
        sim.close()


def test_regression_trajectory_is_bounded_allowlisted_and_segmented(tmp_path):
    import json
    from backend.regression import RegressionSequence
    sequence=RegressionSequence();sequence.directory=tmp_path.resolve();sequence.state.update(sequence_id="sequence",current_index=0)
    recording=tmp_path/"case"/"recording";recording.mkdir(parents=True)
    sequence.state["cases"][0].update(status="running",recording_directory=str(recording))
    samples=[
        {"position_m":[0.,0.,0.],"wall_s":0.,"manual_placements":0,"collisions":[]},
        {"position_m":[.3,.4,0.],"wall_s":1.,"manual_placements":0,"collisions":[{"force_n":1}]},
        {"position_m":[1.,1.,0.],"wall_s":2.,"manual_placements":1,"collisions":[]}]
    (recording/"trajectory.jsonl").write_text("".join(json.dumps(sample)+"\n" for sample in samples),encoding="utf-8")
    route=sequence.trajectory("parking")
    assert route["sample_count"]==3 and route["distance_m"]==pytest.approx(.5)
    assert [point["segment"] for point in route["points"]]==[0,0,1] and len(route["contacts"])==1
    assert sequence.public()["cases"][0]["trajectory_url"].endswith("/parking/trajectory")
    with pytest.raises(ValueError,match="Unknown"):
        sequence.trajectory("missing")
    outside=tmp_path.parent/"outside-recording";outside.mkdir(exist_ok=True)
    sequence.state["cases"][0]["recording_directory"]=str(outside)
    with pytest.raises(ValueError,match="outside"):
        sequence.trajectory("parking")


def test_latest_completed_regression_restores_without_resuming(tmp_path):
    import json
    from backend.regression import RegressionSequence
    directory=tmp_path/f"regression-{SUITE_ID}-20260919T000000Z-restored";directory.mkdir()
    state={"suite_id":SUITE_ID,"sequence_id":"restored","phase":"completed","finished_at":"2026-09-19T00:00:00Z",
        "cases":[{**case,"status":"failed"} for case in CASES]}
    (directory/"sequence.json").write_text(json.dumps({**state,"active":False}),encoding="utf-8")
    sequence=RegressionSequence()
    assert sequence.restore_latest(tmp_path) and sequence.public()["sequence_id"]=="restored"
    assert not sequence.active and all("trajectory_url" in case for case in sequence.public()["cases"])


@pytest.mark.parametrize("contents", [None, "", "{"])
def test_regression_path_api_handles_unavailable_recordings_and_recovers(tmp_path, monkeypatch, contents):
    import json
    from fastapi.testclient import TestClient
    from backend.app import app, lab
    from backend.regression import RegressionSequence

    monkeypatch.setenv("MILO_RENDERER", "tiny")
    with TestClient(app) as client:
        sequence = RegressionSequence()
        sequence.directory = tmp_path
        recording = tmp_path / "recording"
        recording.mkdir()
        sequence.state["cases"][0].update(status="running", recording_directory=str(recording))
        monkeypatch.setattr(lab, "regression", sequence)
        trajectory = recording / "trajectory.jsonl"
        if contents is not None:
            trajectory.write_text(contents, encoding="utf-8")
        initial = client.get("/api/state").json()["snapshot"]
        endpoint = "/api/regression/cases/parking/trajectory"
        response = client.get(endpoint)
        assert response.status_code == 404
        assert response.json() == {"detail": "Regression path unavailable"}
        assert client.get("/api/regression/cases/unknown/trajectory").status_code == 404
        trajectory.write_text(json.dumps({"position_m": [0., 0., 0.], "wall_s": 0., "collisions": []}) + "\n",
            encoding="utf-8")
        response = client.get(endpoint)
        assert response.status_code == 200 and response.json()["sample_count"] == 1
        assert response.headers["cache-control"] == "no-store"
        assert client.get("/api/state").json()["snapshot"] == initial


def test_previous_suite_results_are_retained_without_mixing_with_expanded_suite(tmp_path):
    import json
    from backend.regression import RegressionSequence

    directory = tmp_path / "regression-observable-navigation-v1-20260919T000000Z-legacy"
    directory.mkdir()
    report = directory / "sequence.json"
    original = json.dumps({"suite_id": "observable-navigation-v1", "sequence_id": "legacy",
        "phase": "completed", "finished_at": "2026-09-19T00:00:00Z",
        "cases": [{**case, "status": "failed"} for case in CASES if not case["id"].startswith("parking-")]})
    report.write_text(original, encoding="utf-8")
    sequence = RegressionSequence()
    assert not sequence.restore_latest(tmp_path)
    assert sequence.public()["suite_id"] == SUITE_ID
    assert all(case["status"] == "pending" for case in sequence.public()["cases"])
    assert report.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("interrupt", ["stop", "disconnect"])
async def test_sequence_cancel_during_scene_load_never_starts_next_agent(tmp_path, monkeypatch, interrupt):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from backend.agent import FoundryConfig
    from backend.regression import RegressionSequence
    monkeypatch.setattr("backend.regression.source_hashes", lambda: {"frozen": "hash"})
    sequence = RegressionSequence()
    entered, release = asyncio.Event(), asyncio.Event()
    worker = SimpleNamespace(latest={"run_id": "old", "busy": False}, epoch=1, powered=True, rendering="test", stop=Mock())
    agent = SimpleNamespace(config=FoundryConfig(endpoint="https://test.openai.azure.com"), active=False,
        recording_evidence="scripted_test", halt=AsyncMock(), record_sessions=False, recording_root=None)
    async def reset(*args, **kwargs):
        entered.set()
        await release.wait()
        assert kwargs["reuse_saved_map"] is False
    lab = SimpleNamespace(worker=worker, agent=agent, connections=1, robot_on=True, lock=asyncio.Lock(),
        recording_status=lambda: {"active": False}, interaction_mode="chat", recording_enabled=False,
        recording_root=tmp_path, reset=reset)
    start_agent = AsyncMock()
    sequence.start(lab, RegressionStart(run_id="old", episode_epoch=1), start_agent)
    await entered.wait()
    if interrupt == "disconnect":
        lab.connections = 0
    sequence.cancel(lab, interrupt)
    release.set()
    await sequence.wait()
    start_agent.assert_not_called()
    assert sequence.public()["phase"] == "cancelled" and not sequence.active
    assert sequence.state["cases"][0]["status"] == "cancelled"
    assert all(case["status"] == "not_run" for case in sequence.state["cases"][1:])
    assert lab.recording_enabled is False and lab.recording_root == tmp_path
    worker.stop.assert_called()


def test_regression_api_rejects_unconnected_stale_and_cross_origin_start(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app import app
    monkeypatch.setenv("MILO_RENDERER", "tiny")
    with TestClient(app) as client:
        state = client.get("/api/state").json()
        payload = {"run_id": state["run_id"], "episode_epoch": state["episode_epoch"]}
        assert client.get("/api/regression").json()["suite_id"] == SUITE_ID
        assert client.post("/api/regression/start", json=payload).status_code == 409
        assert client.post("/api/regression/start", json=payload, headers={"Origin": "https://foreign.invalid"}).status_code == 403
        assert client.post("/api/regression/start", json={**payload, "episode_epoch": 999}).status_code == 409
        assert client.get("/api/state").json()["snapshot"] == state["snapshot"]


def test_regression_exclusive_ownership_blocks_settings_and_commands(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app import app, lab
    from backend.regression import RegressionSequence
    monkeypatch.setenv("MILO_RENDERER", "tiny")
    with TestClient(app) as client:
        phase=lab.regression.state["phase"]
        with monkeypatch.context() as active:
            active.setattr(RegressionSequence, "active", property(lambda self: True))
            for endpoint in ("agent/rate", "agent/start", "mission/start", "agent/config", "command", "preferences", "memory", "regression/start"):
                result = client.post(f"/api/{endpoint}", json={})
                assert result.status_code == 409 and "regression" in result.json()["detail"]
        assert lab.regression.state["phase"] == phase and not lab.regression.active


@pytest.mark.parametrize("ending", ["complete", "between_stop", "source_drift", "recording_failure"])
async def test_sequence_serializes_cases_retains_failures_and_never_restarts_after_interrupt(tmp_path, monkeypatch, ending):
    import asyncio
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock
    from backend.agent import FoundryConfig
    from backend.challenges import get_challenge, furniture_circuit
    from backend.regression import RegressionSequence
    source = {"frozen": "hash"}
    monkeypatch.setattr("backend.regression.source_hashes", lambda: dict(source))
    sequence = RegressionSequence()
    worker = SimpleNamespace(latest={"run_id": "old", "busy": False}, epoch=1, powered=True, rendering="test", stop=Mock(),
        call=AsyncMock(return_value={"kind": "arrival", "remaining_m": 1., "physics_complete": False}))
    agent = SimpleNamespace(config=FoundryConfig(endpoint="https://test.openai.azure.com"), active=False,
        recording_evidence="scripted_test", halt=AsyncMock(), record_sessions=False, recording_root=None)
    calls = []
    async def reset(identifier, target, direction, **kwargs):
        assert not calls or lab.agent.task.done()
        assert kwargs["reuse_saved_map"] is False
        worker.epoch += 1
        worker.latest = {"run_id": f"run-{worker.epoch}", "busy": False, "challenge": {"status": "in_progress"}}
        worker.challenge = furniture_circuit(target, direction) if identifier == "furniture_circuit" else get_challenge(identifier)
        agent.recording_directory = tmp_path / f"record-{worker.epoch}"
        agent.recording_directory.mkdir()
        agent.recording_error = None
        agent.state = {"session_id": f"session-{worker.epoch}", "input_tokens": 10, "output_tokens": 2,
            "outcome": {"kind": "limited", "message": "Scripted fixture"}, "error": None}
    async def start(settings):
        calls.append(settings)
        assert settings.goal == worker.challenge.goal and settings.mission_budget_s == CASES[len(calls)-1]["budget_s"]
        async def finish():
            score = {"complete_recording": ending != "recording_failure", "operator_assisted": False, "dropped_records": 0,
                "contact_episodes": 0, "final_physics_success": False}
            (agent.recording_directory / "scorecard.json").write_text(json.dumps(score))
            if ending == "source_drift":
                source["frozen"] = "changed"
        agent.task = asyncio.create_task(finish())
    async def between(seconds):
        assert seconds == 3.
        if ending == "between_stop":
            sequence.cancel(lab, "Stop between cases")
    monkeypatch.setattr("backend.regression.asyncio.sleep", between)
    lab = SimpleNamespace(worker=worker, agent=agent, connections=1, robot_on=True, lock=asyncio.Lock(),
        recording_status=lambda: {"active": False}, interaction_mode="chat", recording_enabled=False,
        recording_root=tmp_path, reset=reset)
    sequence.start(lab, RegressionStart(run_id="old", episode_epoch=1), start)
    await sequence.wait()
    assert len(calls) == (len(CASES) if ending == "complete" else 1)
    assert sequence.state["phase"] == ("completed" if ending == "complete" else "cancelled" if ending == "between_stop" else "error")
    assert sequence.state["cases"][0]["status"] == ("invalid" if ending in {"source_drift", "recording_failure"} else "failed")
    assert lab.recording_enabled is False and lab.recording_root == tmp_path
    saved = json.loads((sequence.directory / "sequence.json").read_text())
    assert saved["active"] is False and saved["finished_at"]
    assert "source_hashes" not in sequence.public() and saved["source_hashes"] == {"frozen": "hash"}
    assert saved["cases"][0]["evaluation"]["progress_pct"] == 0.
    assert saved["cases"][0]["evaluation"]["evaluation_only"]