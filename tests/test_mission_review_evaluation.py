import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.probe_mission_review import (EvaluationWorker, GOAL, ROOM_PROTOCOL, RoomEvaluator, SETTINGS,
    analyze, assess_report, room_results, self_check, stop_category, summarize)


def test_timing_uses_pose_and_exact_union_intersections():
    samples = [{"wall_s": index, "pose": [max(0, index - 1) * .1, 0, 0], "known_cells": 10 + index,
                "linear_speed_mps": .5, "inference_pending": False, "stop_reason": "thinking",
                "route_active": False} for index in range(3)]
    result = summarize(samples, [{"start_s": .5, "end_s": 1.5}, {"start_s": 1., "end_s": 1.75}])
    assert result["moving_s"] == 1.
    assert result["stationary_s"] == 1.
    assert result["inference_s"] == 1.25
    assert result["moving_during_inference_s"] == .75
    assert result["stationary_during_inference_s"] == .5
    assert result["stationary_inference_by_category_s"] == {"initial_authorization": .5}
    assert result["uncertain_s"] == 2.
    assert result["net_known_area_m2"] == pytest.approx(.02)
    assert result["net_free_area_m2"] is None


def test_probe_self_check_without_model_calls():
    assert self_check() == {"status": "passed", "model_calls": 0}


@pytest.mark.parametrize("reason,active,prior,expected", [
    ("CLEARANCE_STOP", False, False, "collision_related_guard"),
    ("contact", False, True, "collision_related_guard"),
    ("periodic_checkpoint", False, True, "periodic_checkpoint"),
    ("thinking", False, False, "initial_authorization"),
    ("thinking", False, True, "inference_without_active_authorization"),
    ("BUFFER_EXPIRED", True, True, "other_stop")])
def test_stop_categories_do_not_equate_inference_or_guards_with_collisions(reason, active, prior, expected):
    assert stop_category({"stop_reason": reason, "route_active": active}, prior, True) == expected


@pytest.mark.parametrize("text,expected", [
    ("I see a bathroom with a toilet and bathtub visible.", True),
    ("Find and identify the bathroom from visible toilet fixtures.", None),
    ("This is not a bathroom. I see a toilet.", None),
    ("I see a bathroom; maybe a toilet is visible.", None),
    ("I see a bathroom with a sink visible.", False),
    ("I see a bathroom with a toilet and sink visible.", False),
    ("Salle de bains avec baignoire.", None)])
def test_report_rubric_is_bounded_and_requires_independent_matching_fixtures(text, expected):
    frame = {"paired": True, "annotation_available": True, "image_verified": True, "visible_groups": ["toilet", "bathtub"]}
    result = assess_report(text, frame)
    assert result["correct_identification"] is expected
    assert assess_report(text, None)["correct_identification"] is None
    assert assess_report(text, {**frame, "image_verified": False})["correct_identification"] is None


def test_unknown_initial_annotation_never_implies_discovery(tmp_path):
    image = b"paired-image"
    (tmp_path / "paired.png").write_bytes(image)
    frame = {"sequence": 7, "wall_s": 2., "paired": True, "annotation_available": True,
        "image_file": "paired.png", "image_sha256": hashlib.sha256(image).hexdigest(), "visible_groups": ["toilet"], "initial": False}
    artifact = {"protocol": ROOM_PROTOCOL, "frames": [frame]}
    assert room_results(artifact, source=tmp_path)["discovery"]["value"] is None
    frame["initial"] = True
    measured = room_results(artifact, source=tmp_path)["discovery"]
    assert measured["value"] is True and measured["initial_visible"] is True and measured["newly_discovered"] is False
    frame["annotation_available"] = False
    assert room_results(artifact, source=tmp_path)["discovery"]["value"] is None
    assert all(room_results()[key]["value"] is None for key in ("discovery", "correct_identification", "arrival"))


def test_trace_identification_ignores_plans_goals_history_and_unpaired_images(tmp_path):
    image = b"exact-head-image"
    for name in ("paired.png", "frame-0002.png"):
        (tmp_path / name).write_bytes(image)
    frame = {"sequence": 7, "wall_s": 2., "simulated_s": 1., "run_id": "run", "episode_epoch": 0,
        "paired": True, "annotation_available": True, "image_file": "paired.png", "initial": False,
        "odometry_m_rad": [0., 0., 0.], "head_rad": [0., 0.],
        "image_sha256": hashlib.sha256(image).hexdigest(), "visible_groups": ["toilet"]}
    artifact = {"protocol": ROOM_PROTOCOL, "frames": [frame], "review_sources": [{"observation_seq": 8, "spatial_sequence": 7,
        "image_sha256": frame["image_sha256"], "run_id": "run", "episode_epoch": 0, "reviewed_at_unix_s": 100., "sensor_age_s": .1}]}
    text = "I see a bathroom with a toilet visible."
    trace = {"events": [{"id": 1, "kind": "session", "payload": {"goal": text}},
        {"id": 2, "kind": "feedback", "payload": {"image_roles": ["current_head", "observed_map"],
            "observation": {"seq": 8, "run_id": "run", "episode_epoch": 0, "simulated_time_s": 1.,
                "odometry_m_rad": [0., 0., 0.], "head_rad": [0., 0.]}, "history": text}},
        {"id": 3, "kind": "response", "timestamp": 101., "payload": {"status": "completed", "calls": [
            {"name": "guide_mission", "arguments": json.dumps({"action": "plan", "plan": {"target": text}})}]}}]}
    assert room_results(artifact, trace, tmp_path)["correct_identification"]["value"] is None
    trace["events"][-1]["payload"]["calls"][0]["arguments"] = json.dumps({"action": "report_observation", "evidence_text": text})
    result = room_results(artifact, trace, tmp_path)
    assert result["correct_identification"]["value"] is True
    assert result["discovery"]["value"] is None and result["arrival"]["value"] is None
    assert result["reports"][0]["feedback_event_id"] == 2
    for field, invalid in (("spatial_sequence", 99), ("sensor_age_s", 20.)):
        original = artifact["review_sources"][0][field]
        artifact["review_sources"][0][field] = invalid
        assert room_results(artifact, trace, tmp_path)["correct_identification"]["value"] is None
        artifact["review_sources"][0][field] = original
    trace["events"][-1]["timestamp"] = 116.
    assert room_results(artifact, trace, tmp_path)["correct_identification"]["value"] is None
    trace["events"][-1]["timestamp"] = 101.
    for field, invalid in (("simulated_time_s", 2.), ("head_rad", [1., 0.]), ("odometry_m_rad", [1., 0., 0.])):
        observation = trace["events"][1]["payload"]["observation"]
        original = observation[field]
        observation[field] = invalid
        assert room_results(artifact, trace, tmp_path)["correct_identification"]["value"] is None
        observation[field] = original
    (tmp_path / "frame-0002.png").write_bytes(b"different-head")
    result = room_results(artifact, trace, tmp_path)
    assert result["reports"][0]["semantic_claim"] is True
    assert result["correct_identification"]["value"] is None


@pytest.mark.parametrize("failure", [None, "gap", "contact", "moving", "frozen"])
def test_arrival_dwell_independent_of_semantics_and_receipts(failure):
    samples = [{"wall_s": index*.1, "simulated_s": index*.05, "eligible": True, "contact": False,
                "contact_episodes": 0} for index in range(12)]
    if failure == "gap":
        samples = samples[:4] + samples[7:]
    elif failure == "contact":
        samples[6]["contact"] = True
        samples[6]["contact_episodes"] = 1
    elif failure == "moving":
        samples[6]["eligible"] = False
    elif failure == "frozen":
        for sample in samples:
            sample["simulated_s"] = 0.
    result = room_results({"protocol": ROOM_PROTOCOL, "arrival_samples": samples})
    assert result["arrival"]["value"] is (failure is None)
    assert result["discovery"]["value"] is None and result["correct_identification"]["value"] is None


@pytest.mark.parametrize("failure", [False, True])
def test_initial_evaluator_rays_share_sensor_capture_and_restore_camera_hook(monkeypatch, failure):
    evaluator = RoomEvaluator()
    events = []
    original_pose = lambda: ("eye", "rotation")
    sim = SimpleNamespace(camera_pose=original_pose)

    def capture_spatial(sequence):
        events.append("sensor_timestamp")
        sim.camera_pose()
        sim.camera_pose()
        if failure:
            raise RuntimeError("Renderer unavailable")
        return "sensor", b"image"

    def capture(source, sequence, eye, rotation, *, initial=False):
        assert events == ["sensor_timestamp"]
        assert source is sim and sequence == 1 and initial
        assert (eye, rotation) == ("eye", "rotation")
        events.append("rays")

    sim.capture_spatial = capture_spatial
    monkeypatch.setattr(evaluator, "capture", capture)
    monkeypatch.setattr(evaluator, "receive", lambda sensor, image: events.append("received"))
    evaluator.capture_initial(sim)
    assert sim.camera_pose is original_pose
    assert events == ["sensor_timestamp", "rays"] + ([] if failure else ["received"])
    assert bool(evaluator.errors) == failure and evaluator.initial_capture_s >= 0.


@pytest.mark.parametrize("view,visible", [("kitchen", False), ("bathroom", True), ("away", False)])
@pytest.mark.parametrize("rendering", ["tiny", "enhanced"])
def test_actual_head_pixel_rays_require_frustum_and_no_wall_occlusion(view, visible, rendering, record_property, tmp_path):
    import numpy as np
    import pybullet as bullet
    from backend.challenges import get_challenge
    from backend.simulation import BulletSimulation
    sim = BulletSimulation(challenge=get_challenge("kitchen_bathroom"), rendering=rendering, width=160, height=120)
    evaluator = RoomEvaluator()
    try:
        sim._ticks(7)
        evaluator.capture_initial(sim)
        assert evaluator.frames[0]["paired"] and evaluator.frames[0]["image_file"] == "room-initial.png"
        assert not evaluator.frames[0]["visible_groups"]
        assert evaluator.initial_capture_s > 0
        evaluator.last_capture = -float("inf")
        if view != "kitchen":
            height = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[0][2]
            bullet.resetBasePositionAndOrientation(sim.robot, [1.4, 1.15, height],
                bullet.getQuaternionFromEuler([0, 0, 3.14159265 if view == "away" else 0]), physicsClientId=sim.client)
        eye, matrix = sim.camera_pose()
        evaluator.capture(sim, 1, eye, matrix)
        sensor, image = sim.capture_spatial(1)
        evaluator.receive(sensor, image)
        assert len(evaluator.frames) == 2
        frame = evaluator.frames[-1]
        assert frame["paired"] and frame["annotation_available"]
        assert bool(frame["visible_groups"]) is visible, frame
        assert frame["image_sha256"] == hashlib.sha256(image).hexdigest()
        assert all(abs(match["axial_depth_m"]-match["sensor_depth_m"]) <= .03 for match in frame["pixel_matches"])
        record_property("evidence", f"scripted real {rendering} renderer physics; not Luna validation")
        record_property("raycast_s", frame["raycast_s"])
        record_property("initial_capture_s", evaluator.initial_capture_s)
        record_property("visible_groups", json.dumps(frame["visible_groups"]))
        for name, captured_image in evaluator.images.items():
            (tmp_path / name).write_bytes(captured_image)
        (tmp_path / "room_evaluation.json").write_text(json.dumps(evaluator.artifact(), indent=2))
        record_property("head_image", str(tmp_path / frame["image_file"]))
        discovery = room_results(evaluator.artifact(), source=tmp_path)["discovery"]
        assert discovery["value"] is visible and discovery["newly_discovered"] is visible
        evaluator.capture(sim, 2, eye, matrix)
        assert 2 not in evaluator.pending
        if visible:
            evaluator.last_capture = -float("inf")
            evaluator.capture(sim, 3, eye, matrix)
            sensor = sensor.model_copy(update={"sequence": 3, "depth_m": np.full(160*120, float("nan")).tolist()})
            evaluator.receive(sensor, image)
            assert not evaluator.frames[-1]["visible_groups"]
    finally:
        sim.close()


def test_real_challenge_arrival_requires_grounded_full_base_and_rest():
    import pybullet as bullet
    from backend.challenges import get_challenge
    from backend.simulation import BulletSimulation
    sim = BulletSimulation(challenge=get_challenge("kitchen_bathroom"), rendering="tiny")
    evaluator = RoomEvaluator()
    physical = SimpleNamespace(contact_active=False, contacts=0)
    try:
        height = bullet.getBasePositionAndOrientation(sim.robot, physicsClientId=sim.client)[0][2]
        for position, velocity, expected in [([1.4, 1.15, height], [0, 0, 0], True),
                ([.82, 1.15, height], [0, 0, 0], False), ([1.4, 1.15, height+.5], [0, 0, 0], False),
                ([1.4, 1.15, height], [.1, 0, 0], False)]:
            bullet.resetBasePositionAndOrientation(sim.robot, position, [0, 0, 0, 1], physicsClientId=sim.client)
            bullet.resetBaseVelocity(sim.robot, linearVelocity=velocity, angularVelocity=[0, 0, 0], physicsClientId=sim.client)
            if expected:
                sim._ticks(120)
            bullet.performCollisionDetection(physicsClientId=sim.client)
            sim._update_challenge()
            evaluator.sample_arrival(sim, physical, sim.ticks/240.)
            assert evaluator.arrival_samples[-1]["eligible"] is expected
    finally:
        sim.close()


def test_offline_analysis_preserves_raw_and_rejects_unowned_overwrite(tmp_path, monkeypatch):
    import scripts.probe_mission_review as probe
    source, output = tmp_path / "baseline", tmp_path / "review"
    source.mkdir()
    contents = {"plan": {"evidence": "real_model_diagnostic"}, "measurements": [], "inference": [], "trace": {"events": []},
        "result": {"controller": {"phase": "completed", "outcome": {"kind": "limited"}, "input_tokens": 10, "output_tokens": 2},
            "task_completion": {"phase": "blocked"}, "travel_m": 7.435, "moving_s": 999}}
    for name, value in contents.items():
        (source / f"{name}.json").write_text(json.dumps(value))
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    monkeypatch.setattr(probe, "load_dotenv", lambda *args, **kwargs: pytest.fail("Offline analysis must not read credentials"))
    monkeypatch.setattr(probe, "EvaluationWorker", lambda *args, **kwargs: pytest.fail("Offline analysis must not start physics"))
    report = analyze(source, output)
    assert report["timing"]["moving_s"] == 0 and report["controller_outcome"]["kind"] == "limited"
    assert report["raw_outcome"]["task_completion"]["phase"] == "blocked"
    assert report["room_results"]["arrival"]["value"] is None
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}
    with pytest.raises(FileExistsError):
        analyze(source, output)
    assert analyze(source, output, overwrite=True)["raw_sha256"] == report["raw_sha256"]
    with pytest.raises(ValueError):
        analyze(source, source)
    (output / "analysis.json").write_text('{"schema":"another-owner"}')
    with pytest.raises(FileExistsError):
        analyze(source, output, overwrite=True)


async def test_actual_model_payload_excludes_evaluator_annotations(tmp_path):
    from backend.agent import AgentStart, FoundryModel
    from backend.challenges import get_challenge
    from backend.home_mapping import MapStore
    from backend.home_mission import HomeMission
    from tests.test_agent import controller_for, model_response
    captured = []

    async def respond(**payload):
        captured.append(copy.deepcopy(payload))
        return model_response("guide_mission", '{"action":"plan","plan":{"kind":"room","target":"Bathroom"}}')

    async def close():
        return None

    model = FoundryModel.__new__(FoundryModel)
    model.client = SimpleNamespace(responses=SimpleNamespace(create=respond), close=close)
    model.credential = None
    model.unified_mission = True
    worker = EvaluationWorker(challenge=get_challenge("kitchen_bathroom"), rendering="tiny", pace=True)
    controller = controller_for(model)
    evaluator = RoomEvaluator()
    try:
        await asyncio.wrap_future(worker.ready)
        worker.home_mission = HomeMission(worker, MapStore(tmp_path / "maps.sqlite3"))
        worker.room_evaluator = evaluator
        settings = AgentStart(run_id=worker.sim.run_id, episode_epoch=worker.epoch,
            **{**SETTINGS, "max_turns": 1, "goal": GOAL})
        controller.start(worker, settings)
        await asyncio.wait_for(controller.task, 50.)
        assert len(captured) == 1, controller.trace()
        assert evaluator.frames and not evaluator.errors
        payload = captured[0]
        feedback = next(event for event in controller.trace()["events"] if event["kind"] == "feedback")
        supplied = payload["input"][-1]
        data = json.loads(supplied["content"][0]["text"])
        assert data["observation"] == feedback["payload"]["observation"]
        assert "User goal: " + GOAL in payload["instructions"]
        user_text = json.dumps(payload["input"])
        for forbidden in ("camera_eye", "pixel_matches", "annotation_available", "evaluation_protocol", "bathroom_basin",
                          "toilet_bowl", "tub_interior", "Base and wheels stopped inside the bathroom"):
            assert forbidden not in user_text
        public_observation = copy.deepcopy(data["observation"])
        assert public_observation["spatial"].pop("environment_id") == "standalone:kitchen_bathroom"
        assert "bathroom" not in json.dumps(public_observation).lower()
        assert worker.home_mission.home is not None
        assert "bathroom" not in json.dumps(worker.home_mission.home.places).lower()
        evaluator.frames[0]["private_sentinel"] = {"label": "hidden_bathroom_oracle", "coordinates": [123.456, 789.012]}
        await model.respond(SimpleNamespace(deployment=payload["model"]), "low", GOAL, payload["input"])
        assert captured[1] == captured[0]
        assert "hidden_bathroom_oracle" not in json.dumps(captured)
    finally:
        await controller.halt()
        await worker.close()