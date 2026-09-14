from types import SimpleNamespace

import pytest

from backend import route_supervisor
from backend.spatial import calibration, measure_visible_region


def test_generic_region_measurement_is_only_paired_visible_depth():
    sensor = SimpleNamespace(calibration=calibration(24, 16), depth_m=[1.] * (24 * 16),
        head_rad=[0., 0.], odometry_m_rad=[0., 0., 0.], sequence=1, captured_at=10.)
    measured = measure_visible_region(sensor, [.25, .25, .75, .75])
    assert measured["source"] == "paired_head_depth" and measured["measured_points"] >= 8
    assert measured["surface_median_xyz_m"][0] == pytest.approx(1.055)
    assert measured["captured_at"] == 10. and measured["sequence"] == 1
    for invalid in ([.8, .2, .3, .7], [float("nan"), 0., 1., 1.], [-1., 0., 1., 1.]):
        with pytest.raises(ValueError):
            measure_visible_region(sensor, invalid)
    sensor.depth_m = [None] * (24 * 16)
    with pytest.raises(ValueError, match="insufficient"):
        measure_visible_region(sensor, [0., 0., 1., 1.])


async def test_generic_wait_yields_to_latest_command(monkeypatch):
    controller = SimpleNamespace(command_revision=1)
    calls = []
    async def wait(controller, worker, settings, duration):
        calls.append(duration)
        controller.command_revision = 2
    monkeypatch.setattr(route_supervisor, "wait_stationary", wait)
    assert not await route_supervisor.command_wait(controller, None, None, 4., 1)
    assert calls == [.5]


@pytest.mark.parametrize("new_goal,final_revision,eligible", [("Original task", 1, True), ("New task", 2, False), ("Original task", 2, False)])
async def test_live_recording_does_not_verify_revised_task_with_old_scorer(tmp_path, new_goal, final_revision, eligible):
    import io
    import json
    from backend.session_recording import finish_session
    directory = tmp_path / "case"
    directory.mkdir()
    async def call(operation):
        return operation(None)
    worker = SimpleNamespace(call=call, recorder=None, camera_frames={}, rendering="enhanced",
        latest={"challenge": {"goal": "Original task"}})
    recorder = SimpleNamespace(stream=io.BytesIO(), finish=lambda metadata: {"final_physics_success": True, "elapsed_wall_s": 3.})
    controller = SimpleNamespace(command_revision=final_revision, trace=lambda: {"events": []},
        state={"goal": new_goal, "model_id": "luna", "phase": "completed", "input_tokens": 10, "output_tokens": 5, "turns": 2})
    manifest = {"evidence": "scripted_test", "session_id": "session", "command_revision": 1,
        "ai_generated_routes": True, "cases": [{"case_id": "case", "challenge": "park", "environment": "standalone"}],
        "design": {"code_sha256": {}}}
    await finish_session(controller, worker, SimpleNamespace(goal="Original task"), tmp_path, recorder, manifest, "in_progress", None)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert report["verification_eligible"] == eligible
    assert report["initial_goal"] == "Original task" and report["final_goal"] == new_goal
    assert report["final_command_revision"] == final_revision


def test_generic_current_camera_is_labelled_before_its_image(monkeypatch):
    from backend import agent
    monkeypatch.setattr(agent, "feedback_message", lambda observation, image: {"role": "user", "content": [
        {"type": "input_text", "text": "paired sensors"}, {"type": "input_image", "image_url": "original-png"}]})
    message = route_supervisor.current_camera_message(None, b"")
    assert message["content"][1]["text"].startswith("IMAGE 1: CURRENT HEAD CAMERA")
    assert message["content"][2]["image_url"] == "original-png"


def test_route_decision_separates_measurement_motion_and_task_limitation():
    for action in ("continue", "look", "wait", "finish", "report_limit"):
        decision = route_supervisor.RouteDecision(action=action, reason="A visible observation supports this decision")
        assert not decision.waypoints_m and decision.image_bounds is None
    with pytest.raises(ValueError):
        route_supervisor.RouteDecision(action="measure", reason="Measure the visible object before planning")
    with pytest.raises(ValueError):
        route_supervisor.RouteDecision(action="route", image_bounds=[0., 0., 1., 1.],
            waypoints_m=[[1., 0.]], reason="Do not combine arbitrary boxes and motion")