import json

import pytest

from backend.agent import feedback_message
from backend.feedback import compact_numbers, feedback_json, model_tool_result, observation_collision
from tests.test_agent import observation


@pytest.mark.parametrize("directions,expected", [(["front"], "short, slow reverse"), (["left"], "short, slow reverse"),
                                               (["rear"], "do not reverse"), (["front", "rear"], "do not reverse")])
def test_collision_guidance_uses_measured_direction(directions, expected):
    raw = observation().model_copy(update={"bumpers": directions})
    message = feedback_message(raw, b"camera")
    collision = json.loads(message["content"][-1]["text"])["collision_feedback"]
    assert expected in collision["guidance"]
    assert collision["contacts"] == [{"direction": direction} for direction in directions]
    assert collision["source"] == "current"
    assert "object" not in collision
    assert observation_collision(observation().model_dump()) is None


def test_model_result_is_compact_but_retains_errors_and_transient_contacts():
    result = {"action_id": "original-id", "status": "error", "error": "NO_PROGRESS", "message": "Inspect first",
        "actual_duration_s": .9999999997, "observation": observation().model_dump(), "interruption_reason": "NO_PROGRESS",
        "sensor_deltas": {"odometry_m_rad": [.00000018, 1.2345678, -.0000001],
                          "contacts_during_action": [{"direction": "front", "force_n": 64.1641234}]}}
    before = json.dumps(result)
    compact = model_tool_result(result)
    assert "observation" not in compact and "action_id" not in compact
    assert compact["observation_seq"] == result["observation"]["seq"]
    assert compact["error"] == "NO_PROGRESS" and compact["actual_duration_s"] == 1
    assert compact["sensor_deltas"]["odometry_m_rad"] == [0, 1.235, 0]
    assert "contacts_during_action" not in compact["sensor_deltas"]
    assert compact["collision_feedback"]["source"] == "during_action"
    assert compact["collision_feedback"]["contacts"][0]["force_n"] == 64.164
    assert "short, slow reverse" in compact["collision_feedback"]["guidance"]
    assert json.dumps(result) == before
    assert len(feedback_json(compact)) < len(json.dumps(result))


def test_missing_contacts_do_not_fabricate_a_collision():
    assert model_tool_result({"status": "ok", "error": None, "message": "", "observation": observation().model_dump()}) == {
        "status": "ok", "observation_seq": 1}
    assert compact_numbers({"values": [True, False, None, "1.23456789", .00001]}) == {
        "values": [True, False, None, "1.23456789", 0]}


def test_current_rear_contact_overrides_earlier_front_contact():
    result = {"status": "ok", "observation": {"seq": 2, "bumpers": ["rear"]},
              "sensor_deltas": {"contacts_during_action": [{"direction": "front", "force_n": 12}]}}
    collision = model_tool_result(result)["collision_feedback"]
    assert collision["source"] == "current"
    assert "do not reverse" in collision["guidance"]
    assert collision["contacts"] == [{"direction": "rear"}]


def test_retained_context_budget_keeps_complete_pairs_without_old_images():
    from backend.feedback import context_token_estimate, retain_context, text_context
    turns = [[{"role": "user", "content": [{"type": "input_text", "text": f"frame {index}"},
                                              {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}]},
              {"type": "function_call", "call_id": str(index), "name": "observe", "arguments": "{}"},
              {"type": "function_call_output", "call_id": str(index), "output": '{"status":"ok"}'}]
             for index in range(3)]
    before = json.dumps(turns)
    cost = context_token_estimate(text_context(turns[-1]))
    kept, used = retain_context(turns, cost * 2)
    assert len(kept) == 2 and used == cost * 2
    assert kept[0][1]["call_id"] == "1" and kept[-1][2]["call_id"] == "2"
    assert "input_image" not in json.dumps(kept)
    assert retain_context(turns, cost - 1) == ([], 0)
    assert retain_context(turns, 0) == ([], 0)
    assert len(retain_context(turns * 4, cost * 12)[0]) == 12
    assert json.dumps(turns) == before
    unicode_turn = [{"role": "user", "content": "\u4f60\u597d"}]
    assert context_token_estimate(unicode_turn) >= len("\u4f60\u597d".encode("utf-8"))


def test_camera_batch_bounds_deduplicates_and_labels_paired_historical_sensors():
    from backend.feedback import camera_batch
    frames = [feedback_message(observation(index).model_copy(update={"wall_timestamp": float(index)}),
                               f"camera-{index}".encode()) for index in range(1, 5)]
    message, selected = camera_batch(frames[-1], frames[:-1], 3, frames[0])
    assert selected == [frames[3], frames[0], frames[2]]
    assert message["content"][:2] == frames[3]["content"][:2]
    historical = [json.loads(part["text"])["historical_camera_observation"] for part in message["content"]
                  if part["type"] == "input_text" and "historical_camera_observation" in part["text"]]
    assert [entry["seq"] for entry in historical] == [1, 3]
    assert len([part for part in message["content"] if part["type"] == "input_image"]) == 3
    assert camera_batch(frames[-1], frames, 1, frames[0])[1] == [frames[-1]]
    assert camera_batch(frames[-1], [frames[-1]] * 8, 8)[1] == [frames[-1]]


def history_sample(sequence, color="gray", head=(0., .1), pose=(0., 0., 0.), flash=False):
    from io import BytesIO
    from PIL import Image, ImageDraw
    from backend.contracts import SpatialObservation
    from backend.spatial import calibration
    image = Image.new("RGB", (160, 120), color)
    if flash:
        ImageDraw.Draw(image).rectangle((75, 35, 95, 55), fill="red")
    output = BytesIO()
    image.save(output, format="PNG")
    sensor = SpatialObservation(run_id="history-test", episode_epoch=0, sequence=sequence,
        captured_at=sequence * .2, simulated_time_s=sequence * .2, head_rad=list(head), odometry_m_rad=list(pose),
        calibration=calibration(160, 120), depth_m=[None] * (160 * 120))
    return sensor, output.getvalue()


def test_motion_camera_history_preserves_brief_sighting_across_long_review_interval():
    from backend.visual_history import CameraHistory
    now = [0.]
    history = CameraHistory("history-test", 0, clock=lambda: now[0])
    flash_image = None
    for sequence in range(1, 301):
        sensor, image = history_sample(sequence, flash=sequence == 5)
        now[0] = sensor.captured_at
        assert history.record(sensor, image)
        if sequence == 5:
            flash_image = image
    metadata, sheet = history.review(300)
    assert len(history.recent) == 50 and len(history.keyframes) == 2
    assert [frame["sequence"] for frame in metadata["frames"]] == [1, 5]
    assert sheet is not None and metadata["frames"][1]["age_s"] > 50
    assert history.original(metadata["frames"][1]["frame_id"])[1] == flash_image
    assert history.review(300)[0]["frames"] == metadata["frames"]
    history.acknowledge(metadata["generation"], 300)
    assert history.review(300)[1] is None
    assert history.original(metadata["frames"][1]["frame_id"])[1] == flash_image
    for sequence in range(301, 321):
        sensor, image = history_sample(sequence)
        now[0] = sensor.captured_at
        history.record(sensor, image)
    history.acknowledge(metadata["generation"], 300)
    assert history.review(320)[1] is None


def test_motion_camera_history_selects_pose_and_visual_diversity_and_keeps_native_pixels():
    from io import BytesIO
    from PIL import Image
    from backend.visual_history import CameraHistory
    now = [0.]
    history = CameraHistory("history-test", 0, clock=lambda: now[0])
    for sequence in range(1, 25):
        sensor, image = history_sample(sequence, head=(sequence * .2, .1), pose=(sequence * .15, 0., 0.))
        now[0] = sensor.captured_at
        history.record(sensor, image)
    metadata, sheet = history.review(24)
    assert len(history.keyframes) <= 6 and len(metadata["frames"]) == 4
    assert metadata["frames"][0]["sequence"] == 1
    assert [frame["sequence"] for frame in metadata["frames"]] == sorted(frame["sequence"] for frame in metadata["frames"])
    assert all(frame["sequence"] < 24 and not frame["authorizes_motion"] for frame in metadata["frames"])
    with Image.open(BytesIO(sheet)) as montage:
        assert montage.size == (332, 332)
        for index, frame in enumerate(metadata["frames"]):
            left, top = 4 + index % 2 * 164, 54 + index // 2 * 154
            with Image.open(BytesIO(history.original(frame["frame_id"])[1])) as original:
                assert montage.crop((left, top, left + 160, top + 120)).tobytes() == original.convert("RGB").tobytes()


def test_motion_camera_history_rejects_stale_wrong_episode_and_invalidates_originals():
    from backend.visual_history import CameraHistory
    now = [1.]
    history = CameraHistory("history-test", 0, clock=lambda: now[0])
    sensor, image = history_sample(5)
    assert not history.record(sensor.model_copy(update={"episode_epoch": 1}), image)
    assert not history.record(sensor.model_copy(update={"captured_at": 2.}), image)
    assert not history.record(sensor.model_copy(update={"captured_at": -1.}), image)
    assert history.record(sensor, image)
    assert not history.record(sensor, image)
    metadata, _ = history.review(6)
    identifier = metadata["frames"][0]["frame_id"]
    with pytest.raises(ValueError, match="generation"):
        history.acknowledge("old-generation", 5)
    with pytest.raises(ValueError, match="not offered"):
        CameraHistory("history-test", 1).original(identifier)
    now[0] = 302.
    with pytest.raises(ValueError, match="expired"):
        history.original(identifier)
    assert not history.recent and not history.keyframes and history.reference is None


def test_motion_camera_history_bounds_originals_and_acknowledges_only_reviewed_capture():
    from backend.visual_history import CameraHistory
    now = [0.]
    history = CameraHistory("history-test", 0, clock=lambda: now[0])
    pending = None
    for sequence in range(1, 101):
        sensor, image = history_sample(sequence, pose=(sequence * .4, 0., 0.))
        now[0] = sensor.captured_at
        history.record(sensor, image)
        if sequence % 5 == 0 and sequence <= 90:
            review, _ = history.review(sequence)
            assert len(history.retained) <= 24 and len(history.recent) <= 50 and len(history.keyframes) <= 6 and len(history.pending_keyframes) <= 6
            for frame in review["available_frames"]:
                assert history.original(frame["frame_id"])[1] == image
            if sequence == 90:
                pending = review
    history.acknowledge(pending["generation"], pending["through_sequence"])
    review, _ = history.review(100)
    assert review["frames"] and all(90 < frame["sequence"] < 100 for frame in review["frames"])