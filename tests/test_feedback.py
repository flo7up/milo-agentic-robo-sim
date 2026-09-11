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
    assert json.dumps(turns) == before
    unicode_turn = [{"role": "user", "content": "\u4f60\u597d"}]
    assert context_token_estimate(unicode_turn) >= len("\u4f60\u597d".encode("utf-8"))