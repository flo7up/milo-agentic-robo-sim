import base64
from io import BytesIO
import json

import httpx
from PIL import Image

from scripts.probe_hybrid import VISION_FORMAT, probe_actions, probe_vision, validate_action_calls, vision_completion


async def test_ollama_camera_probe_disables_thinking_and_omits_private_trace():
    def handle(request):
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["model"] == "gemma4:e2b-it-qat"
        assert payload["think"] is False and payload["stream"] is False
        assert payload["format"] == VISION_FORMAT
        assert payload["options"]["num_ctx"] == 4096
        assert payload["messages"][1]["content"] == ""
        assert base64.b64decode(payload["messages"][1]["images"][0]) == b"camera"
        return httpx.Response(200, json={
            "model": payload["model"], "message": {"content": '{"room":"kitchen","fixtures":["oven"]}',
                                                    "thinking": "private-marker"},
            "done": True, "done_reason": "stop", "total_duration": 1500000000,
            "prompt_eval_count": 100, "eval_count": 10})

    async with httpx.AsyncClient(base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handle)) as client:
        result = await vision_completion(client, "gemma4:e2b-it-qat", None, b"camera", "none", "ollama")
    assert result["thinking_disabled"] and result["thinking_present"]
    assert "private-marker" not in json.dumps(result)
    assert result["timing_s"]["total_duration"] == 1.5
    assert result["usage"] == {"input_tokens": 100, "output_tokens": 10}


async def test_changed_image_probes_are_distinct_and_labelled(tmp_path):
    source = Image.new("RGB", (640, 480))
    source.putdata([(column % 256, row % 256, 0) for row in range(480) for column in range(640)])
    source.save(tmp_path / "kitchen.png")
    images = []

    def handle(request):
        payload = json.loads(request.content)
        images.append(payload["messages"][1]["images"][0])
        assert Image.open(BytesIO(base64.b64decode(images[-1]))).size == (320, 240)
        return httpx.Response(200, json={"model": payload["model"], "done": True, "done_reason": "stop",
            "message": {"content": '{"room":"kitchen","fixtures":["sink"]}'}})

    async with httpx.AsyncClient(base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handle)) as client:
        records = await probe_vision(client, "gemma4:e2b-it-qat", tmp_path, 320, "none", 3,
                                    frames=tmp_path, rooms=["kitchen"], provider="ollama", vary_crops=True)
    assert len(set(images)) == 3
    assert all(record["image_kind"] == "derived_crop" and record["exact_json_correct"] for record in records)


async def test_ollama_tool_dry_run_checks_canonical_limits_without_execution():
    replies = [
        {"name": "set_head", "arguments": {"yaw_rad": .5, "pitch_rad": 0, "duration_s": .5}},
        {"name": "set_head", "arguments": {"yaw_rad": -.5, "pitch_rad": 0, "duration_s": .5}},
        {"name": "stop", "arguments": {}}]

    def handle(request):
        payload = json.loads(request.content)
        assert payload["think"] is False
        assert "format" not in payload
        assert {tool["function"]["name"] for tool in payload["tools"]} == {"set_head", "drive_base", "stop", "observe"}
        return httpx.Response(200, json={"done": True, "done_reason": "stop", "model": payload["model"],
            "message": {"content": "", "tool_calls": [{"function": replies.pop(0)}]}})

    async with httpx.AsyncClient(base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handle)) as client:
        records = await probe_actions(client, "gemma4:e2b-it-qat", b"camera", 1)
    assert len(records) == 3
    assert all(record["valid_single_action"] and record["matched_instruction"] for record in records)
    assert validate_action_calls([], "left") == (False, False)
    assert validate_action_calls([{"function": {"name": "unknown"}}], "left") == (False, False)
    invalid = {"function": {"name": "drive_base", "arguments": {"linear_mps": 99, "angular_radps": 0, "duration_s": 1}}}
    assert validate_action_calls([invalid], "left") == (False, False)
    assert validate_action_calls([invalid, invalid], "left") == (False, False)