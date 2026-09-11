import base64
from io import BytesIO
import json
import numpy as np
import pytest

import httpx
from PIL import Image

from scripts.probe_hybrid import VISION_FORMAT, probe_actions, probe_vision, validate_action_calls, vision_completion


def test_smolvla_smoke_checks_base_action_shape_without_milo_execution():
    from scripts.probe_smolvla import checked_actions
    values = np.zeros((1, 50, 6), dtype=np.float32)
    report = checked_actions(values, 50, 6)
    assert report["finite"] and report["shape"] == [1, 50, 6]
    assert len(report["actions"]) == 50 and len(report["first_action"]) == 6
    with pytest.raises(ValueError):
        checked_actions(np.zeros((1, 50, 7)), 50, 6)
    values[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        checked_actions(values, 50, 6)


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


async def test_control_benchmark_measures_validated_varied_images_and_rejects_truncation():
    from scripts.benchmark_control import distribution, image_to_skill, summarize
    images = []

    def handle(request):
        payload = json.loads(request.content)
        images.append(payload["messages"][1]["images"][0])
        assert Image.open(BytesIO(base64.b64decode(images[-1]))).size == (320, 240)
        assert payload["options"]["num_ctx"] == 4096
        assert len(payload["messages"]) == 2 and payload["think"] is False
        assert "observation" not in payload["messages"][1]
        return httpx.Response(200, json={"model": payload["model"], "done": True,
            "done_reason": "stop" if len(images) == 1 else "length", "message": {
                "content": '{"skill":"approach","target_visible":true,"target_x":0.3}'}})

    records = []
    async with httpx.AsyncClient(base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handle)) as client:
        for color in ("red", "blue"):
            buffer = BytesIO()
            Image.new("RGB", (640, 480), color).save(buffer, format="PNG")
            records.append(await image_to_skill(client, "test-model", buffer.getvalue(), deadline_s=0))
    summary = summarize(records)
    assert summary["requests"] == 2 and summary["valid_actions"] == 1
    assert summary["unique_images"] == 2 and summary["stale_actions"] == 2
    assert summary["usable_within_deadline"] == 0
    assert records[0]["latency_s"] >= records[0]["preprocess_s"] > 0
    assert distribution(list(range(1, 101))) == {"count": 100, "p50": 50, "p95": 95, "max": 100}
    assert distribution([])["p95"] is None


async def test_control_benchmark_invalid_action_is_not_a_latency_success():
    from scripts.benchmark_control import image_to_skill, summarize
    buffer = BytesIO()
    Image.new("RGB", (160, 120)).save(buffer, format="PNG")

    def handle(request):
        return httpx.Response(200, json={"done": True, "done_reason": "stop", "message": {
            "content": '{"skill":"approach","target_visible":true,"target_x":2}'}})

    async with httpx.AsyncClient(base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handle)) as client:
        record = await image_to_skill(client, "test-model", buffer.getvalue())
    assert not record["valid"] and record["error"] == "ValidationError"
    assert summarize([record])["valid_image_to_action_s"]["count"] == 0


async def test_control_benchmark_instruments_real_worker_and_records_burst_gaps():
    from scripts.benchmark_control import TimedWorker, cadence, submit
    worker = TimedWorker(pace=False)
    try:
        await worker.instrument()
        await submit(worker, "set_head", {"yaw_rad": .4, "pitch_rad": .2, "duration_s": .2}, navigation=False)
        metrics = worker.measurements()
        assert metrics["physics_step_240hz"]["gap_s"]["count"] == 47
        assert metrics["physics_steps_against_100hz_gap_budget"]["gap_s"]["count"] == 47
        assert metrics["worker_publish_20hz"]["gap_s"]["count"] == 3
        assert metrics["capture_cost_s"]["count"] > 0
        assert cadence([0, .005, .06], .01)["missed_slots"] == 5
    finally:
        await worker.close()


async def test_control_benchmark_openai_compatible_runtime_uses_image_and_schema():
    from scripts.benchmark_control import image_to_skill
    buffer = BytesIO()
    Image.new("RGB", (640, 480), "red").save(buffer, format="PNG")

    def handle(request):
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
        assert len(payload["messages"][1]["content"]) == 2
        assert payload["max_tokens"] == 96
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {
            "content": '{"skill":"inspect","target_visible":false,"target_x":0.5}'}}]})

    async with httpx.AsyncClient(base_url="http://127.0.0.1:8080", transport=httpx.MockTransport(handle)) as client:
        record = await image_to_skill(client, "cosmos-candidate", buffer.getvalue(), provider="openai")
    assert record["valid"] and record["decision"]["skill"] == "inspect"


async def test_control_benchmark_live_capture_counts_acquisition_and_staleness():
    from types import SimpleNamespace
    from scripts.benchmark_control import TimedWorker, live_image_to_skill
    worker = TimedWorker(pace=False)

    def handle(request):
        return httpx.Response(200, json={"done": True, "done_reason": "stop", "message": {
            "content": '{"skill":"hold","target_visible":false,"target_x":0.5}'}})

    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:11434", transport=httpx.MockTransport(handle)) as client:
            record = await live_image_to_skill(worker, client, SimpleNamespace(
                model="test", width=320, provider="ollama", deadline=0))
        assert record["latency_s"] >= record["acquisition_s"] + record["request_after_capture_s"]
        assert record["valid"] and record["stale"]
        assert record["observation_seq"] > 0 and record["observation_wall_timestamp"] > 0
    finally:
        await worker.close()