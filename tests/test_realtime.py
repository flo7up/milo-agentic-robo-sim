import asyncio
import base64
import json

import pytest
from pydantic import ValidationError
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from backend.agent import FoundryConfig
from backend.realtime import FoundryRealtime, RealtimeChannel, RealtimeConfig, VoiceController, VoiceStart, realtime_session
from backend.worker import SimulationWorker


def test_voice_configuration_requires_resource_endpoint_and_deployment():
    assert not RealtimeConfig().public()["configured"]
    assert RealtimeConfig().public()["target_model"] == "gpt-realtime-2"
    assert RealtimeConfig().public()["reasoning_effort"] == "low"
    assert RealtimeConfig().deployment == ""
    config = RealtimeConfig(endpoint="https://test.openai.azure.com", deployment="my-realtime")
    assert config.public()["configured"] and config.endpoint.endswith("/openai/v1/")
    for endpoint in ("https://example.com", "http://test.openai.azure.com", "https://test.services.ai.azure.com/api/projects/test"):
        with pytest.raises(ValidationError):
            RealtimeConfig(endpoint=endpoint)


def test_voice_session_has_manual_turns_pcm_audio_and_only_robot_tools():
    session = realtime_session(RealtimeConfig())
    assert session["reasoning"] == {"effort": "low"}
    assert "truncation" not in session
    assert session["audio"]["input"]["turn_detection"] is None
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["output"]["voice"] == "alloy"
    assert {tool["name"] for tool in session["tools"]} == {
        "observe", "wait", "drive_base", "set_head", "set_arm_joints", "move_end_effector", "set_gripper", "stop"}


async def test_realtime_uses_ga_transport_and_keeps_keys_on_server(monkeypatch):
    from importlib import import_module
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-only-key")
    model = FoundryRealtime(RealtimeConfig(endpoint="https://test.openai.azure.com", deployment="test-voice"))
    try:
        captured = {}
        def connect(url, **kwargs):
            captured["url"] = url
            captured.update(kwargs)
            return RealtimeConnection([])
        monkeypatch.setattr(import_module("backend.realtime"), "connect", connect)
        async with model.connect():
            pass
        assert captured["url"] == "wss://test.openai.azure.com/openai/v1/realtime?model=test-voice"
        assert captured["additional_headers"]["api-key"] == "test-only-key"
        assert "test-only-key" not in str(RealtimeConfig().public())
    finally:
        await model.close()


async def test_realtime_channel_sends_and_receives_json_over_real_websocket():
    received = []

    async def peer(socket):
        received.append(json.loads(await socket.recv()))
        await socket.send(json.dumps({"type": "session.updated"}))

    async with serve(peer, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with connect(f"ws://127.0.0.1:{port}") as socket:
            channel = RealtimeChannel(socket)
            settings = {"type": "session.update", "session": realtime_session(RealtimeConfig())}
            await channel.send(settings)
            assert await channel.recv() == {"type": "session.updated"}
    assert received == [settings]


class VoiceBrowser:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.audio = []

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, value):
        await self.outgoing.put(value)

    async def send_bytes(self, value):
        self.audio.append(value)

    async def send(self, value):
        await self.incoming.put({"type": "websocket.receive", **({"bytes": value} if isinstance(value, bytes) else {"text": json.dumps(value)})})

    async def wait(self, kind):
        async with asyncio.timeout(5):
            while True:
                value = await self.outgoing.get()
                if value["type"] == kind:
                    return value


class RealtimeConnection:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.incoming = asyncio.Queue()
        self.sent = []
        self.response_number = 0
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def send(self, packet):
        self.sent.append(packet)
        if packet["type"] == "session.update":
            await self.incoming.put({"type": "session.updated"})
        elif packet["type"] == "input_audio_buffer.commit":
            await self.incoming.put({"type": "conversation.item.input_audio_transcription.completed", "transcript": "Move forward"})
        elif packet["type"] == "response.create":
            self.response_number += 1
            identifier = f"voice-response-{self.response_number}"
            await self.incoming.put({"type": "response.created", "response": {"id": identifier}})
            reply = next(self.replies)
            if reply == "wait":
                return
            if not reply:
                reply = [{"type": "message", "content": [{"type": "audio", "transcript": "Movement complete."}]}]
                await self.incoming.put({"type": "response.output_audio.delta", "response_id": identifier,
                                         "delta": base64.b64encode(b"\x01\x00" * 2400).decode()})
            await self.incoming.put({"type": "response.done", "response": {"id": identifier,
                "status": "completed", "output": reply, "usage": {"input_tokens": 10, "output_tokens": 5}}})

    async def recv(self):
        return await self.incoming.get()


class RealtimeModel:
    def __init__(self, replies):
        self.connection = RealtimeConnection(replies)
        self.closed = False

    def connect(self):
        return self.connection

    async def close(self):
        self.closed = True


def tool_call(arguments=None, name="drive_base", call_id="voice-call"):
    return [{"type": "function_call", "name": name, "call_id": call_id,
             "arguments": json.dumps(arguments if arguments is not None else {"linear_mps": .2, "angular_radps": 0, "duration_s": .5})}]


async def start_voice(model, pace=False):
    worker = SimulationWorker(pace=pace)
    await asyncio.wrap_future(worker.ready)
    browser = VoiceBrowser()
    controller = VoiceController(FoundryConfig(), RealtimeConfig(endpoint="https://test.openai.azure.com", deployment="test-realtime"), lambda config: model)
    controller.start_voice(worker, VoiceStart(run_id=worker.latest["run_id"], episode_epoch=worker.latest["episode_epoch"], feedback_interval_s=.25), browser)
    await browser.wait("ready")
    return worker, browser, controller


async def test_voice_idle_wakes_for_camera_or_new_utterance_without_recording_automatically():
    import pybullet as bullet
    from tests.test_agent import wait_for_phase
    model = RealtimeModel([[], [], []])
    worker, browser, controller = await start_voice(model)
    controller.idle_delay_s, controller.camera_poll_s = .08, .02
    try:
        await browser.send({"type": "listen"})
        await browser.send(b"\0\0" * 4800)
        await browser.send({"type": "commit"})
        await browser.wait("ready")
        await wait_for_phase(controller, "sleeping")
        assert not controller.recording and model.connection.response_number == 1
        await worker.call(lambda sim: bullet.resetJointState(sim.robot, sim.joints["head_pitch"], .8, physicsClientId=sim.client))
        await browser.wait("waking")
        await browser.wait("ready")
        await wait_for_phase(controller, "sleeping")
        assert model.connection.response_number == 2 and not controller.recording
        responses = [packet for packet in model.connection.sent if packet["type"] == "response.create"]
        assert responses[-1]["response"]["tool_choice"] == "none"
        assert worker.sim.ticks == 0
        await browser.send({"type": "listen"})
        await wait_for_phase(controller, "listening")
        assert not controller.state["auto_wake"]
        await browser.send(b"\0\0" * 4800)
        await browser.send({"type": "commit"})
        await browser.wait("ready")
        assert model.connection.response_number == 3
        await controller.halt()
        assert controller.idle_task is None and not controller.state["auto_wake"]
    finally:
        await controller.halt()
        await worker.close()


async def test_voice_audio_drives_real_robot_and_returns_spoken_reply():
    model = RealtimeModel([tool_call(), []])
    worker, browser, controller = await start_voice(model)
    try:
        await browser.send({"type": "listen"})
        await browser.send(b"\x01\x00" * 4800)
        await browser.send({"type": "commit"})
        await browser.wait("ready")
        assert worker.latest["observation"]["odometry_m_rad"][0] > .02
        assert browser.audio and controller.state["phase"] == "voice_ready"
        assert controller.state["reasoning"] == "low"
        events = controller.trace()["events"]
        assert any(event["title"] == "Spoken command" for event in events)
        assert any(event["kind"] == "result" and event["payload"]["result"]["status"] == "ok" for event in events)
        sent = model.connection.sent
        image_message = next(packet["item"] for packet in sent if packet["type"] == "conversation.item.create" and packet["item"]["type"] == "message")
        assert image_message["content"][1]["image_url"].startswith("data:image/png;base64,")
        assert "geometry" not in json.dumps(sent) and "snapshot" not in json.dumps(sent)
        assert any(packet.get("item", {}).get("type") == "function_call_output" for packet in sent)
    finally:
        await controller.halt()
        await worker.close()
    assert model.closed and model.connection.closed


async def test_realtime2_keeps_commentary_and_final_answer_without_private_reasoning():
    response = [
        {"type": "message", "phase": "commentary", "content": [{"type": "audio", "transcript": "Checking the camera."}]},
        {"type": "reasoning", "encrypted_content": "private-reasoning-marker"},
        {"type": "message", "phase": "final_answer", "content": [{"type": "audio", "transcript": "I can see the cube."}]},
    ]
    worker, browser, controller = await start_voice(RealtimeModel([response]))
    try:
        await browser.send({"type": "listen"})
        await browser.send(b"\0\0" * 4800)
        await browser.send({"type": "commit"})
        await browser.wait("ready")
        assert controller.state["message"] == "Checking the camera.\nI can see the cube."
        trace = json.dumps(controller.trace())
        assert "Checking the camera." in trace and "I can see the cube." in trace
        assert "private-reasoning-marker" not in trace
        assert worker.latest["snapshot"]["simulated_time_s"] == 0
    finally:
        await controller.halt()
        await worker.close()


async def test_voice_disconnect_cancels_pending_inference_without_motion():
    model = RealtimeModel(["wait"])
    worker, browser, controller = await start_voice(model)
    await browser.send({"type": "listen"})
    await browser.send(b"\0\0" * 4800)
    await browser.send({"type": "commit"})
    await browser.send({"type": "close"})
    await asyncio.wait_for(controller.task, 5)
    assert worker.latest["snapshot"]["simulated_time_s"] == 0
    assert worker.latest["stopped"] and not controller.active and model.closed
    await worker.close()


async def test_voice_disconnect_brakes_before_cloud_close_completes():
    closing = asyncio.Event()
    release_close = asyncio.Event()

    class SlowClose(RealtimeConnection):
        async def __aexit__(self, *args):
            closing.set()
            await release_close.wait()
            await super().__aexit__(*args)

    model = RealtimeModel([])
    model.connection = SlowClose([tool_call({"linear_mps": .2, "angular_radps": 0, "duration_s": 2})])
    worker, browser, controller = await start_voice(model, pace=True)
    try:
        await browser.send({"type": "listen"})
        await browser.send(b"\0\0" * 4800)
        await browser.send({"type": "commit"})
        async with asyncio.timeout(5):
            while worker.latest["snapshot"]["simulated_time_s"] == 0:
                await asyncio.sleep(.01)
            await browser.send({"type": "close"})
            await closing.wait()
        assert worker.latest["stopped"] and controller.cancelled
        await worker.call(lambda sim: sim.hold_current())
        assert worker.latest["snapshot"]["simulated_time_s"] < 2
        release_close.set()
        await asyncio.wait_for(controller.task, 5)
        results = [entry["payload"]["result"] for entry in controller.trace()["events"] if entry["kind"] == "result"]
        assert len(results) == 1 and results[0]["status"] == "cancelled"
    finally:
        release_close.set()
        await controller.halt()
        await worker.close()


async def test_voice_rejects_audio_without_explicit_microphone_activation():
    worker, browser, controller = await start_voice(RealtimeModel([]))
    await browser.send(b"\0\0" * 4800)
    await asyncio.wait_for(controller.task, 5)
    assert controller.state["phase"] == "error" and worker.latest["stopped"]
    assert worker.latest["snapshot"]["simulated_time_s"] == 0
    await worker.close()


@pytest.mark.parametrize("reply", [
    tool_call({"linear_mps": 100, "angular_radps": 0, "duration_s": .5}),
    tool_call(name="teleport"),
    tool_call() + tool_call(call_id="another-call"),
    tool_call() + [{"type": "message", "content": [{"type": "refusal", "refusal": "Declined"}]}],
])
async def test_voice_invalid_or_refused_tools_do_not_move(reply):
    model = RealtimeModel([reply, []])
    worker, browser, controller = await start_voice(model)
    try:
        await browser.send({"type": "listen"})
        await browser.send(b"\0\0" * 4800)
        await browser.send({"type": "commit"})
        async with asyncio.timeout(5):
            while True:
                packet = await browser.outgoing.get()
                if packet["type"] in ("ready", "ended"):
                    break
        assert worker.latest["snapshot"]["simulated_time_s"] == 0
    finally:
        await controller.halt()
        await worker.close()


async def test_voice_tool_stop_cannot_be_followed_by_unsolicited_motion():
    model = RealtimeModel([tool_call(name="stop", arguments={}), tool_call(call_id="unexpected-motion")])
    worker, browser, controller = await start_voice(model)
    try:
        await browser.send({"type": "listen"})
        await browser.send(b"\0\0" * 4800)
        await browser.send({"type": "commit"})
        await browser.wait("ended")
        assert worker.latest["snapshot"]["simulated_time_s"] == 0
        assert worker.latest["stopped"] and controller.state["phase"] == "error"
        requests = [packet for packet in model.connection.sent if packet["type"] == "response.create"]
        assert requests[-1]["response"]["tool_choice"] == "none"
    finally:
        await controller.halt()
        await worker.close()