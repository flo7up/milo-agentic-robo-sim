import asyncio
import base64
from contextlib import asynccontextmanager
import json
import os
import time
from typing import Literal
from urllib.parse import urlencode

from azure.identity import DefaultAzureCredential
from pydantic import Field, field_validator
from starlette.websockets import WebSocketDisconnect
from websockets.asyncio.client import connect

from backend.agent import AgentController, AgentStart, FeedbackRate, FoundryConfig, INSTRUCTIONS, ModelProfile, feedback_message, robot_tools
from backend.contracts import Command, StrictModel, TOOL_MODELS
from backend.feedback import compact_numbers, feedback_json, model_tool_result, observation_collision

REALTIME_MODEL = "gpt-realtime-2"
REALTIME_REASONING_EFFORT = "low"


class RealtimeConfig(StrictModel):
    endpoint: str = ""
    deployment: str = Field(default="", max_length=120)
    voice: Literal["alloy", "echo", "shimmer", "ash", "ballad", "coral", "sage", "verse"] = "alloy"

    @field_validator("endpoint")
    @classmethod
    def resource_endpoint(cls, value):
        endpoint = FoundryConfig(endpoint=value).endpoint
        if "/api/projects/" in endpoint:
            raise ValueError("Realtime needs the Foundry resource endpoint, not a project endpoint")
        return endpoint

    @classmethod
    def from_environment(cls):
        return cls(endpoint=os.getenv("FOUNDRY_REALTIME_ENDPOINT", ""),
                   deployment=os.getenv("FOUNDRY_REALTIME_DEPLOYMENT", ""),
                   voice=os.getenv("FOUNDRY_REALTIME_VOICE", "alloy"))

    def public(self):
        return {**self.model_dump(), "target_model": REALTIME_MODEL, "reasoning_effort": REALTIME_REASONING_EFFORT,
            "configured": bool(self.endpoint and self.deployment.strip())}


VOICE_INSTRUCTIONS = INSTRUCTIONS + """
You are Milo's voice interface. The user explicitly switches on their microphone to
give you a command. Respond aloud, briefly, in the user's language. Do not act until
the user finishes their utterance. Only carry out movement requested by the user.
Use at most one tool per response. Explain what happened using actual tool results,
not predictions. Ask a spoken clarification if a command is ambiguous. A stop tool
brakes the robot; afterwards you may confirm aloud and wait for the next command.
Do not end the connection after each reply. Wait for the next push-to-talk utterance.
"""


def realtime_session(config, goal=""):
    instructions = VOICE_INSTRUCTIONS + ("\nCurrent challenge goal: " + goal if goal else "")
    return {"type": "realtime", "instructions": instructions,
            "reasoning": {"effort": REALTIME_REASONING_EFFORT},
            "output_modalities": ["audio"], "max_output_tokens": 2048,
            "tools": [{key: value for key, value in tool.items() if key != "strict"} for tool in robot_tools()],
            "tool_choice": "auto", "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": 24000},
                          "transcription": {"model": "whisper-1"}, "turn_detection": None},
                "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": config.voice}}}


class RealtimeChannel:
    def __init__(self, connection):
        self.connection = connection

    async def send(self, packet):
        await self.connection.send(json.dumps(packet, allow_nan=False))

    async def recv(self):
        packet = json.loads(await self.connection.recv())
        if not isinstance(packet, dict) or not isinstance(packet.get("type"), str):
            raise ValueError("Invalid Foundry Realtime event")
        return packet


class FoundryRealtime:
    def __init__(self, config):
        self.config = config
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.credential = None if self.api_key else DefaultAzureCredential(exclude_interactive_browser_credential=True)

    @asynccontextmanager
    async def connect(self):
        if self.api_key:
            headers = {"api-key": self.api_key}
        else:
            token = await asyncio.to_thread(self.credential.get_token, "https://ai.azure.com/.default")
            headers = {"Authorization": f"Bearer {token.token}"}
        endpoint = self.config.endpoint.replace("https://", "wss://", 1)
        url = endpoint + "realtime?" + urlencode({"model": self.config.deployment})
        async with connect(url, additional_headers=headers, open_timeout=15, max_size=2097152) as connection:
            yield RealtimeChannel(connection)

    async def close(self):
        if self.credential:
            await asyncio.to_thread(self.credential.close)


class VoiceStart(FeedbackRate):
    type: Literal["start"] = "start"
    run_id: str = Field(min_length=1, max_length=80)
    episode_epoch: int = Field(ge=0)
    max_turns: int = Field(default=30, ge=1, le=200)


class VoiceController(AgentController):
    def __init__(self, config, voice_config, voice_factory=FoundryRealtime):
        super().__init__(config)
        self.voice_config = voice_config
        self.voice_factory = voice_factory
        self.recording = False
        self.audio_bytes = 0
        self.responding = False
        self.response_id = None
        self.seen_calls = set()
        self.last_feedback = None
        self.observation = None
        self.actions = 0
        self.pending_command = None
        self.pending_call_id = None
        self.idle_outcome = None

    def start_voice(self, worker, settings: VoiceStart, browser):
        if not self.voice_config.public()["configured"]:
            raise ValueError("Configure a Foundry Realtime resource endpoint and deployment")
        goal = worker.challenge.goal if worker.challenge else "Follow the user's spoken robot commands."
        self.settings = AgentStart(**settings.model_dump(exclude={"type"}), model_id="realtime",
                       reasoning=REALTIME_REASONING_EFFORT, goal=goal)
        profile = ModelProfile(id="realtime", label=self.voice_config.deployment,
                               deployment=self.voice_config.deployment, reasoning_efforts=[REALTIME_REASONING_EFFORT])
        self._begin_session(worker, self.settings, profile, realtime_session(self.voice_config, goal)["instructions"], "voice")
        self.task = asyncio.create_task(self._run_voice(worker, browser))

    def _check_voice_live(self, worker):
        if (self.cancelled or worker.closed or worker.latest["run_id"] != self.settings.run_id or
                worker.latest["episode_epoch"] != self.settings.episode_epoch):
            raise asyncio.CancelledError

    async def _respond(self, worker, connection, allow_tools=True, call_id=None):
        if self.state["turns"] >= self.settings.max_turns:
            raise ValueError("Voice session turn limit reached")
        await self._wait_for_feedback(self.last_feedback)
        self._check_voice_live(worker)
        self.state["turns"] += 1
        self.observation, image = await self._feedback(worker)
        self._check_voice_live(worker)
        now = time.monotonic()
        self.state.update(phase="thinking", last_feedback_at=time.time(), next_feedback_at=None,
                          observed_interval_s=None if self.last_feedback is None else now - self.last_feedback)
        self.last_feedback = now
        message = feedback_message(self.observation, image)
        message["content"][1].pop("detail", None)
        await connection.send({"type": "conversation.item.create", "item": {"type": "message", **message}})
        self._trace("feedback", "Camera + sensors submitted", {
            "observation": compact_numbers(self.observation.model_dump()), "image_detail": "auto", "history_turns": [],
            "collision_feedback": observation_collision(self.observation.model_dump()),
            "context_mode": "realtime_conversation", "input_items": 1, "images_in_request": 1,
            "tool_result_call_ids": [call_id] if call_id else []}, image=image)
        self.response_started = time.monotonic()
        self.responding = True
        self.allow_tools = allow_tools
        self.response_id = None
        await connection.send({"type": "response.create", "response": {"tool_choice": "auto" if allow_tools else "none"}})

    async def _browser_input(self, worker, browser, connection):
        while True:
            packet = await asyncio.wait_for(browser.receive(), 120)
            if packet["type"] == "websocket.disconnect":
                raise WebSocketDisconnect
            self._check_voice_live(worker)
            if packet.get("bytes") is not None:
                audio = packet["bytes"]
                if not self.recording or len(audio) > 16384 or len(audio) % 2:
                    raise ValueError("Invalid microphone audio packet")
                self.audio_bytes += len(audio)
                if self.audio_bytes > 30 * 24000 * 2:
                    raise ValueError("Microphone recording exceeds 30 seconds")
                await connection.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(audio).decode("ascii")})
                continue
            text = packet.get("text", "")
            if len(text) > 256:
                raise ValueError("Invalid voice control message")
            message = json.loads(text)
            if message == {"type": "listen"} and self.state["phase"] in {"voice_ready", "sleeping"}:
                self._cancel_idle()
                self.idle_outcome = None
                self.state["outcome"] = None
                self.state.update(idle_since=None, idle_reason=None, wake_reason="Voice interaction", camera_unchanged_s=0)
                self.recording, self.audio_bytes, self.actions = True, 0, 0
                await worker.resume_manual()
                self._check_voice_live(worker)
                await connection.send({"type": "input_audio_buffer.clear"})
                self.state["phase"] = "listening"
            elif message == {"type": "commit"} and self.recording:
                self.recording = False
                if self.audio_bytes < 4800:
                    await connection.send({"type": "input_audio_buffer.clear"})
                    self.state["phase"] = "voice_ready"
                    await browser.send_json({"type": "notice", "message": "Recording too short. Try again."})
                    await browser.send_json({"type": "ready"})
                    continue
                self._trace("session", "Microphone audio submitted", {"message": f"{self.audio_bytes / 48000:.2f} s mono PCM audio"})
                await connection.send({"type": "input_audio_buffer.commit"})
                await self._respond(worker, connection)
            elif message == {"type": "close"}:
                return
            else:
                raise ValueError("Voice input is not available during a robot response")

    async def _receive_model(self, worker, browser, connection):
        while True:
            packet = await asyncio.wait_for(connection.recv(), 45 if self.responding else 120)
            self._check_voice_live(worker)
            kind = packet["type"]
            if kind == "error":
                raise ValueError("Foundry Realtime rejected the request. Check deployment, access, and audio support.")
            if kind == "conversation.item.input_audio_transcription.completed":
                text = packet.get("transcript", "")[:4000]
                self._trace("session", "Spoken command", {"message": text})
                await browser.send_json({"type": "transcript", "speaker": "user", "text": text})
            elif kind == "response.created" and self.responding:
                self.response_id = packet["response"]["id"]
            elif kind == "response.output_audio.delta" and self.responding and packet.get("response_id") == self.response_id:
                audio = base64.b64decode(packet["delta"], validate=True)
                if len(audio) > 192000 or len(audio) % 2:
                    raise ValueError("Invalid Realtime output audio")
                self.state["phase"] = "speaking"
                await browser.send_bytes(audio)
            elif kind == "response.output_audio_transcript.delta" and self.responding and packet.get("response_id") == self.response_id:
                await browser.send_json({"type": "transcript_delta", "text": packet.get("delta", "")[:4000]})
            elif kind == "response.done" and self.responding and packet["response"]["id"] == self.response_id:
                await self._finish_response(worker, browser, connection, packet["response"])

    async def _finish_response(self, worker, browser, connection, response):
        self._check_voice_live(worker)
        self.responding = False
        output = response.get("output", [])
        text = "\n".join(content.get("transcript") or content.get("text", "") for item in output
                         if item["type"] == "message" for content in item.get("content", []))[:16000]
        calls = [item for item in output if item["type"] == "function_call"]
        usage = response.get("usage") or {}
        self.state["input_tokens"] += usage.get("input_tokens", 0)
        self.state["output_tokens"] += usage.get("output_tokens", 0)
        latency = time.monotonic() - self.response_started
        self.state.update(message=text[:2000], inference_latency_s=latency)
        self._trace("response", "Spoken LLM response", {
            "text": text, "text_truncated": False, "status": response["status"], "latency_s": latency,
            "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
            "calls": [{"name": call["name"], "call_id": call["call_id"], "arguments": call["arguments"][:8000]} for call in calls[:8]],
            "calls_truncated": len(calls) > 8, "refusals": []})
        await browser.send_json({"type": "transcript", "speaker": "robot", "text": text})
        refused = any(content.get("type") == "refusal" for item in output if item["type"] == "message"
                  for content in item.get("content", []))
        if response["status"] != "completed" or refused or len(calls) > 1 or (calls and not self.allow_tools):
            raise ValueError("Incomplete or multiple-action voice response; robot stopped")
        if not calls:
            self.state["phase"] = "voice_ready"
            if self.state["outcome"] is None:
                self._set_outcome("ended", text or "Voice reply finished", "agent")
            await browser.send_json({"type": "ready"})
            async def wake():
                self._check_voice_live(worker)
                if self.recording or self.responding:
                    return
                await browser.send_json({"type": "waking"})
                self.state["outcome"] = None
                await connection.send({"type": "conversation.item.create", "item": {"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": "The head-camera scene changed while idle. Briefly assess the new view without moving; wait for a new spoken command."}]}})
                await self._respond(worker, connection, allow_tools=False)
            self._arm_idle(worker, self.idle_outcome or "Reply finished", wake, worker.stop_revision)
            return
        call = calls[0]
        call_id, name = call["call_id"], call["name"]
        if call_id in self.seen_calls:
            raise ValueError("Repeated voice tool call; robot stopped")
        self.seen_calls.add(call_id)
        arguments = None
        try:
            if name not in {tool["name"] for tool in robot_tools()} or len(call["arguments"]) > 8000:
                raise ValueError("Invalid tool")
            arguments = json.loads(call["arguments"])
            TOOL_MODELS[name].model_validate(arguments)
        except (ValueError, TypeError):
            arguments = None
            result = {"status": "error", "error": "INVALID_ARGUMENT", "message": "Use a published robot tool and valid arguments"}
        else:
            self._check_voice_live(worker)
            self.state["phase"] = "acting"
            self._trace("tool", "Voice robot tool requested", {"tool": name, "arguments": arguments, "call_id": call_id})
            self.pending_command = Command(run_id=self.settings.run_id, episode_epoch=self.settings.episode_epoch,
                                           action_id=f"{self.state['session_id']}:{self.state['turns']}",
                                           observation_seq=self.observation.seq, tool=name, arguments=arguments)
            self.pending_call_id = call_id
            result = (await worker.execute(self.pending_command, assisted=False)).model_dump()
            self.pending_command = None
        self.actions += 1
        if name == "stop" and result["status"] == "ok":
            self.idle_outcome = "Task unachievable" if arguments.get("task_status") == "unachievable" else "Task ended"
            self._set_outcome(arguments.get("task_status") or "ended", arguments.get("reason") or text or "Robot stopped", "agent")
        model_result = model_tool_result(result)
        self._trace("result", "Robot tool result", {"tool": name, "call_id": call_id,
            "result": compact_numbers(result), "model_result": model_result})
        self._check_voice_live(worker)
        await connection.send({"type": "conversation.item.create", "item": {
            "type": "function_call_output", "call_id": call_id, "output": feedback_json(model_result)}})
        await self._respond(worker, connection, allow_tools=name != "stop" and self.actions < 8, call_id=call_id)

    async def _run_voice(self, worker, browser):
        model = None
        tasks = []
        try:
            model = self.voice_factory(self.voice_config)
            async with asyncio.timeout(600):
                async with model.connect() as connection:
                    await connection.send({"type": "session.update", "session": realtime_session(self.voice_config, self.settings.goal)})
                    async with asyncio.timeout(20):
                        while True:
                            event = await connection.recv()
                            if event["type"] == "session.updated":
                                break
                            if event["type"] == "error":
                                raise ValueError("Foundry Realtime session configuration failed")
                    self._check_voice_live(worker)
                    self.state["phase"] = "voice_ready"
                    await browser.send_json({"type": "ready", "session_id": self.state["session_id"]})
                    tasks = [asyncio.create_task(self._browser_input(worker, browser, connection)),
                             asyncio.create_task(self._receive_model(worker, browser, connection))]
                    try:
                        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in done:
                            await task
                    finally:
                        self.cancelled = True
                        worker.stop()
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
        except (asyncio.CancelledError, WebSocketDisconnect):
            self.state.update(phase="stopped", message="Voice control ended")
        except TimeoutError:
            self.state.update(phase="error", error="Voice session timed out; robot stopped")
        except ValueError as error:
            self.state.update(phase="error", error=str(error))
        except Exception:
            self.state.update(phase="error", error="Foundry Realtime connection failed. Check the resource endpoint, deployment, Azure access, and network.")
        finally:
            idle_task = self._cancel_idle()
            if self.state["phase"] == "error":
                self._set_outcome("error", self.state["error"])
            elif self.state["outcome"] is None:
                self._set_outcome("interrupted", "Voice control ended")
            if idle_task:
                await asyncio.gather(idle_task, return_exceptions=True)
            worker.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if not worker.closed:
                await worker.call(lambda sim: sim.hold_current())
            if self.pending_command is not None:
                result = worker.latest.get("result")
                if result and result.get("action_id") == self.pending_command.action_id:
                    self._trace("result", "Robot tool result", {"tool": self.pending_command.tool,
                                "call_id": self.pending_call_id, "result": result})
            try:
                if model:
                    await model.close()
            finally:
                self.active = False
                if self.state["phase"] not in ("error", "stopped"):
                    self.state.update(phase="stopped", message="Voice control ended")
                self.state["next_feedback_at"] = None
                self._trace("session", "Voice session ended", {"status": self.state["phase"],
                            "message": self.state["error"] or "Voice connection closed"})
                try:
                    await browser.send_json({"type": "ended", "error": self.state["error"]})
                except (RuntimeError, WebSocketDisconnect):
                    pass