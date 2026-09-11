import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.agent import AgentController, AgentStart, ChatStart, FeedbackRate, FoundryConfig, InteractionMode, robot_tools
from backend.challenges import ChallengeLoad, PRESETS, get_challenge
from backend.contracts import Command, ManualPlacement, tool_schemas
from backend.local_progress import read_navigation_progress
from backend.robot import calibration
from backend.materials import TEXTURE_NAMES, TEXTURE_ROOT
from backend.policy import PolicyConfig, check_policy_readiness
from backend.realtime import FoundryRealtime, RealtimeConfig, VoiceController, VoiceStart
from backend.simulation import MotionError
from backend.worker import SimulationWorker


class Lab:
    def __init__(self):
        self.worker = None
        self.epoch = 0
        self.lock = asyncio.Lock()
        self.connections = 0
        self.agent = AgentController()
        self.realtime_config = RealtimeConfig()
        self.voice_factory = FoundryRealtime
        self.challenge_id = "bench"
        self.interaction_mode = "chat"

    def state(self):
        return {**self.worker.latest, "agent": self.agent.public(), "realtime": self.realtime_config.public(),
            "interaction_mode": self.interaction_mode}

    async def reset(self, challenge_id=None):
        async with self.lock:
            await self.agent.halt("Episode reset")
            if self.worker:
                await self.worker.close()
            self.epoch += 1
            selected = self.challenge_id if challenge_id is None else challenge_id
            replacement = SimulationWorker(epoch=self.epoch, challenge=get_challenge(selected))
            await asyncio.wrap_future(replacement.ready)
            self.worker = replacement
            self.challenge_id = selected
            self.agent = AgentController(self.agent.config, self.agent.model_factory, policy_factory=self.agent.policy_factory)
            self.interaction_mode = "chat"
        return self.state()


lab = Lab()


@asynccontextmanager
async def lifespan(app):
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    configuration_error = None
    try:
        lab.agent = AgentController(FoundryConfig.from_environment())
    except ValueError:
        lab.agent = AgentController()
        configuration_error = "Invalid Foundry configuration. Check the endpoint and model profiles."
    await lab.reset("bench")
    try:
        lab.realtime_config = RealtimeConfig.from_environment()
    except ValueError:
        lab.realtime_config = RealtimeConfig()
        configuration_error = "Invalid Foundry Realtime configuration. Use a resource endpoint and a Realtime deployment."
    lab.agent.state["error"] = configuration_error
    yield
    await lab.agent.halt("Server shutdown")
    await lab.worker.close()


app = FastAPI(title="Embodied Robot Lab", lifespan=lifespan)


@app.middleware("http")
async def local_origin_guard(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and origin not in {f"http://{request.headers.get('host')}", f"https://{request.headers.get('host')}"}:
        return Response("Same-origin access required", status_code=403)
    return await call_next(request)


@app.get("/api/state")
async def state():
    return lab.state()


@app.get("/api/local-navigation/status")
def local_navigation_status(response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return {"test": read_navigation_progress()}
    except (OSError, ValueError) as error:
        raise HTTPException(503, "Local navigation status is temporarily unavailable") from error


@app.get("/api/textures/{name}.png")
async def texture(name: str):
    if name not in TEXTURE_NAMES:
        raise HTTPException(404, "Unknown material")
    return FileResponse(TEXTURE_ROOT / f"{name}.png", media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/calibration")
async def robot_calibration():
    return calibration()


@app.get("/api/tools")
async def tools(execution_mode: Literal["single_step", "navigation_plan", "supervised_policy"] = "single_step"):
    return robot_tools(execution_mode) if execution_mode != "single_step" else tool_schemas()


@app.post("/api/command")
async def command(envelope: Command):
    if envelope.tool == "stop":
        await lab.agent.halt()
        lab.worker.stop()
        return await lab.worker.execute(envelope)
    if lab.agent.active or lab.lock.locked() or lab.worker.latest.get("busy"):
        raise HTTPException(409, "Take manual control before issuing a command; another controller is active")
    async with lab.lock:
        return await lab.worker.execute(envelope)


@app.post("/api/robot/placement")
async def place_robot(placement: ManualPlacement):
    if lab.agent.active or lab.lock.locked() or lab.worker.latest.get("busy"):
        raise HTTPException(409, "Take manual control and wait for motion to finish before positioning the robot.")
    async with lab.lock:
        try:
            await lab.worker.reposition(placement)
        except MotionError as error:
            raise HTTPException(409 if error.code in {"STALE_STATE", "CANCELLED", "OBJECT_HELD"} else 422, str(error)) from error
    return lab.state()


@app.post("/api/stop")
async def stop():
    lab.worker.stop()
    await lab.agent.halt()
    return {"stopped": True}


@app.post("/api/resume")
@app.post("/api/agent/takeover")
async def resume():
    lab.worker.stop()
    await lab.agent.halt("Manual takeover")
    async with lab.lock:
        await lab.worker.resume_manual()
    return lab.state()


@app.post("/api/reset")
async def reset():
    lab.worker.stop()
    await lab.agent.halt("Episode reset")
    return await lab.reset()


@app.get("/api/challenges")
async def challenges():
    return [preset.public() for preset in PRESETS.values()]


@app.post("/api/challenges/load")
async def load_challenge(selection: ChallengeLoad):
    lab.worker.stop()
    await lab.agent.halt("Challenge changed")
    return await lab.reset(selection.challenge_id)


@app.get("/api/agent")
async def agent_state():
    return lab.agent.public()


@app.get("/api/agent/trace")
async def agent_trace(after: int = Query(default=0, ge=0), session_id: str | None = None):
    if session_id is not None and session_id != lab.agent.state["session_id"]:
        raise HTTPException(409, "Trace session changed")
    return lab.agent.trace(after)


@app.get("/api/agent/trace/{session_id}/{event_id}/frame")
async def agent_trace_frame(session_id: str, event_id: int, index: int = Query(default=0, ge=0, le=7)):
    if session_id != lab.agent.state["session_id"] or event_id not in lab.agent.trace_images:
        raise HTTPException(404, "Frame no longer retained in this exchange feed")
    images = lab.agent.trace_image_batches.get(event_id, [lab.agent.trace_images[event_id]])
    if index >= len(images):
        raise HTTPException(404, "Camera batch index not retained")
    return Response(images[index], media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/api/agent/config")
async def configure_agent(config: FoundryConfig):
    if lab.agent.active or lab.lock.locked():
        raise HTTPException(409, "Stop LLM control before changing its connection")
    if lab.agent.state["auto_wake"]:
        await lab.agent.halt("Model configuration changed")
    lab.agent.config = config
    lab.agent.state["error"] = None
    return lab.agent.public()


@app.post("/api/agent/mode")
async def change_interaction_mode(selection: InteractionMode):
    if selection.run_id != lab.worker.latest["run_id"] or selection.episode_epoch != lab.worker.latest["episode_epoch"]:
        raise HTTPException(409, "Episode changed before mode switch")
    if lab.lock.locked():
        raise HTTPException(409, "Wait for the current manual operation to finish")
    async with lab.lock:
        lab.worker.stop()
        await lab.agent.halt("Interaction mode changed")
        lab.interaction_mode = selection.mode
    return lab.state()


@app.post("/api/policy/check")
async def policy_readiness(config: PolicyConfig):
    return await check_policy_readiness(config, lab.agent.policy_factory)


@app.post("/api/agent/chat")
async def chat_message(settings: ChatStart):
    return await start_agent(settings)


@app.post("/api/agent/start")
async def start_agent(settings: AgentStart):
    if lab.interaction_mode != "chat":
        raise HTTPException(409, "Switch to Chat before sending a text request")
    if lab.lock.locked():
        raise HTTPException(409, "A manual command or episode transition is active")
    if not lab.connections:
        raise HTTPException(409, "Keep an operator interface connected while the LLM controls the robot")
    async with lab.lock:
        if settings.execution_mode == "supervised_policy":
            if lab.agent.active or lab.worker.latest.get("busy"):
                raise HTTPException(409, "Stop current control before starting a policy session")
            worker = lab.worker
            stop_revision = worker.stop_revision
            readiness = await check_policy_readiness(settings.policy, lab.agent.policy_factory)
            if not readiness["ready"]:
                raise HTTPException(409, readiness["message"])
            if worker is not lab.worker or worker.stop_revision != stop_revision or not lab.connections:
                raise HTTPException(409, "Policy start invalidated by Stop or episode/operator change")
        try:
            lab.agent.start(lab.worker, settings)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
    return lab.agent.public()


@app.post("/api/agent/rate")
async def agent_rate(settings: FeedbackRate):
    lab.agent.set_rate(settings)
    return lab.agent.public()


@app.post("/api/voice/config")
async def configure_voice(config: RealtimeConfig):
    if lab.agent.active or lab.lock.locked():
        raise HTTPException(409, "Stop robot control before changing the voice connection")
    lab.realtime_config = config
    return config.public()


@app.websocket("/api/voice")
async def voice_session(socket: WebSocket):
    host = socket.headers.get("host")
    if socket.headers.get("origin") not in (None, f"http://{host}", f"https://{host}"):
        await socket.close(code=1008)
        return
    await socket.accept()
    controller = None
    try:
        async with asyncio.timeout(10):
            raw = await socket.receive_text()
        if len(raw) > 1024:
            raise ValueError("Invalid voice start request")
        settings = VoiceStart.model_validate_json(raw)
        if lab.lock.locked() or not lab.connections or lab.interaction_mode != "voice":
            raise ValueError("Keep the robot view connected and wait for manual motion to finish")
        async with lab.lock:
            if settings.run_id != lab.worker.latest["run_id"] or settings.episode_epoch != lab.worker.latest["episode_epoch"]:
                raise ValueError("Episode changed before voice takeover")
            if lab.agent.active and lab.agent.state["mode"] == "voice":
                raise ValueError("Another microphone already controls the robot")
            if not lab.realtime_config.public()["configured"]:
                raise ValueError("Configure the Foundry Realtime resource endpoint and deployment first")
            await lab.agent.halt("Voice takeover")
            controller = VoiceController(lab.agent.config, lab.realtime_config, lab.voice_factory)
            controller.start_voice(lab.worker, settings, socket)
            lab.agent = controller
        await asyncio.shield(controller.task)
    except (ValueError, RuntimeError):
        await socket.send_json({"type": "ended", "error": "Cannot start voice control. Check the Realtime connection, episode, and controller availability."})
    except TimeoutError:
        await socket.send_json({"type": "ended", "error": "Voice connection timed out"})
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        if controller and controller.active:
            await controller.halt("Voice disconnected")
        try:
            await socket.close()
        except (RuntimeError, WebSocketDisconnect):
            pass


@app.get("/api/camera/{run_id}/{reference}")
async def live_camera_frame(run_id: str, reference: str):
    worker = lab.worker
    image = worker.camera_frames.get(reference) if not worker.closed and run_id == worker.latest.get("run_id") else None
    if image is None:
        raise HTTPException(404, "Live camera frame no longer available")
    return Response(image, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/frames/{run_id}/{reference}")
async def frame(run_id: str, reference: str):
    worker = lab.worker
    if run_id != worker.latest.get("run_id") or reference not in worker.sim.frames:
        raise HTTPException(404, "Frame no longer in live memory")
    return Response(worker.sim.frame(reference), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


@app.websocket("/api/live")
async def live(socket: WebSocket):
    host = socket.headers.get("host")
    if socket.headers.get("origin") not in (None, f"http://{host}", f"https://{host}"):
        await socket.close(code=1008)
        return
    await socket.accept()
    lab.connections += 1
    try:
        while True:
            await socket.send_json(lab.state())
            await asyncio.sleep(.05)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        lab.connections -= 1
        if lab.connections == 0:
            lab.worker.stop()
            await lab.agent.halt("Operator disconnected")


dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")