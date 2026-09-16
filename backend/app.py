import asyncio
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from anyio import CancelScope
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.agent import AgentController, AgentStart, ChatStart, FeedbackRate, FoundryConfig, InteractionMode, RunInstruction, robot_tools
from backend.challenges import ChallengeLoad, PRESETS, furniture_circuit, get_challenge, shared_apartment
from backend.contracts import Command, ManualPlacement, SpatialSettings, StrictModel, tool_schemas
from backend.continuous_navigation import ContinuousScan, ContinuousTarget
from backend.local_progress import read_navigation_progress
from backend.memory_session import MemoryRequest, configure_memory, memory_command, memory_state
from backend.local_navigation import ResidentNavigationModel
from backend.robot import calibration
from backend.materials import TEXTURE_NAMES, TEXTURE_ROOT
from backend.policy import PolicyConfig, check_policy_readiness
from backend.realtime import FoundryRealtime, RealtimeConfig, VoiceController, VoiceStart
from backend.simulation import MotionError
from backend.worker import SimulationWorker
from backend.ros_navigation import RosBridgeStatus, RosGoalResult, RosStart, RosVelocity
from backend.home_mission import HomeRequest
from backend.preferences import PreferenceStore, PreferencesPatch


class Lab:
    def __init__(self):
        self.worker = None
        self.render_resources = None
        self.epoch = 0
        self.lock = asyncio.Lock()
        self.connections = 0
        self.local_navigation = ResidentNavigationModel()
        self.agent = AgentController(local_navigation_factory=self.local_navigation)
        self.realtime_config = RealtimeConfig()
        self.voice_factory = FoundryRealtime
        self.challenge_id = "bench"
        self.environment = "standalone"
        self.environment_instance_id = None
        self.orbit_target = "table"
        self.orbit_direction = "clockwise"
        self.interaction_mode = "chat"
        self.home_recording = None
        self.recording_enabled = True
        self.recording_root = None
        self.reuse_saved_map = True
        self.preference_error = None
        self.power_off_pending = False
        self.robot_on = True

    async def finish_home_recording(self, reason=None):
        recording = self.home_recording
        if recording:
            await asyncio.shield(recording.finish(reason))
            self.agent.recording_directory = recording.recorder.directory
            self.agent.recording_error = recording.error
            if self.home_recording is recording:
                self.home_recording = None

    def state(self):
        mission = self.worker.home_mission
        home = mission.home if mission else None
        return {**self.worker.latest, "power": self.worker.power_state(), "agent": self.agent.public(), "realtime": self.realtime_config.public(),
            "memory": {"scope": self.worker.memory.scope.model_dump(), "profile": self.worker.memory_profile,
                "error": self.worker.memory.error} if self.worker.memory else None,
            "recording": self.recording_status(),
            "preference_error": self.preference_error,
            "map_setup": {"reuse_saved_map": self.reuse_saved_map, "map_id": home.identity if home else None,
                "name": home.name if home else None, "revision": home.revision if home else None,
                "localization": mission.localization["status"] if mission else "unlocalized"},
            "interaction_mode": self.interaction_mode, "local_navigation_model": self.local_navigation.public()}

    def recording_status(self):
        from backend.session_recording import SESSION_RESULTS_ROOT
        recorder = self.worker.recorder if self.worker else None
        home = self.home_recording
        active = self.agent.recording_active or bool(home and not home.finished)
        directory = recorder.directory if recorder else home.recorder.directory if home else self.agent.recording_directory
        return {"enabled": self.recording_enabled, "directory": str(self.recording_root or SESSION_RESULTS_ROOT),
            "active": active, "status": "recording" if recorder else "finalizing" if active else "ready" if self.recording_enabled else "off",
            "run_directory": str(directory) if directory else None,
            "samples": recorder.sample_sequence if recorder else None,
            "error": (home.error if home else None) or self.agent.recording_error or (self.worker.latest.get("recording_error") if self.worker else None)}

    def configure_recording(self, preferences):
        from backend.session_recording import recording_root
        self.recording_enabled = preferences.get("recording_enabled", True)
        self.recording_root = recording_root(preferences["recording_directory"]) if preferences.get("recording_directory") else None
        self.agent.record_sessions = self.recording_enabled
        self.agent.recording_root = self.recording_root

    async def reset(self, challenge_id=None, orbit_target=None, orbit_direction=None, environment=None, reuse_saved_map=None, environment_instance_id=None):
        async with self.lock:
            await self.agent.halt("Episode reset")
            await self.finish_home_recording("episode_reset")
            spatial_enabled = bool(self.worker and not self.worker.closed and self.worker.spatial_enabled)
            if self.worker:
                await self.worker.close()
            self.epoch += 1
            selected = self.challenge_id if challenge_id is None else challenge_id
            target = self.orbit_target if challenge_id is None else orbit_target or "table"
            direction = self.orbit_direction if challenge_id is None else orbit_direction or "clockwise"
            layout = self.environment if challenge_id is None else environment or "standalone"
            instance_id = self.environment_instance_id if challenge_id is None else environment_instance_id
            reuse_map = self.reuse_saved_map if reuse_saved_map is None else reuse_saved_map
            rendering = os.environ.get("MILO_RENDERER", "enhanced")
            if rendering == "enhanced":
                from backend.camera import EnhancedResources
                if self.render_resources and (self.render_resources.renderer.closed or self.render_resources.renderer.process.poll() is not None):
                    await asyncio.to_thread(self.render_resources.close)
                    self.render_resources = None
                if self.render_resources is None:
                    self.render_resources = await asyncio.to_thread(EnhancedResources)
            challenge = (shared_apartment(selected, direction) if layout == "shared_apartment_v1" else
                furniture_circuit(target, direction) if selected == "furniture_circuit" else get_challenge(selected))
            replacement = SimulationWorker(epoch=self.epoch, challenge=challenge,
                rendering=rendering, render_resources=self.render_resources if rendering == "enhanced" else None)
            await asyncio.wrap_future(replacement.ready)
            home_state = await replacement.home_state()
            saved_home = next((item for item in home_state["maps"] if item["environment_id"] == home_state["environment_id"]
                and item["environment_revision"] == "legacy"), None)
            if saved_home and reuse_map:
                await replacement.home_command(HomeRequest(run_id=replacement.latest["run_id"], episode_epoch=self.epoch,
                    action="load_map", map_id=saved_home["map_id"]))
            await configure_memory(replacement, instance_id=instance_id, fresh=not reuse_map, name="Fresh exploration" if not reuse_map else "Default knowledge", reuse=reuse_map)
            if spatial_enabled:
                await replacement.configure_spatial(SpatialSettings(run_id=replacement.latest["run_id"],
                    episode_epoch=self.epoch, enabled=True))
            if not self.robot_on:
                await replacement.set_power(False)
            self.worker = replacement
            self.challenge_id = selected
            self.environment = layout
            self.environment_instance_id = instance_id
            self.reuse_saved_map = reuse_map
            self.orbit_target, self.orbit_direction = target, direction
            recording_evidence = self.agent.recording_evidence
            self.agent = AgentController(self.agent.config, self.agent.model_factory, policy_factory=self.agent.policy_factory,
                local_navigation_factory=self.agent.local_navigation_factory)
            from backend.agent import ConfiguredModel
            self.agent.record_sessions = self.recording_enabled
            self.agent.recording_root = self.recording_root
            self.agent.recording_evidence = "real_model" if self.agent.model_factory is ConfiguredModel else (
                "scripted_test" if recording_evidence == "scripted_test" else "unknown")
            self.interaction_mode = "chat"
        return self.state()


lab = Lab()


@asynccontextmanager
async def lifespan(app):
    lab.robot_on = True
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    configuration_error = None
    lab.local_navigation = ResidentNavigationModel()
    try:
        lab.agent = AgentController(FoundryConfig.from_environment(), local_navigation_factory=lab.local_navigation)
    except ValueError:
        lab.agent = AgentController(local_navigation_factory=lab.local_navigation)
        configuration_error = "Invalid Foundry configuration. Check the endpoint and model profiles."
    lab.preference_error = None
    try:
        saved = await asyncio.to_thread(PreferenceStore().read)
        saved_scene = saved["scene"]
        lab.configure_recording(saved["preferences"])
    except (OSError, sqlite3.Error, ValueError):
        saved_scene = None
        lab.configure_recording({})
        lab.preference_error = "Saved preferences are unavailable; the original store was left unchanged."
    selection = ChallengeLoad.model_validate(saved_scene or {"challenge_id": "bench"})
    await lab.reset(selection.challenge_id, selection.orbit_target, selection.orbit_direction,
        selection.environment, selection.reuse_saved_map, selection.environment_instance_id)
    if saved_scene is not None:
        lab.worker.stop()
        await lab.worker.hold_stopped()
    try:
        lab.realtime_config = RealtimeConfig.from_environment()
    except ValueError:
        lab.realtime_config = RealtimeConfig()
        configuration_error = "Invalid Foundry Realtime configuration. Use a resource endpoint and a Realtime deployment."
    lab.agent.state["error"] = configuration_error
    try:
        yield
    finally:
        try:
            await lab.agent.halt("Server shutdown")
            await lab.finish_home_recording("server_shutdown")
        finally:
            try:
                await lab.local_navigation.close()
            finally:
                await lab.worker.close()
                if lab.render_resources:
                    await asyncio.to_thread(lab.render_resources.close)
                    lab.render_resources = None


app = FastAPI(title="Embodied Robot Lab", lifespan=lifespan)


@app.middleware("http")
async def local_origin_guard(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and origin not in {f"http://{request.headers.get('host')}", f"https://{request.headers.get('host')}"}:
        return Response("Same-origin access required", status_code=403)
    if request.method == "POST" and lab.worker and not lab.worker.powered and request.url.path not in {
            "/api/power", "/api/stop", "/api/command", "/api/reset", "/api/challenges/load", "/api/preferences",
            "/api/agent/config", "/api/voice/config", "/api/memory"}:
        return JSONResponse({"detail": "Turn the robot on before starting work"}, status_code=409)
    ros = getattr(lab.worker, "ros_navigation", None)
    if request.method == "POST" and ros and ros.active and request.url.path not in {
            "/api/ros/velocity", "/api/ros/heartbeat", "/api/ros/result", "/api/agent/instruction",
            "/api/power", "/api/stop", "/api/resume", "/api/agent/takeover", "/api/reset", "/api/challenges/load", "/api/preferences"}:
        return Response("Stop ROS navigation before changing control", status_code=409)
    response = await call_next(request)
    if request.url.path in {"/", "/index.html"}:
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/state")
async def state():
    return lab.state()


@app.get("/api/preferences")
async def read_preferences():
    try:
        return JSONResponse(await asyncio.to_thread(PreferenceStore().read), headers={"Cache-Control": "no-store"})
    except (OSError, sqlite3.Error, ValueError):
        raise HTTPException(503, "Saved preferences are unavailable; the original store was left unchanged.") from None


@app.post("/api/preferences")
async def save_preferences(patch: PreferencesPatch):
    if patch.model_fields_set & {"recording_enabled", "recording_directory"}:
        if lab.lock.locked() or lab.agent.active or lab.agent.recording_active or lab.worker.latest.get("busy") or lab.worker.recorder:
            raise HTTPException(409, "Wait for the current run and recording to finish before changing recording settings")
        async with lab.lock:
            try:
                from backend.session_recording import check_recording_directory
                if patch.recording_directory is not None:
                    await asyncio.to_thread(check_recording_directory, patch.recording_directory)
                saved = await asyncio.to_thread(PreferenceStore().update, patch)
                lab.configure_recording(saved["preferences"])
                return saved
            except (OSError, sqlite3.Error, ValueError):
                raise HTTPException(400, "Recording folder could not be saved. Choose a writable local folder.") from None
    try:
        return await asyncio.to_thread(PreferenceStore().update, patch)
    except (OSError, sqlite3.Error, ValueError):
        raise HTTPException(503, "Preferences could not be saved.") from None


@app.get("/api/recording")
async def recording_status():
    return JSONResponse(lab.recording_status(), headers={"Cache-Control": "no-store"})


@app.get("/api/test-variant")
async def test_variant(execution_mode: Literal["luna_continuous", "luna_navigation"] = "luna_continuous",
    model_id: str = "luna", reasoning: Literal["none", "low", "medium", "high"] = "high",
        skill_composer: bool = False, navigation_backend: Literal["builtin", "nav2"] = "builtin"):
    from backend.experiment_variants import variant_snapshot
    profile = next((entry for entry in lab.agent.config.models if entry.id == model_id), None)
    if profile is None:
        raise HTTPException(404, "Model profile unavailable")
    try:
        settings = AgentStart(run_id=lab.worker.latest["run_id"], episode_epoch=lab.worker.epoch,
            execution_mode=execution_mode, model_id=model_id, reasoning=reasoning,
            skill_composer=skill_composer, navigation_backend=navigation_backend, goal="Architecture preview only")
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    return JSONResponse({**variant_snapshot(settings, profile), "supports_ai_generated_routes": False,
        "supports_skill_composer": True, "supports_navigation_backend": True, "nav2": ros_status()},
        headers={"Cache-Control": "no-store"})


@app.get("/api/test-results")
async def test_results():
    from backend.saved_results import saved_results
    return JSONResponse(await asyncio.to_thread(saved_results), headers={"Cache-Control": "no-store"})


@app.get("/api/test-results/{batch_id}/images/{trial_index}")
async def test_result_image(batch_id: str, trial_index: int):
    from backend.saved_results import terminal_image
    try:
        path = await asyncio.to_thread(terminal_image, batch_id, trial_index)
    except (OSError, ValueError):
        raise HTTPException(404, "Saved camera image unavailable")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


@app.get("/api/test-results/{batch_id}/trajectories/{trial_index}")
async def test_result_trajectory(batch_id: str, trial_index: int):
    from backend.saved_results import recorded_trajectory
    try:
        route = await asyncio.to_thread(recorded_trajectory, batch_id, trial_index)
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        raise HTTPException(404, "Saved trajectory unavailable")
    return JSONResponse(route, headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


@app.get("/api/test-results/{batch_id}/replay/{trial_index}")
async def test_result_replay(batch_id: str, trial_index: int):
    from backend.saved_results import recorded_replay
    try:
        result = await asyncio.to_thread(recorded_replay, batch_id, trial_index)
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        raise HTTPException(404, "Recorded replay unavailable")


@app.get("/api/test-results/{batch_id}/replay/{trial_index}/media/{name}")
async def test_result_replay_media(batch_id: str, trial_index: int, name: str):
    from backend.saved_results import recorded_replay_media
    try:
        data, content_type = await asyncio.to_thread(recorded_replay_media, batch_id, trial_index, name)
        headers = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}
        return Response(data, media_type=content_type, headers=headers) if isinstance(data, bytes) else JSONResponse(data, headers=headers)
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        raise HTTPException(404, "Recorded media unavailable")


@app.get("/api/spatial")
async def spatial_state(response: Response, request: Request):
    response.headers["Cache-Control"] = "no-store"
    worker = lab.worker
    try:
        return await worker.call(lambda sim: worker.spatial_state(include_motion_zones=request.headers.get("x-milo-motion-zones") == "1"))
    except RuntimeError as error:
        if worker.closed or worker is not lab.worker:
            raise HTTPException(409, "Spatial sensor episode changed; refresh state") from error
        raise


@app.get("/api/home")
async def home_state(compact: bool = False):
    try:
        return JSONResponse(await lab.worker.home_state(compact=compact), headers={"Cache-Control": "no-store"})
    except RuntimeError as error:
        raise HTTPException(409, "Home map episode changed") from error


@app.get("/api/memory")
async def get_memory():
    try:
        return JSONResponse(await memory_state(lab.worker), headers={"Cache-Control": "no-store"})
    except (ValueError, RuntimeError) as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/memory")
async def change_memory(request: MemoryRequest):
    worker = lab.worker
    if lab.lock.locked() or lab.agent.active or worker.latest.get("busy") or not lab.connections:
        raise HTTPException(409, "Keep the operator connected and stop the active task before changing knowledge")
    async with lab.lock:
        try:
            return await memory_command(worker, request)
        except (ValueError, RuntimeError, OSError, sqlite3.Error) as error:
            raise HTTPException(409, str(error)) from error


@app.post("/api/home")
async def home_operation(request: HomeRequest):
    worker = lab.worker
    if request.run_id != worker.latest["run_id"] or request.episode_epoch != worker.epoch:
        raise HTTPException(409, "Home request belongs to another episode")
    if request.action == "cancel_task":
        worker.stop()
        await lab.agent.halt("Home task cancelled")
        result = await worker.home_command(request)
        await lab.finish_home_recording("cancelled")
        return result
    if lab.lock.locked() or lab.agent.active or worker.latest.get("busy") or not lab.connections:
        raise HTTPException(409, "Keep the operator connected and take manual control before changing the map")
    async with lab.lock:
        from backend.session_recording import HomeSessionRecording
        stop_revision, task_revision = worker.stop_revision, worker.task_revision
        starts = request.action in {"start_mapping", "start_exploration", "continue_mapping", "guided_to", "load_map", "localize", "navigate_to", "explore", "explore_frontier"}
        if starts and lab.home_recording and (lab.home_recording.done.is_set() or request.action != "start_mapping"):
            await lab.finish_home_recording("workflow_transition")
        if lab.home_recording and lab.home_recording.finished:
            lab.home_recording = None
        if starts and lab.home_recording is None and lab.recording_enabled:
            evidence = "scripted_test" if lab.agent.recording_evidence == "scripted_test" else "operator_session"
            try:
                lab.home_recording = HomeSessionRecording(worker, request, evidence, root=lab.recording_root)
                await lab.home_recording.start()
            except (OSError, ValueError, RuntimeError) as error:
                lab.home_recording = None
                raise HTTPException(503, "Spatial recording could not start; no map command executed") from error
        recording = lab.home_recording
        if recording:
            recording.executing = True
        try:
            result = await worker.home_command(request, stop_revision, task_revision, operator_review=request.action == "review_room")
            if recording:
                await worker.call(lambda sim: recording.recorder.capture(worker))
                if request.action in {"save_map", "localize", "load_map"}:
                    await lab.finish_home_recording(request.action)
            return result
        except (ValueError, MotionError, RuntimeError) as error:
            if recording:
                recording.request_error = str(error)
                await lab.finish_home_recording("rejected")
            raise HTTPException(409, str(error)) from error
        finally:
            if recording:
                recording.executing = False


@app.get("/api/home/{map_id}/rooms/{observation_id}/image.png")
async def home_room_image(map_id: str, observation_id: str):
    mission = lab.worker.home_mission
    if mission is None or mission.home is None or mission.home.identity != map_id:
        raise HTTPException(404, "Load the matching map to view room evidence")
    try:
        image = await asyncio.to_thread(mission.store.room_image, map_id, observation_id)
        return Response(image, media_type="image/png", headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


@app.get("/api/home/{map_id}/objects/{observation_id}/image.png")
async def home_object_image(map_id: str, observation_id: str):
    mission = lab.worker.home_mission
    if mission is None or mission.home is None or mission.home.identity != map_id:
        raise HTTPException(404, "Load the matching map to view object evidence")
    try:
        image = await asyncio.to_thread(mission.store.object_image, map_id, observation_id)
        return Response(image, media_type="image/png", headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


def require_ros():
    if os.environ.get("MILO_ROS_ENABLED") != "1":
        raise HTTPException(404, "ROS bridge is disabled")


def require_ros_owner():
    worker = lab.worker
    session = worker.ros_navigation
    agent_owned = bool(session and session.active and session.owner == "agent"
        and lab.agent.state.get("navigation_backend") == "nav2")
    if lab.lock.locked() or (lab.agent.active and not agent_owned) or not lab.connections:
        raise HTTPException(409, "ROS motion ownership is unavailable")
    return worker


@app.get("/api/ros/status")
def ros_status():
    enabled = os.environ.get("MILO_ROS_ENABLED") == "1"
    return {**lab.worker.nav2_status(), "enabled": enabled} if enabled else {
        "enabled": False, "ready": False, "message": "Start with -Nav2 to enable the bridge"}


@app.post("/api/ros/heartbeat")
async def ros_heartbeat(status: RosBridgeStatus):
    require_ros()
    try:
        return await lab.worker.ros_heartbeat(status)
    except (MotionError, ValueError, RuntimeError) as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/ros/result")
async def ros_result(result: RosGoalResult):
    require_ros()
    worker = require_ros_owner()
    try:
        return await worker.ros_result(result)
    except (MotionError, ValueError, RuntimeError) as error:
        raise HTTPException(409, str(error)) from error


@app.get("/api/ros/odometry")
async def ros_odometry():
    require_ros()
    try:
        return JSONResponse(await lab.worker.ros_odometry(), headers={"Cache-Control": "no-store"})
    except RuntimeError as error:
        raise HTTPException(409, "ROS odometry episode changed") from error


@app.get("/api/ros/sensors")
async def ros_sensors(response: Response):
    require_ros()
    response.headers["Cache-Control"] = "no-store"
    worker = lab.worker
    try:
        return JSONResponse(await worker.ros_sensors(), headers={"Cache-Control": "no-store"})
    except RuntimeError as error:
        raise HTTPException(409, "ROS sensor episode changed") from error


@app.post("/api/ros/start")
async def ros_start(request: RosStart):
    require_ros()
    if lab.lock.locked() or lab.agent.active or lab.worker.latest.get("busy") or not lab.connections:
        raise HTTPException(409, "Keep the operator interface connected and take manual control")
    async with lab.lock:
        try:
            return await lab.worker.start_ros(request)
        except (MotionError, ValueError, RuntimeError) as error:
            raise HTTPException(409, str(error)) from error


@app.post("/api/ros/velocity")
async def ros_velocity(command: RosVelocity):
    require_ros()
    worker = require_ros_owner()
    try:
        return await worker.ros_velocity(command)
    except (MotionError, ValueError, RuntimeError) as error:
        raise HTTPException(409, str(error)) from error


@app.post("/api/spatial")
async def configure_spatial(settings: SpatialSettings):
    if lab.lock.locked() or (not settings.enabled and (lab.agent.active or lab.worker.latest.get("busy"))):
        raise HTTPException(409, "Take manual control before changing spatial sensing")
    async with lab.lock:
        try:
            return await lab.worker.configure_spatial(settings)
        except MotionError as error:
            raise HTTPException(409, str(error)) from error


async def continuous_operation(request, scan=False):
    if lab.agent.active or lab.lock.locked() or lab.worker.latest.get("busy"):
        raise HTTPException(409, "Take manual control and wait for motion to finish")
    if not lab.connections:
        raise HTTPException(409, "Keep the operator interface connected during continuous navigation")
    async with lab.lock:
        try:
            return await (lab.worker.scan_continuous(request) if scan else lab.worker.start_continuous(request))
        except (MotionError, ValueError) as error:
            raise HTTPException(409, str(error)) from error


@app.post("/api/continuous/scan")
async def continuous_scan(request: ContinuousScan):
    return await continuous_operation(request, scan=True)


@app.post("/api/continuous/start")
async def continuous_start(request: ContinuousTarget):
    return await continuous_operation(request)


@app.get("/api/spatial/{run_id}/{sequence}/{kind}")
async def spatial_frame(run_id: str, sequence: int, kind: Literal["rgb.png", "depth.png", "depth.json"]):
    worker = lab.worker
    def read(sim):
        if sim.run_id != run_id or sequence not in worker.spatial_frames:
            raise HTTPException(404, "Spatial frame no longer retained")
        return worker.spatial_frames[sequence]
    try:
        observation, rgb, depth = await worker.call(read)
    except RuntimeError as error:
        if worker.closed or worker is not lab.worker:
            raise HTTPException(404, "Spatial frame belongs to a closed episode") from error
        raise
    if kind == "depth.json":
        return Response(observation.model_dump_json(), media_type="application/json", headers={"Cache-Control": "no-store"})
    return Response(rgb if kind == "rgb.png" else depth, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/local-navigation/status")
def local_navigation_status(response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return {"test": read_navigation_progress()}
    except (OSError, ValueError) as error:
        raise HTTPException(503, "Local navigation status is temporarily unavailable") from error


@app.post("/api/local-navigation/unload")
async def unload_local_navigation():
    if lab.agent.active or lab.lock.locked() or lab.worker.latest.get("busy"):
        raise HTTPException(409, "Stop robot control before unloading the local model")
    async with lab.lock:
        await lab.local_navigation.close()
    return lab.local_navigation.public()


@app.get("/api/textures/{name}.png")
async def texture(name: str):
    if name not in TEXTURE_NAMES:
        raise HTTPException(404, "Unknown material")
    return FileResponse(TEXTURE_ROOT / f"{name}.png", media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.get("/api/calibration")
async def robot_calibration():
    return calibration()


@app.get("/api/tools")
async def tools(execution_mode: Literal["single_step", "navigation_plan", "supervised_policy", "local_navigation", "luna_navigation", "luna_continuous"] = "luna_navigation"):
    return robot_tools(execution_mode) if execution_mode != "single_step" else tool_schemas()


@app.post("/api/command")
async def command(envelope: Command):
    if envelope.tool == "stop":
        await lab.agent.halt()
        lab.worker.stop()
        return await lab.worker.execute(envelope)
    if not lab.worker.powered:
        raise HTTPException(409, "Turn the robot on before starting work")
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
            lab.agent.navigation_memory.bind(placement.run_id, placement.episode_epoch, reset=True)
        except MotionError as error:
            raise HTTPException(409 if error.code in {"STALE_STATE", "CANCELLED", "OBJECT_HELD"} else 422, str(error)) from error
    return lab.state()


class RobotPower(StrictModel):
    run_id: str
    episode_epoch: int
    on: bool


@app.post("/api/power")
async def power(settings: RobotPower):
    worker = lab.worker
    if settings.run_id != worker.latest["run_id"] or settings.episode_epoch != worker.epoch:
        raise HTTPException(409, "Power request belongs to another episode")
    if not settings.on:
        if lab.power_off_pending:
            raise HTTPException(409, "Robot power-off is already in progress")
        lab.power_off_pending = True
        lab.robot_on = False
        lab.agent.interrupt("Robot powered off")
        try:
            await worker.set_power(False)
            await lab.agent.halt("Robot powered off")
            await lab.finish_home_recording("powered_off")
            await lab.local_navigation.close()
        finally:
            lab.power_off_pending = False
    else:
        if lab.lock.locked() or lab.power_off_pending:
            raise HTTPException(409, "Wait for the episode transition to finish")
        async with lab.lock:
            if worker is not lab.worker or lab.power_off_pending:
                raise HTTPException(409, "Power request invalidated by an episode or power transition")
            await worker.set_power(True)
            lab.robot_on = worker.powered
    return lab.state()


@app.post("/api/stop")
async def stop():
    lab.worker.stop()
    await lab.agent.halt()
    await lab.finish_home_recording("stopped")
    return {"stopped": True}


@app.post("/api/resume")
@app.post("/api/agent/takeover")
async def resume():
    lab.worker.stop()
    await lab.agent.halt("Manual takeover")
    await lab.finish_home_recording("takeover")
    async with lab.lock:
        await lab.worker.resume_manual()
    return lab.state()


@app.post("/api/reset")
async def reset():
    lab.worker.stop()
    await lab.agent.halt("Episode reset")
    return await lab.reset()


@app.get("/api/challenges")
async def challenges(environment: Literal["standalone", "shared_apartment_v1"] = "standalone"):
    if environment == "shared_apartment_v1":
        return [shared_apartment(identifier).public() for identifier in ("furniture_circuit", "apartment", "flat_kitchen", "recharge")]
    return [preset.public() for preset in PRESETS.values()]


@app.post("/api/challenges/load")
async def load_challenge(selection: ChallengeLoad):
    lab.worker.stop()
    await lab.agent.halt("Challenge changed")
    await lab.reset(selection.challenge_id, selection.orbit_target, selection.orbit_direction, selection.environment, selection.reuse_saved_map, selection.environment_instance_id)
    try:
        await asyncio.to_thread(PreferenceStore().save_scene, selection)
        lab.preference_error = None
    except (OSError, sqlite3.Error, ValueError):
        lab.preference_error = "The scene loaded, but its startup selection could not be saved."
    return lab.state()


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


@app.post("/api/agent/instruction")
async def run_instruction(instruction: RunInstruction):
    if not lab.connections or lab.lock.locked() or lab.interaction_mode != "chat":
        raise HTTPException(409, "Keep the operator connected and wait for any episode transition")
    async with lab.lock:
        try:
            return await lab.agent.redirect(lab.worker, instruction)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(409, str(error)) from error


@app.get("/api/mission/capabilities")
async def mission_capabilities():
    from backend.experiment_variants import variant_snapshot
    from backend.mission import MissionPlan
    settings = AgentStart(run_id="capability", episode_epoch=0, execution_mode="luna_continuous", goal="capability",
        unified_mission=True, images_per_request=2)
    return {"version": 1, "unified_mission": True, "observed_map_version": 1, "backends": ["builtin"],
        "task_kinds": MissionPlan.model_json_schema()["properties"]["kind"]["enum"], "image_slots": 2,
        "review_policy": "bounded_stopped_checkpoints", "qualification": "experimental",
        "architecture": variant_snapshot(settings, None)["architecture"]}


@app.post("/api/mission/start")
async def start_mission(settings: AgentStart):
    if not settings.unified_mission:
        raise HTTPException(422, "The mission endpoint requires unified_mission=true")
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
            await lab.finish_home_recording("luna_handoff")
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
        if not lab.worker.powered or lab.lock.locked() or not lab.connections or lab.interaction_mode != "voice":
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
            controller.local_navigation_factory = lab.agent.local_navigation_factory
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
            await asyncio.sleep(.05 if lab.worker.power_state()["mode"] == "working" else 1.)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        lab.connections -= 1
        if lab.connections == 0:
            lab.worker.stop()
            with CancelScope(shield=True):
                await lab.agent.halt("Operator disconnected")
                await lab.finish_home_recording("disconnected")


dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")