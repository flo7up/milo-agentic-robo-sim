import asyncio
import base64
from collections import deque
import hashlib
import json
import math
import time
from types import SimpleNamespace
from uuid import uuid4

from pydantic import ValidationError

from backend.contracts import Command
from backend.continuous_navigation import ContinuousScan
from backend.continuous_supervisor import execute_object_capability, rotate, wait_stationary
from backend.home_mission import HomeRequest
from backend.map_context import render_observed_map
from backend.mission import MissionDecision, MissionPlan
from backend.simulation import MotionError


INSTRUCTIONS = """You supervise one robot mission. Return exactly one guide_mission call.
First use plan to classify the user's task: explore, object, room, or place; preserve all requested stages including return_home.
For object/room/place give a precise target description including shape, color, support/context. Do not omit a requested return.
The current head image and observation are your only evidence. The additional OBSERVED map contains sensor-built geometry,
not the full world. Gray means unknown, not free. Map numbered markers index the supplied destinations, whose exact IDs are
used in calls. Map labels are hypotheses. Never infer target identity or unseen room geometry from a map label.
Use explore to traverse measured frontiers until a target is visible. Optional frontier_id must come from the current observation.
The executive handles local continuation and pauses at bounded checkpoints for your review. You do not control wheels.
For an object: select_object with CURRENT normalized [left,top,right,bottom] object_bounds and object_label; then approach_object
with the supplied object_goal_id and a reachable front/left/right approach; then verify_object with a NEW post-arrival image box.
Check shape and support, not just color. A decorative box on a counter is not necessarily a cube on a low pedestal.
If identity is uncertain, look or turn from rest. Do not repeat an unchanged rejected view. You may select a new goal after inspection.
For named places or rooms: navigate_place with an exact reachable place_id. Rooms require a new observe_room report at arrival,
with room_matches and evidence_text grounded in current fixtures. Missing room identity is not task completion.
The executive returns to the mission's Home automatically when requested after target verification. Do not manually skip stages.
finish is only valid after all requested receipts; it cannot substitute for object/room/place verification.
All actions are subject to fresh sensing, footprint clearance, original deadlines and Stop. Local failures do not imply arrival.
If a goal is unsupported, do not relabel it as a simpler task. Report uncertainty in reason; do not invent IDs or coordinates.
"""


def tools():
    return [{"type": "function", "name": "guide_mission", "description": "Interpret a mission and select an observed bounded capability.",
        "parameters": MissionDecision.model_json_schema(), "strict": False}]


def authority(worker):
    return (worker.sim.run_id, worker.epoch, worker.stop_revision, worker.task_revision)


async def mapped_operation(controller, worker, settings, action, *, place_id=None, frontier_id=None, returning=False):
    mission = controller.mission
    token = mission.begin("returning" if returning else "exploring" if action == "explore" else "navigating", authority(worker))
    initial = await worker.home_state(compact=True)
    start_pose = list(worker.latest["observation"]["odometry_m_rad"])
    began = time.monotonic()
    try:
        remaining = min(60., mission.deadline-time.monotonic())
        if remaining < 1.:
            raise TimeoutError("Mission budget exhausted")
        request_action = "explore_frontier" if action == "explore" and frontier_id else "explore" if action == "explore" else "navigate_to"
        await worker.home_command(HomeRequest(run_id=settings.run_id, episode_epoch=settings.episode_epoch, action=request_action,
            place_id=place_id, frontier_id=frontier_id, time_budget=remaining), *mission.authority[2:])
        while worker.home_mission.active:
            controller._check_live(worker, settings)
            mission.check(authority(worker))
            if action == "explore":
                pose = worker.latest["observation"]["odometry_m_rad"]
                if time.monotonic()-began >= 8. or math.dist(start_pose[:2], pose[:2]) >= 1.:
                    await worker.call(lambda sim: worker.home_mission.fail("Bounded mission observation checkpoint", "paused"))
                    break
            await asyncio.sleep(.05)
        controller._check_live(worker, settings)
        result = await worker.home_state(compact=True)
        task = result["task"]
        if task["status"] not in {"completed", "paused"}:
            raise ValueError(task["reason"])
        if action != "explore":
            await worker.verify_saved_place(place_id, *mission.authority[2:])
        useful = action == "explore" and math.dist(start_pose[:2], worker.latest["observation"]["odometry_m_rad"][:2]) >= .5
        mission.end_operation(token, authority(worker), "return" if returning else "exploration" if useful else None)
        mission.reason = task["reason"]
        controller._trace("policy", "Mission operation finished", {"mission": mission.state(), "task": task,
            "coverage_before": initial.get("coverage"), "coverage_after": result.get("coverage")})
        return result
    except BaseException:
        if mission.operation == token:
            await worker.call(lambda sim: worker.home_mission.fail("Mission operation interrupted", "cancelled"))
            mission.operation = None
        raise


async def run(controller, worker, settings, model, profile, stop_revision):
    from backend.feedback import retain_context
    mission = controller.mission
    history = deque(maxlen=6)
    trail = deque(maxlen=128)
    last = None
    overview = False
    controller.state.update(phase="acting", message="Preparing observed mission")
    if not await worker.resume_manual(expected_stop_revision=stop_revision):
        raise asyncio.CancelledError
    controller._check_live(worker, settings)
    await worker.scan_continuous(ContinuousScan(run_id=settings.run_id, episode_epoch=settings.episode_epoch, compact_arms=settings.compact_arms))
    controller._check_live(worker, settings)
    mission.home_id = await worker.prepare_mission(mission.identity, *mission.authority[2:])
    if settings.mission_local_only:
        mission.configure(MissionPlan(kind="explore", target="Bounded local exploration"))
    for turn in controller.turn_indices(settings):
        controller._check_live(worker, settings)
        mission.check(authority(worker))
        local_exploration = settings.mission_local_only or bool(mission.plan and mission.plan.kind == "explore" and not mission.rejections)
        object_state = await worker.object_state()
        sensor, image, observation = await worker.mission_feedback()
        if not trail or math.dist(trail[-1], observation.odometry_m_rad[:2]) > .1:
            trail.append(observation.odometry_m_rad[:2])
        controller.navigation_memory.observe(observation)
        images, image_roles = [image], ["current_head"]
        map_details = {"status": "disabled"}
        if settings.map_context and not local_exploration:
            started = time.monotonic()
            try:
                context = await worker.mission_map(sensor, observation, list(trail), overview=overview)
                observation.observed_map = context
                async with asyncio.timeout(1.):
                    rendered = await asyncio.to_thread(render_observed_map, context)
                images.append(rendered)
                image_roles.append("observed_map")
                map_details = {"status": "available", "revision": context.revision, "sha256": hashlib.sha256(rendered).hexdigest(),
                    "assembly_s": time.monotonic()-started, "bytes": len(rendered)}
            except (ValueError, TimeoutError) as error:
                observation.observed_map = None
                map_details = {"status": "unavailable", "reason": str(error) or "Observed map preparation timed out"}
        mission.phase = "interpreting"
        payload = {"observation": observation.model_dump(exclude={"observed_map": {"cells"}}), "mission": mission.state(),
            "object_goal": object_state, "last_execution": last, "map_context": map_details}
        if local_exploration:
            finished = bool(last and last.get("task", {}).get("status") == "completed")
            returning = bool(mission.plan.return_home and "exploration" in mission.receipts and mission.deadline-time.monotonic() < 20.)
            decision = MissionDecision(action="finish" if finished or returning else "explore",
                reason="Local exploration continues under the approved mission plan")
        else:
            message = {"role": "user", "content": [{"type": "input_text", "text": json.dumps(payload)}]}
            for role, frame in zip(image_roles, images):
                message["content"].extend([{"type": "input_text", "text": role},
                    {"type": "input_image", "detail": "high", "image_url": "data:image/png;base64,"+base64.b64encode(frame).decode("ascii")}])
            controller.state.update(phase="thinking", turns=turn+1)
            controller._trace("feedback", "Mission camera and observed map", {**payload, "observed_map_snapshot": observation.observed_map.model_dump() if observation.observed_map else None,
                "image_roles": image_roles, "images_in_request": len(images), "history_turns": [], "tool_result_call_ids": []}, image=image, images=images)
            began = time.monotonic()
            async with asyncio.timeout(controller.request_timeout_s):
                retained, _ = retain_context(list(history), settings.context_tokens)
                response = await model.respond(profile, settings.reasoning, settings.goal, [*(item for turn in retained for item in turn), message])
            controller._check_live(worker, settings)
            mission.check(authority(worker))
            controller.state["inference_latency_s"] = time.monotonic()-began
            if response.usage:
                controller.state["input_tokens"] += response.usage.input_tokens
                controller.state["output_tokens"] += response.usage.output_tokens
            outputs = [item.model_dump(exclude_none=True) for item in response.output]
            calls = [item for item in outputs if item["type"] == "function_call"]
            controller._trace("response", "Luna mission decision", {"calls": calls, "status": response.status, "latency_s": time.monotonic()-began,
                "text": response.output_text[:2000], "refusals": [], "input_tokens": response.usage.input_tokens if response.usage else None,
                "output_tokens": response.usage.output_tokens if response.usage else None})
            if response.status != "completed" or len(calls) != 1 or calls[0]["name"] != "guide_mission":
                raise ValueError("Expected one completed guide_mission response")
            if any(part.get("type") == "refusal" for item in outputs for part in item.get("content", [])):
                raise ValueError("Model declined the request")
            if any(item.get("blocked") for item in (getattr(response, "model_extra", None) or {}).get("content_filters", []) or []):
                raise ValueError("Response blocked by model guardrails")
            if calls[0]["call_id"] in controller.seen_text_calls or len(calls[0]["arguments"]) > 8000:
                raise ValueError("Repeated or oversized mission response")
            controller.seen_text_calls.add(calls[0]["call_id"])
            try:
                decision = MissionDecision.model_validate_json(calls[0]["arguments"])
            except ValidationError:
                mission.rejections += 1
                last = {"status": "rejected", "reason": "Invalid mission decision; use the supplied schema and current IDs"}
                if mission.rejections >= 3:
                    raise ValueError("Mission decision correction limit reached")
                continue
        controller.state.update(phase="acting", message=decision.reason)
        if decision.reason:
            controller.navigation_reply(decision.reason, source="controller" if local_exploration else "model")
        overview = decision.map_view == "overview"
        try:
            if decision.action == "plan":
                if decision.plan is None:
                    raise ValueError("Provide a structured mission plan")
                mission.configure(decision.plan)
                last = {"status": "planned", "plan": mission.plan.model_dump()}
            elif mission.plan is None:
                raise ValueError("Interpret the requested mission before execution")
            elif decision.action == "explore":
                if not 0 <= time.monotonic()-sensor.captured_at <= 15.:
                    raise ValueError("Exploration decision expired; inspect current evidence")
                if decision.frontier_id and decision.frontier_id not in {item["frontier_id"] for item in (observation.spatial or {}).get("frontiers", [])}:
                    raise ValueError("Select a frontier supplied in the current observation")
                last = await mapped_operation(controller, worker, settings, "explore", frontier_id=decision.frontier_id)
            elif decision.action in {"select_object", "approach_object", "verify_object"}:
                if mission.plan.kind != "object":
                    raise ValueError("Object capabilities require an object mission")
                if decision.action == "select_object":
                    mission.receipts.pop("target", None)
                    mission.receipts.pop("return", None)
                token = mission.begin("navigating" if decision.action == "approach_object" else "inspecting", authority(worker))
                guide = SimpleNamespace(**decision.model_dump(), standoff_m=.8)
                last = await execute_object_capability(controller, worker, settings, guide, sensor, *mission.authority[2:])
                receipt = "target" if last["status"] == "object_arrival_verified" else None
                mission.target_id = last.get("goal_id", mission.target_id)
                mission.end_operation(token, authority(worker), receipt)
            elif decision.action == "navigate_place":
                if mission.plan.kind not in {"room", "place"}:
                    raise ValueError("Use the object approach contract for object missions")
                if not 0 <= time.monotonic()-sensor.captured_at <= 15.:
                    raise ValueError("Named-place decision expired; inspect current evidence")
                place = next((place for place in (observation.spatial or {}).get("places", []) if place["place_id"] == decision.place_id and place["reachable"]), None)
                if place is None or (mission.plan.kind == "room" and place["kind"] != "room"):
                    raise ValueError("Select a currently reachable place of the requested kind")
                last = await mapped_operation(controller, worker, settings, "navigate", place_id=decision.place_id)
                mission.target_id = decision.place_id
                if mission.plan.kind == "place":
                    mission.receipts["target"] = {"place_id": decision.place_id, "verified_at": time.monotonic()}
            elif decision.action == "observe_room":
                if mission.plan.kind != "room" or not mission.target_id or decision.place_id != mission.target_id:
                    raise ValueError("Navigate to the requested room before inspecting it")
                last = await worker.home_command(HomeRequest(run_id=settings.run_id, episode_epoch=settings.episode_epoch,
                    action="observe_room", place_id=decision.place_id, room_matches=decision.room_matches,
                    name=mission.plan.target, kind="room", evidence_text=decision.evidence_text, spatial_sequence=sensor.sequence),
                    *mission.authority[2:], selected_evidence=(sensor, image))
                if last["room_verification"]["status"] == "visual_match_reported" and decision.room_matches:
                    mission.receipts["target"] = {"place_id": decision.place_id, "verified_at": time.monotonic()}
            elif decision.action in {"look", "turn", "wait"}:
                token = mission.begin("inspecting", authority(worker))
                if decision.action == "turn":
                    await rotate(controller, worker, settings, stop_revision, decision.turn_rad)
                elif decision.action == "wait":
                    await wait_stationary(controller, worker, settings, decision.duration_s)
                else:
                    result = await worker.execute(Command(run_id=settings.run_id, episode_epoch=settings.episode_epoch,
                        observation_seq=observation.seq, action_id=str(uuid4()), tool="set_head",
                        arguments={"yaw_rad": decision.yaw_rad, "pitch_rad": decision.pitch_rad, "duration_s": 1}), assisted=False)
                    if result.status != "ok":
                        raise ValueError(result.message)
                mission.end_operation(token, authority(worker))
                last = {"status": "observed", "action": decision.action}
            elif decision.action == "finish":
                if mission.plan.kind == "explore" and "exploration" in mission.receipts and mission.plan.return_home:
                    await mapped_operation(controller, worker, settings, "navigate", place_id=mission.home_id, returning=True)
                await worker.call(lambda sim: worker._verify_navigation_rest(*mission.authority[2:], lambda: mission.check(authority(worker))))
                mission.complete(authority(worker))
            if "target" in mission.receipts and "return" not in mission.receipts:
                if mission.plan.return_home:
                    await mapped_operation(controller, worker, settings, "navigate", place_id=mission.home_id, returning=True)
                mission.complete(authority(worker))
            if mission.phase == "completed":
                controller.state["phase"] = "completed"
                controller._set_outcome("completed", mission.reason, "agent")
                return
            mission.rejections = 0
        except (ValueError, MotionError) as error:
            controller._check_live(worker, settings)
            mission.operation = None
            mission.rejections += 1
            mission.phase, mission.reason = "recovering", str(error)
            last = {"status": "rejected", "reason": str(error), "motion_authorized": False}
            if settings.mission_local_only or mission.rejections >= 3 or ("target" in mission.receipts and mission.plan.return_home):
                raise ValueError(str(error)) from error
        controller._trace("policy", "Mission capability feedback", {"mission": mission.state(), "result": last})
        history.append([{"role": "assistant", "content": [{"type": "output_text", "text": decision.model_dump_json()}]},
            {"role": "user", "content": [{"type": "input_text", "text": json.dumps(last)}]}])
    controller.state["phase"] = "completed"
    controller._set_outcome("limited", "Mission decision budget exhausted")