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
Use plan exactly once, only when mission.plan is null, to classify the user's task: explore, object, room, or place;
preserve all requested stages including return_home. Once accepted, keep the plan and select an action from available_actions.
Rejected actions do not erase the accepted plan. Never submit plan again to report observations or recover from a rejection.
For object/room/place use plan.target as a concise target label, at most 160 characters, with only identifying features.
Example: {"action":"plan","plan":{"kind":"room","target":"Bathroom with toilet and sink","return_home":false}}.
Do not copy the full task or its stages into target. The original user goal stays available on every request; honor it in full.
Do not omit a requested return. Put current fixture evidence in evidence_text (at most 500 characters).
The current head image and observation are your only evidence. The additional OBSERVED map contains sensor-built geometry,
not the full world. Gray means unknown, not free. Map numbered markers index the supplied destinations, whose exact IDs are
used in calls. Map labels are hypotheses. Never infer target identity or unseen room geometry from a map label.
Use explore to delegate sensor-driven exploration until a target is visible. Do not choose frontier IDs or individual waypoints.
The local worker selects fresh reachable frontiers, follows routes and retries blocked destinations without another model call.
If exploration is blocked by missing observed clearance, the worker may make one guarded full-circle sensor survey
and retry locally to measure self-occluded directions. It does not rotate before every exploration.
You supervise intent and visual target identity; you do not steer routine travel or control wheels. Reviews occur at completion,
an unresolved blockage, or a checkpoint triggered after 15 seconds or 2 metres. The configured feedback interval spaces requests.
A review alone does not pause exploration: one existing local task may continue while you think. Its persistent objective has
explicit objective_duration_s (1..60, default 60) and objective_travel_m (>0..6, default 6) limits, always within the original
mission deadline. An explore reply renews only a still-valid identical objective using those bounds, never an expired trajectory.
The worker selects new frontiers and enforces fresh observed footprints, collision checks and short trajectory leases independently.
Unknown space still requires sensing. Expired or failed objectives stop and invalidate replies requested while they were active.
After a moving review requests inspection, reporting or a different action, the worker stops and requests NEW stopped evidence;
that moving reply does not authorize a pose-sensitive action. Observation reports remain unverified and never establish arrival.
Prefer an action over a report-only turn when the next safe intent is clear. Keep reason to one short sentence.
For an object: select_object with CURRENT normalized [left,top,right,bottom] object_bounds and object_label; then approach_object
with the supplied object_goal_id and a reachable front/left/right approach; then verify_object with a NEW post-arrival image box.
Check shape and support, not just color. A decorative box on a counter is not necessarily a cube on a low pedestal.
If identity is uncertain, look or turn from rest. Do not repeat an unchanged rejected view. You may select a new goal after inspection.
If the current camera and sensors show a corner or dead end with no observed forward route, request one in-place half-turn:
{"action":"turn","turn_rad":3.14159,"reason":"Observed dead end; turn around in place and inspect the way back"}.
Use -3.14159 for the opposite direction when current evidence favors it. The base turns without forward or reverse travel.
Afterward reassess the NEW camera and observation before choosing a measured route. Do not repeatedly spin at the same dead end,
infer free space behind you, or drive blindly. If rotation is rejected by clearance checks, inspect or wait; never bypass the rejection.
For named places or rooms: navigate_place with an exact reachable place_id. Rooms require a new observe_room report at arrival,
with room_matches and evidence_text grounded in current fixtures. Missing room identity is not task completion.
To describe the starting room or current fixtures before arrival, use report_observation with evidence_text and no place_id.
Example: {"action":"report_observation","evidence_text":"Visible oven, cabinets and worktop suggest a kitchen."}.
This records an unverified visual report only, not arrival or task completion. Then inspect or explore measured frontiers.
observe_room is only for the supplied mission.target_id after navigating to that named room. If no reachable room ID exists,
continue observed exploration; do not invent a room ID, reuse Home as the bathroom, or restart planning.
The executive returns to the mission's Home automatically when requested after target verification. Do not manually skip stages.
finish is only valid after all requested receipts; it cannot substitute for object/room/place verification.
All actions are subject to fresh sensing, footprint clearance, original deadlines and Stop. Local failures do not imply arrival.
If a goal is unsupported, do not relabel it as a simpler task. Report uncertainty in reason; do not invent IDs or coordinates.
"""


def tools():
    schema = MissionDecision.model_json_schema()
    schema["properties"].pop("frontier_id")
    return [{"type": "function", "name": "guide_mission", "description": "Interpret a mission and select an observed bounded capability.",
        "parameters": schema, "strict": False}]


def decision_validation_feedback(error):
    errors = error.errors(include_url=False, include_context=False, include_input=False)[:8]
    details = [{"field": ".".join(str(part) for part in entry["loc"]) or "decision",
        "type": entry["type"], "message": entry["msg"]} for entry in errors]
    return {"status": "rejected", "reason": "; ".join(f"{entry['field']}: {entry['message']}" for entry in details),
        "validation_errors": details}


def authority(worker):
    return (worker.sim.run_id, worker.epoch, worker.stop_revision, worker.task_revision)


def available_actions(mission, observation, object_state):
    if mission.plan is None:
        return ["plan"]
    actions = ["report_observation", "look", "turn", "wait", "explore"]
    if mission.plan.kind == "object":
        actions.append("select_object")
        if object_state:
            actions.extend(["approach_object", "verify_object"])
    elif mission.plan.kind in {"room", "place"}:
        places = (observation.spatial or {}).get("places", [])
        if any(place.get("reachable") and (mission.plan.kind != "room" or place.get("kind") == "room") for place in places):
            actions.append("navigate_place")
        if mission.plan.kind == "room" and mission.target_id:
            actions.append("observe_room")
    receipt = "exploration" if mission.plan.kind == "explore" else "target"
    if receipt in mission.receipts and (not mission.plan.return_home or "return" in mission.receipts or receipt == "exploration"):
        actions.append("finish")
    return actions


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
        checkpoint_s = min(15., remaining - 1.)
        request_action = "explore_frontier" if action == "explore" and frontier_id else "explore" if action == "explore" else "navigate_to"
        await worker.home_command(HomeRequest(run_id=settings.run_id, episode_epoch=settings.episode_epoch, action=request_action,
            place_id=place_id, frontier_id=frontier_id, time_budget=remaining), *mission.authority[2:])
        while worker.home_mission.active:
            controller._check_live(worker, settings)
            mission.check(authority(worker))
            if action == "explore":
                pose = worker.latest["observation"]["odometry_m_rad"]
                if time.monotonic()-began >= checkpoint_s or math.dist(start_pose[:2], pose[:2]) >= 2.:
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


class MappedReviewOperation:
    def __init__(self, controller, worker, settings):
        self.controller, self.worker, self.settings = controller, worker, settings
        self.mission = controller.mission
        self.identity = None
        self.reviewed_at = 0.
        self.reviewed_travel = 0.
        self.initial = None

    async def start(self, decision):
        self.initial = await self.worker.home_state(compact=True)
        self.identity = self.mission.begin("exploring", authority(self.worker))
        try:
            result = await self.worker.mission_objective(self.mission, self.identity, decision=decision)
        except BaseException:
            await self.close("Mission objective start interrupted")
            raise
        self.reviewed_at = time.monotonic()
        self.reviewed_travel = 0.
        return result

    async def poll(self):
        self.controller._check_live(self.worker, self.settings)
        self.mission.check(authority(self.worker))
        return await self.worker.mission_objective(self.mission, self.identity)

    async def review(self):
        while True:
            result = await self.poll()
            if result["task"]["status"] != "running":
                return await self.close(result["task"]["reason"], result=result)
            elapsed = time.monotonic() - self.reviewed_at
            travelled = self.mission.objective.travel_m - self.reviewed_travel
            if (elapsed >= 15. or travelled >= 2.) and elapsed >= self.controller.state["feedback_interval_s"]:
                self.reviewed_at = time.monotonic()
                self.reviewed_travel = self.mission.objective.travel_m
                self.controller._trace("policy", "Mission review checkpoint", {"mission": self.mission.state(),
                    "task": result["task"], "elapsed_s": elapsed, "travel_m": travelled, "paused_for_review": False})
                return result
            await asyncio.sleep(.05)

    async def respond(self, model, profile, inputs):
        pending = asyncio.create_task(model.respond(profile, self.settings.reasoning, self.settings.goal, inputs))
        try:
            async with asyncio.timeout(self.controller.request_timeout_s):
                while not pending.done():
                    self.controller._check_live(self.worker, self.settings)
                    self.mission.check(authority(self.worker))
                    if self.identity:
                        await self.poll()
                    await asyncio.wait({pending}, timeout=.05)
                return await pending
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    async def close(self, reason, *, result=None):
        if self.identity is None:
            return result
        identity = self.identity
        try:
            result = await self.worker.mission_objective(self.mission, identity, end_reason=reason)
            task = result["task"] or {"status": "cancelled", "reason": reason}
            useful = (self.mission.objective and self.mission.objective.travel_m >= .5
                and task["status"] in {"completed", "paused"})
            if (self.mission.operation == identity and authority(self.worker) == self.mission.authority
                    and self.mission.phase not in self.mission.terminal and time.monotonic() < self.mission.deadline):
                self.mission.end_operation(identity, authority(self.worker), "exploration" if useful else None)
            self.mission.reason = task["reason"]
            self.controller._trace("policy", "Mission operation finished", {"mission": self.mission.state(), "task": task,
                "coverage_before": self.initial.get("coverage"), "coverage_after": result.get("coverage")})
            return result
        finally:
            self.identity = None
            if self.mission.operation == identity:
                self.mission.operation = None


async def run(controller, worker, settings, model, profile, stop_revision):
    mapped = MappedReviewOperation(controller, worker, settings)
    try:
        await run_reviews(controller, worker, settings, model, profile, stop_revision, mapped)
    finally:
        await mapped.close("Mission supervision ended")


async def run_reviews(controller, worker, settings, model, profile, stop_revision, mapped):
    from backend.feedback import retain_context
    mission = controller.mission
    history = deque(maxlen=6)
    trail = deque(maxlen=128)
    last = None
    overview = False
    exploration_surveyed = False
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
        if mapped.identity:
            last = await mapped.review()
        local_exploration = settings.mission_local_only or bool(mission.plan and mission.plan.kind == "explore" and not mission.rejections)
        object_state = await worker.object_state()
        sensor, image, observation = await worker.mission_feedback(mission)
        requested_operation = mapped.identity
        if not trail or math.dist(trail[-1], observation.odometry_m_rad[:2]) > .1:
            trail.append(observation.odometry_m_rad[:2])
        controller.navigation_memory.observe(observation)
        images, image_roles = [image], ["current_head"]
        map_details = {"status": "disabled"}
        if settings.map_context and not local_exploration:
            started = time.monotonic()
            try:
                context = await worker.mission_map(sensor, observation, list(trail), overview=overview, mission=mission)
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
        if not mapped.identity:
            mission.phase = "interpreting"
        payload = {"observation": observation.model_dump(exclude={"observed_map": {"cells"}}), "mission": mission.state(),
            "object_goal": object_state, "last_execution": last, "map_context": map_details,
            "available_actions": available_actions(mission, observation, object_state)}
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
            retained, _ = retain_context(list(history), settings.context_tokens)
            response = await mapped.respond(model, profile, [*(item for turn in retained for item in turn), message])
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
            except ValidationError as error:
                mission.rejections += 1
                last = decision_validation_feedback(error)
                controller._trace("policy", "Mission decision rejected", {**last, "attempt": mission.rejections, "limit": 3})
                if mission.rejections >= 3:
                    raise ValueError(f"Mission decision correction limit reached: {last['reason']}") from error
                continue
        controller.state.update(phase="acting", message=decision.reason)
        if decision.reason:
            controller.navigation_reply(decision.reason, source="controller" if local_exploration else "model")
        overview = decision.map_view == "overview"
        try:
            if requested_operation:
                current = await mapped.poll()
                if (current["task"]["status"] != "running" or mission.objective.status != "active"):
                    last = await mapped.close(current["task"]["reason"], result=current)
                    last = {**last, "status": "stale_review_discarded", "motion_authorized": False}
                    controller._trace("policy", "Expired mission review discarded", last)
                    continue
                if decision.action == "explore":
                    last = await worker.mission_objective(mission, requested_operation, decision=decision, renew=True)
                    controller._trace("policy", "Mission objective renewed", {"mission": mission.state(), "task": last["task"]})
                else:
                    last = await mapped.close("Luna requested stopped evidence for " + decision.action)
                    await wait_stationary(controller, worker, settings, .5)
                    fresh_sensor, _, fresh_observation = await worker.mission_feedback(mission)
                    last = {**last, "status": "stopped_for_fresh_review", "discarded_action": decision.action,
                        "source_spatial_sequence": sensor.sequence, "fresh_spatial_sequence": fresh_sensor.sequence,
                        "fresh_observation_seq": fresh_observation.seq, "motion_authorized": False}
                    controller._trace("policy", "Moving mission decision deferred", last)
                    continue
            elif decision.action == "plan":
                if decision.plan is None:
                    raise ValueError("Provide a structured mission plan")
                if mission.plan == decision.plan:
                    last = {"status": "plan_already_accepted", "plan": mission.plan.model_dump(),
                        "reason": "Keep the accepted plan; select a next action instead of planning again",
                        "available_actions": payload["available_actions"]}
                else:
                    mission.configure(decision.plan)
                    last = {"status": "planned", "plan": mission.plan.model_dump()}
            elif mission.plan is None:
                raise ValueError("Interpret the requested mission before execution")
            elif decision.action == "report_observation":
                if not 0 <= time.monotonic()-sensor.captured_at <= 15.:
                    raise ValueError("Observation report expired; inspect current evidence")
                last = {"status": "observation_reported", "evidence_text": decision.evidence_text,
                    "spatial_sequence": sensor.sequence, "observation_seq": observation.seq,
                    "source_frame": "wheel_odometry", "source_run_id": sensor.run_id, "source_episode_epoch": sensor.episode_epoch,
                    "source_captured_at": sensor.captured_at, "source_pose_m_rad": sensor.odometry_m_rad,
                    "identity_verified": False, "arrival_verified": False}
                controller.navigation_reply(decision.evidence_text, source="model")
            elif decision.action == "explore":
                if not 0 <= time.monotonic()-sensor.captured_at <= 15.:
                    raise ValueError("Exploration decision expired; inspect current evidence")
                try:
                    if local_exploration:
                        last = await mapped_operation(controller, worker, settings, "explore")
                    else:
                        last = await mapped.start(decision)
                except ValueError as error:
                    if exploration_surveyed or not str(error).startswith(("UNREACHABLE:", "OBSERVED_PATH_BLOCKED:",
                            "BLOCKED: bounded route retries exhausted; last route: OBSERVED_PATH_BLOCKED:")):
                        raise
                    exploration_surveyed = True
                    controller._trace("policy", "Local exploration survey requested", {"mission_id": mission.identity,
                        "model_calls": 0, "reason": str(error)})
                    token = mission.begin("inspecting", authority(worker))
                    await rotate(controller, worker, settings, stop_revision, 2 * math.pi, panoramic=True, pitch=.85)
                    await worker.scan_continuous(ContinuousScan(run_id=settings.run_id, episode_epoch=settings.episode_epoch))
                    mission.end_operation(token, authority(worker))
                    controller._trace("policy", "Local exploration survey finished", {"mission_id": mission.identity,
                        "model_calls": 0, "reason": "Fresh measured surroundings before autonomous route selection"})
                    if local_exploration:
                        last = await mapped_operation(controller, worker, settings, "explore")
                    else:
                        last = await mapped.start(decision)
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
            await mapped.close(str(error))
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