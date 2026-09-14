import asyncio
import base64
import json
import time
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from backend.contracts import Command, StrictModel
from backend.continuous_navigation import ContinuousScan
from backend.continuous_supervisor import wait_stationary
from backend.simulation import MotionError
from backend.spatial import measure_visible_region


class RouteDecision(StrictModel):
    action: Literal["route", "continue", "measure", "look", "wait", "finish", "report_limit"]
    waypoints_m: list[list[float]] = Field(default_factory=list, max_length=24)
    image_bounds: list[float] | None = Field(default=None, min_length=4, max_length=4)
    reason: str = Field(min_length=10, max_length=500)
    memory: str = Field(default="", max_length=1600)
    yaw_rad: float = Field(default=0., ge=-1.2, le=1.2)
    pitch_rad: float = Field(default=.3, ge=0., le=.85)
    duration_s: float = Field(default=1., ge=.5, le=4.)

    @model_validator(mode="after")
    def route_shape(self):
        import math
        if self.action == "route" and not self.waypoints_m:
            raise ValueError("A route needs observed waypoints")
        if self.action != "route" and self.waypoints_m:
            raise ValueError("Only route decisions accept waypoints")
        if (self.action == "measure") != (self.image_bounds is not None):
            raise ValueError("Only measure decisions require an image box")
        if any(len(point) != 2 or not all(math.isfinite(value) for value in point) for point in self.waypoints_m):
            raise ValueError("Waypoints are finite XY coordinates")
        return self


INSTRUCTIONS = """You supervise Milo's motion from the user's current natural-language task.
Image 1 is CURRENT and paired with the supplied observation. If present, image 2 alone is a
HISTORICAL camera sheet; its authorizes_motion=false field applies only to those past images.
Call guide_route exactly once. There are no task-specific motion routines: construct and revise
the required route yourself from the current head image, paired sensors and observed free-floor map.
Waypoints are XY meters in wheel-odometry coordinates, NOT image pixels or private scene positions.
Choose 1-24 ordered waypoints inside observed support. Do not include the starting position itself.
Keep consecutive points at least 0.15 m apart, total proposed length 0.3-6 m, within mapped space.
The generic worker interpolates a smooth path, validates every sample and follows it locally.
It rejects unknown floor, collisions, expired plans and abrupt moving direction changes.
The current plan remains active during your next review, with a 20 s lease from capture.
Reply with a refreshed route before its endpoint to preserve motion. Include a compatible prefix
through current travel; points already traversed during inference are removed by the worker.
When moving, start along the observed current heading, then curve gradually; do not jump sideways.
Continue keeps only the existing route and does NOT extend its lease or endpoint.
Plan only as far as observed support permits. A stopped robot can reorient for a new path.
Do not drive through an object just because a endpoint behind it is free; curved samples must be free.
You do not need to know the entire task route before starting a safe local observation approach.
Use measure with image_bounds=[left,top,right,bottom] normalized to [0,1] on one visible object's
current image to obtain paired-depth surface coordinates and visible extent. This is a generic
measurement, not identity recognition or proof of full object geometry. Never use historical boxes.
The worker checks clearance: propose useful short routes on measured floor rather than waiting
for a pre-certified complete task path. On ambiguous static scenes, look or measure to gather new
evidence. Repeating identical waits gives no new view. If no progress is possible, report_limit
ends the attempt explicitly without claiming success. Be concise; plan the next useful segment.
Use your memory for task stage, landmarks, loop progress and target identity; update it each review.
For motion relative to objects or a person, identify the intended targets from images and keep their
identity and changing relative positions under review. Never equate an unrelated similar target.
For a moving target, use fresh visual evidence; predict only short motion, keep a safe following
distance, and wait immediately if it is lost, ambiguous, too close or its next position is unknown.
Do not infer moving-target location from old images. The simulator provides no hidden homing state.
Look changes head orientation while stopped; it does not turn the base. Pitch positive looks down.
Wait requests a stationary dwell and also stops the current route. Use it for uncertainty or loss.
Finish requests a stop and is only a MODEL CLAIM, not verification. Require concrete task evidence.
Report_limit requests a stop and reports a blocked or unsupported task, never completion.
Do not mark a loop complete merely because you moved; do not confuse more travel with progress.
Task updates override older instructions. Preserve relevant observation memory, but discard the old
task's plan/progress when incompatible. Follow only the latest command revision. Stop always wins.
Never output program code, wheel velocities, simulator identifiers or invented clearance.
"""


def current_camera_message(observation, image):
    from backend.agent import feedback_message
    message = feedback_message(observation, image)
    image_index = next(index for index, part in enumerate(message["content"]) if part["type"] == "input_image")
    message["content"].insert(image_index, {"type": "input_text",
        "text": "IMAGE 1: CURRENT HEAD CAMERA, paired with the preceding sensors. Use this image for current object selection and measured routes."})
    return message


def tools():
    return [{"type": "function", "name": "guide_route", "description": "Update a generic observed route or request a stopped observation/dwell.",
        "parameters": RouteDecision.model_json_schema(), "strict": False}]


async def current_response(controller, worker, settings, model, profile, inputs, revision):
    response_task = asyncio.create_task(model.respond(profile, settings.reasoning, settings.goal, inputs))
    changed = asyncio.create_task(controller.command_changed.wait())
    try:
        async with asyncio.timeout(controller.request_timeout_s):
            finished, _ = await asyncio.wait([response_task, changed], return_when=asyncio.FIRST_COMPLETED)
            if changed in finished or controller.command_revision != revision:
                response_task.cancel()
                await asyncio.gather(response_task, return_exceptions=True)
                controller._check_live(worker, settings)
                controller._trace("policy", "Superseded AI response discarded", {"command_revision": revision})
                return None
            return await response_task
    finally:
        changed.cancel()
        response_task.cancel()
        await asyncio.gather(changed, response_task, return_exceptions=True)


async def command_wait(controller, worker, settings, duration, revision):
    remaining = duration
    while remaining > 0:
        if controller.command_revision != revision:
            return False
        interval = min(.5, remaining)
        await wait_stationary(controller, worker, settings, interval)
        remaining -= interval
    return controller.command_revision == revision


async def run(controller, worker, settings, model, profile, stop_revision):
    if not await worker.resume_manual(expected_stop_revision=stop_revision):
        raise asyncio.CancelledError
    controller.state["phase"] = "acting"
    await worker.scan_continuous(ContinuousScan(run_id=settings.run_id, episode_epoch=settings.episode_epoch,
        compact_arms=settings.compact_arms))
    memory, last_result = "", None
    history = []
    for turn in controller.turn_indices(settings):
        controller._check_live(worker, settings)
        revision = controller.command_revision
        controller.command_changed.clear()
        current_settings = settings.model_copy(update={"goal": controller.state["goal"]})
        try:
            observation, image, sensor, ticket, support = await worker.ai_route_feedback()
        except MotionError as error:
            controller._trace("policy", "AI route sensing unavailable", {"reason": str(error)})
            if worker.continuous and worker.continuous.active:
                await worker.call(lambda sim: worker.continuous.finish(sim, worker.navigation, "blocked", str(error)))
            await worker.scan_continuous(ContinuousScan(run_id=settings.run_id, episode_epoch=settings.episode_epoch))
            continue
        controller._check_live(worker, settings)
        if controller.command_revision != revision:
            continue
        payload = {"command_revision": revision, "current_task": current_settings.goal, "memory": memory,
            "observed_free_floor_xy_m": support, "map_spacing_m": .25,
            "map_note": "Depth-derived support only; sampled points do not guarantee a free connecting path.",
            "motion": worker.continuous.state() if worker.continuous else None, "previous_result": last_result}
        message = current_camera_message(observation, image)
        camera_review, history_sheet = await worker.camera_history_review(sensor, stop_revision)
        camera_review["enabled"] = getattr(controller, "evaluation_camera_history", True)
        if not camera_review["enabled"]:
            camera_review = {**camera_review, "frames": [], "available_frames": []}
            history_sheet = None
        payload["camera_history"] = camera_review
        message["content"].append({"type": "input_text", "text": json.dumps(payload)})
        images = [image]
        if history_sheet is not None:
            message["content"].extend([{"type": "input_text", "text": "IMAGE 2 ONLY: HISTORICAL camera sheet. Image 1 above remains CURRENT. Past sightings do not authorize current motion."},
                {"type": "input_image", "detail": "high", "image_url": "data:image/png;base64," + base64.b64encode(history_sheet).decode("ascii")}])
            images.append(history_sheet)
        if controller.command_revision != revision:
            continue
        inputs = [*history[-4:], message]
        controller.state.update(phase="thinking", turns=turn + 1, last_feedback_at=time.time())
        controller._trace("feedback", "AI route camera + sensors", {"observation": observation.model_dump(), **payload,
            "images_in_request": len(images)}, image=image, images=images)
        started = time.monotonic()
        response = await current_response(controller, worker, current_settings, model, profile, inputs, revision)
        controller._check_live(worker, settings)
        if response is None:
            last_result = {"status": "superseded", "reason": "A new operator task replaced the pending response"}
            continue
        await worker.acknowledge_camera_history(camera_review, stop_revision)
        if controller.command_revision != revision:
            continue
        if response.usage:
            controller.state["input_tokens"] += response.usage.input_tokens
            controller.state["output_tokens"] += response.usage.output_tokens
        outputs = [entry.model_dump(exclude_none=True) for entry in response.output]
        calls = [entry for entry in outputs if entry["type"] == "function_call"]
        controller._trace("response", "AI route decision", {"status": response.status, "calls": calls,
            "latency_s": time.monotonic() - started, "command_revision": revision,
            "input_tokens": response.usage.input_tokens if response.usage else None,
            "output_tokens": response.usage.output_tokens if response.usage else None})
        if response.status != "completed" or len(calls) != 1 or calls[0]["name"] != "guide_route":
            raise ValueError("AI route control requires exactly one completed guide_route decision")
        if any(item.get("blocked") for item in (getattr(response, "model_extra", None) or {}).get("content_filters", []) or []):
            raise ValueError("Model response blocked by provider guardrails")
        if any(part.get("type") == "refusal" for entry in outputs if entry["type"] == "message" for part in entry.get("content", [])):
            raise ValueError("Model declined the task")
        if calls[0]["call_id"] in controller.seen_text_calls or len(calls[0]["arguments"]) > 8000:
            raise ValueError("Repeated or oversized AI route decision")
        controller.seen_text_calls.add(calls[0]["call_id"])
        decision = RouteDecision.model_validate_json(calls[0]["arguments"])
        memory = decision.memory
        controller.state.update(phase="acting", message=decision.reason)
        controller.navigation_reply(decision.reason)
        try:
            if decision.action == "route":
                last_result = await worker.apply_ai_route(ticket, decision.waypoints_m)
            elif decision.action == "measure":
                if not 0 <= time.monotonic() - sensor.captured_at <= 15.:
                    raise ValueError("Selected image expired; request a fresh measurement")
                last_result = {"status": "measured", **measure_visible_region(sensor, decision.image_bounds)}
            elif decision.action == "continue":
                last_result = worker.continuous.state() if worker.continuous else {"status": "idle"}
            else:
                if worker.continuous and worker.continuous.active:
                    await worker.call(lambda sim: worker.continuous.finish(sim, worker.navigation, "paused", "AI requested stopped feedback"))
                if controller.command_revision != revision:
                    continue
                if decision.action == "look":
                    current, _ = await worker.feedback()
                    reply = await worker.execute(Command(run_id=settings.run_id, episode_epoch=settings.episode_epoch,
                        observation_seq=current.seq, action_id=str(uuid4()), tool="set_head",
                        arguments={"yaw_rad": decision.yaw_rad, "pitch_rad": decision.pitch_rad, "duration_s": 1.}), assisted=False)
                    if reply.status != "ok":
                        raise ValueError(reply.message)
                    last_result = {"status": "observed", "action": "look"}
                elif decision.action == "wait":
                    if not await command_wait(controller, worker, settings, decision.duration_s, revision):
                        continue
                    last_result = {"status": "waited", "duration_s": decision.duration_s}
                elif decision.action == "report_limit":
                    controller.state["phase"] = "completed"
                    controller._set_outcome("unachievable", decision.reason, "agent")
                    return
                else:
                    if not await command_wait(controller, worker, settings, .5, revision):
                        continue
                    controller.state["phase"] = "completed"
                    controller._set_outcome("completed", decision.reason, "agent")
                    return
        except (ValueError, MotionError) as error:
            last_result = {"status": "rejected", "reason": str(error)}
        controller._trace("policy", "AI route execution feedback", {"command_revision": revision, **last_result})
        history.extend([{"role": "assistant", "content": [{"type": "output_text", "text": json.dumps(decision.model_dump())}]},
            {"role": "user", "content": [{"type": "input_text", "text": json.dumps(last_result)}]}])
    controller.state["phase"] = "completed"
    controller._set_outcome("limited", "AI route supervisor turn limit reached")