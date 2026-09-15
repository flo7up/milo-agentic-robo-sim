import asyncio
import base64
import json
import math
import time
from io import BytesIO
from typing import Literal
from uuid import uuid4

from pydantic import Field, TypeAdapter, ValidationError, model_validator
from PIL import Image, ImageDraw

from backend.contracts import Command, StrictModel
from backend.continuous_navigation import ContinuousScan, ContinuousTarget, MotionSkillBudgetError, MotionSkillPlan
from backend.navigation_memory import MEMORY_GUIDANCE, TaskProgress
from backend.spatial import PLACE_GUIDANCE, FloorRegionTracker, PlaceSighting


class SensorCondition(StrictModel):
    sensor: Literal["battery.charge_pct", "battery.charging", "battery.low", "proximity.front_m"]
    comparison: Literal["gte", "lte", "eq"]
    value: float | bool

    @model_validator(mode="after")
    def validate_comparison(self):
        if self.sensor in {"battery.charging", "battery.low"}:
            if not isinstance(self.value, bool) or self.comparison != "eq":
                raise ValueError("Boolean sensor conditions require eq and a boolean value")
        elif isinstance(self.value, bool) or not math.isfinite(self.value) or not 0 <= self.value <= (2 if self.sensor == "proximity.front_m" else 100):
            raise ValueError("Numeric condition is outside the sensor range")
        return self

    def reading(self, observation):
        if self.sensor == "proximity.front_m":
            reading = next((item for item in observation.proximity.distances if item.direction == "front"), None) if observation.proximity else None
            if reading is None or reading.status == "occluded" or reading.distance_m is None:
                raise ValueError("Wait condition has no current front distance reading")
            return reading.distance_m
        if observation.battery is None:
            raise ValueError("Wait condition has no current battery reading")
        return getattr(observation.battery, self.sensor.removeprefix("battery."))

    def matches(self, observation):
        measured = self.reading(observation)
        if self.comparison == "gte":
            return measured >= self.value
        if self.comparison == "lte":
            return measured <= self.value
        return measured == self.value


class GuideContinuous(StrictModel):
    action: Literal["navigate", "explore", "approach_target", "circle", "look", "turn", "scan", "wait", "wait_until", "inspect_history", "inspect_arrival", "finish", "continue", "advance_subgoal", "compose", "navigate_place", "explore_map", "explore_frontier", "cancel_task", "spatial_state", "remember_object", "observe_room"]
    room_name: str = Field(default="", max_length=80)
    return_place_id: str | None = Field(default=None, max_length=80)
    place_kind: Literal["room", "doorway"] = "room"
    connects: list[str] = Field(default_factory=list, max_length=8)
    evidence_text: str = Field(default="", max_length=500)
    room_matches: bool | None = None
    frontier_id: str | None = Field(default=None, max_length=80)
    confidence: float = Field(default=.5, ge=0., le=1., allow_inf_nan=False)
    place_id: str | None = Field(default=None, max_length=80)
    region_id: str | None = Field(default=None, max_length=80)
    time_budget: float = Field(default=60., ge=1., le=300., allow_inf_nan=False)
    skill_plan: MotionSkillPlan | None = None
    wait_condition: SensorCondition | None = None
    wait_guard: SensorCondition | None = None
    wait_timeout_s: float = Field(default=30., ge=.5, le=60., allow_inf_nan=False)
    next_subgoal: str | None = Field(default=None, min_length=5, max_length=180)
    evidence_event_id: str | None = Field(default=None, min_length=1, max_length=80)
    evidence_frame_ref: str | None = Field(default=None, min_length=1, max_length=80)
    history_frame_id: str | None = Field(default=None, min_length=1, max_length=80)
    object_bounds: list[float] | None = Field(default=None, min_length=4, max_length=4)
    object_label: str = Field(default="", max_length=60)
    circle_direction: Literal["clockwise", "counterclockwise"] = "clockwise"
    floor_target_id: str | None = Field(default=None, min_length=1, max_length=80)
    duration_s: float = Field(default=1., ge=.5, le=4.)
    completion_mode: Literal["parking", "visual_inspection"] = "parking"
    entry_confirmed: bool = False
    destination_candidate_id: int | None = Field(default=None, ge=0, le=40)
    candidate_id: int = Field(default=0, ge=0, le=40)
    yaw_rad: float = Field(default=0, ge=-1.2, le=1.2)
    pitch_rad: float = Field(default=.2, ge=0, le=.85)
    turn_rad: float = Field(default=0, ge=-3.15, le=3.15)
    reason: str = Field(default="No explanation supplied.", min_length=10, max_length=500)
    memory: str = Field(default="", max_length=1400)
    place_sighting: PlaceSighting | None = None

    @model_validator(mode="after")
    def compose_requires_plan(self):
        if (self.action == "compose") != (self.skill_plan is not None):
            raise ValueError("Only compose accepts a required skill_plan")
        return self


def parse_continuous_decision(arguments):
    payload = TypeAdapter(dict[str, object]).validate_json(arguments)
    annotations = []
    for field, minimum, maximum in (("reason", 10, 500), ("memory", 0, 1400)):
        if field not in payload:
            continue
        value = payload[field]
        if not isinstance(value, str) or len(value) < minimum:
            payload.pop(field)
            annotations.append({"field": field, "operation": "omitted_invalid_optional_text"})
        elif len(value) > maximum:
            payload[field] = value[:maximum]
            annotations.append({"field": field, "operation": "text_truncated", "original_length": len(value)})
    if payload.get("action") != "advance_subgoal" and payload.get("next_subgoal") is not None:
        payload.pop("next_subgoal")
        annotations.append({"field": "next_subgoal", "operation": "ignored_outside_explicit_transition"})
    return GuideContinuous.model_validate(payload), annotations


NAV2_GUIDANCE = """Nav2 is the selected navigation backend. Use navigate or explore with a CURRENT
reachable_floor_candidates ID; Nav2 plans and follows a bounded destination using sensor maps.
Explore is a bounded destination in this mode, not rolling directional exploration. Review fresh
stopped evidence after each goal. Do not request continue, compose or circle; those capabilities
belong to the separately selected Built-in backup. Look, turn, scan and stationary verification
remain available. A Nav2 failure never authorizes fallback motion or proves task completion.
"""


SKILL_GUIDANCE = """Motion skill composition is enabled for this session.
Prefer action=compose for a useful short observed search route or precise approach, instead
of individual waypoint choices or repeated rolling continue decisions. Supply skill_plan
with 1-3 components. Total travel must fit 1.8 m; there is no automatic continuation beyond it.
Motion must also fit the remaining 20-second lease measured from the camera observation,
including time spent producing this response and validating depth. The worker integrates
the controller's terminal slowdown, acceleration and steering allowances, then converts simulated
duration to wall time using recent active-motion throughput plus a margin (2x fallback).
It rejects plans that do not fit and returns skill_timing with the estimate and measurement scope.
If admission=no_feasible_plan, no legal shorter component fits that rejected-path estimate:
do not shrink components below their legal minimums and do not copy maximum_length_m into a
component. Use look, scan or wait for a fresh stopped observation, then reassess. This result
is not proof that every possible route is infeasible or that the task is complete.
If admission=shorter_plan_required, only shorter_component_options lists legal sizing ranges.
An approach-only option does not permit a shorter cruise or invent an observed candidate.
Every new route still requires fresh timing and clearance validation; a prior ceiling is not
motion authority. This estimate is not a guarantee: clearance, steering or load can add delay.
Choose a legal CURRENT candidate/route or reobserve after rejection. Never assume a rejected
approach or its terminal inspection ran, and never extend the lease or reuse an old candidate.
cruise: length_m (0.2-1.8), along the current heading.
curve: length_m and angle_rad (-1.2 to 1.2, positive LEFT). Heading changes smoothly over
the component, starting in the preceding heading; it is not an in-place turn or constant-radius arc.
Use gentle curves with enough length; sharply curved routes are rejected.
approach: candidate_id from CURRENT reachable_floor_candidates, last moving component.
It joins smoothly to that observed floor position, not an object's unmeasured center.
All moving components accept view=forward/left/right to inspect fixtures while moving;
the local worker aims the head and still requires fresh depth. Pick a view for the USER'S
search target; avoid repeatedly traversing floor with no useful view. This is a camera
orientation policy, not a detector or a guarantee that a passing target will be recognized.
inspect: optional LAST component, yaw_rad, pitch_rad and duration_s (0.5-2).
It stops and holds the chosen view, then requires a fresh model review. It does NOT verify
object identity, target distance, parking or mission success. For visual inspection tasks,
use at least one second of dwell and review that the correct target remains in view.
Example: skill_plan={"components":[{"kind":"cruise","length_m":0.4,"view":"forward"},
{"kind":"curve","length_m":0.8,"angle_rad":0.4,"view":"left"},{"kind":"inspect","yaw_rad":0.6,"pitch_rad":0.2}]}.
The worker compiles and checks the complete route using observed depth; unknown/blocked
floor rejects the plan without motion. Do not infer clearance from the example or floor color.
A blocked plan needs a different fresh observed route or look/turn, not identical retries.
Use look/turn for stopped reorientation when an approach is over 0.65 rad off the current
heading. Use existing approach_target for marked floor regions; compose does not replace
their persistent target/containment rules. Preserve all completion and sensor-wait rules.
The plan runs locally to its end, a safety interruption or its bounded deadline, then you
review fresh feedback. Stop, disconnect and a new instruction invalidate pending motion.
"""


INSTRUCTIONS = """You direct Milo through observed rooms to satisfy the user goal.
Reason and memory are optional commentary; keep them concise. They do not authorize motion.
Use next_subgoal only with advance_subgoal, whose evidence and transition checks remain required.
An ordinary motion lease can expire without ending the task: the worker stops, discards old
motion and asks for fresh stopped feedback, at most three recoveries in this control loop.
No old candidate or delayed reply renews expired authority. Inspect requests while moving
stop the old route for fresh review instead of waiting for its endpoint. Task time/request
limits remain unchanged; operator Stop, reset, disconnect and takeover always end control.
Use only supplied head images, measured sensors, and observation-derived candidate destinations.
Call guide_continuous exactly once per response. Never output code or motor velocities.
The first image is CURRENT, with candidate IDs marked on observed floor at its original resolution.
When camera_history.enabled is false, historical inspection is unavailable; use current views.
An optional HISTORICAL contact sheet shows up to four distinct paired-camera views captured
since the previous review. Read it oldest to newest, left to right then top to bottom.
Each tile names a frame_id, simulated capture time and measured viewing heading. These are
past sightings, not current clearance, current pixel targets or proof of object identity.
Use inspect_history with history_frame_id from camera_history.available_frames to request
one retained original at its native resolution. This reads a stored image without motion.
After an original is shown, use look/turn/scan to reacquire a promising sighting, or wait for
the next fresh current-only decision. Movement/completion selections made while examining
an original are deferred until a subsequent fresh view. Never apply historical pixel boxes
or candidate IDs to the current camera. Only current sensor/map checks authorize navigation.
History selection is bounded and heuristic; an absent tile does not prove an object was absent.
Record place_sighting only from the CURRENT image, after reacquiring any historical sighting.
For a request to circle a named object, first identify its shape in the CURRENT head image.
Use action=circle, object_label naming the visually identified item, object_bounds=[left,top,right,bottom]
normalized to [0,1] around that single object's visible geometry, and circle_direction matching
the user's command. Do not select a floor patch, the robot's arms, multiple objects or background.
If the object is distant, partly hidden or ambiguous, look/approach using observed candidates first.
The worker estimates the object's position from paired depth and executes bounded clear arcs.
It may return blocked when the next section is unobserved: inspect that section rather than
claiming a complete lap. No hidden object coordinates are available. Use the depth-grounded
circle action rather than approximating a circuit with repeated base turns in place.
Colored floor_regions are estimates from head RGB plus measured depth, not task labels.
For a marked parking destination, select its floor_target_id and use approach_target.
This requests one bounded, freshly validated local route toward that persistent region.
Repeat approach_target after reviewing progress until inside_floor_region is true; the
bay may be farther away than one route. Gray approach floor is valid intermediate progress.
The target persists across waits, turns and clearance pauses. Never replace it with an
unrelated exploration destination just because the bay leaves the current camera view.
If no route is offered, inspect toward the selected target or wait for a moving obstacle;
do not invent clearance. Regions with incomplete views cannot establish parking containment.
For a selected marked region, finish/inspect_arrival uses stationary geometric verification
instead of a full rotation. Use parking completion mode and confirm fresh feedback afterward.
For multi-stage goals, task_progress records measured sensor changes and repeated inspection
attempts. Reconcile these measurements with your memory and the user's expected transition.
If the goal specifies an observable change, compare before/after sensors with that condition;
do not deny an event merely because you did not explicitly request wait. Actual motion or
head actions can advance simulated time and dwell; model inference cannot. A battery change
alone is not general proof of a task, and floor color is not proof of target identity.
When evidence supports moving to the next USER-REQUESTED subgoal, use advance_subgoal with
next_subgoal and exactly one evidence_event_id from task_progress.sensor_changes or
evidence_frame_ref equal to task_progress.current_frame_ref. Explain the evidence in reason.
This is a model assessment, not verified mission success. It clears the previous intermediate
floor target, surface reference and inspection flags, retains measured routes/sightings, and
executes NO motion. Select a fresh observed route only on the following decision. Do not skip
unmet user requirements or use this action to evade a safety rejection. Retained routes always
need fresh clearance. For a single-stage parking goal, keep its target until actually parked.
Repeated unchanged verification is limited to three attempts. If the last inspection rejected
containment, repeating it without a changed pose/view will not improve the evidence: inspect
a missing boundary, select a different safe approach, reconcile a justified subgoal transition,
or report the limitation. Successful verification still requires all original safety checks.
For open-ended room/object SEARCH, use explore with candidate_id selecting an observed direction.
The local supervisor rolls along fresh short paths in that direction while you review new images.
It does not stop at that candidate's position: the candidate supplies a direction, NOT a destination.
Choose explore only for useful search travel, never for precise arrival, parking, a marked floor
target, inspection dwell, a completed task or a furniture circuit. Use navigate/approach_target
when a specific destination is sighted and bounded arrival is needed.
When rolling_exploration is true, each response must review current progress. Choose explore
with a compatible current candidate to change search direction, or continue to retain direction.
These decisions renew at most 20 seconds and 6 m from the captured review, never an expired
authorization; the current exploration mission is capped at 180 seconds. The worker renews
only <=1.8 m freshly observed, footprint-valid paths and brakes for unknown floor or obstacles.
If a destination, doorway requiring inspection, or important ambiguous object is sighted, request
navigate, look or inspect_arrival: rolling renewal ends, then fresh stopped feedback is required.
Do not keep rolling past a target merely to maintain speed. A sharp turn also requires stopping.
For ordinary non-exploration handoffs, when planning_while_moving is true the robot is still following its current authorized route.
Choose navigate only for a compatible observed continuation of the same local intent, or
continue to finish the current route. Any inspection/finish decision must wait for fresh
stopped feedback. A continuation cannot renew the current 1.8 m budget or 60-second deadline.
If no offered continuation improves progress, use continue. Never infer that a running
route has already arrived. Safety and Stop invalidate outstanding proposals.
Use wait with duration_s between 0.5 and 4 to hold position while a moving obstacle passes
or to satisfy an inspection/charging dwell. Physics advances only during actions: inference
time is not simulated waiting. After waiting, inspect fresh feedback before navigating.
Use wait_until for a longer stationary dwell until an observable sensor condition is met.
Supply wait_condition={sensor,comparison,value}, optional wait_guard in the same format,
and wait_timeout_s (0.5-60 seconds, capped in BOTH simulated and wall time and by the original
session deadline). Allowed sensors are battery.charge_pct, battery.charging, battery.low,
and proximity.front_m. Numeric comparisons are gte/lte/eq; booleans require eq with true/false.
Choose thresholds from the user's requested goal, not an arbitrary lower completion target.
For a requested battery level while already charging, select battery.charge_pct gte that
level and guard battery.charging eq true. Do not request another room perimeter rotation
merely to wait when current observations already show charging and safe stationary clearance.
This action does not confirm parking, room entry, identity or whole-task success. If the
goal separately requires boundary verification, keep that requirement. No motion is requested.
The worker rechecks current paired sensors, contact, unchanged pose and observed clearance
every half simulated second. A lost guard, missing/stale sensor or safety rejection returns
blocked without continuing the dwell. A timeout returns timed_out, never condition_met.
Review fresh feedback after the result. Inference is not needed for each half-second slice.
After a moving-obstacle clearance stop, wait before choosing a fresh route; never keep
reissuing the blocked destination while the obstacle is crossing. Stop always ends control.
For goals that only require viewing/inspecting an object, use completion_mode=visual_inspection
with inspect_arrival or finish. First approach to the requested distance and aim the head at
the correct object. The controller holds that view for one simulated second, then sends fresh
feedback. Confirm finish with entry_confirmed=true only if the requested object remains
clearly visible at the required distance. Do not rotate away from it or seek a matching floor.
Use the default parking mode for parking, room entry, delivery, crossing and recharge goals;
all the perimeter, surface and parking-margin rules below apply to parking mode only.
Room search is visual: identify fixtures and openings. Never invent a map or room coordinates.
Navigate by choosing a candidate_id from the current reachable_floor_candidates list. Pixel
coordinates are normalized left/top=0, right/bottom=1 in the current image. Candidate target_m
coordinates are wheel-relative measured map coordinates, not hidden room locations.
Candidates with pixel=null come from previously observed, clearance-checked floor in the
accumulated depth map, possibly beside/behind the robot. bearing_from_base_rad tells their
direction: 0 ahead, +1.57 left, -1.57 right, +/-3.14 behind. The controller turns along the
planned path automatically; you do not need to turn first to select a side/rear candidate.
For room search, prefer candidates farther from visited positions and toward mapped openings.
Do not keep scanning a wall when an unvisited mapped destination is available behind you.
The local controller follows each selected path smoothly up to 1.8 m. Prefer the farther
candidate along a clear doorway/hallway, avoiding unnecessary tiny goals. Stop only to inspect
or replan at meaningful changes. If no useful candidate appears, look or turn to expose a new
direction. Head yaw positive looks left, pitch positive looks down; head motion does not turn
the base. turn_rad positive rotates the BASE left. Use base turns of about 1.5 rad to explore
behind you; do not repeat identical stationary views. For movement, looking down at pitch .45
or .65 reveals floor; pitch .1-.2 identifies room fixtures. Remember visited areas and the
route chosen. A clear range beam does not prove a doorway is traversable.
The goal names the destination room/object, NOT a requirement to return to start unless stated.
Historical views include their measured pose and head angles. Compare their doorways and
fixtures to decide which direction to face. Candidate IDs are ONLY for the first CURRENT
image, never a historical image. Turn toward a promising sighting, then select a newly
observed current candidate. scan gathers fresh coverage for the paired-camera history.
An oven, cooktop and green cabinetry with a sink identify the kitchen, not a bathroom. A sink
alone near the already identified kitchen does not establish arrival in another room.
Large pale articulated structures close to the camera are YOUR folded robot arms, not room
partitions or furniture. Gray room walls are opaque continuous surfaces. Use the supplied
measured exploration record instead of inventing which headings are new. Angles are in radians
in wheel-odometry coordinates; view_heading_rad = base yaw + head yaw. The visited-distance
field on each candidate reports proximity to an earlier reached position. Prefer a genuinely
new destination in an observed opening, unless backtracking is necessary. Do not oscillate
between opposite base turns or head views at one position: after a blocked view, continue
scanning in the SAME rotation direction until an uninspected heading is exposed. Your memory
must retain fixture sightings for the requested destination and the odometry/view heading
where they occurred. Follow the destination in the user goal; it is not always the bathroom.
When that destination is sighted, work toward its opening; do not restart a general room search.
For room entry, seeing its fixtures is insufficient: choose an interior floor destination
well beyond the threshold and reach it before finish. Robot footprint includes arms; leave
clearance around the destination. Never claim success based on a floor color alone.
The annotated CURRENT image marks visible candidate IDs precisely on their floor pixels.
Compare the candidate label with the destination room's floor and the doorway edges. A point
in the hallway beside a doorway does NOT enter the room, even if it is farther away. If the
room lies to image right, a central hallway candidate remains in the hallway. Select a point
on that room's interior floor, or turn/look toward it if no such candidate is offered yet.
After arrival, verify the doorway is behind/beside the whole robot and fixtures surround
the robot in the destination room. Seeing a room through an opening still means outside.
The base and wheels extend roughly 0.3 m from the robot center. For room entry, target a
point at least 0.5 m beyond the doorway plane, not the first floor patch across it. You may
need one more short route farther into the room after the first interior-floor arrival.
Park near the CENTER of the destination room's clear floor, well away from walls, fixtures
and doorway edges. Candidate observed_clearance_m is measured clearance to obstacles OR
unobserved floor in the depth map. Once inside the destination, prefer a central candidate
with larger observed clearance, rather than a far point by another wall. Stop near the middle
of the open interior floor so the entire base remains comfortably inside the room.
current_parking.parking_margin_ok must be true before finish: the local controller requires
8 cm of additional observed clearance outside the conservative robot footprint for parking.
If false, choose a candidate INSIDE THE SAME DESTINATION ROOM whose observed_clearance_m
exceeds required_clearance_m. Do not leave the destination for a clearer hallway point.
When the destination room is identified by visible fixtures, set destination_candidate_id
to a candidate ON THAT ROOM'S FLOOR. This records its measured surface color as the persistent
destination-floor reference. Null leaves the prior reference unchanged. Do not set it to a
hallway candidate beside the doorway. destination_floor_reference and current_parking report
whether the robot is on the recorded surface. Finish requires a surface match; a room seen
through a doorway does not put the robot on its floor. Keep the same reference after entry.
Use inspect_arrival to request a fresh low-angle perimeter inspection. Use finish ONLY
after that inspection with entry_confirmed=true when entry is verified. An uncertain
statement requesting another inspection must use inspect_arrival, NOT finish. The supplied
arrival_inspected flag tells you whether the current position has been inspected since movement.
floor_rgb on ALL candidates is sampled RGB from observed floor, including mapped points.
Use it together with fixture evidence to stay in the destination room; do not choose a
neutral gray hallway point as a blue-tiled bathroom interior point. In blue/cyan floor samples
the green/blue channels are substantially higher than red. Colors identify floor surfaces,
not room semantics: first identify the room by fixtures in real images.
Read its labelled views to check whether hallway floor or the doorway threshold is still
under/adjacent to the base. If any part remains outside, navigate farther into the destination
room before finishing again. Preserve and describe both starting and destination fixtures.
Tool errors authorize no movement. Select another candidate or inspect. Give concise visible
evidence in reason and store task stage, landmarks and visited directions in memory.
"""


def record_inspection_progress(controller, progress, status):
    state = progress.record_inspection(status)
    controller._trace("policy", "Inspection progress", state)
    if state["inspection_exhausted"]:
        controller.state.update(phase="completed", message="Repeated verification without a changed pose or view; robot holding position")
        controller._set_outcome("unachievable", controller.state["message"])
        return True
    return False


def tools(skill_composer=False):
    schema = GuideContinuous.model_json_schema()
    if not skill_composer:
        schema["properties"].pop("skill_plan")
        schema["properties"]["action"]["enum"].remove("compose")
        for name in ("MotionSkillPlan", "CruiseSkill", "CurveSkill", "ApproachSkill", "InspectSkill"):
            schema.get("$defs", {}).pop(name, None)
    return [{"type": "function", "name": "guide_continuous", "description": "Select a current observed destination, inspect a retained camera frame, look, rotate, wait, or finish.",
             "parameters": schema, "strict": False}]


def annotate_candidates(image, candidates, regions=(), selected_id=None):
    rendered = Image.open(BytesIO(image)).convert("RGB")
    original = rendered.copy()
    draw = ImageDraw.Draw(rendered)
    for region in regions:
        if not region["visible"] or not region["image_polygon"]:
            continue
        polygon = [(int(point[0] * rendered.width), int(point[1] * rendered.height)) for point in region["image_polygon"]]
        draw.line([*polygon, polygon[0]], fill="yellow" if region["id"] == selected_id else "cyan", width=2)
        horizontal = max(0, min(rendered.width - 65, min(point[0] for point in polygon)))
        vertical = max(0, min(rendered.height - 12, min(point[1] for point in polygon) - 12))
        draw.text((horizontal, vertical), region["id"], fill="black", stroke_width=1, stroke_fill="white")
    for candidate in candidates:
        if candidate["pixel"] is None:
            continue
        horizontal = min(rendered.width - 1, int(candidate["pixel"][0] * rendered.width))
        vertical = min(rendered.height - 1, int(candidate["pixel"][1] * rendered.height))
        candidate["floor_rgb"] = list(original.getpixel((horizontal, vertical)))
        text = str(candidate["id"])
        draw.ellipse((horizontal - 8, vertical - 8, horizontal + 8, vertical + 8), fill="white", outline="black", width=1)
        draw.text((horizontal, vertical), text, fill="black", anchor="mm")
    output = BytesIO()
    rendered.save(output, format="PNG")
    return output.getvalue()


def nearby_panorama(frames, observation):
    now = time.time()
    return [(recorded, image) for recorded, image in frames
        if math.dist(recorded.odometry_m_rad[:2], observation.odometry_m_rad[:2]) < .2
        and 0 <= now - getattr(recorded, "wall_timestamp", now) <= 90
        and recorded.run_id == observation.run_id and recorded.episode_epoch == observation.episode_epoch]

def target_candidate(observation, candidates, floor_target):
    distance = math.dist(observation.odometry_m_rad[:2], floor_target["center_m"])
    progress = [candidate for candidate in candidates if math.dist(candidate["target_m"], floor_target["center_m"]) < distance - .05]
    return min(progress, key=lambda candidate: math.dist(candidate["target_m"], floor_target["center_m"])
        + .1 * candidate.get("path_length_m", candidate["distance_m"])) if progress else None


def recovery_choice(memory, observation, candidates, floor_target=None):
    if memory.recovery_attempts >= 2:
        return None
    failed = [action["parameters"].get("target_m") for action in list(memory.actions)[-8:]
        if action["status"] in {"blocked", "failed", "error"}]
    if floor_target is not None:
        choices = [candidate for candidate in candidates if not any(target is not None
            and math.dist(target, candidate["target_m"]) < .15 for target in failed)]
        choice = target_candidate(observation, choices, floor_target)
        if choice is not None:
            return GuideContinuous(action="navigate", candidate_id=choice["id"], floor_target_id=floor_target["id"],
                reason="Bounded recovery toward the retained observed floor target")
        delta = [floor_target["center_m"][axis] - observation.odometry_m_rad[axis] for axis in (0, 1)]
        bearing = math.atan2(delta[1], delta[0]) - observation.odometry_m_rad[2]
        bearing = math.atan2(math.sin(bearing), math.cos(bearing))
        return GuideContinuous(action="turn" if abs(bearing) > .2 else "look", turn_rad=max(-1.5, min(1.5, bearing)),
            pitch_rad=.3, floor_target_id=floor_target["id"], reason="Inspect toward the retained floor target without switching to unrelated exploration")
    alternatives = [choice for choice in candidates if choice["distance_m"] >= .4
        and choice.get("distance_to_visited_m", 0.) >= .3
        and not any(target is not None and math.dist(target, choice["target_m"]) < .3 for target in failed)]
    if alternatives:
        choice = max(alternatives, key=lambda item: (item["distance_to_visited_m"], -item.get("path_length_m", item["distance_m"])))
        return GuideContinuous(action="navigate", candidate_id=choice["id"], reason="Bounded recovery toward a currently reachable unvisited position")
    angle = memory.next_scan_turn(observation)
    if angle is not None:
        return GuideContinuous(action="turn", turn_rad=angle, reason="Bounded recovery to inspect a missing heading in the same rotation direction")
    for route in reversed(memory.routes):
        backtrack = [choice for choice in candidates if choice["distance_m"] >= .4
            and math.dist(choice["target_m"], route["start_m"]) <= .3
            and not any(target is not None and math.dist(target, choice["target_m"]) < .3 for target in failed)]
        if backtrack:
            choice = min(backtrack, key=lambda item: item.get("path_length_m", item["distance_m"]))
            return GuideContinuous(action="navigate", candidate_id=choice["id"], reason="Bounded backtrack to a previously reached position through a currently validated route")
    return None


async def adaptive_scan(controller, worker, settings, stop_revision):
    observation, _ = await worker.feedback()
    controller._check_live(worker, settings)
    angle = controller.navigation_memory.next_scan_turn(observation)
    if angle is None:
        return []
    frames = await rotate(controller, worker, settings, stop_revision, angle, panoramic=True)
    controller._check_live(worker, settings)
    await worker.scan_continuous(ContinuousScan(run_id=settings.run_id, episode_epoch=settings.episode_epoch))
    controller._check_live(worker, settings)
    return frames


def matching_floor(first, second):
    if first is None or second is None or not sum(first) or not sum(second):
        return False
    return math.dist([channel / sum(first) for channel in first], [channel / sum(second) for channel in second]) < .04


async def rotate(controller, worker, settings, stop_revision, angle, panoramic=False, pitch=.15):
    from scripts.navigation_policy import apply
    controller._check_live(worker, settings)
    if panoramic:
        current, _ = await worker.feedback()
        controller._check_live(worker, settings)
        reply = await worker.execute(Command(run_id=settings.run_id, episode_epoch=settings.episode_epoch,
            observation_seq=current.seq, action_id=str(uuid4()), tool="set_head",
            arguments={"yaw_rad": 0., "pitch_rad": pitch, "duration_s": 1.}), assisted=False)
        if reply.status != "ok":
            raise ValueError(reply.message)
    controller._check_live(worker, settings)
    await worker.begin_navigation(stop_revision)
    await worker.call(lambda sim: apply(worker.navigation, sim, "begin_local_subgoal", {"goal": "Inspect observed room headings"}))
    initial, image = await worker.feedback()
    target = initial.odometry_m_rad[2] + angle
    frames = [(initial, image)] if panoramic else []
    while True:
        controller._check_live(worker, settings)
        current = await worker.call(lambda sim: sim.observe(render=False))
        remaining = target - current.odometry_m_rad[2]
        if panoramic and len(frames) < 4 and abs(current.odometry_m_rad[2] - initial.odometry_m_rad[2]) >= len(frames) * math.pi / 2:
            current, image = await worker.feedback()
            frames.append((current, image))
            if hasattr(controller, "navigation_memory"):
                controller.navigation_memory.observe(current)
        if abs(remaining) < .035:
            break
        await worker.call(lambda sim: apply(worker.navigation, sim, "replace_motion_buffer", {"segments": [
            {"kind": "drive", "linear_mps": 0., "angular_radps": max(-.5, min(.5, 2 * remaining)), "duration_s": 1.}]}))
        await asyncio.sleep(.1)
    await worker.call(lambda sim: worker.navigation.cancel(sim, "Viewpoint rotation finished"))
    final = await settle(worker, settings)
    if panoramic:
        final, image = await worker.feedback()
        frames.append((final, image))
    if hasattr(controller, "navigation_memory"):
        controller.navigation_memory.remember(initial, final, {"action": "scan" if panoramic else "turn", "turn_rad": angle}, {"status": "ok"})
    return frames


async def settle(worker, settings):
    observation, _ = await worker.feedback()
    result = await worker.execute(Command(run_id=settings.run_id, episode_epoch=settings.episode_epoch,
        observation_seq=observation.seq, action_id=str(uuid4()), tool="drive_base",
        arguments={"linear_mps": 0., "angular_radps": 0., "duration_s": .5}), assisted=False)
    if result.status != "ok":
        raise ValueError(result.message)
    return result.observation


async def wait_stationary(controller, worker, settings, duration_s):
    remaining = duration_s
    while remaining > 1e-6:
        controller._check_live(worker, settings)
        observation, _ = await worker.feedback()
        controller._check_live(worker, settings)
        interval = min(.5, remaining)
        result = await worker.execute(Command(run_id=settings.run_id, episode_epoch=settings.episode_epoch,
            observation_seq=observation.seq, action_id=str(uuid4()), tool="wait",
            arguments={"duration_s": interval}), assisted=False)
        controller._check_live(worker, settings)
        if result.status != "ok":
            raise ValueError(result.message)
        remaining -= interval


async def wait_for_sensor(controller, worker, settings, condition, timeout_s, guard=None, floor_target=None, destination_floor=None):
    from backend.simulation import MotionError
    if not math.isfinite(timeout_s) or not .5 <= timeout_s <= 60:
        raise ValueError("Sensor wait must be bounded to 0.5-60 seconds")
    started = time.monotonic()
    initial = None
    elapsed_simulated = 0.

    def outcome(status, observation, reason):
        return {"status": status, "action": "wait_until", "reason": reason,
            "condition": condition.model_dump(), "guard": guard.model_dump() if guard else None,
            "elapsed_simulated_s": round(elapsed_simulated, 3), "elapsed_wall_s": round(time.monotonic() - started, 3),
            "frame_ref": observation.frame_ref, "mission_success_verified": False}

    while True:
        controller._check_live(worker, settings)
        observation, _ = await worker.feedback()
        controller._check_live(worker, settings)
        if initial is None:
            initial = observation
        try:
            parking = await worker.parking_clearance(floor_target, stationary_request=settings)
        except (ValueError, MotionError) as error:
            controller._check_live(worker, settings)
            return outcome("blocked", observation, str(error))
        controller._check_live(worker, settings)
        observation, _ = await worker.feedback()
        controller._check_live(worker, settings)
        elapsed_simulated = observation.simulated_time_s - initial.simulated_time_s
        if (observation.run_id, observation.episode_epoch) != (settings.run_id, settings.episode_epoch):
            raise asyncio.CancelledError
        if not 0 <= time.time() - observation.wall_timestamp <= 1.:
            return outcome("blocked", observation, "Current paired sensor observation is stale")
        if observation.bumpers or (observation.proximity and observation.proximity.collisions):
            return outcome("blocked", observation, "Contact detected during stationary wait")
        yaw_delta = observation.odometry_m_rad[2] - initial.odometry_m_rad[2]
        if (math.dist(observation.odometry_m_rad[:2], initial.odometry_m_rad[:2]) > .02
                or abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta))) > .05):
            return outcome("blocked", observation, "Stationary wait pose changed")
        if ((worker.continuous and worker.continuous.active)
                or (observation.navigation and observation.navigation.status in {"running", "awaiting_feedback"})
                or (observation.skill and observation.skill.status in {"running", "awaiting_policy"})):
            return outcome("blocked", observation, "Stationary wait requires inactive motion controllers")
        if (not parking["parking_margin_ok"] or (floor_target is not None and not parking.get("inside_floor_region"))
                or (destination_floor is not None and not matching_floor(parking["floor_rgb"], destination_floor))):
            return outcome("blocked", observation, "Observed stationary clearance or selected-floor containment is not confirmed")
        try:
            if guard is not None and not guard.matches(observation):
                return outcome("blocked", observation, "Required sensor wait guard no longer holds")
            matched = condition.matches(observation)
        except ValueError as error:
            return outcome("blocked", observation, str(error))
        if time.monotonic() - started >= timeout_s:
            return outcome("timed_out", observation, "Bounded sensor wait wall-time limit reached")
        if matched:
            return {**outcome("condition_met", observation, "Requested sensor condition observed"),
                "measured_value": condition.reading(observation)}
        remaining = min(timeout_s - elapsed_simulated, timeout_s - (time.monotonic() - started))
        if remaining <= 1e-6:
            return outcome("timed_out", observation, "Bounded sensor wait simulation-time limit reached")
        await wait_stationary(controller, worker, settings, min(.5, remaining))


async def circle_observed_object(controller, worker, settings, sensor, guide, stop_revision):
    from backend.simulation import MotionError
    if guide.object_bounds is None or not guide.object_label.strip():
        raise ValueError("Identify the object and supply its current image bounding box")
    orbit = await worker.locate_observed_orbit(sensor, guide.object_bounds, guide.object_label, guide.circle_direction)
    current, _ = await worker.feedback()
    result = await worker.execute(Command(run_id=settings.run_id, episode_epoch=settings.episode_epoch, observation_seq=current.seq,
        action_id=str(uuid4()), tool="set_head", arguments={"yaw_rad": 0., "pitch_rad": .45, "duration_s": 1.}), assisted=False)
    if result.status != "ok":
        raise ValueError(result.message)
    rescans = 0
    freshness_retries = 0
    async def retry_freshness(reason):
        nonlocal freshness_retries
        controller._check_live(worker, settings)
        if freshness_retries >= 2:
            return False
        freshness_retries += 1
        controller._trace("policy", "Circuit sensing recovery", {"attempt": freshness_retries,
            "source": "controller", "object": guide.object_label, "reason": reason})
        await asyncio.sleep(.1)
        controller._check_live(worker, settings)
        return True

    for _ in range(32):
        controller._check_live(worker, settings)
        try:
            progress = await worker.start_observed_orbit_step(orbit, stop_revision)
        except MotionError as error:
            controller._check_live(worker, settings)
            if error.code == "SPATIAL_STALE" and await retry_freshness(f"{error.code}: {error}"):
                continue
            if error.code == "OBSERVED_PATH_BLOCKED" and rescans < 2:
                rescans += 1
                controller._trace("policy", "Circuit floor rescan", {"attempt": rescans, "source": "controller", "reason": str(error)})
                current, _ = await worker.feedback()
                controller._check_live(worker, settings)
                path, complete = orbit.path(current.odometry_m_rad)
                if not complete:
                    delta = [path[-1][axis] - current.odometry_m_rad[axis] for axis in (0, 1)]
                    bearing = math.atan2(delta[1], delta[0]) - current.odometry_m_rad[2]
                    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
                    if abs(bearing) > .2:
                        await rotate(controller, worker, settings, stop_revision, bearing)
                await worker.scan_continuous(ContinuousScan(run_id=settings.run_id, episode_epoch=settings.episode_epoch))
                continue
            return {"status": "blocked", "reason": f"{error.code}: {error}", "object": guide.object_label}
        controller._trace("policy", "Observed furniture circuit", progress)
        if progress["complete"]:
            await wait_stationary(controller, worker, settings, .5)
            return {"status": "arrived", **progress, "completion_source": "measured_orbit_not_semantic_proof"}
        while worker.continuous.active:
            controller._check_live(worker, settings)
            continuation = getattr(worker.continuous, "orbit_progress", None)
            if continuation is not None and continuation is not progress:
                progress = continuation
                controller._trace("policy", "Observed furniture circuit", progress)
            await asyncio.sleep(.05)
        controller._check_live(worker, settings)
        if worker.continuous.status != "arrived":
            if worker.continuous.reason.startswith("SPATIAL_STALE:") and await retry_freshness(worker.continuous.reason):
                continue
            return {"status": "blocked", "reason": worker.continuous.reason, **progress}
    raise ValueError("Circuit segment budget exhausted; inspect and reassess")


async def execute_motion_skills(controller, worker, settings, sensor, guide, candidates, stop_revision, task_revision):
    from backend.simulation import MotionError
    try:
        await worker.start_motion_skills(sensor, guide.skill_plan, candidates, stop_revision, task_revision)
        while worker.continuous.active:
            controller._check_live(worker, settings)
            await asyncio.sleep(.05)
        controller._check_live(worker, settings)
        result = {**worker.continuous.state(), "mission_success_verified": False, "inspection_ready": False}
        if result["status"] != "arrived":
            return result
        await settle(worker, settings)
        inspection = guide.skill_plan.components[-1]
        if inspection.kind == "inspect":
            current, _ = await worker.feedback()
            reply = await worker.execute(Command(run_id=settings.run_id, episode_epoch=settings.episode_epoch,
                observation_seq=current.seq, action_id=str(uuid4()), tool="set_head",
                arguments={"yaw_rad": inspection.yaw_rad, "pitch_rad": inspection.pitch_rad, "duration_s": 1.}), assisted=False)
            controller._check_live(worker, settings)
            if reply.status != "ok":
                raise ValueError(reply.message)
            await wait_stationary(controller, worker, settings, inspection.duration_s)
            observed, _ = await worker.feedback()
            controller._check_live(worker, settings)
            result.update(inspection_ready=inspection.duration_s >= 1., inspection_frame_ref=observed.frame_ref,
                inspection_duration_s=inspection.duration_s)
        return result
    except MotionSkillBudgetError as error:
        controller._check_live(worker, settings)
        return {"status": "blocked", "error_code": "PLAN_BUDGET", "reason": str(error),
            "admission": error.timing["admission"], "skill_timing": error.timing,
            "motion_authorized": False, "mission_success_verified": False, "inspection_ready": False}
    except (MotionError, ValueError) as error:
        controller._check_live(worker, settings)
        return {"status": "blocked", "reason": str(error), "mission_success_verified": False, "inspection_ready": False}


async def execute_home_capability(controller, worker, settings, guide, sensor, stop_revision, task_revision, image=None):
    from backend.home_mission import HomeRequest
    actions = {"navigate_place": "navigate_to", "explore_map": "explore", "explore_frontier": "explore_frontier",
        "cancel_task": "cancel_task", "spatial_state": "get_spatial_state", "remember_object": "remember_object", "observe_room": "observe_room"}
    controller._check_live(worker, settings)
    if guide.action in {"navigate_place", "explore_map", "explore_frontier"} and not 0 <= time.monotonic() - sensor.captured_at <= 15.:
        raise ValueError("STALE_PLAN: mapped task selection expired during inference")
    state = await worker.home_state(compact=True)
    request = HomeRequest(run_id=settings.run_id, episode_epoch=settings.episode_epoch,
        action=actions[guide.action], map_id=state["map_id"], place_id=guide.place_id, return_place_id=guide.return_place_id,
        region_id=guide.region_id, time_budget=guide.time_budget, object_label=guide.object_label,
        object_bounds=guide.object_bounds, confidence=guide.confidence,
        name=guide.room_name, kind=guide.place_kind, connects=guide.connects,
        evidence_text=guide.evidence_text, room_matches=guide.room_matches, frontier_id=guide.frontier_id,
        spatial_sequence=sensor.sequence if guide.action in {"remember_object", "observe_room"} else None)
    await worker.home_command(request, stop_revision, task_revision,
        selected_evidence=(sensor, image) if guide.action in {"remember_object", "observe_room"} and image is not None else None)
    while worker.home_mission.active:
        controller._check_live(worker, settings)
        await asyncio.sleep(.1)
    controller._check_live(worker, settings)
    return await worker.home_state(compact=True)


async def run(controller, worker, settings, model, profile, stop_revision):
    from backend.agent import feedback_message
    if not await worker.resume_manual(expected_stop_revision=stop_revision):
        raise asyncio.CancelledError
    history = []
    memory = ""
    task_progress = TaskProgress()
    visited = []
    views = []
    finish_checked = False
    inspection_view_checked = False
    destination_floor = None
    floor_tracker = FloorRegionTracker(settings.run_id, settings.episode_epoch)
    floor_target_id = None
    last_execution = None
    requested_history_frame = None
    schema_rejections = 0
    decision_validation = None
    recovered_routes = set()

    def recover_expired_route():
        control = worker.continuous
        if (control is None or control.active or control.identity in recovered_routes or getattr(control, "observed_orbit", None) is not None
                or not control.reason.startswith(("MOTION_LEASE_EXPIRED:", "EXPLORATION_REVIEW_EXPIRED:", "BUFFER_EXPIRED:"))):
            return None
        recovered_routes.add(control.identity)
        result = {**control.state(), "motion_authorized": False, "recovery_count": len(recovered_routes),
            "recovery_limit": 3, "fresh_stopped_feedback_required": True}
        controller._trace("policy", "Motion lease recovery", result)
        if len(recovered_routes) > 3:
            raise ValueError("Motion lease recovery limit reached after 3 recoveries; robot remains stopped")
        return result

    controller.state["phase"] = "acting"
    preparation_started = time.monotonic()
    revisiting = bool(controller.navigation_memory.actions)
    await worker.scan_continuous(ContinuousScan(run_id=settings.run_id, episode_epoch=settings.episode_epoch, compact_arms=settings.compact_arms))
    controller._check_live(worker, settings)
    panorama = []
    if not settings.adaptive_navigation and not revisiting:
        await rotate(controller, worker, settings, stop_revision, 2 * math.pi, panoramic=True, pitch=.85)
        panorama = await rotate(controller, worker, settings, stop_revision, 2 * math.pi, panoramic=True)
    else:
        observation, image = await worker.feedback()
        controller.navigation_memory.observe(observation)
        panorama = [(observation, image)]
    controller._trace("policy", "Exploration preparation", {"adaptive": settings.adaptive_navigation,
        "full_startup_rotations": 0 if settings.adaptive_navigation or revisiting else 2,
        "wall_s": time.monotonic() - preparation_started})
    for turn in controller.turn_indices(settings):
        controller._check_live(worker, settings)
        expired = recover_expired_route()
        if expired is not None:
            last_execution = expired
            memory += " Previous motion authority expired; no arrival or inspection was established. Choose only from fresh stopped evidence."
        scored = worker.latest.get("challenge")
        if scored and scored["id"] == "furniture_circuit" and scored["status"] in {"completed", "failed"}:
            controller.state.update(phase="completed", message="Furniture circuit verified by physics" if scored["status"] == "completed" else "Furniture circuit failed: contact")
            controller._set_outcome(scored["status"], controller.state["message"], "physics")
            return
        floor_target = next((region for region in floor_tracker.public() if region["id"] == floor_target_id), None)
        proposal = None
        if worker.continuous and worker.continuous.active and (settings.continuous_handoff or worker.continuous.exploration is not None):
            try:
                sensor, image, candidates, observation, proposal = await worker.moving_candidates()
            except ValueError:
                continue
        else:
            sensor, image, candidates, observation = await worker.continuous_candidates(visited, floor_target)
        try:
            regions = await asyncio.to_thread(floor_tracker.update, sensor, image)
        except ValueError:
            regions = floor_tracker.public()
        controller._check_live(worker, settings)
        floor_target = next((region for region in regions if region["id"] == floor_target_id), None)
        panorama = nearby_panorama(panorama, observation)
        episode_memory = controller.navigation_memory.observe(observation)
        progress_feedback = task_progress.observe(observation)
        parking = await worker.parking_clearance(floor_target)
        parking["destination_surface_match"] = matching_floor(parking["floor_rgb"], destination_floor)
        position = observation.odometry_m_rad
        if not visited or min(math.dist(position[:2], point) for point in visited) > .35:
            visited.append(position[:2])
        sightings = await worker.place_sightings()
        for choice in candidates:
            choice["distance_to_visited_m"] = round(min(math.dist(choice["target_m"], point) for point in visited), 2)
            choice["nearby_place_sightings"] = [item["id"] for item in sightings
                if math.dist(choice["target_m"], item["position_m"]) < .8]
            choice["distance_to_sightings_m"] = {item["id"]: round(math.dist(choice["target_m"], item["position_m"]), 2)
                for item in sightings}
        view_heading = position[2] + observation.head_rad[0]
        views.append({"turn": turn + 1, "position_m": position[:2], "view_heading_rad": round(math.atan2(math.sin(view_heading), math.cos(view_heading)), 2)})
        annotated = annotate_candidates(image, candidates, regions, floor_target_id)
        camera_review, history_sheet = await worker.camera_history_review(sensor, stop_revision)
        camera_review["enabled"] = getattr(controller, "evaluation_camera_history", True)
        if not camera_review["enabled"]:
            camera_review = {**camera_review, "frames": [], "available_frames": []}
            history_sheet = None
        historical_original = None
        if requested_history_frame is not None:
            try:
                historical_original = await worker.inspect_camera_history(requested_history_frame, stop_revision)
            except ValueError as error:
                memory += f" Historical original unavailable: {error}. Inspect a fresh view."
            requested_history_frame = None
        controller._check_live(worker, settings)
        message = feedback_message(observation, annotated)
        message["content"].append({"type": "input_text", "text": json.dumps({"reachable_floor_candidates": candidates,
            "floor_regions": regions, "selected_floor_target": floor_target,
            "memory": memory, "arrival_inspected": finish_checked, "current_parking": parking,
            "task_progress": progress_feedback,
            "decision_validation": decision_validation,
            "visual_inspection_checked": inspection_view_checked,
            "destination_floor_reference": destination_floor,
            "planning_while_moving": proposal is not None,
            "rolling_exploration": bool(proposal and proposal.get("exploration")),
            "measured_episode_memory": episode_memory,
            "place_sightings": sightings,
            "sighting_route_connections": controller.navigation_memory.sighting_routes(sightings),
            "adaptive_exploration": settings.adaptive_navigation,
            "camera_history": camera_review,
            "historical_original": historical_original[0] if historical_original else None,
            "fresh_view_required_before_motion": historical_original is not None,
            "measured_exploration": {"visited_positions_m": visited[-30:], "recent_views": views[-12:]},
            "local_execution": worker.continuous.state() if proposal is not None else last_execution})})
        model_images = [annotated]
        if history_sheet is not None:
            message["content"].extend([{"type": "input_text", "text": "HISTORICAL contact sheet. Native-size views, chronological 2x2. Frame IDs refer only to stored past images; current candidate IDs and pixel selections do not apply."},
                {"type": "input_image", "detail": "high", "image_url": "data:image/png;base64," + base64.b64encode(history_sheet).decode("ascii")}])
            model_images.append(history_sheet)
        if historical_original is not None:
            message["content"].extend([{"type": "input_text", "text": "REQUESTED HISTORICAL ORIGINAL. " + json.dumps(historical_original[0]) + " Reacquire a fresh view before choosing movement or completion."},
                {"type": "input_image", "detail": "high", "image_url": "data:image/png;base64," + base64.b64encode(historical_original[1]).decode("ascii")}])
            model_images.append(historical_original[1])
        inputs = [*history[-8:], message]
        inputs.insert(0, {"role": "user", "content": [{"type": "input_text", "text": MEMORY_GUIDANCE + PLACE_GUIDANCE
            + " Saved-home capabilities: use observation.spatial.places and room_graph for named-room tasks. navigate_place selects an exact reachable place_id; never invent coordinates or IDs. operator_confirmed labels have operator review; tentative and operator_named_unreviewed labels are not confirmed room identities. If the requested room is unknown, choose a supplied frontier_id with explore_frontier (time_budget1-300s) or explore_map; inspect fresh images to identify rooms. Unknown room names do not authorize arbitrary destinations or completion. observe_room records room_name, place_kind, confidence and evidence_text from the CURRENT image at the measured robot position; optional connects references supplied IDs and requires observed routes. This creates tentative labels, never operator confirmation. For an existing room use place_id. After navigate_place arrives, a NEW post-arrival image and observe_room(place_id,room_matches=true/false,evidence_text) are required before reporting visual room identification. Physical arrival alone and old room photographs are insufficient; mismatch means reassess, not success. Room reports remain model claims, not independent semantic verification. room_graph connections describe current clearance of recorded paths, not proof that a door is open. spatial_state obtains progress; cancel_task stops. remember_object stores object_label, CURRENT normalized object_bounds and confidence with measured surface position and original image. Historical observations never authorize motion. The worker owns planning, wheels, retries and timeouts. Unlocalized maps require operator localization."
            + (NAV2_GUIDANCE if settings.navigation_backend == "nav2" else "")}]})
        inputs[0]["content"].append({"type": "input_text", "text": "For room-and-return requests, navigate_place(place_id=room,return_place_id=Home,time_budget=...) starts one bounded workflow. After physical arrival, observe_room with a NEW current image reports match or mismatch. A match authorizes automatic worker return to Home; a mismatch fails the workflow. A completed round trip requires room_workflow.status=completed. Semantic identity remains a reported annotation."})
        task_revision = worker.task_revision
        controller.state.update(phase="thinking", turns=turn + 1, last_feedback_at=time.time())
        controller._trace("feedback", "Continuous navigation camera + sensors", {"observation": observation.model_dump(),
            "task_progress": progress_feedback,
            "decision_validation": decision_validation,
            "measured_episode_memory": episode_memory, "place_sightings": sightings,
            "sighting_route_connections": controller.navigation_memory.sighting_routes(sightings),
            "adaptive_exploration": settings.adaptive_navigation,
            "candidates": candidates, "images_in_request": len(model_images), "image_detail": "high", "history_turns": [],
            "camera_history": camera_review, "historical_original": historical_original[0] if historical_original else None,
            "floor_regions": regions, "selected_floor_target": floor_target,
            "current_parking": parking,
            "input_items": len(inputs), "tool_result_call_ids": []}, image=annotated, images=model_images)
        started = time.monotonic()
        async with asyncio.timeout(controller.request_timeout_s):
            response = await model.respond(profile, settings.reasoning, settings.goal, inputs)
        controller._check_live(worker, settings)
        controller.state["inference_latency_s"] = time.monotonic() - started
        if response.usage:
            controller.state["input_tokens"] += response.usage.input_tokens
            controller.state["output_tokens"] += response.usage.output_tokens
        outputs = [entry.model_dump(exclude_none=True) for entry in response.output]
        calls = [entry for entry in outputs if entry["type"] == "function_call"]
        controller._trace("response", "Luna continuous decision", {"text": response.output_text[:2000], "status": response.status,
            "latency_s": time.monotonic() - started, "calls": calls, "text_truncated": False, "calls_truncated": False, "refusals": [],
            "input_tokens": response.usage.input_tokens if response.usage else None, "output_tokens": response.usage.output_tokens if response.usage else None})
        if response.status != "completed" or len(calls) != 1 or calls[0]["name"] != "guide_continuous":
            raise ValueError("Luna must return one completed guide_continuous decision")
        if any(item.get("blocked") for item in (getattr(response, "model_extra", None) or {}).get("content_filters", []) or []):
            raise ValueError("Model response blocked by Foundry guardrails")
        if any(part.get("type") == "refusal" for entry in outputs if entry["type"] == "message" for part in entry.get("content", [])):
            raise ValueError("Model declined the request; no motion was executed")
        if calls[0]["call_id"] in controller.seen_text_calls or len(calls[0]["arguments"]) > 8000:
            raise ValueError("Repeated or oversized continuous decision")
        controller.seen_text_calls.add(calls[0]["call_id"])
        expired = recover_expired_route()
        if expired is not None:
            last_execution = expired
            memory += " Motion expired during the previous decision; that reply was discarded. Reassess this fresh stopped view."
            controller._trace("policy", "Expired motion decision discarded", {"call_id": calls[0]["call_id"], "motion_authorized": False})
            continue
        try:
            guide, annotations = parse_continuous_decision(calls[0]["arguments"])
        except ValidationError as error:
            schema_rejections += 1
            decision_validation = {"status": "schema_rejected", "motion_authorized": False,
                "rejections": schema_rejections, "corrections_remaining": max(0, 3 - schema_rejections),
                "instruction": "Return one corrected guide_continuous decision using this fresh stopped view. The rejected action was not executed. Keep reason within 500 characters.",
                "errors": [{"field": ".".join(str(part) for part in item["loc"]), "message": item["msg"][:200]}
                    for item in error.errors(include_input=False, include_context=False, include_url=False)[:6]]}
            controller._trace("policy", "Continuous decision rejected", decision_validation)
            await worker.pause_continuous(ContinuousScan(run_id=settings.run_id, episode_epoch=settings.episode_epoch),
                stop_revision, task_revision, "Invalid model decision; fresh stopped correction required")
            controller._check_live(worker, settings)
            if schema_rejections > 2:
                raise ValueError("Continuous decision schema failed after 2 correction attempts") from error
            last_execution = decision_validation
            continue
        decision_validation = None
        if annotations:
            controller._trace("policy", "Decision commentary normalized", {"fields": annotations,
                "execution_parameters_changed": False, "raw_response_retained": True})
        if guide.action == "compose" and not settings.skill_composer:
            raise ValueError("Motion skill composition is not enabled for this session")
        await worker.acknowledge_camera_history(camera_review, stop_revision)
        controller._check_live(worker, settings)
        if guide.place_sighting is not None and historical_original is None:
            await worker.record_place_sighting(observation, guide.place_sighting, stop_revision)
            controller._check_live(worker, settings)
        rolling = bool(proposal and proposal.get("exploration"))
        if proposal is not None and ((rolling and guide.action not in {"explore", "continue"})
                                     or (not rolling and guide.action != "navigate")):
            await worker.pause_continuous(ContinuousScan(run_id=settings.run_id, episode_epoch=settings.episode_epoch),
                stop_revision, task_revision, "Stopped for fresh supervisor inspection")
            controller._check_live(worker, settings)
            last_execution = {"status": "stopped_for_review", "requested_action": guide.action,
                "motion_authorized": False, "inspection_ready": False}
            memory = guide.memory + " Rolling travel ended for the requested " + guide.action + ". Reassess with fresh stopped evidence before acting."
            controller._trace("policy", "Moving decision deferred", {"action": guide.action, "reason": "Fresh stopped feedback required"})
            continue
        memory = guide.memory
        if guide.action == "inspect_history":
            try:
                if not camera_review["enabled"]:
                    raise ValueError("Historical inspection is disabled for this evaluation; inspect a current view")
                if guide.history_frame_id is None:
                    raise ValueError("Select a retained history_frame_id from camera_history.available_frames")
                historical, _ = await worker.inspect_camera_history(guide.history_frame_id, stop_revision)
                requested_history_frame = guide.history_frame_id
                result = {"status": "historical_image_requested", **historical}
            except ValueError as error:
                result = {"status": "unavailable", "reason": str(error)}
            controller._check_live(worker, settings)
            controller.navigation_reply(guide.reason)
            controller._trace("policy", "Historical frame inspection", result)
            history.extend([{"role": "assistant", "content": [{"type": "output_text", "text": json.dumps(guide.model_dump())}]},
                {"role": "user", "content": [{"type": "input_text", "text": json.dumps(result)}]}])
            last_execution = result
            continue
        if historical_original is not None and guide.action not in {"look", "turn", "scan", "wait"}:
            memory += " Historical inspection finished. Select movement or completion only after this new current view; old image coordinates are not actionable."
            controller._trace("policy", "Historical motion selection deferred", {"action": guide.action, "frame_id": historical_original[0]["frame_id"], "reason": memory})
            continue
        if guide.action in {"navigate_place", "explore_map", "explore_frontier", "cancel_task", "spatial_state", "remember_object", "observe_room"}:
            controller.state.update(phase="acting", message=guide.reason)
            try:
                last_execution = await execute_home_capability(controller, worker, settings, guide, sensor, stop_revision, task_revision, image)
            except (ValueError, MotionError) as error:
                last_execution = {"status": "rejected", "reason": str(error), "motion_authorized": False}
            controller._check_live(worker, settings)
            controller.navigation_reply(guide.reason)
            controller._trace("policy", "Saved home mission", last_execution)
            history.extend([{"role": "assistant", "content": [{"type": "output_text", "text": json.dumps(guide.model_dump())}]},
                {"role": "user", "content": [{"type": "input_text", "text": json.dumps(last_execution)}]}])
            continue
        if guide.action == "advance_subgoal":
            try:
                if guide.next_subgoal is None:
                    raise ValueError("Name the next user-requested subgoal")
                transition = task_progress.advance(guide.next_subgoal,
                    event_id=guide.evidence_event_id, frame_ref=guide.evidence_frame_ref)
            except ValueError as error:
                last_execution = {"status": "rejected", "reason": str(error), "motion_authorized": False}
                controller._trace("policy", "Subgoal transition rejected", last_execution)
            else:
                floor_target_id = floor_target = destination_floor = None
                finish_checked = inspection_view_checked = False
                last_execution = {"status": "subgoal_advanced", **transition}
                memory += " Previous intermediate target released. Reassess fresh observations before acting; no motion or mission completion was authorized."
                controller._trace("policy", "Subgoal advanced", last_execution)
            controller.navigation_reply(guide.reason)
            history.extend([{"role": "assistant", "content": [{"type": "output_text", "text": json.dumps(guide.model_dump())}]},
                {"role": "user", "content": [{"type": "input_text", "text": json.dumps(last_execution)}]}])
            continue
        if guide.floor_target_id is not None:
            selected = next((region for region in floor_tracker.public() if region["id"] == guide.floor_target_id), None)
            if selected is None:
                memory += " Requested floor target is unavailable or expired. Inspect a fresh camera view."
                controller._trace("policy", "Floor goal rejected", {"reason": "Unknown or expired observed region"})
                if guide.action in {"finish", "inspect_arrival"} and record_inspection_progress(controller, task_progress, "target_unavailable"):
                    return
                continue
            if floor_target_id != selected["id"]:
                finish_checked = False
            floor_target_id, floor_target = selected["id"], selected
        if guide.action == "approach_target":
            if floor_target is None:
                memory += " Select a current observed floor_target_id before approach_target."
                continue
            sensor, image, candidates, observation = await worker.continuous_candidates(visited, floor_target)
            controller._check_live(worker, settings)
            choice = target_candidate(observation, candidates, floor_target)
            if choice is None:
                memory += " No fresh clearance-valid approach toward the retained target. Inspect toward it or wait; do not switch to unrelated exploration."
                controller.navigation_memory.remember(observation, observation, guide.model_dump(), {"status": "blocked", "reason": memory})
                controller._trace("policy", "Floor goal blocked", {"floor_target_id": floor_target_id, "reason": memory})
                delta = [floor_target["center_m"][axis] - observation.odometry_m_rad[axis] for axis in (0, 1)]
                bearing = math.atan2(delta[1], delta[0]) - observation.odometry_m_rad[2]
                guide = guide.model_copy(update={"action": "look", "yaw_rad": max(-1.2, min(1.2, math.atan2(math.sin(bearing), math.cos(bearing)))),
                    "pitch_rad": .3, "reason": "Inspect the blocked approach toward the retained floor target"})
            else:
                controller._trace("policy", "Floor goal approach", {"floor_target_id": floor_target_id, "target_m": choice["target_m"], "path_length_m": choice["path_length_m"]})
                guide = guide.model_copy(update={"action": "navigate", "candidate_id": choice["id"]})
        recovery = False
        progress = controller.navigation_memory.progress()
        if settings.adaptive_navigation and proposal is None and historical_original is None and progress["recovery_needed"] and guide.action not in {"wait", "wait_until", "finish", "inspect_arrival"}:
            replacement = recovery_choice(controller.navigation_memory, observation, candidates, floor_target)
            controller._trace("policy", "No-progress recovery", {"proposed": guide.model_dump(), "progress": progress,
                "replacement": replacement.model_dump() if replacement else None})
            if replacement is None:
                controller.state.update(phase="completed", message="No progress after bounded recovery; robot holding position")
                controller._set_outcome("unachievable", controller.state["message"])
                return
            guide = replacement
            recovery = True
            controller.navigation_memory.recovery_attempts += 1
        if guide.destination_candidate_id is not None:
            reference = next((candidate for candidate in candidates if candidate["id"] == guide.destination_candidate_id), None)
            if reference and reference["floor_rgb"]:
                destination_floor = reference["floor_rgb"]
                parking["destination_surface_match"] = matching_floor(parking["floor_rgb"], destination_floor)
        controller.state.update(message=guide.reason, phase="acting")
        if recovery:
            controller.navigation_reply(guide.reason, source="controller")
        else:
            controller.navigation_reply(guide.reason)
        controller._trace("policy", "Continuous goal selected", {**guide.model_dump(), "source": "controller" if recovery else "model"})
        if guide.action in {"finish", "inspect_arrival"}:
            home_state = await worker.home_state(compact=True) if worker.home_mission else {}
            room_workflow = home_state.get("room_workflow")
            if room_workflow:
                if guide.action == "finish" and room_workflow["status"] == "completed":
                    controller.state["phase"] = "completed"
                    controller._set_outcome("completed", room_workflow["reason"], "agent")
                    return
                last_execution = {"status": "rejected", "reason": "Room workflow is not complete; use fresh room evidence or report the limitation", "room_workflow": room_workflow}
                memory += " Physical arrival alone cannot complete the round trip. " + room_workflow["reason"]
                controller._trace("policy", "Room workflow completion rejected", last_execution)
                continue
            if (worker.latest.get("challenge") or {}).get("id") == "furniture_circuit":
                memory = "Circling is not a room-entry task. Identify the requested object and use circle; the independent circuit scorer ends the scenario after a full lap and rest."
                continue
            if floor_target_id is not None and guide.completion_mode == "parking":
                if floor_target is None:
                    finish_checked = False
                    memory += " The selected floor target expired. Reacquire its boundaries before confirming parking."
                    controller._trace("policy", "Floor parking verification", {"status": "target_expired"})
                    if record_inspection_progress(controller, task_progress, "target_expired"):
                        return
                    continue
                parking = await worker.parking_clearance(floor_target)
                controller._check_live(worker, settings)
                if not parking["inside_floor_region"] or not parking["parking_margin_ok"]:
                    finish_checked = False
                    memory += " Parking not confirmed: the full robot footprint plus uncertainty must fit in the observed target region with fresh clearance. Continue toward its center or inspect missing boundaries."
                    controller._trace("policy", "Floor parking verification", {"status": "not_contained", **parking})
                    if record_inspection_progress(controller, task_progress, "not_contained"):
                        return
                    continue
                if not finish_checked or guide.action == "inspect_arrival" or not guide.entry_confirmed:
                    await wait_stationary(controller, worker, settings, 1.)
                    finish_checked = True
                    memory += " Stationary floor-region check complete. Review fresh feedback; keep this pose if containment remains true."
                    controller._trace("policy", "Floor parking verification", {"status": "confirmation_required", **parking})
                    if record_inspection_progress(controller, task_progress, "confirmation_required"):
                        return
                    continue
                controller.state["phase"] = "completed"
                controller._set_outcome("completed", guide.reason, "agent")
                return
            if guide.completion_mode == "visual_inspection":
                if not inspection_view_checked or guide.action == "inspect_arrival" or not guide.entry_confirmed:
                    await wait_stationary(controller, worker, settings, 1.)
                    inspection_view_checked = True
                    memory = guide.memory + " Review fresh feedback after the stationary one-second inspection. Confirm only if the correct target remains visible at the requested distance."
                    controller._trace("policy", "Visual inspection verification", {"status": "confirmation_required", "duration_s": 1.})
                    if record_inspection_progress(controller, task_progress, "visual_confirmation_required"):
                        return
                    continue
                controller._check_live(worker, settings)
                controller.state["phase"] = "completed"
                controller._set_outcome("completed", guide.reason, "agent")
                return
            inspection_view_checked = False
            if not finish_checked or guide.action == "inspect_arrival" or not guide.entry_confirmed:
                inspection_started = time.monotonic()
                panorama = await rotate(controller, worker, settings, stop_revision, 2 * math.pi, panoramic=True, pitch=.85)
                controller._trace("policy", "Local action timing", {"action": "arrival_inspection", "wall_s": time.monotonic() - inspection_started})
                finish_checked = True
                memory = guide.memory + " Completion NOT accepted yet. Review the fresh downward perimeter views for the doorway/floor boundary beside the base. Move at least half a meter farther into the destination room if entry is incomplete."
                controller._trace("policy", "Room entry verification", {"status": "inspection_required", "reason": memory})
                if record_inspection_progress(controller, task_progress, "room_confirmation_required"):
                    return
                continue
            if not parking["destination_surface_match"]:
                memory = guide.memory + " Arrival rejected: current floor does not match the recorded destination surface, or no reference is selected. Identify a destination-floor candidate using visible room fixtures, then navigate onto that surface."
                controller._trace("policy", "Room entry verification", {"status": "destination_surface_required", **parking})
                if record_inspection_progress(controller, task_progress, "destination_surface_required"):
                    return
                continue
            if not parking["parking_margin_ok"]:
                memory = guide.memory + " Finish rejected: insufficient observed parking margin. Select a more central clear floor point in the same destination room."
                controller._trace("policy", "Room entry verification", {"status": "parking_margin_required", **parking})
                if record_inspection_progress(controller, task_progress, "parking_margin_required"):
                    return
                continue
            await settle(worker, settings)
            controller._check_live(worker, settings)
            controller.state["phase"] = "completed"
            controller._set_outcome("completed", guide.reason, "agent")
            return
        result = {"status": "ok"}
        inspection_view_checked = False
        frames = []
        choice = None
        action_started = time.monotonic()
        try:
            if settings.navigation_backend == "nav2" and guide.action in {"navigate", "explore"}:
                choice = next((item for item in candidates if item["id"] == guide.candidate_id), None)
                if choice is None:
                    raise ValueError("Select a current observed destination for Nav2")
                finish_checked = False
                await worker.start_nav2_target(sensor, choice["target_m"], stop_revision, task_revision)
                while worker.ros_navigation.active:
                    controller._check_live(worker, settings)
                    await asyncio.sleep(.05)
                controller._check_live(worker, settings)
                result = {**worker.ros_navigation.state(), "mission_success_verified": False}
                if result["status"] == "arrived":
                    await settle(worker, settings)
            elif settings.navigation_backend == "nav2" and guide.action in {"compose", "circle", "continue"}:
                raise ValueError("This action requires the Built-in backup; Nav2 accepts bounded observed destinations")
            elif guide.action == "compose":
                if floor_target is not None:
                    raise ValueError("Use approach_target for retained marked floor destinations")
                finish_checked = False
                result = await execute_motion_skills(controller, worker, settings, sensor, guide, candidates, stop_revision, task_revision)
                inspection_view_checked = result.get("inspection_ready", False)
            elif guide.action == "circle":
                if proposal is not None:
                    raise ValueError("Wait for the current route before identifying a circuit target")
                result = await circle_observed_object(controller, worker, settings, sensor, guide, stop_revision)
            elif guide.action in {"navigate", "explore"} or (rolling and guide.action == "continue"):
                finish_checked = False
                choice = next((item for item in candidates if item["id"] == guide.candidate_id), None) if guide.action != "continue" else None
                if choice is None and guide.action != "continue":
                    raise ValueError("Candidate is unavailable; inspect another direction")
                if guide.action == "explore" and (floor_target is not None or guide.destination_candidate_id is not None):
                    raise ValueError("Use a bounded destination approach for selected floor or room targets")
                if rolling:
                    await worker.review_exploration(proposal, guide.candidate_id if guide.action == "explore" else None)
                    controller._trace("policy", "Rolling exploration review", worker.continuous.state())
                elif proposal is not None:
                    await worker.retarget_continuous(proposal, guide.candidate_id)
                else:
                    await worker.start_continuous(ContinuousTarget(run_id=settings.run_id, episode_epoch=settings.episode_epoch,
                        spatial_sequence=sensor.sequence, pixel=choice["pixel"] or [.5, .5]), selected_sensor=sensor,
                        selected_target=choice["target_m"] if choice["pixel"] is None else None, exploration=guide.action == "explore")
                while worker.continuous.active:
                    controller._check_live(worker, settings)
                    if worker.continuous.exploration is not None and worker.continuous.updates > 2:
                        break
                    if (settings.continuous_handoff and worker.continuous.handoffs < 2 and worker.continuous.updates > 2
                            and worker.continuous.distance_m > .35 and worker.navigation.travel < 1.3):
                        break
                    await asyncio.sleep(.05)
                result = worker.continuous.state()
                controller._check_live(worker, settings)
                if result["status"] == "arrived":
                    await settle(worker, settings)
            elif guide.action == "wait":
                await wait_stationary(controller, worker, settings, guide.duration_s)
                result = {"status": "ok", "action": "wait", "duration_s": guide.duration_s}
            elif guide.action == "wait_until":
                if guide.wait_condition is None:
                    raise ValueError("Select an explicit permitted sensor wait condition")
                result = await wait_for_sensor(controller, worker, settings, guide.wait_condition, guide.wait_timeout_s,
                    guide.wait_guard, floor_target, destination_floor)
            elif guide.action == "look":
                current, _ = await worker.feedback()
                reply = await worker.execute(Command(run_id=settings.run_id, episode_epoch=settings.episode_epoch,
                    observation_seq=current.seq, action_id=str(uuid4()), tool="set_head",
                    arguments={"yaw_rad": guide.yaw_rad, "pitch_rad": guide.pitch_rad, "duration_s": 1.}), assisted=False)
                if reply.status != "ok":
                    raise ValueError(reply.message)
            elif guide.action in {"turn", "scan"}:
                if settings.adaptive_navigation and (guide.action == "scan" or recovery):
                    frames = await adaptive_scan(controller, worker, settings, stop_revision)
                    panorama = nearby_panorama([*panorama, *frames][-6:], observation)
                else:
                    if guide.action == "scan":
                        await rotate(controller, worker, settings, stop_revision, 2 * math.pi, panoramic=True, pitch=.85)
                    frames = await rotate(controller, worker, settings, stop_revision,
                        2 * math.pi if guide.action == "scan" else guide.turn_rad, panoramic=guide.action == "scan")
                    if frames:
                        panorama = frames
                    await worker.scan_continuous(ContinuousScan(run_id=settings.run_id, episode_epoch=settings.episode_epoch))
        except ValueError as error:
            result = {"status": "blocked", "reason": str(error)}
            if proposal is not None:
                if rolling:
                    try:
                        await worker.review_exploration(proposal, ending=True)
                    except ValueError:
                        pass
                while worker.continuous and worker.continuous.active:
                    controller._check_live(worker, settings)
                    await asyncio.sleep(.05)
        last_execution = result
        controller._trace("policy", "Continuous execution feedback", result)
        controller._trace("policy", "Local action timing", {"action": guide.action, "recovery": recovery,
            "wall_s": time.monotonic() - action_started, "status": result["status"]})
        after, _ = await worker.feedback()
        if guide.action not in {"turn", "scan"}:
            decision = guide.model_dump()
            decision["source"] = "controller" if recovery else "model"
            if guide.action in {"navigate", "explore"} and choice is not None:
                decision["target_m"] = choice["target_m"]
            controller.navigation_memory.remember(observation, after, decision, result)
        elif settings.adaptive_navigation and (guide.action == "scan" or recovery) and not frames:
            controller.navigation_memory.remember(observation, after, {**guide.model_dump(), "source": "controller" if recovery else "model"}, result)
        if recovery:
            controller.navigation_memory.stagnant_actions = 0
        history.extend([{"role": "assistant", "content": [{"type": "output_text", "text": json.dumps(guide.model_dump())}]},
            {"role": "user", "content": [{"type": "input_text", "text": json.dumps(result)}]}])
    controller.state.update(phase="completed", message="Continuous supervision turn limit reached")
    controller._set_outcome("limited", controller.state["message"])