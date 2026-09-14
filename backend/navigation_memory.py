from collections import deque
import math
import time


class TaskProgress:
    inspection_limit = 3
    transition_limit = 8

    def __init__(self):
        self.episode = None
        self.current = None
        self.sensor_changes = deque(maxlen=4)
        self.change_revision = 0
        self.subgoal = None
        self.transitions = []
        self.used_evidence = set()
        self.inspection_anchor = None
        self.inspection_attempts = 0
        self.last_inspection = None

    def observe(self, observation):
        episode = observation.run_id, observation.episode_epoch
        if self.episode != episode:
            self.__init__()
            self.episode = episode
        current = {"frame_ref": observation.frame_ref, "simulated_time_s": observation.simulated_time_s,
            "odometry_m_rad": list(observation.odometry_m_rad), "head_rad": list(observation.head_rad),
            "battery": observation.battery.model_dump() if observation.battery is not None else None}
        previous = self.current
        if previous and previous["battery"] is not None and current["battery"] is not None:
            before, after = previous["battery"], current["battery"]
            delta = after["charge_pct"] - before["charge_pct"]
            if abs(delta) >= 5 or any(before[field] != after[field] for field in ("low", "charging")):
                self.change_revision += 1
                self.sensor_changes.append({"id": f"sensor-change-{self.change_revision}",
                    "from_frame": previous["frame_ref"], "to_frame": current["frame_ref"],
                    "simulated_elapsed_s": round(current["simulated_time_s"] - previous["simulated_time_s"], 3),
                    "battery_before": before, "battery_after": after, "charge_delta_pct": round(delta, 3)})
        anchor = self.inspection_anchor
        if anchor:
            yaw_delta = current["odometry_m_rad"][2] - anchor["odometry_m_rad"][2]
            changed_view = (math.dist(current["odometry_m_rad"][:2], anchor["odometry_m_rad"][:2]) >= .15
                or abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta))) >= .25
                or max(abs(current["head_rad"][axis] - anchor["head_rad"][axis]) for axis in (0, 1)) >= .2)
            if changed_view:
                self.inspection_anchor = None
                self.inspection_attempts = 0
                self.last_inspection = None
        self.current = current
        return self.summary()

    def record_inspection(self, status):
        if self.current is None:
            raise ValueError("Inspection requires a current paired observation")
        if self.inspection_anchor is None:
            self.inspection_anchor = self.current
        self.inspection_attempts += 1
        self.last_inspection = {"status": status, "frame_ref": self.current["frame_ref"]}
        return self.summary()

    def advance(self, subgoal, *, event_id=None, frame_ref=None):
        if (event_id is None) == (frame_ref is None):
            raise ValueError("Cite one retained sensor-change ID or the current paired frame")
        if event_id is not None:
            if not any(change["id"] == event_id for change in self.sensor_changes):
                raise ValueError("Sensor-change evidence is unavailable in this session")
            evidence = "event:" + event_id
        else:
            if self.current is None or frame_ref != self.current["frame_ref"]:
                raise ValueError("Visual subgoal evidence must use the current paired frame")
            evidence = "frame:" + frame_ref
        if evidence in self.used_evidence or subgoal == self.subgoal or len(self.transitions) >= self.transition_limit:
            raise ValueError("Repeated or exhausted subgoal transition; gather different evidence or stop")
        transition = {"from_subgoal": self.subgoal, "to_subgoal": subgoal, "evidence": evidence,
            "source": "model_assessment_not_verified", "motion_authorized": False}
        self.transitions.append(transition)
        self.used_evidence.add(evidence)
        self.subgoal = subgoal
        self.inspection_anchor = None
        self.inspection_attempts = 0
        self.last_inspection = None
        return transition

    def summary(self):
        return {"current_subgoal": self.subgoal, "current_frame_ref": self.current["frame_ref"] if self.current else None,
            "sensor_changes": list(self.sensor_changes),
            "recent_transitions": self.transitions[-4:], "transitions_remaining": self.transition_limit - len(self.transitions),
            "inspection_attempts_here": self.inspection_attempts, "inspection_limit": self.inspection_limit,
            "inspection_exhausted": self.inspection_attempts >= self.inspection_limit,
            "last_inspection": self.last_inspection,
            "note": "Sensor changes are measured evidence, not task success. Subgoal transitions are model assessments; no motion or completion is authorized."}


class NavigationMemory:
    def __init__(self):
        self.episode = None
        self.positions = deque(maxlen=64)
        self.views = deque(maxlen=128)
        self.actions = deque(maxlen=48)
        self.instructions = deque(maxlen=8)
        self.assessments = deque(maxlen=4)
        self.revision = 0
        self.last_observation = None
        self.stagnant_actions = 0
        self.recovery_attempts = 0
        self.scan_direction = 1
        self.routes = deque(maxlen=32)
        self.reached_positions = deque(maxlen=64)

    def bind(self, run_id, epoch, reset=False):
        if reset or self.episode != (run_id, epoch):
            self.__init__()
            self.episode = (run_id, epoch)

    def observe(self, observation):
        self.bind(observation.run_id, observation.episode_epoch)
        self.last_observation = observation
        position = list(observation.odometry_m_rad[:2])
        if not self.positions or math.dist(position, self.positions[-1]) >= .15:
            self.positions.append(position)
        heading = observation.odometry_m_rad[2] + observation.head_rad[0]
        heading = math.atan2(math.sin(heading), math.cos(heading))
        sector = round(heading / (math.pi / 12)) % 24
        view = {"position_m": position, "heading_sector": sector, "pitch_rad": round(observation.head_rad[1], 1),
            "wall_timestamp": getattr(observation, "wall_timestamp", time.time())}
        if not self.views or (view["heading_sector"] != self.views[-1]["heading_sector"]
                or view["pitch_rad"] != self.views[-1]["pitch_rad"] or math.dist(position, self.views[-1]["position_m"]) > .15):
            self.views.append(view)
            self.revision += 1
        elif view["wall_timestamp"] > self.views[-1]["wall_timestamp"]:
            self.views[-1] = view
        return self.summary(observation)

    def remember(self, before, after, decision, result):
        if (before.run_id, before.episode_epoch) != (after.run_id, after.episode_epoch):
            raise ValueError("Cannot remember motion across episodes")
        self.bind(after.run_id, after.episode_epoch)
        if not self.reached_positions:
            self.reached_positions.append(list(before.odometry_m_rad[:2]))
        novel = min(math.dist(after.odometry_m_rad[:2], point) for point in self.reached_positions) >= .3
        if novel:
            self.reached_positions.append(list(after.odometry_m_rad[:2]))
            self.stagnant_actions = 0
            self.recovery_attempts = 0
        elif decision.get("action", decision.get("motion")) != "operator_redirect":
            self.stagnant_actions += 1
        travelled = math.dist(before.odometry_m_rad[:2], after.odometry_m_rad[:2])
        if travelled >= .2 and result.get("status") in {"ok", "arrived"}:
            self.routes.append({"start_m": list(before.odometry_m_rad[:2]), "end_m": list(after.odometry_m_rad[:2]),
                "wall_timestamp": getattr(after, "wall_timestamp", time.time()), "source": "executed_odometry",
                "clearance": "must_revalidate"})
        self.observe(after)
        if decision.get("memory") or decision.get("reason"):
            self.assessments.append({"source": "controller_recovery" if decision.get("source") == "controller" else "model_assessment_not_verified", "position_m": list(after.odometry_m_rad[:2]),
                "text": str(decision.get("memory") or decision["reason"])[:1400]})
        parameters = {key: value for key, value in decision.items() if key not in {"reason", "memory", "entry_confirmed", "destination_candidate_id"}}
        self.actions.append({"action": decision.get("action", decision.get("motion", "unknown")),
            "parameters": parameters, "start_odometry_m_rad": list(before.odometry_m_rad),
            "end_odometry_m_rad": list(after.odometry_m_rad),
            "distance_m": round(math.dist(before.odometry_m_rad[:2], after.odometry_m_rad[:2]), 3),
            "turn_rad": round(after.odometry_m_rad[2] - before.odometry_m_rad[2], 3),
            "status": result.get("status", "unknown"), "reason": str(result.get("reason", result.get("message", "")))[:240],
            "simulated_time_s": after.simulated_time_s})
        self.revision += 1

    def recent_sectors(self, observation, now=None, pitch=None):
        now = time.time() if now is None else now
        return {view["heading_sector"] for view in self.views
            if math.dist(view["position_m"], observation.odometry_m_rad[:2]) < .25
            and 0 <= now - view["wall_timestamp"] <= 90
            and (pitch is None or abs(view["pitch_rad"] - pitch) <= .2)}

    def next_scan_turn(self, observation, now=None):
        inspected = self.recent_sectors(observation, now, pitch=.15)
        heading = round(observation.odometry_m_rad[2] / (math.pi / 12)) % 24
        for quarter in (1, 2, 3):
            target = (heading + self.scan_direction * 6 * quarter) % 24
            if not any(min((target - sector) % 24, (sector - target) % 24) <= 2 for sector in inspected):
                return self.scan_direction * min(quarter, 2) * math.pi / 2
        return None

    def progress(self):
        return {"stagnant_actions": self.stagnant_actions, "recovery_needed": self.stagnant_actions >= 4,
            "recovery_attempts": self.recovery_attempts, "recovery_limit": 2}

    def sighting_routes(self, sightings):
        now = time.time()
        return [{**route,
            "from_sighting_ids": [item["id"] for item in sightings if math.dist(item["position_m"], route["start_m"]) < .8],
            "to_sighting_ids": [item["id"] for item in sightings if math.dist(item["position_m"], route["end_m"]) < .8]}
            for route in list(self.routes)[-12:] if 0 <= now - route["wall_timestamp"] <= 1800]

    def summary(self, observation=None):
        position = observation.odometry_m_rad[:2] if observation else (self.positions[-1] if self.positions else [0., 0.])
        nearby_actions = [action for action in self.actions if math.dist(action["start_odometry_m_rad"][:2], position) < .25]
        return {"revision": self.revision, "frame": "wheel_odometry", "visited_positions_m": list(self.positions)[-24:],
            "progress": self.progress(), "observed_routes": self.sighting_routes([]),
            "recent_heading_sectors_here": sorted(self.recent_sectors(observation)) if observation else [],
            "inspected_heading_sectors_here": sorted({view["heading_sector"] for view in self.views if math.dist(view["position_m"], position) < .25}),
            "sector_size_deg": 15, "recent_actions": list(self.actions)[-8:],
            "blocked_actions_here": [action for action in nearby_actions if action["status"] in {"blocked", "error", "cancelled"}][-4:],
            "rotation_without_translation_rad": round(sum(abs(action["turn_rad"]) for action in nearby_actions if action["distance_m"] < .08), 2),
            "operator_instructions": list(self.instructions)[-4:],
            "recent_model_assessments": list(self.assessments),
            "caution": "Measured history, not a room map or success proof. Repeating a turn can be necessary; prefer an uninspected direction or another route when observations have not changed."}


MEMORY_GUIDANCE = """Use measured_episode_memory to remember actions that actually executed.
It survives operator redirection within this episode; reset starts a new memory. Do not
confuse a planned route with movement completed. Read blocked_actions_here and inspected
headings before repeating a turn/scan. Repeating an identical failed route without changed
observations is unlikely to help. After a full rotation with little translation, prefer
an unvisited reachable direction, inspect a specific missing view, or report blocked.
Backtracking and repeating an observation can be necessary; do not forbid them blindly.
Earlier operator instructions are historical context; the latest user goal has priority.
recent_heading_sectors_here expires after 90 seconds and only describes nearby views.
In adaptive exploration, scan turns at most 180 degrees toward missing coverage. Choose
an existing useful candidate before requesting more scanning. A bounded recovery may
replace repetitive actions; its trace identifies the controller intervention.
sighting_route_connections associates unverified sightings with measured route endpoints.
These are historical transitions, not straight-line routes, doorway boundaries or current
clearance. Use them to choose current reachable candidates and reobserve the destination.
"""