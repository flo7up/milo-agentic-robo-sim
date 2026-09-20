"""Measured place/passage history. This is exploration evidence, never a motion plan."""
from collections import deque
import math

from pydantic import Field, FiniteFloat, model_validator

from backend.contracts import StrictModel


class PassagePlace(StrictModel):
    id: str = Field(pattern=r"^P[0-9]+$", max_length=12)
    position_m: list[FiniteFloat] = Field(min_length=2, max_length=2)
    visits: int = Field(default=1, ge=1, le=1000000)


class TraversedPassage(StrictModel):
    start: str
    end: str
    traversals: int = Field(default=1, ge=1, le=1000000)
    distance_m: FiniteFloat = Field(ge=0)


class PassageAttempt(StrictModel):
    id: str = Field(max_length=24)
    origin: str
    sector: int = Field(ge=0, lt=8)
    attempts: int = Field(default=0, ge=0, le=1000000)
    no_progress: int = Field(default=0, ge=0, le=1000000)
    last_result: str = Field(default="untried", pattern=r"^(untried|reached|blocked)$")
    new_area_m2: FiniteFloat = Field(default=0, ge=0)


class PassageDocument(StrictModel):
    version: int = Field(default=1, ge=1, le=1)
    frame_revision: str = Field(default="1", min_length=1, max_length=80)
    places: list[PassagePlace] = Field(default_factory=list, max_length=512)
    passages: list[TraversedPassage] = Field(default_factory=list, max_length=2048)
    branches: list[PassageAttempt] = Field(default_factory=list, max_length=4096)
    loops: int = Field(default=0, ge=0)
    saturated: bool = False

    @model_validator(mode="after")
    def references(self):
        ids = {place.id for place in self.places}
        if len(ids) != len(self.places) or any(place.id != f"P{index+1}" for index, place in enumerate(self.places)):
            raise ValueError("Passage place identities must be unique and ordered")
        keys = {(edge.start, edge.end) for edge in self.passages}
        if len(keys) != len(self.passages) or any(edge.start not in ids or edge.end not in ids or edge.start == edge.end for edge in self.passages):
            raise ValueError("Passages require distinct recorded places")
        if len({branch.id for branch in self.branches}) != len(self.branches) or any(
                branch.origin not in ids or branch.id != f"{branch.origin}:{branch.sector}" for branch in self.branches):
            raise ValueError("Branches must reference a recorded place and direction")
        return self


class PassageMemory:
    def __init__(self, document=None):
        self.data = PassageDocument.model_validate(document or {})
        self.places = {place.id: place for place in self.data.places}
        self.edges = {(edge.start, edge.end): edge for edge in self.data.passages}
        self.branches = {branch.id: branch for branch in self.data.branches}
        self.offers = {}
        self.recent = deque(maxlen=32)
        self.loop_active = False
        self.break_tracking()

    def break_tracking(self):
        self.current = None
        self.previous = None
        self.previous_at = None
        self.segment_distance = 0.
        self.recent.clear()
        self.loop_active = False

    def document(self):
        return self.data.model_dump(mode="json")

    def bind_frame(self, revision):
        if self.data.frame_revision != revision:
            self.__init__({"frame_revision": revision})

    def nearest(self, point, clear, radius=.65):
        candidates = sorted(self.places.values(), key=lambda place: math.dist(place.position_m, point))
        return next((place for place in candidates if math.dist(place.position_m, point) <= radius
            and clear(place.position_m, point)), None)

    def observe(self, pose, timestamp, clear):
        if len(pose) < 2 or not all(math.isfinite(value) for value in [*pose, timestamp]):
            self.break_tracking()
            return
        point = list(pose[:2])
        if not clear(point, point):
            self.break_tracking()
            return
        if self.previous is not None:
            step = math.dist(self.previous, point)
            age = timestamp-self.previous_at
            if age < 0 or (age > 2. and step > .05) or step > .5 or not clear(self.previous, point):
                self.break_tracking()
            else:
                self.segment_distance += step
        self.previous, self.previous_at = point, timestamp
        previous_place = self.places.get(self.current)
        if previous_place and math.dist(previous_place.position_m, point) < .85 and clear(previous_place.position_m, point):
            return  # Dwell, rotation and sensor frequency are not repeat visits.
        place = self.nearest(point, clear)
        novel = place is None
        if novel:
            if len(self.places) >= 512:
                self.data.saturated = True
                self.break_tracking()
                return
            place = PassagePlace(id=f"P{len(self.places)+1}", position_m=point)
            self.places[place.id] = place
            self.data.places.append(place)
            self.loop_active = False
        elif place.id == self.current:
            return
        else:
            place.visits = min(1000000, place.visits+1)
        if previous_place and self.segment_distance > .1:
            key = (previous_place.id, place.id)
            edge = self.edges.get(key)
            if edge:
                edge.traversals = min(1000000, edge.traversals+1)
                edge.distance_m += self.segment_distance
            elif len(self.edges) < 2048:
                edge = TraversedPassage(start=key[0], end=key[1], distance_m=self.segment_distance)
                self.edges[key] = edge
                self.data.passages.append(edge)
            else:
                self.data.saturated = True
            recent = list(self.recent)
            cycle = place.id in recent[:-2] or (edge is not None and edge.traversals >= 2)
            if not novel and cycle and not self.loop_active:
                self.data.loops += 1
                self.loop_active = True
        self.current = place.id
        self.segment_distance = 0.
        self.recent.append(place.id)

    def annotate(self, candidates, pose, clear, known_near, *, recovery=False):
        origin = self.nearest(pose[:2], clear, radius=.9)
        result = []
        for candidate in candidates:
            item = dict(candidate)
            target = item["position_m"]
            visited = self.nearest(target, clear, radius=.8)
            item["destination_visits"] = visited.visits if visited else 0
            branch = None
            if origin:
                angle = math.atan2(target[1]-origin.position_m[1], target[0]-origin.position_m[0])
                sector = int(math.floor((angle+math.pi/8)/(math.pi/4))) % 8
                identity = f"{origin.id}:{sector}"
                branch = self.branches.get(identity)
                if branch is None and len(self.branches) < 4096:
                    branch = PassageAttempt(id=identity, origin=origin.id, sector=sector)
                    self.branches[identity] = branch
                    self.data.branches.append(branch)
                if branch:
                    item.update(passage_id=identity, passage_attempts=branch.attempts,
                        passage_no_progress=branch.no_progress, passage_last_result=branch.last_result)
                    self.offers[item["frontier_id"]] = (identity, list(target), known_near(target))
            if recovery:
                penalty = 2.*min(item["destination_visits"], 3)
                if branch:
                    penalty += min(6., 2.*branch.no_progress)
                item["exploration_score"] = item["search_score"]-penalty
                item["selection_basis"] = "new_observed_area_after_measured_loop"
            result.append(item)
        # Offers are ephemeral evidence; persistent branch counts do not depend on a frontier's changing grid ID.
        self.offers = dict(list(self.offers.items())[-128:])
        return sorted(result, key=lambda item: -item["exploration_score"]) if recovery else result

    def finish(self, frontier_id, known_near, result="reached"):
        offer = self.offers.pop(frontier_id, None)
        if not offer:
            return
        identity, target, before = offer
        branch = self.branches[identity]
        branch.attempts = min(1000000, branch.attempts+1)
        branch.new_area_m2 = max(0., known_near(target)-before)
        branch.no_progress = min(1000000, branch.no_progress+1) if branch.new_area_m2 < .25 else 0
        branch.last_result = result

    def summary(self, map_id):
        place = self.places.get(self.current)
        return {"source": "measured_localized_movement", "frame": "map", "map_id": map_id,
            "current_place_id": self.current, "current_place_visits": place.visits if place else 0,
            "places_reached": len(self.places), "passages_travelled": len(self.edges),
            "loop_detected": self.loop_active, "loops_detected": self.data.loops,
            "recent_place_ids": list(self.recent)[-10:], "history_saturated": self.data.saturated,
            "branches_here": [branch.model_dump() for branch in self.branches.values() if branch.origin == self.current],
            "motion_authorized": False,
            "caution": "Visits are not fully explored branches. Reuse known passages to reach unseen areas; revalidate every route. No-progress attempts do not prove a dead end."}
