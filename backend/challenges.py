from typing import Annotated, Literal

from pydantic import Field

from backend.contracts import BatterySensor, StrictModel


ChallengeId = Literal["bench", "park", "tidy", "sort", "recharge", "apartment", "kitchen_bathroom", "clinic_delivery",
                      "warehouse", "inspection", "workshop"]
Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]


class ChallengeLoad(StrictModel):
    challenge_id: ChallengeId


class Objective(StrictModel):
    label: str
    body: str
    center: Vector3
    size: Annotated[list[Annotated[float, Field(gt=0)]], Field(min_length=2, max_length=2)]
    color: Annotated[list[float], Field(min_length=4, max_length=4)]
    require_lift: bool = True
    visible_zone: bool = True
    transit: bool = False
    dwell_s: float = Field(default=0, ge=0, le=10)


class Challenge(StrictModel):
    id: ChallengeId
    title: str
    skill: str
    category: Literal["Navigation", "Perception", "Manipulation"] = "Navigation"
    difficulty: Literal["Foundation", "Advanced"] = "Foundation"
    goal: str
    objects: list[dict]
    objectives: list[Objective]
    suggested_turn_limit: int = Field(default=30, ge=1, le=200)
    initial_head_pitch: float = Field(default=0, ge=-.7, le=1.15)
    initial_xy: Annotated[list[float], Field(min_length=2, max_length=2)] = Field(default_factory=lambda: [0, 0])
    search_target: str | None = None
    ordered_objectives: bool = False
    floor_size_m: Annotated[list[Annotated[float, Field(ge=4, le=20)]], Field(min_length=2, max_length=2)] = Field(default_factory=lambda: [6, 6])

    def public(self):
        return {"id": self.id, "title": self.title, "skill": self.skill, "goal": self.goal,
            "objectives": [objective.label for objective in self.objectives], "suggested_turn_limit": self.suggested_turn_limit,
            "category": self.category, "difficulty": self.difficulty}

    def scene(self):
        objects = [
            {"name": "floor", "size": [*self.floor_size_m, .1], "position": [0, 0, -.05], "color": [.77, .80, .79, 1]},
            *([] if any(item["name"] == "back_wall" for item in self.objects) else [
                {"name": "back_wall", "size": [.1, 6, 1], "position": [2.5, 0, .5], "color": [.50, .57, .56, 1]}]),
            *self.model_dump()["objects"],
        ]
        zones = set()
        for index, objective in enumerate(self.objectives):
            if self.search_target or not objective.visible_zone:
                continue
            footprint = (tuple(objective.center), tuple(objective.size))
            if footprint in zones:
                continue
            zones.add(footprint)
            objects.append({"name": f"zone_{index}", "size": [*objective.size, .002],
                            "position": [*objective.center[:2], .001], "color": objective.color,
                            "marker": True})
        return objects


PRESETS = {
        "apartment": Challenge(
                id="apartment", title="Apartment Search", skill="Room exploration", category="Perception",
                suggested_turn_limit=100, initial_head_pitch=.15, search_target="yellow_target",
                goal="Find the yellow cube on a short pedestal in this small apartment. Explore the rooms through the open doorways; the red cube is not the target. Approach within 0.9 m of the yellow cube and stop with it clearly visible in the head camera for one simulated second. Use wait while looking at it. You do not need to pick it up.",
                objects=[
                    {"name": "back_wall", "size": [.1, 5.2, 1.05], "position": [2.5, 0, .525], "color": [.50, .57, .56, 1]},
                        {"name": "west_wall", "size": [.1, 5.2, 1.05], "position": [-1.6, 0, .525], "color": [.70, .73, .75, 1]},
                        *[{"name": f"outer_{side}", "size": [4.1, .1, 1.05], "position": [.45, sign * 2.6, .525], "color": [.70, .73, .75, 1]}
                            for side, sign in (("south", -1), ("north", 1))],
                        {"name": "hall_partition", "size": [.1, 1.7, 1.05], "position": [.65, 0, .525], "color": [.82, .85, .86, 1]},
                        *[{"name": f"door_end_{side}", "size": [.1, .45, 1.05], "position": [.65, sign * 2.375, .525], "color": [.82, .85, .86, 1]}
                            for side, sign in (("south", -1), ("north", 1))],
                        {"name": "room_divider", "size": [1.85, .1, 1.05], "position": [1.575, 0, .525], "color": [.82, .85, .86, 1]},
                        {"name": "study_floor", "size": [1.7, 2.4, .002], "position": [1.55, 1.3, .001], "color": [.63, .75, .82, 1], "marker": True},
                        {"name": "kitchen_floor", "size": [1.7, 2.4, .002], "position": [1.55, -1.3, .001], "color": [.74, .81, .68, 1], "marker": True},
                        {"name": "sofa_seat", "size": [.55, 1.1, .28], "position": [-1.12, 1.35, .14], "color": [.26, .42, .50, 1]},
                        {"name": "sofa_back", "size": [.15, 1.1, .60], "position": [-1.4, 1.35, .30], "color": [.26, .42, .50, 1]},
                        {"name": "kitchen_counter", "size": [.35, .85, .42], "position": [2.22, -.65, .21], "color": [.54, .60, .55, 1]},
                        *[{"name": f"pedestal_{side}", "size": [.32, .32, .34], "position": [1.95, sign * 1.7, .17], "color": [.35, .39, .40, 1]}
                            for side, sign in (("south", -1), ("north", 1))],
                        {"name": "yellow_target", "size": [.18, .18, .18], "position": [1.95, 1.7, .43], "color": [.98, .78, .05, 1]},
                        {"name": "red_decoy", "size": [.18, .18, .18], "position": [1.95, -1.7, .43], "color": [.85, .12, .18, 1]},
                ],
                objectives=[Objective(label="Find and inspect the yellow cube", body="robot", center=[1.95, 1.7, 0],
                                                            size=[1.8, 1.8], color=[.98, .78, .05, 1], require_lift=False)]),
    "park": Challenge(
        id="park", title="Park in the Bay", skill="Navigation",
        goal="Drive to the green parking bay beyond the two yellow posts. Stop with the entire base and both wheels inside the green area.",
        objects=[{"name": f"gate_{side}", "size": [.10, .10, .25], "position": [.65, sign * .53, .125],
                  "color": [.97, .70, .13, 1]} for side, sign in (("left", 1), ("right", -1))],
        objectives=[Objective(label="Base and wheels parked in the green bay", body="robot", center=[1.15, 0, 0],
                              size=[.75, .82], color=[.21, .63, .36, 1], require_lift=False)]),
    "tidy": Challenge(
        id="tidy", title="Tidy the Cube", skill="Pick and place", category="Manipulation",
        goal="Pick up the red cube from the floor, lift it clear of the ground, and place it entirely inside the blue drop zone. Release it and let it settle on the floor; do not just push it into the zone.",
        objects=[{"name": "red_cube", "size": [.06, .06, .06], "position": [.36, .25, .032],
                  "color": [.86, .17, .27, 1], "mass": .08}],
        objectives=[Objective(label="Red cube lifted, released, and settled in the blue zone", body="red_cube",
                              center=[.32, .50, 0], size=[.18, .18], color=[.14, .49, .86, 1])]),
    "sort": Challenge(
        id="sort", title="Color Sort", skill="Two-object manipulation", category="Manipulation",
        goal="Tidy both cubes into their matching floor zones: the red cube in the red zone and the blue cube in the blue zone. Lift each cube clear of the ground, release it fully inside its zone, and let both settle. Pushing or holding a cube above a zone does not count.",
        objects=[{"name": f"{name}_cube", "size": [.06, .06, .06], "position": [.36, sign * .25, .032],
                  "color": color, "mass": .08}
                 for name, sign, color in (("red", 1, [.86, .17, .27, 1]), ("blue", -1, [.14, .49, .86, 1]))],
        objectives=[Objective(label=f"{name.title()} cube lifted and settled in its matching zone", body=f"{name}_cube",
                              center=[.32, sign * .50, 0], size=[.18, .18], color=color)
                    for name, sign, color in (("red", 1, [.86, .17, .27, 1]), ("blue", -1, [.14, .49, .86, 1]))]),
    "recharge": Challenge(
        id="recharge", title="Remember and Recharge", skill="Spatial memory and energy management",
        suggested_turn_limit=100, initial_head_pitch=.9,
        goal="Remember the cyan charging pad where you start and the landmarks around it. Navigate around the gray screen to the orange survey zone. Park fully inside the orange zone and wait for one simulated second to complete an energy-intensive scan. The scan consumes 60 percentage points of battery. When the battery sensor reports low, find your way back to the original cyan pad using your observations and remembered route. Park fully on the pad and use wait until the battery reaches at least 90%, then stop. There is no homing tool; reaching another zone or staying at the charger from the start does not count.",
        objects=[
            {"name": "screen", "size": [.12, .65, .85], "position": [.75, 0, .425], "color": [.38, .42, .43, 1]},
            {"name": "charger_beacon", "size": [.10, .10, .55], "position": [-.55, .60, .275], "color": [.08, .68, .73, 1]},
            {"name": "survey_beacon", "size": [.10, .10, .55], "position": [1.95, .60, .275], "color": [.96, .46, .10, 1]},
        ],
        objectives=[
            Objective(label="Leave the charger and complete the survey", body="robot", center=[1.45, 0, 0],
                      size=[.70, .80], color=[.96, .46, .10, 1], require_lift=False),
            Objective(label="Return to the original charger after the low-battery warning", body="robot", center=[0, 0, 0],
                      size=[.80, .80], color=[.08, .68, .73, 1], require_lift=False),
            Objective(label="Recharge to at least 90% while parked", body="robot", center=[0, 0, 0],
                      size=[.80, .80], color=[.08, .68, .73, 1], require_lift=False),
        ]),
}


PRESETS["kitchen_bathroom"] = Challenge(
        id="kitchen_bathroom", title="Kitchen to Bathroom", skill="Room recognition and navigation",
        suggested_turn_limit=100, initial_xy=[1.1, -1.5], initial_head_pitch=.12,
        goal="Look around and identify your current room from its visible fixtures. Find the bathroom by navigating through the open doorways, then stop with your entire base and wheels inside it. Describe the fixtures that identify the starting room and the bathroom when you arrive. Do not move or pick up furniture.",
        objects=[
                *[dict(item) for item in PRESETS["apartment"].objects if item["name"] in {
                        "back_wall", "west_wall", "outer_south", "outer_north", "hall_partition", "door_end_south", "door_end_north", "room_divider"}],
                {"name": "kitchen_floor", "size": [1.7, 2.4, .002], "position": [1.55, -1.3, .001], "color": [.87, .86, .81, 1], "marker": True},
                {"name": "bathroom_floor", "size": [1.7, 2.4, .002], "position": [1.55, 1.3, .001], "color": [.62, .83, .88, 1], "marker": True},
                {"name": "fridge_body", "size": [.42, .57, .98], "position": [2.2, -2.17, .49], "color": [.94, .96, .95, 1]},
                {"name": "fridge_freezer_door", "size": [.022, .53, .27], "position": [1.976, -2.17, .82], "color": [.82, .86, .87, 1]},
                {"name": "fridge_cooler_door", "size": [.022, .53, .60], "position": [1.976, -2.17, .365], "color": [.86, .90, .91, 1]},
                *[{"name": f"fridge_handle_{index}", "size": [.045, .025, height], "position": [1.95, -1.98, center], "color": [.25, .29, .31, 1]}
                    for index, height, center in ((0, .16, .81), (1, .28, .43))],
                {"name": "sink_cabinet", "size": [.47, .62, .47], "position": [2.17, -1.47, .235], "color": [.29, .48, .41, 1]},
                {"name": "kitchen_sink_basin", "size": [.33, .43, .015], "position": [2.13, -1.47, .475], "color": [.29, .39, .43, 1]},
                *[{"name": f"sink_counter_{index}", "size": size, "position": position, "color": [.80, .83, .82, 1]}
                    for index, size, position in ((0, [.055, .64, .045], [1.948, -1.47, .497]), (1, [.10, .64, .045], [2.385, -1.47, .497]),
                                                                             (2, [.38, .08, .045], [2.17, -1.19, .497]), (3, [.38, .08, .045], [2.17, -1.75, .497]))],
                {"name": "kitchen_faucet_stem", "shape": "cylinder", "size": [.026, .026, .18], "position": [2.34, -1.47, .59], "color": [.55, .61, .63, 1]},
                {"name": "kitchen_faucet_spout", "size": [.15, .026, .026], "position": [2.28, -1.47, .68], "color": [.55, .61, .63, 1]},
                {"name": "stove_body", "size": [.48, .59, .48], "position": [2.16, -.77, .24], "color": [.90, .91, .92, 1]},
                {"name": "stove_hob", "size": [.49, .60, .022], "position": [2.16, -.77, .49], "color": [.12, .15, .17, 1]},
                *[{"name": f"hob_burner_{index}", "shape": "cylinder", "size": [.13, .13, .012], "position": [center_x, center_y, .508], "color": [.43, .47, .49, 1]}
                    for index, (center_x, center_y) in enumerate(((2.04, -.92), (2.28, -.92), (2.04, -.62), (2.28, -.62)))],
                {"name": "oven_window", "size": [.022, .43, .27], "position": [1.908, -.77, .24], "color": [.12, .19, .22, 1]},
                {"name": "oven_handle", "size": [.047, .40, .026], "position": [1.889, -.77, .417], "color": [.52, .57, .59, 1]},
                *[{"name": f"stove_knob_{index}", "size": [.025, .042, .042], "position": [1.90, center_y, .455], "color": [.20, .23, .25, 1]}
                    for index, center_y in enumerate((-.95, -.83, -.71, -.59))],
                {"name": "kitchen_backsplash", "material": "tile", "size": [.018, 1.32, .34], "position": [2.438, -1.13, .69], "color": [.75, .83, .76, 1]},
                {"name": "bathroom_tile_wall", "material": "tile", "size": [.018, 2.45, .9], "position": [2.438, 1.28, .45], "color": [.72, .89, .92, 1]},
                {"name": "vanity_cabinet", "size": [.42, .54, .41], "position": [2.20, .49, .205], "color": [.32, .47, .56, 1]},
                {"name": "bathroom_basin", "size": [.35, .40, .018], "position": [2.15, .49, .419], "color": [.37, .63, .70, 1]},
                *[{"name": f"basin_rim_{index}", "size": size, "position": position, "color": [.96, .97, .96, 1]}
                    for index, size, position in ((0, [.05, .57, .07], [1.97, .49, .44]), (1, [.07, .57, .07], [2.405, .49, .44]),
                                                                             (2, [.40, .055, .07], [2.19, .23, .44]), (3, [.40, .055, .07], [2.19, .75, .44]))],
                {"name": "bathroom_faucet", "shape": "cylinder", "size": [.024, .024, .16], "position": [2.36, .49, .55], "color": [.56, .63, .65, 1]},
                {"name": "bathroom_tap_spout", "size": [.13, .025, .025], "position": [2.31, .49, .63], "color": [.56, .63, .65, 1]},
                {"name": "mirror_frame", "size": [.035, .54, .44], "position": [2.405, .49, .82], "color": [.30, .35, .39, 1]},
                {"name": "mirror_glass", "size": [.012, .48, .38], "position": [2.38, .49, .82], "color": [.71, .84, .89, 1]},
                {"name": "toilet_foot", "size": [.24, .24, .19], "position": [2.16, 1.31, .095], "color": [.94, .95, .93, 1]},
                {"name": "toilet_bowl", "shape": "cylinder", "size": [.38, .38, .20], "position": [2.12, 1.31, .24], "color": [.94, .95, .93, 1]},
                {"name": "toilet_seat", "shape": "cylinder", "size": [.40, .40, .028], "position": [2.12, 1.31, .355], "color": [.98, .98, .97, 1]},
                {"name": "toilet_opening", "shape": "cylinder", "size": [.25, .25, .008], "position": [2.12, 1.31, .374], "color": [.31, .50, .57, 1]},
                {"name": "toilet_cistern", "size": [.17, .42, .57], "position": [2.34, 1.31, .285], "color": [.93, .95, .94, 1]},
                {"name": "toilet_lid", "size": [.19, .44, .035], "position": [2.34, 1.31, .588], "color": [.99, .99, .98, 1]},
                {"name": "toilet_flush", "size": [.014, .065, .023], "position": [2.246, 1.18, .525], "color": [.42, .48, .51, 1]},
                {"name": "tub_base", "size": [1.39, .60, .14], "position": [1.66, 2.20, .07], "color": [.91, .95, .96, 1]},
                {"name": "tub_interior", "size": [1.20, .40, .015], "position": [1.66, 2.20, .15], "color": [.45, .70, .77, 1]},
                *[{"name": f"tub_side_{index}", "size": size, "position": position, "color": [.96, .98, .97, 1]}
                    for index, size, position in ((0, [1.4, .08, .33], [1.66, 1.94, .165]), (1, [1.4, .08, .33], [1.66, 2.46, .165]),
                                                                             (2, [.08, .48, .33], [.999, 2.20, .165]), (3, [.08, .48, .33], [2.32, 2.20, .165]))],
                {"name": "bathroom_towel", "material": "fabric", "size": [.025, .25, .34], "position": [2.38, .96, .62], "color": [.20, .48, .61, 1]},
        ],
        objectives=[Objective(label="Base and wheels stopped inside the bathroom", body="robot", center=[1.40, 1.15, 0],
                                                    size=[1.20, 1.48], color=[.62, .83, .88, 1], require_lift=False, visible_zone=False)],
)


PRESETS["clinic_delivery"] = Challenge(
    id="clinic_delivery", title="Clinic Supply Delivery", skill="Visual pathfinding and route recovery", difficulty="Advanced",
    suggested_turn_limit=160, initial_xy=[-2.30, -1.70], initial_head_pitch=.08, ordered_objectives=True,
    goal="Navigate from clinic reception to the green treatment-room delivery bay. The central hallway is closed by a maintenance barrier, so inspect the surroundings and find a detour through diagnostics. Pass the blue hand-sanitizer station, continue through the diagnostics doorway, then stop with the base and wheels fully inside the green bay. Blue wall panels identify the public route; green panels identify treatment. No carried supplies or object manipulation is required. Do not move furniture or cross barriers.",
    objects=[
        {"name": "back_wall", "size": [.10, 5.8, 1.2], "position": [2.90, 0, .60], "color": [.72, .76, .77, 1]},
        {"name": "west_wall", "size": [.10, 5.8, 1.2], "position": [-2.90, 0, .60], "color": [.72, .76, .77, 1]},
        {"name": "south_wall", "size": [5.8, .10, 1.2], "position": [0, -2.90, .60], "color": [.72, .76, .77, 1]},
        {"name": "north_wall", "size": [5.8, .10, 1.2], "position": [0, 2.90, .60], "color": [.72, .76, .77, 1]},
                {"name": "entry_left", "size": [.50, .10, 1.2], "position": [-2.65, -2.15, .60], "color": [.84, .87, .87, 1]},
        {"name": "entry_right", "size": [1.05, .10, 1.2], "position": [-.25, -2.15, .60], "color": [.84, .87, .87, 1]},
                {"name": "reception_partition", "size": [.10, 1.15, 1.2], "position": [-1.05, -1.35, .60], "color": [.84, .87, .87, 1]},
                {"name": "reception_desk", "size": [.48, 1.05, .62], "position": [-1.35, -1.40, .31], "color": [.35, .48, .50, 1], "material": "wood"},
                {"name": "desk_counter", "size": [.62, 1.12, .08], "position": [-1.35, -1.40, .66], "color": [.72, .67, .57, 1], "material": "wood"},
                {"name": "desk_monitor", "size": [.08, .34, .28], "position": [-1.66, -1.40, .84], "color": [.10, .15, .17, 1]},
                {"name": "waiting_bench_1", "size": [.42, .92, .30], "position": [-2.35, .75, .15], "color": [.19, .48, .62, 1], "material": "fabric"},
                {"name": "waiting_bench_2", "size": [.42, .92, .30], "position": [-2.35, 1.75, .15], "color": [.19, .48, .62, 1], "material": "fabric"},
                {"name": "waiting_bench_back_1", "size": [.12, .92, .65], "position": [-2.60, .75, .42], "color": [.19, .48, .62, 1], "material": "fabric"},
                {"name": "waiting_bench_back_2", "size": [.12, .92, .65], "position": [-2.60, 1.75, .42], "color": [.19, .48, .62, 1], "material": "fabric"},
                {"name": "plant_pot", "shape": "cylinder", "size": [.28, .28, .30], "position": [-2.48, 2.55, .15], "color": [.46, .34, .22, 1]},
                {"name": "plant_stem", "shape": "cylinder", "size": [.08, .08, .56], "position": [-2.48, 2.55, .58], "color": [.20, .48, .22, 1]},
                {"name": "plant_crown", "shape": "cylinder", "size": [.42, .42, .22], "position": [-2.48, 2.55, .91], "color": [.24, .58, .28, 1]},
        {"name": "hall_wall_south", "size": [2.90, .10, 1.2], "position": [.45, -.75, .60], "color": [.88, .89, .86, 1]},
        {"name": "hall_wall_north", "size": [2.90, .10, 1.2], "position": [.45, .75, .60], "color": [.88, .89, .86, 1]},
        {"name": "hall_end_south", "size": [.10, 1.25, 1.2], "position": [1.90, -1.525, .60], "color": [.88, .89, .86, 1]},
        {"name": "hall_end_north", "size": [.10, .72, 1.2], "position": [1.90, 1.11, .60], "color": [.88, .89, .86, 1]},
        {"name": "maintenance_barrier", "size": [.16, 1.30, .82], "position": [1.28, 0, .41], "color": [.96, .56, .08, 1]},
        {"name": "barrier_stripe_1", "size": [.025, .28, .84], "position": [1.19, -.38, .42], "color": [.14, .16, .17, 1]},
        {"name": "barrier_stripe_2", "size": [.025, .28, .84], "position": [1.19, .38, .42], "color": [.14, .16, .17, 1]},
        {"name": "maintenance_cart", "size": [.50, .55, .52], "position": [1.58, -.40, .26], "color": [.94, .66, .11, 1]},
        {"name": "cart_handle", "size": [.06, .62, .55], "position": [1.80, -.40, .58], "color": [.24, .27, .28, 1]},
                {"name": "detour_wall_west_south", "size": [.10, .55, 1.2], "position": [-.55, 1.175, .60], "color": [.84, .87, .87, 1]},
                {"name": "detour_wall_west_north", "size": [.10, .35, 1.2], "position": [-.55, 2.725, .60], "color": [.84, .87, .87, 1]},
        {"name": "diagnostics_south", "size": [1.42, .10, 1.2], "position": [.16, 1.62, .60], "color": [.84, .87, .87, 1]},
                {"name": "sanitizer_column", "size": [.18, .18, .92], "position": [-1.65, 1.72, .46], "color": [.20, .55, .80, 1]},
                {"name": "sanitizer_dispenser", "size": [.22, .14, .28], "position": [-1.55, 1.72, .88], "color": [.83, .94, .96, 1]},
                {"name": "diagnostics_scanner", "size": [.44, .28, .72], "position": [.30, 2.68, .36], "color": [.78, .84, .85, 1]},
                {"name": "scanner_screen", "size": [.035, .24, .24], "position": [.065, 2.68, .70], "color": [.08, .38, .48, 1]},
                {"name": "scanner_stool", "shape": "cylinder", "size": [.30, .30, .38], "position": [.70, 2.65, .19], "color": [.22, .51, .58, 1]},
        {"name": "east_cross_wall_south", "size": [.10, 1.35, 1.2], "position": [.87, .925, .60], "color": [.84, .87, .87, 1]},
        {"name": "east_cross_wall_north", "size": [.10, .50, 1.2], "position": [.87, 2.65, .60], "color": [.84, .87, .87, 1]},
                {"name": "isolation_wall", "size": [1.18, .10, 1.2], "position": [1.49, 1.35, .60], "color": [.88, .82, .81, 1]},
        {"name": "isolation_door", "size": [.10, .72, 1.2], "position": [2.03, 1.98, .60], "color": [.66, .24, .24, 1]},
        {"name": "isolation_window", "size": [.035, .56, .42], "position": [2.075, 1.03, .76], "color": [.74, .38, .38, 1]},
        {"name": "treatment_wall", "size": [1.18, .10, 1.2], "position": [1.49, 2.55, .60], "color": [.80, .88, .82, 1]},
        {"name": "treatment_bed", "size": [1.12, .52, .46], "position": [2.18, 2.16, .23], "color": [.44, .68, .57, 1], "material": "fabric"},
        {"name": "bed_pillow", "size": [.28, .42, .12], "position": [2.49, 2.16, .52], "color": [.90, .94, .90, 1], "material": "fabric"},
        {"name": "iv_pole", "shape": "cylinder", "size": [.05, .05, 1.05], "position": [2.55, 1.72, .525], "color": [.50, .56, .57, 1]},
        {"name": "iv_bag", "size": [.08, .18, .24], "position": [2.55, 1.72, .91], "color": [.70, .90, .86, 1]},
        {"name": "blue_corridor_sign", "size": [.04, .58, .24], "position": [-.98, -.28, .91], "color": [.08, .43, .72, 1]},
        {"name": "blue_detour_sign", "size": [.04, .52, .24], "position": [-.50, .88, .91], "color": [.08, .43, .72, 1]},
        {"name": "green_treatment_sign", "size": [.04, .54, .24], "position": [.92, 2.28, .91], "color": [.12, .62, .34, 1]},
        {"name": "red_isolation_sign", "size": [.04, .54, .24], "position": [1.94, 1.18, .91], "color": [.74, .13, .15, 1]},
                {"name": "delivery_bay", "material": "tile", "size": [1.00, .85, .002], "position": [1.25, 2.00, .001], "color": [.18, .66, .38, 1], "marker": True},
    ],
    objectives=[
                Objective(label="Reach the signed detour beside the sanitizer station", body="robot", center=[-1.15, 1.95, 0],
                          size=[1.00, 1.00], color=[.08, .43, .72, 1], require_lift=False, visible_zone=False, transit=True),
                Objective(label="Pass through the diagnostics doorway", body="robot", center=[.78, 2.05, 0],
                          size=[.60, .70], color=[.08, .43, .72, 1], require_lift=False, visible_zone=False, transit=True),
                Objective(label="Stop fully inside the treatment-room delivery bay", body="robot", center=[1.25, 2.00, 0],
                          size=[1.00, .85], color=[.18, .66, .38, 1], require_lift=False, visible_zone=False),
    ])


def enclosure(width, depth):
    return [
        {"name": name, "size": [.1, depth, 1.3], "position": [sign * (width / 2 - .1), 0, .65], "color": [.72, .77, .78, 1]}
        for name, sign in (("back_wall", 1), ("west_wall", -1))
    ] + [
        {"name": name, "size": [width, .1, 1.3], "position": [0, sign * (depth / 2 - .1), .65], "color": [.80, .83, .82, 1]}
        for name, sign in (("north_wall", 1), ("south_wall", -1))
    ]


PRESETS["warehouse"] = Challenge(
    id="warehouse", title="Warehouse Dispatch Circuit", skill="Long-route planning and precision parking", difficulty="Advanced",
    floor_size_m=[10, 8], initial_xy=[-3.6, -2.8], initial_head_pitch=.12, suggested_turn_limit=200,
    ordered_objectives=True,
    goal="Navigate the warehouse aisles in order: pass fully through the blue inspection square, then the orange stock-check square, and finish in the green dispatch bay. Rack rows block the direct route. Find clear aisles using the head camera and proximity sensors, allowing space for both arms when turning. Stop fully inside the green bay and wait one simulated second. Do not move stock or take shortcuts through racks.",
    objects=[
        *enclosure(10, 8),
        *[{"name": f"rack_{row}_shelf_{level}", "size": [.72, 4.4, .08], "position": [center_x, center_y, height],
           "color": [.26, .35, .40, 1]} for row, center_x, center_y in ((0, -1.5, -.5), (1, 1.5, .5))
          for level, height in enumerate((.10, .60, 1.15))],
        *[{"name": f"rack_{row}_upright_{side}_{end}", "size": [.08, .08, 1.45],
           "position": [center_x + side * .32, center_y + end * 2.15, .725], "color": [.94, .61, .13, 1]}
          for row, center_x, center_y in ((0, -1.5, -.5), (1, 1.5, .5)) for side in (-1, 1) for end in (-1, 1)],
        *[{"name": f"stock_{row}_{level}_{slot}", "size": [.56, .55, .35],
           "position": [center_x, center_y + offset, height], "material": "wood", "color": [.68, .60, .47, 1]}
          for row, center_x, center_y in ((0, -1.5, -.5), (1, 1.5, .5)) for level, height in enumerate((.315, .815))
          for slot, offset in enumerate((-1.7, -.85, 0, .85, 1.7))],
        *[{"name": f"stock_label_{row}_{slot}", "size": [.012, .23, .12],
           "position": [center_x - .367, center_y + offset, .60], "color": [.89, .94, .93, 1]}
          for row, center_x, center_y in ((0, -1.5, -.5), (1, 1.5, .5)) for slot, offset in enumerate((-1.7, 0, 1.7))],
        {"name": "packing_counter", "size": [.42, 1.4, .65], "position": [4.52, .4, .325], "material": "wood", "color": [.48, .59, .56, 1]},
        {"name": "packing_scales", "size": [.30, .40, .06], "position": [4.5, .4, .68], "color": [.24, .28, .29, 1]},
        {"name": "dispatch_terminal", "size": [.12, .32, .85], "position": [4.50, 2.8, .425], "color": [.16, .55, .40, 1]},
        {"name": "dispatch_screen", "size": [.015, .25, .20], "position": [4.43, 2.8, .77], "color": [.54, .87, .81, 1]},
        *[{"name": f"lane_stripe_{index}", "size": [.025, 5.5, .002], "position": [center_x, 0, .002],
           "color": [.96, .80, .25, 1], "marker": True} for index, center_x in enumerate((-2.3, -.7, .7, 2.3))],
    ],
    objectives=[
        Objective(label="Pass the blue inspection square", body="robot", center=[0, 2.8, 0], size=[1.3, 1.2],
                  color=[.12, .51, .85, 1], require_lift=False, transit=True),
        Objective(label="Pass the orange stock-check square", body="robot", center=[0, -2.8, 0], size=[1.3, 1.2],
                  color=[.95, .46, .12, 1], require_lift=False, transit=True),
        Objective(label="Park in dispatch and wait 1 s", body="robot", center=[3.5, 2.7, 0], size=[1.2, 1.3],
                  color=[.14, .65, .36, 1], require_lift=False, dwell_s=1),
    ])


PRESETS["inspection"] = Challenge(
    id="inspection", title="Service Gallery Inspection", skill="Occluded search and visual discrimination", category="Perception", difficulty="Advanced",
     floor_size_m=[8, 8], initial_xy=[-2.7, -2.7], initial_head_pitch=.12, suggested_turn_limit=180,
     search_target="inspection_target",
     goal="Find the yellow inspection cube on a low pedestal in the service gallery. The red cube marks a different station and is not the target. Explore around the equipment partitions and through their open passages. Approach within 0.9 m of the yellow cube, stop with it clearly visible in the head camera, and wait one simulated second. No picking up, wall crossing or moving equipment is required.",
     objects=[
          *enclosure(8, 8),
          {"name": "service_partition", "size": [.14, 4.8, 1.35], "position": [-.6, -1, .675], "color": [.53, .64, .65, 1]},
          {"name": "equipment_screen", "size": [2.3, .14, 1.35], "position": [2.65, .5, .675], "color": [.76, .81, .78, 1]},
          *[{"name": f"electrical_cabinet_{index}", "size": [.45, .8, 1.2], "position": [-3.55, center_y, .6],
              "color": [.30, .47, .54, 1]} for index, center_y in enumerate((-2.3, -.9, .5))],
          *[{"name": f"cabinet_display_{index}", "size": [.016, .25, .18], "position": [-3.317, center_y, .87],
              "color": [.37, .81, .73, 1]} for index, center_y in enumerate((-2.3, -.9, .5))],
          *[{"name": f"cabinet_vent_{index}_{slat}", "size": [.018, .5, .025],
              "position": [-3.315, center_y, .25 + slat * .055], "color": [.12, .22, .26, 1]}
             for index, center_y in enumerate((-2.3, -.9, .5)) for slat in range(4)],
          *[{"name": f"service_tank_{index}", "shape": "cylinder", "size": [.60, .60, 1.1],
              "position": [center_x, -3.35, .55], "color": [.59, .68, .69, 1]}
             for index, center_x in enumerate((1.5, 2.5, 3.4))],
          *[{"name": f"tank_valve_{index}", "shape": "cylinder", "size": [.18, .18, .08],
              "position": [center_x, -3.35, 1.14], "color": [.81, .25, .23, 1]}
             for index, center_x in enumerate((1.5, 2.5, 3.4))],
          {"name": "maintenance_counter", "size": [1.3, .45, .65], "position": [2.6, 3.52, .325], "material": "wood", "color": [.52, .58, .46, 1]},
          {"name": "tool_case", "size": [.4, .28, .18], "position": [2.55, 3.50, .75], "color": [.89, .59, .14, 1]},
          *[{"name": f"inspection_pedestal_{index}", "size": [.36, .36, .34], "position": [center_x, center_y, .17],
              "color": [.38, .42, .44, 1]} for index, (center_x, center_y) in enumerate(((2.8, -2.3), (2.8, 2.6)))],
          {"name": "inspection_target", "size": [.18, .18, .18], "position": [2.8, -2.3, .43], "color": [.98, .78, .05, 1]},
          {"name": "inspection_decoy", "size": [.18, .18, .18], "position": [2.8, 2.6, .43], "color": [.85, .12, .18, 1]},
          {"name": "service_walkway", "size": [1.2, 6.0, .002], "position": [-2.5, 0, .001], "color": [.56, .66, .65, 1], "marker": True},
          *[{"name": f"partition_safety_band_{index}", "size": [.016, .35, .16], "position": [-.679, center_y, .9],
              "color": [.96, .76, .15, 1]} for index, center_y in enumerate((-2.8, -.6, 1.1))],
     ],
     objectives=[Objective(label="Find and inspect the yellow cube for 1 s", body="robot", center=[2.8, -2.3, 0],
                                  size=[1.8, 1.8], color=[.98, .78, .05, 1], require_lift=False, visible_zone=False)])


PRESETS["workshop"] = Challenge(
    id="workshop", title="Cluttered Assembly Workshop", skill="Two-arm precision placement around fixtures", category="Manipulation", difficulty="Advanced",
     initial_head_pitch=.65, suggested_turn_limit=120,
     goal="At this assembly station, lift the small red and blue cubes and place each fully inside its matching colored square. Release both cubes and let them settle on the floor. Keep the base parked, leave the large gray fixture blocks in place, and avoid the divider and storage fixtures. The two small cubes are reachable by the corresponding left and right arms. Pushing a cube or holding it above its square does not count.",
     objects=[
          *enclosure(6, 6),
          *[{"name": f"{name}_cube", "size": [.06, .06, .06], "position": [.36, sign * .25, .032],
              "color": color, "mass": .08} for name, sign, color in (("red", 1, [.86, .17, .27, 1]), ("blue", -1, [.14, .49, .86, 1]))],
          {"name": "workstation_divider", "size": [.10, .12, .48], "position": [.70, 0, .24], "color": [.67, .71, .69, 1]},
          *[{"name": f"fixture_block_{side}", "size": [.22, .20, .22], "position": [.83, sign * .55, .11],
              "color": [.44, .48, .49, 1]} for side, sign in (("left", 1), ("right", -1))],
          *[{"name": f"parts_rack_{side}_shelf_{level}", "size": [1.2, .50, .07], "position": [.5, sign * 1.8, height],
              "color": [.29, .42, .43, 1]} for side, sign in (("left", 1), ("right", -1)) for level, height in enumerate((.1, .65, 1.15))],
          *[{"name": f"parts_rack_{side}_post_{end}", "size": [.06, .50, 1.2], "position": [.5 + end * .57, sign * 1.8, .6],
              "color": [.45, .54, .54, 1]} for side, sign in (("left", 1), ("right", -1)) for end in (-1, 1)],
          *[{"name": f"parts_bin_{side}_{index}", "size": [.25, .35, .22], "position": [center_x, sign * 1.8, .80],
              "color": [.82, .67, .21, 1]} for side, sign in (("left", 1), ("right", -1)) for index, center_x in enumerate((.10, .5, .9))],
          {"name": "assembly_counter", "size": [.50, 1.65, .68], "position": [2.3, 0, .34], "material": "wood", "color": [.55, .63, .56, 1]},
          {"name": "bench_vise_base", "size": [.27, .35, .08], "position": [2.28, .45, .73], "color": [.25, .32, .35, 1]},
          *[{"name": f"vise_jaw_{index}", "size": [.08, .27, .14], "position": [center_x, .45, .84],
              "color": [.35, .48, .52, 1]} for index, center_x in enumerate((2.2, 2.36))],
          {"name": "assembly_toolbox", "size": [.35, .45, .25], "position": [2.3, -.50, .84], "color": [.77, .23, .24, 1]},
          {"name": "workstation_mat", "size": [1.2, 1.6, .002], "position": [.38, 0, .0005], "color": [.64, .68, .66, 1], "marker": True},
     ],
     objectives=[Objective(label=f"{name.title()} cube lifted, released and settled in its square", body=f"{name}_cube",
                                  center=[.32, sign * .50, 0], size=[.18, .18], color=color)
                     for name, sign, color in (("red", 1, [.86, .17, .27, 1]), ("blue", -1, [.14, .49, .86, 1]))])


def get_challenge(identifier):
    if identifier == "bench":
        return None
    return PRESETS[identifier].model_copy(deep=True)


class ChallengeProgress:
    def __init__(self, challenge):
        self.challenge = challenge
        self.lifted = set()
        self.status = None
        self.battery = BatterySensor(charge_pct=100, low=False, charging=False) if challenge.id == "recharge" else None
        self.last_time = 0
        self.last_travel = 0
        self.survey_dwell_s = 0
        self.survey_complete = False
        self.low_away = False
        self.returned = False
        self.search_dwell_s = 0
        self.ordered_stage = 0
        self.route_dwell_s = 0

    @staticmethod
    def _parked(measurement, objective):
        lower, upper = measurement["bounds"]
        return (all(lower[axis] >= objective.center[axis] - objective.size[axis] / 2 and
                    upper[axis] <= objective.center[axis] + objective.size[axis] / 2 for axis in (0, 1)) and
                measurement["grounded"] and measurement["speed"] < .025 and measurement["angular_speed"] < .15)

    def _update_recharge(self, measurement, simulated_time_s, travel_m):
        elapsed = max(0, simulated_time_s - self.last_time)
        distance = max(0, travel_m - self.last_travel)
        self.last_time, self.last_travel = simulated_time_s, travel_m
        docked = self._parked(measurement, self.challenge.objectives[1])
        surveying = self._parked(measurement, self.challenge.objectives[0])
        charge = self.battery.charge_pct
        if docked:
            charge = min(100, charge + elapsed * 20)
        else:
            charge = max(0, charge - distance * 4 - elapsed * .1)
        if not self.survey_complete:
            self.survey_dwell_s = self.survey_dwell_s + elapsed if surveying and charge > 0 else 0
            if self.survey_dwell_s >= 1:
                self.survey_complete = True
                charge = max(0, charge - 60)
        self.low_away = self.low_away or (self.survey_complete and charge <= 35 and not docked)
        self.returned = self.returned or (self.low_away and docked)
        self.battery = BatterySensor(charge_pct=charge, low=charge <= 35, charging=docked and charge < 100)
        complete = self.returned and docked and charge >= 90
        failed = charge <= 0 and not docked
        checks = [self.survey_complete, self.returned, complete]
        details = ["Complete" if self.survey_complete else "Park in the orange zone and wait 1 s",
                   "Complete" if self.returned else "Remember the route back to the cyan pad",
                   "Complete" if complete else "Charging" if self.battery.charging and self.returned else "Park on the original charger"]
        objectives = [{"label": objective.label, "complete": checks[index], "detail": "Battery empty: reset the episode" if failed else details[index]}
                      for index, objective in enumerate(self.challenge.objectives)]
        self.status = {**self.challenge.public(), "status": "failed" if failed else "completed" if complete else "in_progress",
                       "completed_objectives": sum(checks), "progress": objectives}
        return self.status

    def update(self, measurements, held, simulated_time_s=0, travel_m=0):
        if self.challenge.search_target:
            measurement = measurements["robot"]
            elapsed = max(0, simulated_time_s - self.last_time)
            self.last_time = simulated_time_s
            stationary = (measurement["grounded"] and measurement["speed"] < .025 and
                          measurement["angular_speed"] < .15 and measurement["head_stationary"])
            inspecting = measurement["target_nearby"] and measurement["target_visible"] and stationary
            self.search_dwell_s = min(1, self.search_dwell_s + elapsed) if inspecting else 0
            complete = self.search_dwell_s >= 1
            detail = ("Complete" if complete else "Explore the rooms" if not measurement["target_nearby"] else
                      "Look at the yellow cube" if not measurement["target_visible"] else
                      "Stop and hold the camera steady" if not stationary else "Inspecting / hold for 1 s")
            self.status = {**self.challenge.public(), "status": "completed" if complete else "in_progress",
                           "completed_objectives": int(complete), "progress": [
                               {"label": self.challenge.objectives[0].label, "complete": complete, "detail": detail}]}
            return self.status
        if self.battery is not None:
            return self._update_recharge(measurements["robot"], simulated_time_s, travel_m)
        if self.challenge.ordered_objectives:
            measurement = measurements["robot"]
            elapsed = max(0, simulated_time_s - self.last_time)
            self.last_time = simulated_time_s
            objective = self.challenge.objectives[self.ordered_stage]
            lower, upper = measurement["bounds"]
            inside = all(lower[axis] >= objective.center[axis] - objective.size[axis] / 2 and
                         upper[axis] <= objective.center[axis] + objective.size[axis] / 2 for axis in (0, 1))
            eligible = measurement["grounded"] and inside and (objective.transit or self._parked(measurement, objective))
            self.route_dwell_s = self.route_dwell_s + elapsed if eligible else 0
            reached = eligible and self.route_dwell_s >= objective.dwell_s
            complete = reached and self.ordered_stage == len(self.challenge.objectives) - 1
            if reached and not complete:
                self.ordered_stage += 1
                self.route_dwell_s = 0
            objectives = []
            for index, objective in enumerate(self.challenge.objectives):
                checked = index < self.ordered_stage or complete
                detail = ("Complete" if checked else "Current route objective" if index == self.ordered_stage else "Complete the previous route objective first")
                objectives.append({"label": objective.label, "complete": checked, "detail": detail})
            self.status = {**self.challenge.public(), "status": "completed" if complete else "in_progress",
                           "completed_objectives": self.ordered_stage + int(complete), "progress": objectives}
            return self.status
        objectives = []
        for objective in self.challenge.objectives:
            measurement = measurements[objective.body]
            lower, upper = measurement["bounds"]
            holding = objective.body in held
            if holding and lower[2] >= .08:
                self.lifted.add(objective.body)
            lifted = not objective.require_lift or objective.body in self.lifted
            inside = all(lower[axis] >= objective.center[axis] - objective.size[axis] / 2 and
                         upper[axis] <= objective.center[axis] + objective.size[axis] / 2 for axis in (0, 1))
            grounded = measurement["grounded"]
            settled = measurement["speed"] < .025 and measurement["angular_speed"] < .15
            complete = lifted and inside and not holding and grounded and settled
            detail = ("Complete" if complete else "Lift clear of the floor" if not lifted else
                      ("Move fully inside the bathroom" if self.challenge.id == "kitchen_bathroom" else "Move fully inside the zone") if not inside else "Release the cube" if holding else
                      "Lower to the floor" if not grounded else "Come to rest")
            objectives.append({"label": objective.label, "complete": complete, "detail": detail})
        self.status = {**self.challenge.public(), "status": "completed" if all(item["complete"] for item in objectives) else "in_progress",
                       "completed_objectives": sum(item["complete"] for item in objectives), "progress": objectives}
        return self.status