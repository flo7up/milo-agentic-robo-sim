import json


def compact_numbers(value):
    if isinstance(value, float):
        rounded = round(value, 3)
        return int(rounded) if rounded.is_integer() else rounded
    if isinstance(value, dict):
        return {key: compact_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact_numbers(item) for item in value]
    return value


def feedback_json(value):
    return json.dumps(compact_numbers(value), separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def text_context(items):
    return [{**item, "content": [part for part in item["content"] if part.get("type") != "input_image"]}
            if isinstance(item.get("content"), list) else dict(item) for item in items]


def context_token_estimate(items):
    return len(json.dumps(items, separators=(",", ":"), ensure_ascii=False).encode("utf-8")) + 16 * len(items) if items else 0


def retain_context(history, budget):
    retained, used = [], 0
    for turn in reversed(history):
        text = text_context(turn)
        cost = context_token_estimate(text)
        if used + cost > budget:
            break
        retained.insert(0, text)
        used += cost
    return retained, used


def camera_batch(current, previous, limit, memory=None):
    selected = []
    current_url = current["content"][1]["image_url"]
    seen = {current_url}
    candidates = ([memory] if memory is not None else []) + list(reversed(previous))
    for candidate in candidates:
        if len(selected) >= limit - 1:
            break
        image_url = candidate["content"][1]["image_url"]
        if image_url not in seen:
            selected.append(candidate)
            seen.add(image_url)
    selected.sort(key=lambda item: json.loads(item["content"][0]["text"])["wall_timestamp"])
    content = list(current["content"])
    for frame in selected:
        sensors = json.loads(frame["content"][0]["text"])
        content.extend([{"type": "input_text", "text": feedback_json({"historical_camera_observation": sensors})},
                        frame["content"][1]])
    return {**current, "content": content}, [current, *selected]


def collision_feedback(contacts, source="current"):
    if not contacts:
        return None
    rear_contact = any(contact["direction"] == "rear" for contact in contacts)
    guidance = ("Rear contact: do not reverse. Inspect other directions and stop if clearance is uncertain."
                if rear_contact else
                "Contact detected: stop pushing. Inspect rear clearance; a short, slow reverse may help. "
                "Reverse only if clear; distance beams alone do not guarantee clearance.")
    return compact_numbers({"source": source, "contacts": contacts, "guidance": guidance})


def observation_collision(observation):
    proximity = observation.get("proximity") or {}
    contacts = proximity.get("collisions") or [{"direction": direction} for direction in observation.get("bumpers", [])]
    return collision_feedback(contacts)


def model_tool_result(result):
    observation = result.get("observation") or {}
    compact = {key: result[key] for key in ("status", "error", "message", "actual_duration_s", "sensor_deltas")
               if key in result and result[key] is not None and result[key] != "" and result[key] != {}}
    if "seq" in observation:
        compact["observation_seq"] = observation["seq"]
    contacts = result.get("sensor_deltas", {}).get("contacts_during_action")
    collision = observation_collision(observation) or collision_feedback(contacts, "during_action")
    if contacts:
        compact["sensor_deltas"] = {key: value for key, value in compact["sensor_deltas"].items()
                                    if key != "contacts_during_action"}
    if collision:
        compact["collision_feedback"] = collision
    return compact_numbers(compact)