import argparse
import asyncio
import base64
from io import BytesIO
import json
from pathlib import Path
import statistics
import time

import httpx
from openai import APIError, AsyncOpenAI
from PIL import Image


async def camera_feedback(worker):
    def capture(sim):
        observation = sim.observe()
        return observation, sim.frame(observation.frame_ref)
    return await worker.call(capture)


def local_feedback(observation, image, sensors="full"):
    content = []
    if sensors != "none":
        values = observation.model_dump() if sensors == "full" else observation.model_dump(
            include={"head_rad", "odometry_m_rad", "proximity", "bumpers"})
        content.append({"type": "text", "text": json.dumps(values)})
    content.append(
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")}})
    return {"role": "user", "content": content}


VISION_PROMPT = (
    "Identify the room using only the supplied camera image. Return one JSON object "
    "with room (kitchen, bathroom, other, or uncertain) and fixtures (a short list of "
    "visible fixtures). Do not invent objects. No explanation outside JSON.")
VISION_FORMAT = {"type": "object", "properties": {
    "room": {"type": "string", "enum": ["kitchen", "bathroom", "other", "uncertain"]},
    "fixtures": {"type": "array", "items": {"type": "string"}, "maxItems": 6}},
    "required": ["room", "fixtures"], "additionalProperties": False}


def ollama_result(data):
    if data.get("error"):
        raise ValueError("Ollama returned an inference error")
    return {"answer": data.get("message", {}).get("content", ""), "response_model": data.get("model"),
        "finish_reason": data.get("done_reason"), "done": data.get("done", False),
        "thinking_disabled": True, "thinking_present": bool(data.get("message", {}).get("thinking")),
        "usage": {"input_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count")},
        "timing_s": {key: round(data[key] / 1e9, 4) for key in
            ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration") if key in data}}


async def vision_completion(client, model, observation, image, sensors, provider):
    feedback = local_feedback(observation, image, sensors)
    if provider == "ollama":
        response = await client.post("/api/chat", json={
            "model": model, "think": False, "stream": False, "keep_alive": "5m",
            "format": VISION_FORMAT,
            "options": {"temperature": 0, "num_predict": 128, "num_ctx": 4096},
            "messages": [{"role": "system", "content": VISION_PROMPT},
                {"role": "user", "content": "\n".join(
                    part["text"] for part in feedback["content"] if part["type"] == "text"),
                 "images": [base64.b64encode(image).decode("ascii")]}]})
        response.raise_for_status()
        return ollama_result(response.json())
    response = await client.chat.completions.create(
        model=model, temperature=0, max_tokens=128,
        messages=[{"role": "system", "content": VISION_PROMPT}, feedback])
    return {"answer": response.choices[0].message.content or "", "response_model": response.model,
        "finish_reason": response.choices[0].finish_reason,
        "usage": response.usage.model_dump() if response.usage else None}


async def probe_vision(client, model, output, image_width, sensors, repetitions, frames=None,
                       rooms=("kitchen", "bathroom"), provider="openai", vary_crops=False):
    records = []
    for room, position in (("kitchen", [1.1, -1.5]), ("bathroom", [1.2, 1.4])):
        if room not in rooms:
            continue
        worker = None
        try:
            if frames:
                observation, image = None, (frames / f"{room}.png").read_bytes()
            else:
                from backend.challenges import get_challenge
                from backend.worker import SimulationWorker
                preset = get_challenge("kitchen_bathroom")
                preset.initial_xy = position
                worker = SimulationWorker(challenge=preset, pace=True)
                observation, image = await camera_feedback(worker)
            if image_width != 640:
                resized = Image.open(BytesIO(image)).resize((image_width, image_width * 3 // 4))
                buffer = BytesIO()
                resized.save(buffer, format="PNG")
                image = buffer.getvalue()
            (output / f"{room}.png").write_bytes(image)
            for repetition in range(repetitions):
                request_image = image
                if vary_crops:
                    with Image.open(BytesIO(image)) as source:
                        inset = (repetition + 1) * 2
                        cropped = source.crop((inset, inset, source.width - inset, source.height - inset))
                        buffer = BytesIO()
                        cropped.resize(source.size).save(buffer, format="PNG")
                        request_image = buffer.getvalue()
                    (output / f"{room}-crop-{repetition}.png").write_bytes(request_image)
                started = time.perf_counter()
                try:
                    response = await vision_completion(client, model, observation, request_image, sensors, provider)
                except (APIError, httpx.HTTPError, ValueError) as error:
                    records.append({"expected_room": room, "repetition": repetition,
                        "latency_s": round(time.perf_counter() - started, 3),
                        "error": type(error).__name__, "exact_json_correct": False})
                    print(json.dumps(records[-1]), flush=True)
                    return records
                elapsed = time.perf_counter() - started
                answer = response["answer"]
                try:
                    parsed = json.loads(answer)
                except json.JSONDecodeError:
                    parsed = None
                records.append({"expected_room": room, "repetition": repetition,
                    "image_kind": "derived_crop" if vary_crops else "same_frame",
                    "latency_s": round(elapsed, 3), **response,
                    "exact_json_correct": isinstance(parsed, dict) and parsed.get("room") == room
                        and isinstance(parsed.get("fixtures"), list)
                        and all(isinstance(fixture, str) for fixture in parsed["fixtures"])
                        and response["finish_reason"] == "stop" and not response.get("thinking_present", False)})
                print(json.dumps(records[-1]), flush=True)
        finally:
            if worker:
                await worker.close()
    return records


def validate_action_calls(calls, instruction):
    from backend.contracts import TOOL_MODELS
    if len(calls) != 1:
        return False, False
    function = calls[0].get("function", {})
    name, arguments = function.get("name"), function.get("arguments", {})
    if name not in {"observe", "stop", "set_head", "drive_base"}:
        return False, False
    try:
        TOOL_MODELS[name].model_validate(arguments)
    except ValueError:
        return False, False
    matched = (name == "stop" if instruction == "stop" else
               name == "set_head" and (arguments["yaw_rad"] > 0 if instruction == "left" else arguments["yaw_rad"] < 0))
    return True, matched


async def probe_actions(client, model, image, repetitions):
    from backend.contracts import tool_schemas
    tools = [{"type": "function", "function": {key: value for key, value in schema.items() if key != "type"}}
             for schema in tool_schemas() if schema["name"] in {"observe", "stop", "set_head", "drive_base"}]
    records = []
    for instruction, prompt in (("left", "Look to your left to inspect this room. Keep the base stationary."),
                                ("right", "Look to your right to inspect this room. Keep the base stationary."),
                                ("stop", "Stop all robot motion now.")):
        for repetition in range(repetitions):
            started = time.perf_counter()
            try:
                response = await client.post("/api/chat", json={
                    "model": model, "think": False, "stream": False, "keep_alive": "5m",
                    "options": {"temperature": 0, "num_predict": 128, "num_ctx": 4096}, "tools": tools,
                    "messages": [{"role": "system", "content":
                        "You control a simulated robot through the provided tools. Use exactly one tool. "
                        "Head angles are radians; positive yaw looks left, negative yaw looks right, "
                        "and positive pitch looks down. No physical command will be executed in this dry run. "
                        "Only the attached head camera describes the scene. Do not invent map coordinates."},
                        {"role": "user", "content": prompt,
                         "images": [base64.b64encode(image).decode("ascii")]}]})
                response.raise_for_status()
                data = response.json()
                result = ollama_result(data)
                calls = data.get("message", {}).get("tool_calls") or []
                valid, matched = validate_action_calls(calls, instruction)
                completed = result["done"] and result["finish_reason"] == "stop" and not result["thinking_present"]
                record = {"instruction": instruction, "repetition": repetition,
                    "latency_s": round(time.perf_counter() - started, 3), **result, "tool_calls": calls,
                    "valid_single_action": valid and completed, "matched_instruction": matched and completed}
            except (httpx.HTTPError, ValueError) as error:
                records.append({"instruction": instruction, "repetition": repetition,
                    "latency_s": round(time.perf_counter() - started, 3), "error": type(error).__name__,
                    "valid_single_action": False, "matched_instruction": False})
                print(json.dumps(records[-1]), flush=True)
                return records
            records.append(record)
            print(json.dumps(record), flush=True)
    return records


async def main(options):
    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"model": options.model, "stage": options.stage, "live_model": True,
              "provider": options.provider, "thinking_disabled": options.provider == "ollama",
              "vary_crops": options.vary_crops,
              "robot_motion_executed": False, "image_width": options.image_width,
              "sensors": options.sensors, "frame_source": "saved_camera" if options.frames else "fresh_simulation",
              "rooms": options.rooms, options.stage: []}
    try:
        client = (httpx.AsyncClient(base_url=options.local_endpoint, timeout=options.timeout, trust_env=False)
                  if options.provider == "ollama" else AsyncOpenAI(
                      base_url=options.local_endpoint.rstrip("/") + "/v1",
                      api_key="not-needed", timeout=options.timeout, max_retries=0))
        async with client:
            if options.stage == "actions":
                image = (options.frames / f"{options.rooms[0]}.png").read_bytes()
                with Image.open(BytesIO(image)) as source:
                    buffer = BytesIO()
                    source.resize((options.image_width, options.image_width * 3 // 4)).save(buffer, format="PNG")
                    image = buffer.getvalue()
                report["actions"] = await probe_actions(client, options.model, image, options.repetitions)
            else:
                report["vision"] = await probe_vision(client, options.model, output, options.image_width,
                                                     options.sensors, options.repetitions, options.frames, options.rooms,
                                                     options.provider, options.vary_crops)
        records = report[options.stage]
        latencies = [record["latency_s"] for record in records]
        report["summary"] = {"samples": len(latencies), "median_s": statistics.median(latencies),
            "max_s": max(latencies), "errors": sum("error" in record for record in records)}
        for key in (("valid_single_action", "matched_instruction") if options.stage == "actions" else ("exact_json_correct",)):
            report["summary"][key] = sum(record[key] for record in records)
    finally:
        (output / f"{options.stage}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated live local vision-model robot feasibility probe")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--stage", choices=["vision", "actions"], default="vision")
    parser.add_argument("--vary-crops", action="store_true", help="Use distinct derived crops to avoid identical-image repeats")
    parser.add_argument("--local-endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=".runtime/hybrid-probe")
    parser.add_argument("--image-width", type=int, choices=[320, 640], default=640)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--sensors", choices=["full", "navigation", "none"], default="full")
    parser.add_argument("--repetitions", type=int, choices=range(1, 11), default=3)
    parser.add_argument("--frames", type=Path, help="Replay saved camera PNGs without importing the simulator")
    parser.add_argument("--rooms", choices=["kitchen", "bathroom"], nargs="+", default=["kitchen", "bathroom"])
    options = parser.parse_args()
    if options.stage == "actions" and (options.provider != "ollama" or not options.frames or len(options.rooms) != 1 or options.vary_crops):
        parser.error("Action dry runs require Ollama, one saved room, and no crop variants")
    if options.frames:
        if options.sensors != "none":
            parser.error("Saved frames require --sensors none; no synthetic sensors are supplied")
        if any(not (options.frames / f"{room}.png").is_file() for room in options.rooms):
            parser.error("Each requested room must have a saved PNG in --frames")
    asyncio.run(main(options))