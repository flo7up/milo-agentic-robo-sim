import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = {
    "single_step": ("tool-step", "Tool-step control", "0.1.0"),
    "navigation_plan": ("buffered-plan", "Buffered navigation plan", "0.1.0"),
    "luna_continuous": ("observed-continuous", "Observed continuous control", "0.6.0"),
    "unified_mission": ("unified-map-mission", "Unified map-aware mission", "0.4.2"),
    "hybrid_task_supervision": ("qwen-luna-task-supervision", "Qwen with Luna task supervision", "0.3.2"),
    "nav2": ("nav2-supervised", "Nav2 observed goal control", "0.1.0"),
    "skill_composer": ("motion-skills", "Composed motion skills", "0.6.0"),
    "ai_routes": ("ai-waypoints", "AI-generated waypoint control", "0.1.0"),
    "luna_navigation": ("luna-smolvla", "Luna + SmolVLA navigation", "0.1.0"),
    "local_navigation": ("local-policy", "Local navigation policy", "0.1.0"),
    "supervised_policy": ("supervised-policy", "Supervised manipulation policy", "0.1.0"),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def variant_snapshot(settings, profile, root=ROOT, effective_configuration=None):
    mode = getattr(settings, "execution_mode", "single_step")
    architecture_key = "ai_routes" if mode == "luna_continuous" and getattr(settings, "ai_generated_routes", False) else mode
    if mode == "luna_continuous" and getattr(settings, "skill_composer", False):
        architecture_key = "skill_composer"
    if mode == "luna_continuous" and getattr(settings, "navigation_backend", "builtin") == "nav2":
        architecture_key = "nav2"
    if getattr(settings, "unified_mission", False):
        architecture_key = "hybrid_task_supervision" if getattr(settings, "task_supervisor_model_id", None) else "unified_mission"
    identifier, name, version = ARCHITECTURES.get(architecture_key, (mode, mode, "0.0.0"))
    paths = [*sorted((root / "backend").glob("*.py")), root / "assets/milo.urdf", root / "pyproject.toml"]
    if architecture_key == "nav2":
        paths.extend([*sorted((root / "ros").glob("*.py")), *sorted((root / "ros").glob("*.yaml")),
            root / "ros/Dockerfile", root / "start.ps1"])
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}
    implementation = digest(hashes)
    configuration = {field: getattr(settings, field, None) for field in (
        "reasoning", "images_per_request", "context_tokens", "feedback_interval_s", "max_turns",
        "compact_arms", "continuous_handoff", "adaptive_navigation", "ai_generated_routes", "skill_composer", "navigation_backend",
        "unified_mission", "mission_local_only", "map_context", "mission_budget_s")}
    configuration.update(task_supervisor_model_id=getattr(settings, "task_supervisor_model_id", None),
        task_supervisor_reasoning=getattr(settings, "task_supervisor_reasoning", None),
        max_task_supervisor_requests=getattr(settings, "max_task_supervisor_requests", None),
        max_task_supervisor_tokens=getattr(settings, "max_task_supervisor_tokens", None))
    if mode == "supervised_policy" and getattr(settings, "policy", None) is not None:
        configuration["policy"] = settings.policy.model_dump()
    configuration.update(effective_configuration or {})
    if getattr(profile, "provider", None) == "ollama":
        configuration["local_context_window"] = getattr(profile, "context_window", None) or max(16384,
            ((getattr(settings, "context_tokens", 4096) + 8192 + getattr(settings, "images_per_request", 1) * 1024 + 8191) // 8192) * 8192)
        configuration["local_output_tokens"] = 1024
        configuration["local_transport"] = "mission_json_v1" if getattr(settings, "unified_mission", False) else "native_tools"
        configuration["automatic_fallback"] = False
        if getattr(settings, "unified_mission", False):
            configuration.update(semantic_review_interval_s=3., semantic_review_travel_m=.75,
                semantic_review_latency_backoff="max(3, 1.25*(recent_max_input_s+recent_max_response_s)+0.25)",
                semantic_reviews_require_stopped_evidence=True)
    model = {"provider": getattr(profile, "provider", "unknown"), "profile_id": getattr(profile, "id", "unknown"),
        "deployment": getattr(profile, "deployment", "unknown"), "configuration": configuration,
        "underlying_identity_verified": False}
    if getattr(settings, "task_supervisor_model_id", None):
        model["task_supervision"] = {"profile_id": settings.task_supervisor_model_id,
            "scope": "immutable_task_plan_and_non_actuating_guidance", "operational_authority": False,
            "observation_inputs": False, "automatic_fallback": False}
    model["revision"] = digest(model)[:12]
    architecture = {"id": identifier, "name": name, "version": version,
        "revision": implementation[:12], "implementation_sha256": implementation,
        "version_key": f"{identifier}@{version}+{implementation[:12]}"}
    return {"architecture": architecture, "model_variant": model, "code_sha256": hashes,
        "variant_id": f"{architecture['version_key']}/{model['revision']}"}