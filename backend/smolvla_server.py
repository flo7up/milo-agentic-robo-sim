import argparse
import base64
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from pydantic import Field, model_validator

from backend.contracts import AgentObservation, StrictModel
from backend.policy import ACTION_NAMES, PolicyChunk, PolicyMetadata, PolicyTicket


class PolicyRequest(StrictModel):
    ticket: PolicyTicket
    observation: AgentObservation
    image: str = Field(min_length=1, max_length=2_000_000)
    instruction: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def paired_observation(self):
        if (self.ticket.run_id != self.observation.run_id or
                self.ticket.episode_epoch != self.observation.episode_epoch or
                self.ticket.observation_seq != self.observation.seq):
            raise ValueError("Image observation and policy ticket must refer to the same episode and frame")
        return self


def state_vector(observation):
    positions = {joint.name: joint.position for joint in observation.joints}
    try:
        return [positions[name] for name in ACTION_NAMES[:6]] + [observation.grippers["left"].aperture_m]
    except KeyError as error:
        raise ValueError("Observation lacks Milo left-arm proprioception") from error


class SmolVLABackend:
    def __init__(self, checkpoint, device="cuda", dtype="bfloat16"):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        checkpoint = Path(checkpoint).resolve(strict=True)
        self.metadata = PolicyMetadata.model_validate_json((checkpoint / "milo-policy.json").read_text(encoding="utf-8"))
        if not self.metadata.trained_for_milo:
            raise ValueError("Checkpoint manifest does not declare Milo task training; execution disabled")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in the policy environment")
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint)).to(device).eval()
        config = self.policy.config
        if (tuple(config.input_features["observation.state"].shape) != (7,) or
                tuple(config.output_features["action"].shape) != (7,) or
                set(config.image_features) != {self.metadata.camera_key} or config.n_obs_steps != 1 or
                not 1 <= config.chunk_size <= 50):
            raise ValueError("Checkpoint features are incompatible with milo-left-arm-v1; do not reshape a base checkpoint")
        self.preprocess, self.postprocess = make_pre_post_processors(config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}})
        self.device, self.dtype = device, dtype

    def predict(self, request):
        import numpy as np
        import torch
        with Image.open(BytesIO(base64.b64decode(request.image, validate=True))) as source:
            if source.width > 640 or source.height > 480:
                raise ValueError("Policy camera exceeds the supported head-camera resolution")
            pixels = np.array(source.convert("RGB"), dtype=np.float32) / 255
        frame = {"observation.state": torch.tensor(state_vector(request.observation), dtype=torch.float32),
                 self.metadata.camera_key: torch.from_numpy(pixels).permute(2, 0, 1), "task": request.instruction}
        self.policy.reset()
        batch = self.preprocess(frame)
        precision = (torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" and self.dtype == "bfloat16"
                     else nullcontext())
        with torch.inference_mode(), precision:
            prediction = self.policy.predict_action_chunk(batch)
            actions = [self.postprocess(prediction[:, index, :])[0].detach().cpu().tolist()
                       for index in range(prediction.shape[1])]
        return PolicyChunk(ticket=request.ticket, actions=actions)


def create_app(backend):
    app = FastAPI(title="Milo SmolVLA Policy Server")
    inference_lock = threading.Lock()

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        from fastapi.responses import Response
        origin = request.headers.get("origin")
        if origin and origin != f"http://{request.headers.get('host')}":
            return Response("Same-origin requests required", status_code=403)
        if int(request.headers.get("content-length", "0")) > 2_100_000:
            return Response("Request too large", status_code=413)
        return await call_next(request)

    @app.get("/policy")
    def metadata():
        return backend.metadata

    @app.post("/predict")
    def predict(request: PolicyRequest):
        age = time.time() - request.observation.wall_timestamp
        if age < -1 or age > 2:
            raise HTTPException(409, "Observation expired")
        if not backend.metadata.trained_for_milo:
            raise HTTPException(409, "A Milo-trained checkpoint is required")
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(429, "Policy inference already in flight; do not queue old frames")
        try:
            return backend.predict(request)
        except (ValueError, KeyError) as error:
            raise HTTPException(422, "Invalid policy input or action output") from error
        finally:
            inference_lock.release()

    return app


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Local SmolVLA server for a trained Milo left-arm checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--port", type=int, default=8085)
    options = parser.parse_args()
    backend = SmolVLABackend(options.checkpoint, options.device, options.dtype)
    uvicorn.run(create_app(backend), host="127.0.0.1", port=options.port, limit_concurrency=4)