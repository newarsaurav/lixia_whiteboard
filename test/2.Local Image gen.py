"""
Local, self-hosted image generation for the whiteboard's create_sketch_image tool.
Runs an SDXL pipeline (+ SDXL-Lightning for speed, + your own optional style LoRA)
on your own GPU instead of calling Gemini's hosted image model — free, private,
and retrainable whenever you want a different sketch style.

Tuned for a 16GB card (e.g. RTX 5060 Ti): SDXL base + a 4-step Lightning LoRA
generates a 1024x1024 image in roughly 2-4 seconds, with enable_model_cpu_offload()
keeping VRAM headroom for your own style LoRA on top.

Install (pick the torch line for your CUDA version — check https://pytorch.org):
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install diffusers transformers accelerate safetensors pillow peft huggingface_hub

First run will download the base SDXL weights (~7GB) and the Lightning LoRA (~400MB)
from Hugging Face and cache them under ~/.cache/huggingface.

── Bringing your own style ────────────────────────────────────────────────
Train a LoRA on ~20-50 of your own pencil sketches with diffusers' official
SDXL LoRA training script, e.g.:

    accelerate launch train_text_to_image_lora_sdxl.py \\
      --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \\
      --train_data_dir="/path/to/your/sketch/dataset" \\
      --caption_column="text" \\
      --resolution=1024 --train_batch_size=1 --rank=16 \\
      --learning_rate=1e-4 --max_train_steps=1500 \\
      --output_dir="/path/to/output/my-sketch-lora"

(script lives in the diffusers repo under examples/text_to_image/). Each training
image needs a short caption describing it, e.g. "a pencil sketch of a bicycle".
Once trained, point LOCAL_STYLE_LORA_PATH at the resulting .safetensors file and
restart the server — no code changes needed.
"""

import base64
import io
import os
import threading

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download

BASE_MODEL = os.environ.get("LOCAL_BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

# SDXL-Lightning: a distilled LoRA that gets good results in 4 steps instead of the
# usual ~30-50, which is what makes this fast enough to feel "live" on a single GPU.
LIGHTNING_REPO = os.environ.get("LOCAL_LIGHTNING_REPO", "ByteDance/SDXL-Lightning")
LIGHTNING_CKPT = os.environ.get("LOCAL_LIGHTNING_CKPT", "sdxl_lightning_4step_lora.safetensors")
LIGHTNING_STEPS = int(os.environ.get("LOCAL_LIGHTNING_STEPS", "4"))

# Your own fine-tuned pencil-sketch-style LoRA, layered on top. Optional — without
# it you still get sketch-ish results from prompting alone, just less consistent.
STYLE_LORA_PATH = os.environ.get("LOCAL_STYLE_LORA_PATH")
STYLE_LORA_SCALE = float(os.environ.get("LOCAL_STYLE_LORA_SCALE", "0.8"))

_pipe = None
_lock = threading.Lock()


def _load_pipeline():
    """Lazy-loads once, on first request, so the server starts instantly and only
    pays the (large) model-load cost when a sketch is actually needed."""
    global _pipe
    if _pipe is not None:
        return _pipe
    with _lock:
        if _pipe is not None:
            return _pipe

        print(f"[local_image_gen] loading base model {BASE_MODEL} ...")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, variant="fp16"
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )

        print("[local_image_gen] loading SDXL-Lightning speed LoRA ...")
        lightning_path = hf_hub_download(LIGHTNING_REPO, LIGHTNING_CKPT)
        pipe.load_lora_weights(lightning_path, adapter_name="lightning")

        adapters, weights = ["lightning"], [1.0]
        if STYLE_LORA_PATH and os.path.exists(STYLE_LORA_PATH):
            print(f"[local_image_gen] loading your style LoRA: {STYLE_LORA_PATH}")
            pipe.load_lora_weights(STYLE_LORA_PATH, adapter_name="sketch_style")
            adapters.append("sketch_style")
            weights.append(STYLE_LORA_SCALE)
        pipe.set_adapters(adapters, adapter_weights=weights)

        # Keeps VRAM comfortable on 16GB while leaving room for the style LoRA.
        pipe.enable_model_cpu_offload()

        _pipe = pipe
        print("[local_image_gen] ready")
    return _pipe


SKETCH_PROMPT_TEMPLATE = (
    "black and white pencil sketch on white paper of {subject}, {style_notes} "
    "hand drawn, graphite pencil texture, loose expressive linework, sketchbook doodle, "
    "monochrome, simple plain background"
)
NEGATIVE_PROMPT = (
    "color, photo, photorealistic, 3d render, digital art, watermark, text, "
    "signature, blurry, extra limbs, deformed"
)


def generate_sketch_image_local(subject: str, style_notes: str = "") -> dict:
    """Synchronous, GPU-bound — call this via asyncio.to_thread from async code
    so it doesn't block the FastAPI event loop."""
    try:
        pipe = _load_pipeline()
        prompt = SKETCH_PROMPT_TEMPLATE.format(subject=subject, style_notes=style_notes or "")
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=LIGHTNING_STEPS,
            guidance_scale=0.0,  # Lightning-distilled models want CFG off
            height=1024,
            width=1024,
        ).images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return {"mime_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode("utf-8")}
    except Exception as e:
        return {"error": str(e)}