"""
RunPod Serverless handler for ACE-Step 1.5 with hot-swappable LoRA support.

Cold start: loads base model from /runpod-volume/checkpoints
Per request: optionally loads/switches LoRA, generates audio, returns base64 MP3.

Request format:
{
    "input": {
        "prompt": "dark ambient pad with reverb",
        "duration": 30,
        "bpm": 120,
        "key": "Cm",
        "genre": "none",
        "guidance_scale": 7.0,
        "seed": -1,
        "inference_steps": 8,
        "lora_name": "chinese-new-year",   // optional, name of dir under /runpod-volume/loras/
        "lora_scale": 0.5                  // optional, 0.0-1.0
    }
}

Response:
{
    "audio_base64": "...",
    "format": "mp3",
    "duration": 30,
    "seed": 123456,
    "model": "acestep-v15-xl-turbo",
    "lora": "chinese-new-year",
    "lora_scale": 0.5,
    "generation_time": 3.52
}
"""

import os
import sys
import time
import base64
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

import runpod
import torch

# --- Environment / paths ---
VOLUME_ROOT = os.environ.get("ACESTEP_VOLUME_ROOT", "/runpod-volume")
CHECKPOINTS_DIR = os.path.join(VOLUME_ROOT, "checkpoints")
LORAS_DIR = os.path.join(VOLUME_ROOT, "loras")
CONFIG_PATH = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-xl-turbo")
LM_MODEL = os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-0.6B")

# Make sure ACE-Step is importable
sys.path.insert(0, "/app/ACE-Step-1.5")

from acestep.handler import AceStepHandler  # noqa: E402

# --- Globals (loaded once on cold start) ---
_handler: Optional[AceStepHandler] = None
_loaded_loras: Dict[str, str] = {}  # lora_name -> path (PEFT cache)
_current_lora: Optional[str] = None
_current_lora_scale: float = 1.0


# --- Genre / key expansion (mirrors Python and JS clients) ---
GENRE_TAGS = {
    "drill": "uk drill, dark trap, 808 bass, aggressive, minor key",
    "plugg": "plugg, dreamy, melodic trap, spacey, lush",
    "hyperpop": "hyperpop, glitchy, distorted, energetic, experimental",
    "jersey_club": "jersey club, bouncy, dance, club, rhythmic",
    "rage": "rage beat, aggressive, distorted 808, hard trap",
    "lofi": "lo-fi hip hop, chill, jazzy, warm, dusty vinyl",
    "rnb": "r&b, smooth, neo soul, groovy, sensual",
    "indie": "indie, alternative, organic, warm, emotional",
    "house": "house music, four on the floor, groovy, dance, deep house",
    "synthwave": "synthwave, 80s, retro, neon, analog synth, driving",
    "festival_edm": "edm, festival, big room, euphoric, drop, build-up",
    "techno": "techno, dark, minimal, hypnotic, industrial, repetitive",
    "dnb": "drum and bass, breakbeat, heavy bass, fast, energetic",
}

KEY_NAMES = {
    "Cm": "C minor", "Dm": "D minor", "Em": "E minor", "Fm": "F minor",
    "Gm": "G minor", "Am": "A minor", "Bbm": "Bb minor", "Ebm": "Eb minor",
    "C": "C major", "D": "D major", "F": "F major", "G": "G major",
    "A": "A major", "Bb": "Bb major", "Eb": "Eb major", "Ab": "Ab major",
}


def expand_prompt(user_prompt: str, bpm: Optional[int], key: Optional[str], genre: Optional[str]) -> str:
    parts = [user_prompt.strip()]
    if bpm:
        parts.append(f"{bpm} bpm")
    if key and key in KEY_NAMES:
        parts.append(f"{KEY_NAMES[key]} key")
    if genre and genre in GENRE_TAGS:
        parts.append(GENRE_TAGS[genre])
    parts.append("instrumental")
    return ", ".join(parts)


def initialize_model():
    """Cold start: load the base ACE-Step model. Called once per worker."""
    global _handler

    if _handler is not None:
        return

    print(f"[Init] Loading ACE-Step model: {CONFIG_PATH}")
    print(f"[Init] Checkpoints dir: {CHECKPOINTS_DIR}")
    print(f"[Init] LoRAs dir: {LORAS_DIR}")

    if not os.path.isdir(CHECKPOINTS_DIR):
        raise RuntimeError(
            f"Checkpoints directory not found at {CHECKPOINTS_DIR}. "
            f"Make sure the network volume is mounted and contains the model."
        )

    _handler = AceStepHandler()

    # Detect VRAM and decide offload strategy
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[Init] GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")
        # 24GB+ can run XL without offload
        offload = vram_gb < 20.0
    else:
        vram_gb = 0
        offload = True

    print(f"[Init] CPU offload: {offload}")

    status, ok = _handler.initialize_service(
        project_root=VOLUME_ROOT,
        config_path=CONFIG_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_flash_attention=True,
        compile_model=False,
        offload_to_cpu=offload,
        offload_dit_to_cpu=offload,
    )

    print(f"[Init] {status}")
    if not ok:
        raise RuntimeError(f"Failed to initialize model: {status}")

    print("[Init] Model ready")


def ensure_lora(lora_name: Optional[str], lora_scale: float):
    """Load + activate the requested LoRA. Unloads if name is None/empty."""
    global _current_lora, _current_lora_scale, _loaded_loras

    # Unload case
    if not lora_name:
        if _current_lora is not None:
            print(f"[LoRA] Unloading {_current_lora}")
            _handler.unload_lora()
            _current_lora = None
        return

    # Already active and same scale — nothing to do
    if _current_lora == lora_name and _current_lora_scale == lora_scale:
        return

    lora_path = os.path.join(LORAS_DIR, lora_name)
    if not os.path.isdir(lora_path):
        raise ValueError(f"LoRA '{lora_name}' not found at {lora_path}")

    config_file = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(config_file):
        raise ValueError(f"LoRA '{lora_name}' missing adapter_config.json")

    # Load (PEFT will deep-copy the base decoder on first load)
    print(f"[LoRA] Loading {lora_name} from {lora_path}")
    result = _handler.load_lora(lora_path)
    if not result.startswith("✅"):
        raise RuntimeError(result)

    _handler.set_lora_scale(lora_scale)
    _current_lora = lora_name
    _current_lora_scale = lora_scale


def generate(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single generation."""
    prompt = input_data.get("prompt", "").strip()
    if not prompt:
        return {"error": "prompt is required"}

    duration = float(input_data.get("duration", 30))
    bpm = input_data.get("bpm")
    key = input_data.get("key", "")
    genre = input_data.get("genre", "")
    guidance_scale = float(input_data.get("guidance_scale", 7.0))
    inference_steps = int(input_data.get("inference_steps", 8))
    seed = input_data.get("seed", -1)
    use_random = (seed == -1 or seed is None)

    lora_name = input_data.get("lora_name") or None
    lora_scale = float(input_data.get("lora_scale", 1.0))

    task_type = input_data.get("task_type", "text2music")
    track_name = input_data.get("track_name")
    repaint_start = float(input_data.get("repaint_start", 0.0))
    repaint_end = float(input_data.get("repaint_end", 0.0))

    # Handle base64-encoded source audio (for cover/repaint/lego/extract/complete)
    src_audio_path = None
    src_audio_b64 = input_data.get("src_audio_base64")
    if src_audio_b64:
        fmt = input_data.get("src_audio_format", "wav")
        tmp_path = f"/tmp/acestep_src_{int(time.time())}.{fmt}"
        with open(tmp_path, "wb") as f:
            f.write(base64.b64decode(src_audio_b64))
        src_audio_path = tmp_path
        print(f"[Generate] Decoded {len(src_audio_b64)} base64 chars to {tmp_path}")

    # LoRA management
    try:
        ensure_lora(lora_name, lora_scale)
    except Exception as e:
        return {"error": f"LoRA error: {str(e)}"}

    # Expand prompt with metadata
    expanded = expand_prompt(prompt, bpm, key, genre)
    print(f"[Generate] task={task_type} track={track_name} prompt={expanded}")

    start = time.time()
    result = _handler.generate_music(
        captions=expanded,
        lyrics="",
        bpm=bpm,
        audio_duration=duration,
        guidance_scale=guidance_scale,
        inference_steps=inference_steps,
        use_random_seed=use_random,
        seed=seed if not use_random else -1,
        batch_size=1,
        task_type=task_type,
        src_audio=src_audio_path,
        repainting_start=repaint_start,
        repainting_end=repaint_end if repaint_end > 0 else -1,
    )
    elapsed = time.time() - start

    # Clean up temp input file
    if src_audio_path and os.path.exists(src_audio_path):
        try: os.remove(src_audio_path)
        except OSError: pass

    if not result.get("success", False):
        return {"error": result.get("error", "unknown generation error")}

    audios = result.get("audios", [])
    if not audios:
        return {"error": "no audio in result"}

    # Read the first audio file and base64 encode it
    audio_path = audios[0].get("path")
    if not audio_path or not os.path.exists(audio_path):
        return {"error": f"audio file not found: {audio_path}"}

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

    # Determine format from extension
    ext = os.path.splitext(audio_path)[1].lstrip(".").lower() or "mp3"

    # Clean up the file
    try:
        os.remove(audio_path)
    except OSError:
        pass

    return {
        "audio_base64": audio_b64,
        "format": ext,
        "duration": duration,
        "expanded_prompt": expanded,
        "seed": audios[0].get("seed_value", "?"),
        "model": CONFIG_PATH,
        "lora": _current_lora,
        "lora_scale": _current_lora_scale if _current_lora else None,
        "generation_time": round(elapsed, 2),
    }


def list_loras() -> Dict[str, Any]:
    """List available LoRAs in the volume."""
    if not os.path.isdir(LORAS_DIR):
        return {"loras": []}
    loras = []
    for name in sorted(os.listdir(LORAS_DIR)):
        path = os.path.join(LORAS_DIR, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
            loras.append(name)
    return {"loras": loras}


def handler(event):
    """RunPod serverless entrypoint."""
    try:
        # Lazy init on first request to avoid blocking worker startup
        if _handler is None:
            initialize_model()

        input_data = event.get("input", {})

        # Action routing
        action = input_data.get("action", "generate")

        if action == "list_loras":
            return list_loras()
        elif action == "generate":
            return generate(input_data)
        else:
            return {"error": f"unknown action: {action}"}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}


# --- RunPod entrypoint ---
if __name__ == "__main__":
    print("[Startup] RunPod ACE-Step handler starting...")
    # Eager init to warm the worker (cold-start cost paid once)
    try:
        initialize_model()
        print("[Startup] Model initialized, ready for requests")
    except Exception as e:
        print(f"[Startup] WARNING: eager init failed, will retry on first request: {e}")

    runpod.serverless.start({"handler": handler})
