# RunPod Serverless ACE-Step 1.5

Hot-swappable LoRA endpoint for ACE-Step XL-turbo on RunPod serverless.

## Architecture

```
┌──────────────────────────────────────────────┐
│  Your VST3 / app                             │
└────────────────┬─────────────────────────────┘
                 │ HTTPS (RunPod API)
                 ▼
┌──────────────────────────────────────────────┐
│  RunPod Serverless Endpoint                   │
│  ├── Worker (24GB GPU, e.g. L40S)            │
│  ├── Docker image (handler.py + ACE-Step)    │
│  └── Network volume (persistent)              │
│       ├── checkpoints/                        │
│       │   ├── acestep-v15-xl-turbo/  (~19GB) │
│       │   ├── vae/                            │
│       │   └── acestep-5Hz-lm-0.6B/            │
│       └── loras/                              │
│           ├── chinese-new-year/               │
│           ├── ambient-dream/                  │
│           ├── raspy-vocal/                    │
│           ├── acoustic-guitar/                │
│           └── symphonic-metal/                │
└──────────────────────────────────────────────┘
```

## Setup (one-time)

### 1. Create a RunPod network volume
- Go to RunPod → Storage → New Network Volume
- Pick the same datacenter your endpoint will run in (e.g. EU-RO)
- Size: **40 GB** (model + LoRAs + headroom)
- Note the volume ID

### 2. Populate the volume
Spin up a cheap pod with the volume mounted:
- Image: any PyTorch image (e.g. `runpod/pytorch:2.7.1-py3.11-cuda12.8.1-cudnn-devel-ubuntu24.04`)
- Mount your network volume at `/runpod-volume`
- Open a web terminal and run:
```bash
cd /workspace
wget https://raw.githubusercontent.com/YOU/runpod-acestep/main/setup_volume.py
python setup_volume.py
```
This downloads ~25 GB to the volume. Takes 10-30 minutes depending on bandwidth.
**Stop the pod when done** — the volume persists.

### 3. Build and push the Docker image

```bash
# Build locally (or use RunPod's GitHub integration)
docker build -t YOUR_DOCKERHUB/acestep-runpod:latest .
docker push YOUR_DOCKERHUB/acestep-runpod:latest
```

Or use RunPod's GitHub repo build feature: connect this repo and let RunPod build from the Dockerfile.

### 4. Create the serverless endpoint
In RunPod Console → Serverless → New Endpoint:
- **Container image:** `YOUR_DOCKERHUB/acestep-runpod:latest`
- **GPU:** L40S 24GB or RTX A5000 24GB (XL needs 20GB+ for no-offload)
- **Network Volume:** the one you created in step 1, mounted at `/runpod-volume`
- **Container disk:** 20 GB
- **Min workers:** 0 (saves money when idle)
- **Max workers:** 1-3
- **Idle timeout:** 30s (shorter = less cost, longer = fewer cold starts)
- **Flashboot:** ENABLE (cuts cold start by ~50%)
- **Execution timeout:** 600s

Note the endpoint ID and copy your API key from RunPod settings.

### 5. Test it

```bash
export RUNPOD_API_KEY=your_key
export RUNPOD_ENDPOINT_ID=your_endpoint_id
python test_request.py "dark ambient pad with reverb" ambient-dream 0.5
```

## API

All requests go to `POST https://api.runpod.ai/v2/{endpoint_id}/run`.

### Generate a song
```json
{
  "input": {
    "action": "generate",
    "prompt": "bouncy trap beat with 808 bass",
    "duration": 30,
    "bpm": 140,
    "key": "Gm",
    "genre": "drill",
    "guidance_scale": 7.0,
    "inference_steps": 8,
    "seed": -1,
    "lora_name": "symphonic-metal",
    "lora_scale": 0.5
  }
}
```

Response:
```json
{
  "audio_base64": "...",
  "format": "mp3",
  "duration": 30,
  "expanded_prompt": "...",
  "seed": "1234567890",
  "model": "acestep-v15-xl-turbo",
  "lora": "symphonic-metal",
  "lora_scale": 0.5,
  "generation_time": 4.2
}
```

### List available LoRAs
```json
{ "input": { "action": "list_loras" } }
```

Response:
```json
{ "loras": ["ambient-dream", "chinese-new-year", "..."] }
```

## LoRA hot-swap behavior

- The handler caches the loaded LoRA in memory between requests on the same warm worker
- If you request the **same** LoRA as the previous call → instant (no reload)
- If you request a **different** LoRA → unloads old, loads new (~5-10s overhead)
- If you request **no LoRA** (omit `lora_name` or set to null) → unloads any active LoRA
- LoRAs **only work without quantization** — the handler runs without INT8/FP8

## Cost estimates (rough, L40S serverless ~$0.86/hr)

| Scenario | Cost per song |
|----------|---------------|
| Cold start + generation (~70s) | ~$0.017 |
| Warm worker, no LoRA swap (~5s) | ~$0.0012 |
| Warm worker, LoRA swap (~12s) | ~$0.0029 |

If you keep `min_workers: 1` (always warm), you pay ~$15-20/month idle but every request is fast.

## Adding new LoRAs

Just upload a new directory under `/runpod-volume/loras/<name>/` containing `adapter_model.safetensors` and `adapter_config.json`. The endpoint will pick it up automatically — no rebuild or redeploy needed.

## Files

- `Dockerfile` — Container with PyTorch 2.7 + CUDA 12.8 + ACE-Step + RunPod SDK
- `handler.py` — RunPod serverless entrypoint with LoRA hot-swap
- `setup_volume.py` — One-time script to populate the network volume
- `test_request.py` — Local test client
