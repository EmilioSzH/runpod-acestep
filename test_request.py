"""
Test script for hitting the deployed RunPod serverless endpoint.

Usage:
    export RUNPOD_API_KEY=your_key
    export RUNPOD_ENDPOINT_ID=your_endpoint_id
    python test_request.py "your prompt here" [lora_name] [lora_scale]
"""

import os
import sys
import time
import json
import base64
import requests

API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")

if not API_KEY or not ENDPOINT_ID:
    print("Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID environment variables")
    sys.exit(1)

BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def submit(payload):
    """Submit an async job."""
    r = requests.post(f"{BASE_URL}/run", headers=HEADERS, json={"input": payload})
    r.raise_for_status()
    return r.json()


def poll(job_id, timeout=600):
    """Poll until job completes."""
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        status = data.get("status")
        print(f"  status={status}")
        if status == "COMPLETED":
            return data.get("output")
        if status == "FAILED":
            raise RuntimeError(f"Job failed: {data}")
        time.sleep(2)
    raise TimeoutError(f"Job {job_id} timed out")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_request.py <prompt> [lora_name] [lora_scale]")
        sys.exit(1)

    prompt = sys.argv[1]
    lora_name = sys.argv[2] if len(sys.argv) > 2 else None
    lora_scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    payload = {
        "prompt": prompt,
        "duration": 20,
        "bpm": 120,
        "guidance_scale": 7.0,
    }
    if lora_name:
        payload["lora_name"] = lora_name
        payload["lora_scale"] = lora_scale

    print(f"Submitting: {json.dumps(payload, indent=2)}")
    job = submit(payload)
    job_id = job["id"]
    print(f"Job ID: {job_id}")

    print("Polling...")
    result = poll(job_id)

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    audio_b64 = result.pop("audio_base64")
    print(f"\nResult metadata: {json.dumps(result, indent=2)}")

    # Save audio
    fmt = result.get("format", "mp3")
    filename = f"runpod_test.{fmt}"
    with open(filename, "wb") as f:
        f.write(base64.b64decode(audio_b64))

    print(f"\nSaved {len(base64.b64decode(audio_b64))} bytes to {filename}")


if __name__ == "__main__":
    main()
