import requests
import base64
import json
import time
import os

# Configuration
IMAGE_PATH = "edge/data/test_image.jpg"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "moondream"

def run_tactical_inference():
    """
    Performs a standalone VLM inference using Moondream.
    Optimized for resource-constrained edge devices (NVIDIA Jetson Nano).
    """
    print(f"[INFO] Initializing AEGIS-Cloud Tactical Reasoning...")
    
    if not os.path.exists(IMAGE_PATH):
        print(f"[ERROR] Local intelligence asset not found: {IMAGE_PATH}")
        return

    try:
        # Step 1: Encode visual data to Base64
        with open(IMAGE_PATH, "rb") as image_file:
            img_payload = base64.b64encode(image_file.read()).decode('utf-8')

        # Step 2: Prepare Tactical Prompt
        payload = {
            "model": MODEL_NAME,
            "prompt": "Act as a defense operator. Provide a concise tactical threat report of this image. Identify objects and risk level.",
            "stream": False,
            "images": [img_payload]
        }

        print(f"[PROCESS] Uplinking to Moondream VLM. Latency expected due to Swap I/O. Waiting...")
        start_ts = time.time()

        # Step 3: Execute POST request (Timeout set to 5 minutes for Cold Starts)
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()

        # Step 4: Output Tactical Report
        execution_time = time.time() - start_ts
        report = response.json().get("response", "Empty response from inference engine.")

        print("\n" + "="*60)
        print(f" TACTICAL INTELLIGENCE REPORT | EXECUTION: {execution_time:.2f}s")
        print("="*60)
        print(report)
        print("="*60 + "\n")

    except requests.exceptions.RequestException as req_err:
        print(f"[CRITICAL] Communication failure with Ollama: {req_err}")
    except Exception as exc:
        print(f"[CRITICAL] System error during inference: {str(exc)}")

if __name__ == "__main__":
    run_tactical_inference()