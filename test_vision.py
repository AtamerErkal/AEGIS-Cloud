import requests
import base64
import json
import time
import cv2
import numpy as np

IMAGE_PATH = "data/test_image.jpg"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "moondream"

def run_tactical_inference():
    print(f"[INFO] Initializing AEGIS-Cloud Tactical Reasoning (Optimization: 224px)...")
    
    # Step 1: Resize image manually to 224px (Lighter for Jetson RAM)
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("[ERROR] Image not found!")
        return
    img_resized = cv2.resize(img, (224, 224))
    _, buffer = cv2.imencode(".jpg", img_resized)
    img_payload = base64.b64encode(buffer).decode('utf-8')

    # Step 2: Payload with strict options
    payload = {
        "model": MODEL_NAME,
        "prompt": "What do you see in this mountain lake image? Describe briefly.",
        "stream": False,
        "options": {
            "num_predict": 100, 
            "temperature": 0.2
        },
        "images": [img_payload]
    }

    print(f"[PROCESS] Uplinking to Moondream. Expecting faster results at 224px...")
    start_ts = time.time()

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        data = response.json()
        
        # print(f"DEBUG RAW DATA: {data}") 

        report = data.get("response", "").strip()
        exec_time = time.time() - start_ts

        print("\n" + "="*60)
        print(f" TACTICAL REPORT | EXECUTION: {exec_time:.2f}s")
        print("="*60)
        if report:
            print(report)
        else:
            print("[CRITICAL] Model returned an empty string. Memory pressure suspected.")
        print("="*60 + "\n")

    except Exception as exc:
        print(f"[CRITICAL] Error: {exc}")

if __name__ == "__main__":
    run_tactical_inference()