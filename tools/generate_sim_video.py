"""
AEGIS-Cloud — Simulation Video Generator
==========================================
Generates a synthetic drone flyby video for testing without a camera.
The video contains a moving drone-like object on a sky background.

Run:
    python tools/generate_sim_video.py

Output:
    data/sim_samples/drone_flyby.mp4
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

OUTPUT_PATH = Path("data/sim_samples/drone_flyby.mp4")
WIDTH, HEIGHT = 640, 480
FPS = 30
DURATION_S = 20    # seconds


def _sky_background(h, w):
    """Gradient sky background (dark blue to light blue)."""
    bg = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / h
        # top: dark blue  ->  bottom: lighter haze
        bg[y, :] = [
            int(80  + 60 * t),   # B
            int(60  + 40 * t),   # G
            int(30  + 20 * t),   # R
        ]
    return bg


def _draw_drone(frame, cx, cy, size=18, angle_deg=0.0):
    """Draw a simple 4-rotor drone silhouette."""
    # Body
    cv2.rectangle(frame,
                  (int(cx - size * 0.3), int(cy - size * 0.15)),
                  (int(cx + size * 0.3), int(cy + size * 0.15)),
                  (40, 40, 40), -1)
    # Arms + rotors
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        arm_ex = int(cx + dx * size * 0.8)
        arm_ey = int(cy + dy * size * 0.6)
        cv2.line(frame, (int(cx), int(cy)), (arm_ex, arm_ey), (50, 50, 50), 2)
        cv2.circle(frame, (arm_ex, arm_ey), int(size * 0.28), (30, 30, 30), -1)
        cv2.circle(frame, (arm_ex, arm_ey), int(size * 0.28), (80, 80, 80), 1)
    # Blinking LED
    cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 200), -1)


def _draw_noise(frame, intensity=6):
    """Add subtle grain to make the background feel real."""
    noise = np.random.randint(-intensity, intensity,
                               frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


def generate():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (WIDTH, HEIGHT))

    if not writer.isOpened():
        print(f"[ERROR] Cannot open VideoWriter for {OUTPUT_PATH}")
        sys.exit(1)

    total_frames = FPS * DURATION_S
    print(f"Generating {DURATION_S}s @ {FPS}fps = {total_frames} frames ...")

    bg = _sky_background(HEIGHT, WIDTH)

    for i in range(total_frames):
        t = i / FPS

        # Drone flight path: figure-eight Lissajous curve
        cx = WIDTH  / 2 + WIDTH  * 0.32 * math.sin(2 * math.pi * 0.18 * t)
        cy = HEIGHT / 2 + HEIGHT * 0.22 * math.sin(2 * math.pi * 0.09 * t)

        # Size varies slightly to simulate depth change
        size = 18 + 6 * math.sin(2 * math.pi * 0.05 * t)

        frame = bg.copy()
        _draw_drone(frame, cx, cy, size=size, angle_deg=t * 30)
        frame = _draw_noise(frame)

        writer.write(frame)

        if i % (FPS * 5) == 0:
            print(f"  {i // FPS}s / {DURATION_S}s")

    writer.release()
    print(f"\n[DONE] Video saved -> {OUTPUT_PATH}")
    print(f"       Run: python run_live.py --video {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
