"""
AEGIS-Cloud — Maritime Simulation Video Generator
===================================================
Generates a synthetic sea-surface patrol video for testing the
maritime surveillance pipeline without real hardware.

Scene:
  - Animated sea surface (gradient + wave texture)
  - 2 civilian vessels (brown/white, small-medium)
  - 1 military warship (grey, large, with mast and radar)
  - Drone gimbal motion (slow pan)

Output: data/sim_samples/maritime_sim.mp4
"""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Output ────────────────────────────────────────────────────────────────────
OUT_PATH = Path("data/sim_samples/maritime_sim.mp4")
W, H     = 1280, 720
FPS      = 25
DURATION = 30          # seconds
N_FRAMES = FPS * DURATION

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (W, H))


# ── Sea background helpers ────────────────────────────────────────────────────
_RNG = np.random.default_rng(42)


def make_sea(frame_idx: int) -> np.ndarray:
    """Animated sea surface — gradient + ripple noise."""
    base = np.zeros((H, W, 3), dtype=np.uint8)

    # Sky gradient (top 40%)
    sky_h = int(H * 0.40)
    for y in range(sky_h):
        t = y / sky_h
        r = int(100 + 40 * t)
        g = int(140 + 40 * t)
        b = int(180 + 50 * t)
        base[y, :] = (b, g, r)

    # Sea gradient (bottom 60%)
    for y in range(sky_h, H):
        t = (y - sky_h) / (H - sky_h)
        r = int(20 + 10 * t)
        g = int(50 + 30 * t)
        b = int(80 + 40 * t)
        base[y, :] = (b, g, r)

    # Wave ripples using sine
    phase = frame_idx * 0.08
    for row_off in range(0, H - sky_h, 18):
        y = sky_h + row_off
        if y >= H:
            break
        amplitude = 1.5 + row_off * 0.003
        for x in range(0, W, 2):
            dy = int(amplitude * math.sin(x * 0.02 + phase + row_off * 0.1))
            wy = min(H - 1, max(0, y + dy))
            base[wy, x] = np.clip(
                base[wy, x].astype(int) + [15, 15, 10], 0, 255
            ).astype(np.uint8)

    # Horizon line
    cv2.line(base, (0, sky_h), (W, sky_h), (160, 170, 180), 1)
    return base


# ── Vessel drawing helpers ────────────────────────────────────────────────────

def draw_civilian_vessel(img, cx, cy, length, heading_deg=180):
    """Small fishing/cargo boat — brown hull, white superstructure."""
    half_l = length // 2
    half_w = max(4, length // 6)

    pts_hull = np.array([
        [cx - half_l,      cy - half_w],
        [cx + half_l - 4,  cy - half_w],
        [cx + half_l,      cy],
        [cx + half_l - 4,  cy + half_w],
        [cx - half_l,      cy + half_w],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts_hull], (30, 60, 120))   # dark hull

    # Superstructure (small white box)
    sx = cx - half_l // 4
    sw = half_l // 3
    sh = half_w
    cv2.rectangle(img, (sx, cy - sh), (sx + sw, cy + sh), (200, 200, 200), -1)

    # Mast
    cv2.line(img, (cx, cy - half_w - 1), (cx, cy - half_w - 12), (180, 180, 180), 1)


def draw_warship(img, cx, cy, length):
    """Military warship — grey hull, radar mast, gun turret silhouette."""
    half_l = length // 2
    half_w = max(8, length // 8)

    # Hull
    pts_hull = np.array([
        [cx - half_l,          cy - half_w],
        [cx + half_l - 6,      cy - half_w],
        [cx + half_l,          cy],
        [cx + half_l - 6,      cy + half_w],
        [cx - half_l,          cy + half_w],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts_hull], (80, 80, 80))    # grey hull

    # Superstructure block
    sx = cx - half_l // 3
    cv2.rectangle(img, (sx, cy - half_w - half_w // 2),
                  (sx + half_l // 2, cy + half_w // 2), (100, 100, 100), -1)

    # Radar mast (tall)
    mast_x = cx - half_l // 6
    cv2.line(img, (mast_x, cy - half_w - half_w // 2),
             (mast_x, cy - half_w - half_w // 2 - 22), (140, 140, 140), 2)
    # Radar dish
    cv2.ellipse(img, (mast_x, cy - half_w - half_w // 2 - 22),
                (8, 4), 0, 0, 180, (160, 160, 160), 1)

    # Gun turret silhouette
    turret_x = cx + half_l // 4
    cv2.circle(img, (turret_x, cy - half_w // 2), 5, (60, 60, 60), -1)
    cv2.line(img, (turret_x, cy - half_w // 2),
             (turret_x + 14, cy - half_w // 2 - 4), (60, 60, 60), 2)

    # Hull number
    cv2.putText(img, "F-511", (cx - half_l // 2, cy + half_w + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)


def draw_horizon_haze(img, sky_h):
    """Subtle haze band at horizon."""
    haze = img.copy()
    cv2.rectangle(haze, (0, sky_h - 8), (W, sky_h + 8), (180, 185, 190), -1)
    cv2.addWeighted(haze, 0.15, img, 0.85, 0, img)


# ── Vessel trajectories ───────────────────────────────────────────────────────
# Each vessel: (start_cx, cy, end_cx, length, is_military)
SKY_H   = int(H * 0.40)
SEA_MID = SKY_H + (H - SKY_H) // 2

VESSELS = [
    # Civilian 1 — small, far away (near horizon, small)
    {"start": -80,  "end": W + 80,  "y": SKY_H + 55,  "length": 45,  "mil": False, "speed": 1.5},
    # Civilian 2 — medium, mid distance
    {"start": W + 100, "end": -100, "y": SKY_H + 140, "length": 65,  "mil": False, "speed": 1.2},
    # Warship — large, appears from right at frame 200, moves left
    {"start": W + 180, "end": 200,  "y": SKY_H + 200, "length": 160, "mil": True,  "speed": 0.6},
]


# ── Render loop ───────────────────────────────────────────────────────────────
print(f"Generating {N_FRAMES} frames → {OUT_PATH} ...")
for fi in range(N_FRAMES):
    sea = make_sea(fi)
    draw_horizon_haze(sea, SKY_H)

    for v in VESSELS:
        t = fi / N_FRAMES
        cx = int(v["start"] + (v["end"] - v["start"]) * t * (v["speed"] * N_FRAMES / N_FRAMES * 1.0))
        # More natural: linear interpolation driven by speed
        cx = int(v["start"] + (v["end"] - v["start"]) * (fi * v["speed"] / N_FRAMES))
        cy = v["y"]
        if cx < -300 or cx > W + 300:
            continue
        if v["mil"]:
            draw_warship(sea, cx, cy, v["length"])
        else:
            draw_civilian_vessel(sea, cx, cy, v["length"])

    # Overlay: minimal HUD (timestamp + lat/lon)
    ts = f"AEGIS-MARITIME-001  |  {fi // FPS:02d}s  |  37.915N 26.340E  ALT 120m"
    cv2.putText(sea, ts, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (180, 220, 255), 1, cv2.LINE_AA)

    writer.write(sea)
    if fi % 50 == 0:
        print(f"  {fi}/{N_FRAMES}", end="\r", flush=True)

writer.release()
print(f"\nDone → {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")
