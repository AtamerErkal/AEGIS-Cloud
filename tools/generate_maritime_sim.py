"""
AEGIS-Cloud — Maritime Simulation Video Generator
===================================================
Generates a synthetic sea-surface patrol video for testing the
maritime surveillance pipeline without real hardware.

Scene:
  - Animated sea surface (gradient + wave texture)
  - 2 civilian vessels (brown/white, small-medium)
  - 1 military warship (grey, large, with mast and radar)
  - Drone gimbal: dynamic pan/tilt scan (NOT fixed at 90°)
  - FOV search area drawn on screen as semi-transparent overlay

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
DURATION = 15          # seconds
N_FRAMES = FPS * DURATION

# Pan world scale: how many pixels a vessel shifts per degree of pan
# Camera looks right → vessels shift left (natural projection)
PAN_WORLD_SCALE = 6

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (W, H))

# ── Sea background helpers ────────────────────────────────────────────────────
_RNG = np.random.default_rng(42)
SKY_H   = int(H * 0.40)   # y coordinate of horizon line
SEA_MID = SKY_H + (H - SKY_H) // 2


def make_sea(frame_idx: int) -> np.ndarray:
    """Animated sea surface — gradient + ripple noise."""
    base = np.zeros((H, W, 3), dtype=np.uint8)

    # Sky gradient (top 40%)
    for y in range(SKY_H):
        t = y / SKY_H
        r = int(100 + 40 * t)
        g = int(140 + 40 * t)
        b = int(180 + 50 * t)
        base[y, :] = (b, g, r)

    # Sea gradient (bottom 60%)
    for y in range(SKY_H, H):
        t = (y - SKY_H) / (H - SKY_H)
        r = int(20 + 10 * t)
        g = int(50 + 30 * t)
        b = int(80 + 40 * t)
        base[y, :] = (b, g, r)

    # Wave ripples using sine
    phase = frame_idx * 0.08
    for row_off in range(0, H - SKY_H, 18):
        y = SKY_H + row_off
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
    cv2.line(base, (0, SKY_H), (W, SKY_H), (160, 170, 180), 1)
    return base


# ── Vessel drawing helpers ────────────────────────────────────────────────────

def draw_civilian_vessel(img, cx, cy, length):
    """Small fishing/cargo boat — dark hull, grey superstructure."""
    half_l = length // 2
    half_w = max(4, length // 6)

    pts_hull = np.array([
        [cx - half_l,      cy - half_w],
        [cx + half_l - 4,  cy - half_w],
        [cx + half_l,      cy],
        [cx + half_l - 4,  cy + half_w],
        [cx - half_l,      cy + half_w],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts_hull], (30, 60, 120))

    # Superstructure
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

    pts_hull = np.array([
        [cx - half_l,          cy - half_w],
        [cx + half_l - 6,      cy - half_w],
        [cx + half_l,          cy],
        [cx + half_l - 6,      cy + half_w],
        [cx - half_l,          cy + half_w],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts_hull], (80, 80, 80))

    # Superstructure
    sx = cx - half_l // 3
    cv2.rectangle(img, (sx, cy - half_w - half_w // 2),
                  (sx + half_l // 2, cy + half_w // 2), (100, 100, 100), -1)

    # Radar mast
    mast_x = cx - half_l // 6
    cv2.line(img, (mast_x, cy - half_w - half_w // 2),
             (mast_x, cy - half_w - half_w // 2 - 22), (140, 140, 140), 2)
    cv2.ellipse(img, (mast_x, cy - half_w - half_w // 2 - 22),
                (8, 4), 0, 0, 180, (160, 160, 160), 1)

    # Gun turret
    turret_x = cx + half_l // 4
    cv2.circle(img, (turret_x, cy - half_w // 2), 5, (60, 60, 60), -1)
    cv2.line(img, (turret_x, cy - half_w // 2),
             (turret_x + 14, cy - half_w // 2 - 4), (60, 60, 60), 2)

    # Hull number
    cv2.putText(img, "F-511", (cx - half_l // 2, cy + half_w + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)


def draw_horizon_haze(img):
    """Subtle haze band at horizon."""
    haze = img.copy()
    cv2.rectangle(haze, (0, SKY_H - 8), (W, SKY_H + 8), (180, 185, 190), -1)
    cv2.addWeighted(haze, 0.15, img, 0.85, 0, img)


# ── Camera model ──────────────────────────────────────────────────────────────
# pan  : horizontal rotation (0–180°, 90° = straight ahead)
# elev : elevation angle (0° = horizontal, 90° = straight down)
#
# Search pattern: pan sweeps left-right while elev oscillates 30°–70°.
# This ensures the camera looks FORWARD (toward horizon) and sweeps the sea.

def camera_angles(fi: int):
    """Return (pan_deg, elev_deg) for this frame."""
    t = fi / N_FRAMES
    # Pan: ±35° sweep around centre (90°), period ~8 s
    pan  = 90 + 35 * math.sin(2 * math.pi * t * (DURATION / 8))
    # Elevation: 30°–70° — never goes to 90° (straight down)
    elev = 50 + 20 * math.sin(2 * math.pi * t * (DURATION / 12) + 0.8)
    return pan, elev


# ── FOV projection helpers ────────────────────────────────────────────────────
# Map camera angles to a trapezoid on the 2-D frame.
#
# elev=30° → looking nearly horizontal → FOV spans upper sea (near horizon)
# elev=70° → looking steeply down     → FOV spans lower sea (close range)
#
# The trapezoid is drawn semi-transparently in green.

FOV_H_DEG = 55   # horizontal field-of-view (degrees)
FOV_V_DEG = 40   # vertical field-of-view (degrees)


def world_to_screen_x(world_x: int, pan_deg: float) -> int:
    """Project world-x to screen-x based on camera pan angle.

    Camera panning right (pan > 90°) shifts the world left on screen.
    This keeps the FOV centred on screen while vessels move naturally.
    """
    pan_offset = (pan_deg - 90) * PAN_WORLD_SCALE
    return world_x - int(pan_offset)


def fov_trapezoid(pan_deg: float, elev_deg: float):
    """
    Return the 4 corners of the FOV trapezoid in screen pixels.

    The FOV is always centred horizontally on screen (cx = W//2).
    Pan movement is reflected by shifting vessel positions, NOT the FOV box,
    so detection and visual overlap are always consistent.

    Returns list of (x, y) tuples: [top-left, top-right, bot-right, bot-left]
    """
    sea_top = SKY_H          # y of horizon
    sea_bot = H              # y of bottom edge

    # Elevation → vertical extent in the sea strip
    norm_elev = (elev_deg - 30) / 40   # 0.0 at 30°, 1.0 at 70°
    norm_elev = max(0.0, min(1.0, norm_elev))

    sea_range = sea_bot - sea_top
    fov_cy = sea_top + int(sea_range * (0.15 + 0.60 * norm_elev))
    half_h = int(sea_range * (0.30 - 0.10 * norm_elev))
    top_y = max(sea_top + 2, fov_cy - half_h)
    bot_y = min(sea_bot - 2, fov_cy + half_h)

    # FOV centre is always the screen centre — pan is handled by vessel projection
    cx = W // 2

    # Width: wider at closer range (high elev), narrower at far (perspective)
    half_w_top = int(W * (0.18 + 0.04 * norm_elev))
    half_w_bot = int(W * (0.28 + 0.06 * norm_elev))

    return [
        (cx - half_w_top, top_y),
        (cx + half_w_top, top_y),
        (cx + half_w_bot, bot_y),
        (cx - half_w_bot, bot_y),
    ]


def point_in_poly(px: int, py: int, poly) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def draw_fov_overlay(img: np.ndarray, poly, any_target: bool) -> None:
    """Draw the semi-transparent FOV trapezoid on the image."""
    pts = np.array(poly, dtype=np.int32)

    # Fill — green tint (darker when no target, bright when target acquired)
    overlay = img.copy()
    colour = (0, 180, 0) if any_target else (0, 100, 40)
    cv2.fillPoly(overlay, [pts], colour)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

    # Border
    border_col = (0, 255, 80) if any_target else (0, 160, 60)
    cv2.polylines(img, [pts], isClosed=True, color=border_col, thickness=2,
                  lineType=cv2.LINE_AA)

    # Corner tick marks
    tick = 8
    for (x, y) in poly:
        cv2.line(img, (x - tick, y), (x + tick, y), border_col, 1)
        cv2.line(img, (x, y - tick), (x, y + tick), border_col, 1)


# ── Vessel trajectories ───────────────────────────────────────────────────────
VESSELS = [
    # world_start / world_end are WORLD coordinates (pan-independent).
    # screen_x = world_x - (pan_deg - 90) * PAN_WORLD_SCALE

    # Civilian 1 — small, near horizon, left-to-right
    {"world_start": -60,   "world_end": W + 60,  "y": SKY_H + 55,  "length": 45,  "mil": False, "speed": 2.5, "label": "CIVILIAN-A"},
    # Civilian 2 — medium, mid distance, right-to-left
    {"world_start": W + 60,"world_end": -60,      "y": SKY_H + 140, "length": 65,  "mil": False, "speed": 2.0, "label": "CIVILIAN-B"},
    # Warship — large, enters from right, reaches centre by ~8 s
    {"world_start": W + 80,"world_end": 220,      "y": SKY_H + 200, "length": 160, "mil": True,  "speed": 2.0, "label": "WARSHIP F-511"},
]


# ── Render loop ───────────────────────────────────────────────────────────────
print(f"Generating {N_FRAMES} frames → {OUT_PATH} ...")
for fi in range(N_FRAMES):
    sea = make_sea(fi)
    draw_horizon_haze(sea)

    # Compute camera angles for this frame
    pan_deg, elev_deg = camera_angles(fi)

    # Compute FOV trapezoid
    fov_poly = fov_trapezoid(pan_deg, elev_deg)

    # Build vessel positions for this frame
    vessel_positions = []
    for v in VESSELS:
        # World coordinate (pan-independent trajectory)
        progress = min(fi * v["speed"] / N_FRAMES, 1.0)
        world_cx = int(v["world_start"] + (v["world_end"] - v["world_start"]) * progress)
        # Project to screen using current camera pan
        screen_cx = world_to_screen_x(world_cx, pan_deg)
        cy = v["y"]
        if screen_cx < -300 or screen_cx > W + 300:
            continue
        # Detection: vessel must be visually ON screen AND inside the FOV polygon
        on_screen = -10 <= screen_cx <= W + 10
        in_fov = on_screen and point_in_poly(screen_cx, cy, fov_poly)
        vessel_positions.append({**v, "cx": screen_cx, "cy": cy, "in_fov": in_fov})

    # Check if any vessel is in FOV
    any_target = any(vp["in_fov"] for vp in vessel_positions)

    # Draw FOV overlay (behind vessels)
    draw_fov_overlay(sea, fov_poly, any_target)

    # Draw vessels
    for vp in vessel_positions:
        cx, cy = vp["cx"], vp["cy"]
        if vp["mil"]:
            draw_warship(sea, cx, cy, vp["length"])
        else:
            draw_civilian_vessel(sea, cx, cy, vp["length"])

        if vp["in_fov"]:
            # Detection box
            hl = vp["length"] // 2
            hw = max(8, vp["length"] // 8) + 4
            box_col = (0, 60, 220) if vp["mil"] else (0, 200, 80)
            cv2.rectangle(sea,
                          (cx - hl - 4, cy - hw - 18),
                          (cx + hl + 4, cy + hw + 4),
                          box_col, 2)
            # Label
            label = vp["label"]
            cv2.putText(sea, label,
                        (cx - hl - 2, cy - hw - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        box_col, 1, cv2.LINE_AA)

    # ── HUD ──────────────────────────────────────────────────────────────────
    # Top-left: camera state
    hud_lines = [
        f"PAN  {pan_deg:5.1f} deg",
        f"ELEV {elev_deg:5.1f} deg",
        f"ALT  120 m",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(sea, line, (12, 22 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)

    # Top-right: status
    status_text = "TARGET ACQUIRED" if any_target else "SEARCHING..."
    status_col  = (0, 255, 100) if any_target else (0, 180, 220)
    (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(sea, status_text, (W - tw - 12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_col, 1, cv2.LINE_AA)

    # FOV label at top of trapezoid
    fov_tx, fov_ty = fov_poly[0]
    cv2.putText(sea, "[ SEARCH AREA ]", (fov_tx + 4, fov_ty - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 210, 80), 1, cv2.LINE_AA)

    # Bottom: timestamp bar
    ts = f"AEGIS-MARITIME-001  |  {fi // FPS:02d}s  |  37.915N 26.340E"
    cv2.putText(sea, ts, (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)

    writer.write(sea)
    if fi % 50 == 0:
        print(f"  {fi}/{N_FRAMES}", end="\r", flush=True)

writer.release()
print(f"\nDone → {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")
