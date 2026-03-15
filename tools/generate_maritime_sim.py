"""
AEGIS-Cloud — Maritime Simulation Video Generator
===================================================
Generates a synthetic sea-surface patrol video for testing the
maritime surveillance pipeline without real hardware.

Scene:
  - Animated sea surface (gradient + wave texture)
  - 2 civilian vessels (brown/white, small-medium)
  - 1 military warship (grey, large, with mast and radar)
  - Drone gimbal: dynamic pan/tilt scan
  - FOV search area moves IN SYNC with pan/tilt servos
  - When warship is detected → camera locks on and tracks it

Camera model:
  - The scene (sea + vessels) is static on screen.
  - pan/tilt angles control WHERE the FOV trapezoid is drawn.
  - pan 90° = FOV centred; pan 55° = FOV left; pan 125° = FOV right.
  - Detection = vessel inside the moving FOV trapezoid AND visible on screen.
  - Tracking = once warship detected, pan/tilt smoothly follows it.

Output: data/sim_samples/maritime_sim.mp4
"""

import math
from pathlib import Path

import cv2
import numpy as np

# ── Output ────────────────────────────────────────────────────────────────────
OUT_PATH = Path("data/sim_samples/maritime_sim.mp4")
W, H     = 1280, 720
FPS      = 25
DURATION = 20          # seconds  (search ~6s → detection → tracking ~14s)
N_FRAMES = FPS * DURATION

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (W, H))

# ── Sea background helpers ────────────────────────────────────────────────────
_RNG  = np.random.default_rng(42)
SKY_H = int(H * 0.40)   # y coordinate of horizon line


def make_sea(frame_idx: int) -> np.ndarray:
    """Animated sea surface — gradient + ripple noise."""
    base = np.zeros((H, W, 3), dtype=np.uint8)

    for y in range(SKY_H):
        t = y / SKY_H
        base[y, :] = (int(180 + 50 * t), int(140 + 40 * t), int(100 + 40 * t))

    for y in range(SKY_H, H):
        t = (y - SKY_H) / (H - SKY_H)
        base[y, :] = (int(80 + 40 * t), int(50 + 30 * t), int(20 + 10 * t))

    phase = frame_idx * 0.08
    for row_off in range(0, H - SKY_H, 18):
        y = SKY_H + row_off
        if y >= H:
            break
        amplitude = 1.5 + row_off * 0.003
        for x in range(0, W, 2):
            dy  = int(amplitude * math.sin(x * 0.02 + phase + row_off * 0.1))
            wy  = min(H - 1, max(0, y + dy))
            base[wy, x] = np.clip(base[wy, x].astype(int) + [15, 15, 10], 0, 255).astype(np.uint8)

    cv2.line(base, (0, SKY_H), (W, SKY_H), (160, 170, 180), 1)
    return base


# ── Vessel drawing helpers ────────────────────────────────────────────────────

def draw_civilian_vessel(img, cx, cy, length):
    half_l = length // 2
    half_w = max(4, length // 6)
    pts = np.array([
        [cx - half_l, cy - half_w], [cx + half_l - 4, cy - half_w],
        [cx + half_l, cy],          [cx + half_l - 4, cy + half_w],
        [cx - half_l, cy + half_w],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], (30, 60, 120))
    sx = cx - half_l // 4
    cv2.rectangle(img, (sx, cy - half_w), (sx + half_l // 3, cy + half_w), (200, 200, 200), -1)
    cv2.line(img, (cx, cy - half_w - 1), (cx, cy - half_w - 12), (180, 180, 180), 1)


def draw_warship(img, cx, cy, length):
    half_l = length // 2
    half_w = max(8, length // 8)
    pts = np.array([
        [cx - half_l,     cy - half_w], [cx + half_l - 6, cy - half_w],
        [cx + half_l,     cy],          [cx + half_l - 6, cy + half_w],
        [cx - half_l,     cy + half_w],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], (80, 80, 80))
    sx = cx - half_l // 3
    cv2.rectangle(img, (sx, cy - half_w - half_w // 2),
                  (sx + half_l // 2, cy + half_w // 2), (100, 100, 100), -1)
    mast_x = cx - half_l // 6
    cv2.line(img, (mast_x, cy - half_w - half_w // 2),
             (mast_x, cy - half_w - half_w // 2 - 22), (140, 140, 140), 2)
    cv2.ellipse(img, (mast_x, cy - half_w - half_w // 2 - 22),
                (8, 4), 0, 0, 180, (160, 160, 160), 1)
    tx = cx + half_l // 4
    cv2.circle(img, (tx, cy - half_w // 2), 5, (60, 60, 60), -1)
    cv2.line(img, (tx, cy - half_w // 2), (tx + 14, cy - half_w // 2 - 4), (60, 60, 60), 2)
    cv2.putText(img, "F-511", (cx - half_l // 2, cy + half_w + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)


def draw_horizon_haze(img):
    haze = img.copy()
    cv2.rectangle(haze, (0, SKY_H - 8), (W, SKY_H + 8), (180, 185, 190), -1)
    cv2.addWeighted(haze, 0.15, img, 0.85, 0, img)


# ── Camera / servo model ──────────────────────────────────────────────────────
#
#  pan  : 0–180°  (90° = straight ahead)
#  elev : 30–70°  (30° = horizon, 70° = steep down)
#
#  The FOV trapezoid IS the search area.  It moves on screen as pan/elev change.
#  Vessels are drawn at fixed screen coordinates; detection fires when a vessel
#  falls inside the moving FOV trapezoid.
#
#  PAN_SCREEN_SCALE: how many pixels the FOV centre shifts per degree of pan.
#  At ±35° the FOV centre travels ±35 % of screen width.

PAN_CENTER       = 90.0
PAN_SWING        = 35.0
ELEV_CENTER      = 50.0
ELEV_SWING       = 20.0
PAN_SCREEN_SCALE = (W * 0.35) / PAN_SWING   # ≈ 12.8 px / degree


def search_pan_elev(fi: int):
    """Search-sweep angles for frame fi."""
    t    = fi / N_FRAMES
    pan  = PAN_CENTER + PAN_SWING  * math.sin(2 * math.pi * t * (DURATION / 8))
    elev = ELEV_CENTER + ELEV_SWING * math.sin(2 * math.pi * t * (DURATION / 12) + 0.8)
    return pan, elev


def pan_elev_for_target(cx: int, cy: int):
    """Compute the pan/elev needed to centre the FOV on screen point (cx, cy)."""
    pan      = PAN_CENTER + (cx - W // 2) / PAN_SCREEN_SCALE
    cy_norm  = max(0.0, min(1.0, (cy - SKY_H) / (H - SKY_H)))
    elev     = 30.0 + cy_norm * 40.0
    return pan, elev


# ── FOV trapezoid ─────────────────────────────────────────────────────────────

def fov_trapezoid(pan_deg: float, elev_deg: float):
    """
    Build the FOV trapezoid that moves with the gimbal.

    - Horizontal centre follows pan_deg.
    - Vertical position / size follows elev_deg.
    Returns [top-left, top-right, bot-right, bot-left] in screen pixels.
    """
    sea_top = SKY_H
    sea_bot = H

    norm_elev = max(0.0, min(1.0, (elev_deg - 30) / 40))
    sea_range = sea_bot - sea_top

    fov_cy = sea_top + int(sea_range * (0.15 + 0.60 * norm_elev))
    half_h = int(sea_range * (0.30 - 0.10 * norm_elev))
    top_y  = max(sea_top + 2, fov_cy - half_h)
    bot_y  = min(sea_bot - 2, fov_cy + half_h)

    # Horizontal centre tracks pan
    cx = W // 2 + int((pan_deg - PAN_CENTER) * PAN_SCREEN_SCALE)

    half_w_top = int(W * (0.18 + 0.04 * norm_elev))
    half_w_bot = int(W * (0.28 + 0.06 * norm_elev))

    return [
        (cx - half_w_top, top_y),
        (cx + half_w_top, top_y),
        (cx + half_w_bot, bot_y),
        (cx - half_w_bot, bot_y),
    ]


def point_in_poly(px: int, py: int, poly) -> bool:
    n, inside, j = len(poly), False, len(poly) - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and px < (xj - xi) * (py - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def draw_fov_overlay(img: np.ndarray, poly, mode: str) -> None:
    """Draw the semi-transparent FOV trapezoid. mode: 'search'|'acquired'|'tracking'."""
    pts = np.array(poly, dtype=np.int32)

    if mode == "tracking":
        fill_col   = (0, 80, 200)
        border_col = (30, 140, 255)
    elif mode == "acquired":
        fill_col   = (0, 180, 0)
        border_col = (0, 255, 80)
    else:
        fill_col   = (0, 100, 40)
        border_col = (0, 160, 60)

    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], fill_col)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    cv2.polylines(img, [pts], isClosed=True, color=border_col, thickness=2,
                  lineType=cv2.LINE_AA)
    tick = 8
    for (x, y) in poly:
        cv2.line(img, (x - tick, y), (x + tick, y), border_col, 1)
        cv2.line(img, (x, y - tick), (x, y + tick), border_col, 1)


# ── Vessel trajectories (screen-space, no pan projection) ────────────────────
#
#  Vessels move linearly across the fixed scene.
#  The camera pan/tilt sweeps the FOV over them — not the other way round.

VESSELS = [
    # Civilian 1 — small, near horizon, left → right
    {"start": -60,    "end": W + 80,  "y": SKY_H + 55,  "length": 45,  "mil": False, "speed": 1.8, "label": "CIVILIAN-A"},
    # Civilian 2 — medium, mid distance, right → left
    {"start": W + 60, "end": -80,     "y": SKY_H + 140, "length": 65,  "mil": False, "speed": 1.4, "label": "CIVILIAN-B"},
    # Warship — enters from right, crosses screen slowly
    {"start": W + 80, "end": -200,    "y": SKY_H + 200, "length": 160, "mil": True,  "speed": 0.9, "label": "WARSHIP F-511"},
]

# ── Camera state ──────────────────────────────────────────────────────────────
tracking  = False          # True once warship is detected
cur_pan, cur_elev = search_pan_elev(0)

# ── Render loop ───────────────────────────────────────────────────────────────
print(f"Generating {N_FRAMES} frames → {OUT_PATH} ...")
for fi in range(N_FRAMES):
    sea = make_sea(fi)
    draw_horizon_haze(sea)

    # ── Vessel screen positions (independent of camera pan) ───────────────────
    vessel_positions = []
    warship_screen   = None          # (cx, cy) of warship if on screen

    for v in VESSELS:
        progress = fi * v["speed"] / N_FRAMES
        cx = int(v["start"] + (v["end"] - v["start"]) * progress)
        cy = v["y"]
        if cx < -300 or cx > W + 300:
            continue
        vessel_positions.append({**v, "cx": cx, "cy": cy, "in_fov": False})
        if v["mil"] and 0 <= cx <= W:
            warship_screen = (cx, cy)

    # ── Update pan / elev ─────────────────────────────────────────────────────
    if tracking and warship_screen:
        # Smoothly track the warship
        tp, te   = pan_elev_for_target(*warship_screen)
        cur_pan  += (tp - cur_pan)  * 0.10
        cur_elev += (te - cur_elev) * 0.10
    elif not tracking:
        # Search sweep
        cur_pan, cur_elev = search_pan_elev(fi)
    # else: tracking=True but warship off-screen → hold last angles

    # ── FOV trapezoid (moves with pan/elev) ───────────────────────────────────
    fov_poly = fov_trapezoid(cur_pan, cur_elev)

    # ── Detection ─────────────────────────────────────────────────────────────
    for vp in vessel_positions:
        on_screen    = 0 <= vp["cx"] <= W
        vp["in_fov"] = on_screen and point_in_poly(vp["cx"], vp["cy"], fov_poly)
        if vp["mil"] and vp["in_fov"] and not tracking:
            tracking = True   # lock on first detection of warship

    any_target = any(vp["in_fov"] for vp in vessel_positions)

    # ── Determine display mode ────────────────────────────────────────────────
    if tracking:
        disp_mode = "tracking"
    elif any_target:
        disp_mode = "acquired"
    else:
        disp_mode = "search"

    # ── Draw scene ────────────────────────────────────────────────────────────
    draw_fov_overlay(sea, fov_poly, disp_mode)

    for vp in vessel_positions:
        cx, cy = vp["cx"], vp["cy"]
        if vp["mil"]:
            draw_warship(sea, cx, cy, vp["length"])
        else:
            draw_civilian_vessel(sea, cx, cy, vp["length"])

        if vp["in_fov"]:
            hl      = vp["length"] // 2
            hw      = max(8, vp["length"] // 8) + 4
            box_col = (0, 60, 220) if vp["mil"] else (0, 200, 80)
            cv2.rectangle(sea,
                          (cx - hl - 4, cy - hw - 18),
                          (cx + hl + 4, cy + hw + 4),
                          box_col, 2)
            cv2.putText(sea, vp["label"],
                        (cx - hl - 2, cy - hw - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_col, 1, cv2.LINE_AA)

    # ── HUD ──────────────────────────────────────────────────────────────────
    hud_lines = [
        f"PAN  {cur_pan:5.1f} deg",
        f"ELEV {cur_elev:5.1f} deg",
        f"ALT  120 m",
        f"MODE {'TRACK' if tracking else 'SEARCH'}",
    ]
    for i, line in enumerate(hud_lines):
        col = (30, 140, 255) if (i == 3 and tracking) else (180, 220, 255)
        cv2.putText(sea, line, (12, 22 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    if disp_mode == "tracking":
        status_text = "TRACKING TARGET"
        status_col  = (30, 140, 255)
    elif disp_mode == "acquired":
        status_text = "TARGET ACQUIRED"
        status_col  = (0, 255, 100)
    else:
        status_text = "SEARCHING..."
        status_col  = (0, 180, 220)

    (tw, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(sea, status_text, (W - tw - 12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_col, 1, cv2.LINE_AA)

    # FOV label (colour matches mode)
    fov_label_col = (30, 140, 255) if disp_mode == "tracking" else (0, 210, 80)
    fov_label     = "[ TRACK AREA ]" if disp_mode == "tracking" else "[ SEARCH AREA ]"
    fov_tx, fov_ty = fov_poly[0]
    cv2.putText(sea, fov_label, (fov_tx + 4, fov_ty - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, fov_label_col, 1, cv2.LINE_AA)

    ts = f"AEGIS-MARITIME-001  |  {fi // FPS:02d}s  |  37.915N 26.340E"
    cv2.putText(sea, ts, (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)

    writer.write(sea)
    if fi % 50 == 0:
        print(f"  {fi}/{N_FRAMES}", end="\r", flush=True)

writer.release()
print(f"\nDone → {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")
