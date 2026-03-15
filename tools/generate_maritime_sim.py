"""
AEGIS-Cloud — Maritime Simulation Video Generator
===================================================
Realistic drone gimbal surveillance simulation.

Camera model (mirrors real pan-tilt gimbal):
  - Scene is a wide panoramic view of the sea.  Vessels move across it.
  - pan/tilt angles control a SMALL FOV window (camera's actual field of view).
  - During SEARCH the gimbal sweeps a raster pattern (left→right, step down,
    right→left, step down …) — like a real search scan.
  - When a military vessel enters the FOV → TRACKING mode: gimbal locks on
    and smoothly follows the target, keeping it centred.

Key principle: the green "SEARCH AREA" box on screen IS where the camera is
looking.  It moves exactly with the pan/tilt values shown in the HUD.

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
DURATION = 18          # seconds
N_FRAMES = FPS * DURATION

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT_PATH), fourcc, FPS, (W, H))

# ── Sea ───────────────────────────────────────────────────────────────────────
SKY_H = int(H * 0.38)


def make_sea(frame_idx: int) -> np.ndarray:
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
        amp = 1.5 + row_off * 0.003
        for x in range(0, W, 2):
            dy = int(amp * math.sin(x * 0.02 + phase + row_off * 0.1))
            wy = min(H - 1, max(0, y + dy))
            base[wy, x] = np.clip(base[wy, x].astype(int) + [15, 15, 10], 0, 255).astype(np.uint8)
    cv2.line(base, (0, SKY_H), (W, SKY_H), (160, 170, 180), 1)
    return base


def draw_horizon_haze(img):
    haze = img.copy()
    cv2.rectangle(haze, (0, SKY_H - 8), (W, SKY_H + 8), (180, 185, 190), -1)
    cv2.addWeighted(haze, 0.15, img, 0.85, 0, img)


# ── Vessel drawing ────────────────────────────────────────────────────────────

def draw_civilian_vessel(img, cx, cy, length):
    hl, hw = length // 2, max(4, length // 6)
    pts = np.array([[cx - hl, cy - hw], [cx + hl - 4, cy - hw],
                    [cx + hl, cy], [cx + hl - 4, cy + hw],
                    [cx - hl, cy + hw]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (30, 60, 120))
    sx = cx - hl // 4
    cv2.rectangle(img, (sx, cy - hw), (sx + hl // 3, cy + hw), (200, 200, 200), -1)
    cv2.line(img, (cx, cy - hw - 1), (cx, cy - hw - 12), (180, 180, 180), 1)


def draw_warship(img, cx, cy, length):
    hl, hw = length // 2, max(8, length // 8)
    pts = np.array([[cx - hl, cy - hw], [cx + hl - 6, cy - hw],
                    [cx + hl, cy], [cx + hl - 6, cy + hw],
                    [cx - hl, cy + hw]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (80, 80, 80))
    sx = cx - hl // 3
    cv2.rectangle(img, (sx, cy - hw - hw // 2),
                  (sx + hl // 2, cy + hw // 2), (100, 100, 100), -1)
    mx = cx - hl // 6
    cv2.line(img, (mx, cy - hw - hw // 2),
             (mx, cy - hw - hw // 2 - 22), (140, 140, 140), 2)
    cv2.ellipse(img, (mx, cy - hw - hw // 2 - 22),
                (8, 4), 0, 0, 180, (160, 160, 160), 1)
    tx = cx + hl // 4
    cv2.circle(img, (tx, cy - hw // 2), 5, (60, 60, 60), -1)
    cv2.line(img, (tx, cy - hw // 2), (tx + 14, cy - hw // 2 - 4), (60, 60, 60), 2)
    cv2.putText(img, "F-511", (cx - hl // 2, cy + hw + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)


# ── Camera / gimbal model ────────────────────────────────────────────────────
#
# The entire visible scene spans:
#   Horizontal (bearing): 0° – 180°  mapped to  x = 0 – W
#   Vertical   (elev)   : 0° – 90°   mapped to  y = SKY_H – H
#
# The gimbal's FOV is SMALL — a realistic camera window.
# During search, the gimbal rasters across the sea.
# Screen coordinate ↔ angle conversion:

BRG_MIN, BRG_MAX = 0.0, 180.0       # bearing range (degrees)
ELEV_MIN, ELEV_MAX = 0.0, 90.0      # elevation range (degrees)

# Camera FOV size (degrees) — small, realistic
CAM_FOV_H = 30.0     # horizontal FOV in degrees
CAM_FOV_V = 22.0     # vertical FOV in degrees

# Search scan limits (degrees) — the area the gimbal sweeps
SCAN_BRG_LO,  SCAN_BRG_HI  = 25.0, 155.0
SCAN_ELEV_LO, SCAN_ELEV_HI = 10.0, 70.0

# Raster scan timing
SCAN_ROW_TIME = 2.5    # seconds per horizontal sweep
SCAN_ROWS     = 3      # number of elevation rows in one full scan
SCAN_PERIOD   = SCAN_ROW_TIME * SCAN_ROWS  # total scan period


def brg_to_x(brg: float) -> int:
    """Convert bearing angle to screen x pixel."""
    return int((brg - BRG_MIN) / (BRG_MAX - BRG_MIN) * W)


def elev_to_y(elev: float) -> int:
    """Convert elevation angle to screen y pixel (high elev = lower on screen)."""
    t = (elev - ELEV_MIN) / (ELEV_MAX - ELEV_MIN)
    return int(SKY_H + t * (H - SKY_H))


def x_to_brg(x: int) -> float:
    """Convert screen x pixel to bearing angle."""
    return BRG_MIN + (x / W) * (BRG_MAX - BRG_MIN)


def y_to_elev(y: int) -> float:
    """Convert screen y pixel to elevation angle."""
    t = max(0.0, min(1.0, (y - SKY_H) / (H - SKY_H)))
    return ELEV_MIN + t * (ELEV_MAX - ELEV_MIN)


def search_angles(fi: int):
    """
    Raster scan pattern.  Returns (bearing_deg, elev_deg) — the gimbal's
    current look direction during search mode.

    Pattern: row 0 sweeps left→right, row 1 right→left, row 2 left→right …
    After all rows → loops back to row 0.
    """
    t_scan = (fi / FPS) % SCAN_PERIOD          # time within one scan cycle
    row    = int(t_scan / SCAN_ROW_TIME)        # which row (0..SCAN_ROWS-1)
    row    = min(row, SCAN_ROWS - 1)
    t_row  = (t_scan - row * SCAN_ROW_TIME) / SCAN_ROW_TIME   # 0..1 within row

    # Smooth the sweep with sinusoidal easing
    t_smooth = 0.5 - 0.5 * math.cos(math.pi * t_row)

    # Horizontal: alternate direction per row
    if row % 2 == 0:
        brg = SCAN_BRG_LO + (SCAN_BRG_HI - SCAN_BRG_LO) * t_smooth
    else:
        brg = SCAN_BRG_HI - (SCAN_BRG_HI - SCAN_BRG_LO) * t_smooth

    # Vertical: step down per row
    elev_step = (SCAN_ELEV_HI - SCAN_ELEV_LO) / max(1, SCAN_ROWS - 1)
    elev = SCAN_ELEV_LO + row * elev_step

    return brg, elev


def fov_rect(brg_deg: float, elev_deg: float):
    """
    Return the 4 corners of the FOV rectangle on screen.
    The rectangle is centred on (brg_deg, elev_deg) with size (CAM_FOV_H, CAM_FOV_V).
    Returns [top-left, top-right, bot-right, bot-left].
    """
    cx = brg_to_x(brg_deg)
    cy = elev_to_y(elev_deg)

    half_w = brg_to_x(BRG_MIN + CAM_FOV_H) // 2
    half_h = (elev_to_y(ELEV_MIN + CAM_FOV_V) - SKY_H) // 2

    x1 = max(0, cx - half_w)
    x2 = min(W, cx + half_w)
    y1 = max(SKY_H, cy - half_h)
    y2 = min(H, cy + half_h)

    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def point_in_rect(px, py, rect):
    """Check if point is inside the FOV rectangle."""
    x1, y1 = rect[0]
    x2, y2 = rect[2]
    return x1 <= px <= x2 and y1 <= py <= y2


def draw_fov_overlay(img, rect, mode):
    """Draw the FOV rectangle overlay. mode: 'search'|'acquired'|'tracking'."""
    pts = np.array(rect, dtype=np.int32)

    if mode == "tracking":
        fill_col, border_col = (0, 80, 200), (30, 140, 255)
    elif mode == "acquired":
        fill_col, border_col = (0, 180, 0), (0, 255, 80)
    else:
        fill_col, border_col = (0, 100, 40), (0, 160, 60)

    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], fill_col)
    cv2.addWeighted(overlay, 0.20, img, 0.80, 0, img)
    cv2.polylines(img, [pts], True, border_col, 2, cv2.LINE_AA)

    # Corner brackets
    x1, y1 = rect[0]; x2, y2 = rect[2]
    bk = 14
    for (cx, cy), (dx, dy) in [((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                                 ((x2, y2), (-1, -1)), ((x1, y2), (1, -1))]:
        cv2.line(img, (cx, cy), (cx + dx * bk, cy), border_col, 2)
        cv2.line(img, (cx, cy), (cx, cy + dy * bk), border_col, 2)


# ── Crosshair for tracking ───────────────────────────────────────────────────

def draw_crosshair(img, cx, cy, size=20):
    """Draw a targeting crosshair at (cx, cy)."""
    col = (30, 140, 255)
    gap = 6
    cv2.line(img, (cx - size, cy), (cx - gap, cy), col, 1, cv2.LINE_AA)
    cv2.line(img, (cx + gap, cy), (cx + size, cy), col, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - size), (cx, cy - gap), col, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + gap), (cx, cy + size), col, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), gap, col, 1, cv2.LINE_AA)


# ── Vessel trajectories ──────────────────────────────────────────────────────
# Screen-space.  Vessels move linearly.  The gimbal sweeps over them.

VESSELS = [
    {"start": -50,    "end": W + 60,  "y": SKY_H + 50,  "length": 40,
     "mil": False, "speed": 1.6, "label": "CIVILIAN-A"},
    {"start": W + 50, "end": -60,     "y": SKY_H + 150, "length": 60,
     "mil": False, "speed": 1.2, "label": "CIVILIAN-B"},
    {"start": W + 60, "end": -100,    "y": SKY_H + 240, "length": 140,
     "mil": True,  "speed": 0.75, "label": "WARSHIP F-511"},
]

# ── State ─────────────────────────────────────────────────────────────────────
tracking   = False
cur_brg    = SCAN_BRG_LO
cur_elev   = SCAN_ELEV_LO

# ── Render ────────────────────────────────────────────────────────────────────
print(f"Generating {N_FRAMES} frames → {OUT_PATH} ...")
for fi in range(N_FRAMES):
    sea = make_sea(fi)
    draw_horizon_haze(sea)

    # ── Vessel positions ──────────────────────────────────────────────────────
    vessel_positions = []
    warship_screen = None

    for v in VESSELS:
        progress = fi * v["speed"] / N_FRAMES
        cx = int(v["start"] + (v["end"] - v["start"]) * progress)
        cy = v["y"]
        if cx < -200 or cx > W + 200:
            continue
        vessel_positions.append({**v, "cx": cx, "cy": cy, "in_fov": False})
        if v["mil"] and 0 <= cx <= W:
            warship_screen = (cx, cy)

    # ── Update gimbal angles ──────────────────────────────────────────────────
    if tracking and warship_screen:
        # Smooth tracking: converge toward target
        target_brg  = x_to_brg(warship_screen[0])
        target_elev = y_to_elev(warship_screen[1])
        cur_brg  += (target_brg  - cur_brg)  * 0.12
        cur_elev += (target_elev - cur_elev) * 0.12
    elif not tracking:
        cur_brg, cur_elev = search_angles(fi)

    # ── FOV rectangle ─────────────────────────────────────────────────────────
    fov = fov_rect(cur_brg, cur_elev)

    # ── Detection ─────────────────────────────────────────────────────────────
    for vp in vessel_positions:
        on_screen = 0 <= vp["cx"] <= W
        vp["in_fov"] = on_screen and point_in_rect(vp["cx"], vp["cy"], fov)
        if vp["mil"] and vp["in_fov"] and not tracking:
            tracking = True

    any_target = any(vp["in_fov"] for vp in vessel_positions)

    if tracking:
        disp_mode = "tracking"
    elif any_target:
        disp_mode = "acquired"
    else:
        disp_mode = "search"

    # ── Draw ──────────────────────────────────────────────────────────────────
    draw_fov_overlay(sea, fov, disp_mode)

    for vp in vessel_positions:
        cx, cy = vp["cx"], vp["cy"]
        if vp["mil"]:
            draw_warship(sea, cx, cy, vp["length"])
        else:
            draw_civilian_vessel(sea, cx, cy, vp["length"])

        if vp["in_fov"]:
            hl = vp["length"] // 2
            hw = max(8, vp["length"] // 8) + 4
            box_col = (0, 60, 220) if vp["mil"] else (0, 200, 80)
            cv2.rectangle(sea,
                          (cx - hl - 4, cy - hw - 18),
                          (cx + hl + 4, cy + hw + 4), box_col, 2)
            cv2.putText(sea, vp["label"],
                        (cx - hl - 2, cy - hw - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_col, 1, cv2.LINE_AA)

    # Crosshair in tracking mode
    if tracking and warship_screen:
        draw_crosshair(sea, *warship_screen)

    # ── HUD ───────────────────────────────────────────────────────────────────
    hud = [
        f"BRG  {cur_brg:6.1f} deg",
        f"ELEV {cur_elev:5.1f} deg",
        f"ALT  120 m",
        f"MODE {'TRACK' if tracking else 'SEARCH'}",
    ]
    for i, line in enumerate(hud):
        col = (30, 140, 255) if (i == 3 and tracking) else (180, 220, 255)
        cv2.putText(sea, line, (12, 22 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    if disp_mode == "tracking":
        st, sc = "TRACKING TARGET", (30, 140, 255)
    elif disp_mode == "acquired":
        st, sc = "TARGET ACQUIRED", (0, 255, 100)
    else:
        st, sc = "SEARCHING...", (0, 180, 220)
    (tw, _), _ = cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(sea, st, (W - tw - 12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, sc, 1, cv2.LINE_AA)

    # FOV label
    lbl_col = (30, 140, 255) if disp_mode == "tracking" else (0, 210, 80)
    lbl_txt = "[ TRACK ]" if disp_mode == "tracking" else "[ SEARCH AREA ]"
    cv2.putText(sea, lbl_txt, (fov[0][0] + 4, fov[0][1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, lbl_col, 1, cv2.LINE_AA)

    # Confidence in tracking mode
    if tracking and warship_screen:
        fov_cx = (fov[0][0] + fov[2][0]) // 2
        fov_cy = (fov[0][1] + fov[2][1]) // 2
        dist = math.hypot(warship_screen[0] - fov_cx, warship_screen[1] - fov_cy)
        max_dist = math.hypot(W // 4, (H - SKY_H) // 4)
        conf = max(0.0, min(100.0, 100.0 * (1.0 - dist / max_dist)))
        conf_txt = f"CONF {conf:4.1f}%"
        cv2.putText(sea, conf_txt, (12, 22 + 4 * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 140, 255), 1, cv2.LINE_AA)

    ts = f"AEGIS-MARITIME-001  |  {fi // FPS:02d}s  |  37.915N 26.340E"
    cv2.putText(sea, ts, (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)

    writer.write(sea)
    if fi % 50 == 0:
        print(f"  {fi}/{N_FRAMES}", end="  ", flush=True)

writer.release()
print(f"\nDone → {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")
