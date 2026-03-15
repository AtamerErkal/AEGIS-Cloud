"""
AEGIS-Cloud — Maritime Surveillance Simulation
===============================================
Realistic drone gimbal model — suitable for overlay on real footage.

Gimbal model (physical, unambiguous):
  gx ∈ [0,1]  — horizontal look direction: 0=far-left, 1=far-right
  gy ∈ [0,1]  — vertical look direction:   0=horizon,  1=nadir (straight down)

  Screen position of FOV centre = (gx*W,  SKY_H + gy*(H-SKY_H))
  This is exact: the green box centre IS the gimbal look-point.

  HUD display angles:
    PAN  = (gx - 0.5) * 160°   →  -80° (left) … 0° (ahead) … +80° (right)
    TILT = -(15 + gy * 50)°    →  -15° (horizon) … -65° (steep-down)

Search: boustrophedon scan — pan AND tilt move simultaneously every frame.
Track:  α-tracker (Kalman-like) centres target inside FOV, maximises confidence.

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
DURATION = 25          # seconds
N_FRAMES = FPS * DURATION

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(OUT_PATH),
                         cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

# ── Scene layout ──────────────────────────────────────────────────────────────
SKY_H = int(H * 0.38)   # horizon line (y-pixel)
SEA_H = H - SKY_H       # sea pixel height

# ── FOV box size (pixels) ────────────────────────────────────────────────────
# Realistic camera FOV: ~16% screen width, ~22% sea height
FOV_W2 = int(W * 0.08)       # half-width  (= 102 px)
FOV_H2 = int(SEA_H * 0.11)   # half-height (=  49 px)


# ── Gimbal ↔ screen conversion ────────────────────────────────────────────────
def gxy_to_px(gx: float, gy: float) -> tuple[int, int]:
    """Normalised gimbal → screen pixel (this IS the FOV centre)."""
    x = int(max(0, min(W,   gx * W)))
    y = int(max(SKY_H, min(H, SKY_H + gy * SEA_H)))
    return x, y


def px_to_gxy(px: int, py: int) -> tuple[float, float]:
    """Screen pixel → normalised gimbal."""
    return px / W, (py - SKY_H) / SEA_H


def gxy_to_hud(gx: float, gy: float) -> tuple[float, float]:
    """Normalised gimbal → display PAN (°) and TILT (°)."""
    pan  = (gx - 0.5) * 160.0
    tilt = -(15.0 + gy * 50.0)
    return pan, tilt


# ── Boustrophedon search (pan + tilt move simultaneously) ────────────────────
SCAN_ROWS   = 5
ROW_SECS    = 3.0
SCAN_PERIOD = SCAN_ROWS * ROW_SECS   # 15 s full scan

GX_LO, GX_HI = 0.07, 0.93   # horizontal search margins
GY_LO, GY_HI = 0.06, 0.90   # vertical   search range


def search_gxy(t_secs: float) -> tuple[float, float]:
    """
    Boustrophedon scan: both pan AND tilt change every frame.
    Row i sweeps left→right (even) or right→left (odd).
    Tilt descends continuously from GY_LO to GY_HI across all rows.
    """
    t      = t_secs % SCAN_PERIOD
    row_f  = t / ROW_SECS               # fractional row index  0 … SCAN_ROWS
    row    = min(int(row_f), SCAN_ROWS - 1)
    t_norm = row_f - row                # 0 … 1 within this row

    # Pan: cosine-eased sweep, direction alternates per row
    sweep = 0.5 - 0.5 * math.cos(math.pi * t_norm)   # 0 → 1, smooth
    if row % 2 == 0:
        gx = GX_LO + (GX_HI - GX_LO) * sweep
    else:
        gx = GX_HI - (GX_HI - GX_LO) * sweep

    # Tilt: linearly descends WITHIN each row as well (smooth, continuous)
    gy_start = GY_LO + (row / SCAN_ROWS) * (GY_HI - GY_LO)
    gy_end   = GY_LO + ((row + 1) / SCAN_ROWS) * (GY_HI - GY_LO)
    gy = gy_start + (gy_end - gy_start) * t_norm

    # Tiny stabiliser micro-jitter (realistic gimbal noise)
    gy += 0.004 * math.sin(t_secs * 7.3)
    gx += 0.003 * math.sin(t_secs * 5.1)

    return float(np.clip(gx, 0, 1)), float(np.clip(gy, 0, 1))


# ── Sea background ────────────────────────────────────────────────────────────
def make_sea(fi: int) -> np.ndarray:
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Sky gradient (top → horizon)
    for y in range(SKY_H):
        t = y / SKY_H
        img[y, :] = (int(175 + 45*t), int(135 + 40*t), int(95 + 45*t))
    # Sea gradient (horizon → close water)
    for y in range(SKY_H, H):
        t = (y - SKY_H) / SEA_H
        img[y, :] = (int(72 + 45*t), int(45 + 30*t), int(18 + 12*t))
    # Wave shimmer
    phase = fi * 0.07
    for row_off in range(0, SEA_H, 16):
        y = SKY_H + row_off
        if y >= H:
            break
        amp = 1.2 + row_off * 0.003
        for x in range(0, W, 2):
            dy = int(amp * math.sin(x * 0.018 + phase + row_off * 0.09))
            wy = min(H - 1, max(0, y + dy))
            img[wy, x] = np.clip(img[wy, x].astype(int) + [12, 12, 8], 0, 255)
    cv2.line(img, (0, SKY_H), (W, SKY_H), (160, 170, 180), 1)
    # Haze at horizon
    haze = img.copy()
    cv2.rectangle(haze, (0, SKY_H - 6), (W, SKY_H + 10), (178, 183, 190), -1)
    cv2.addWeighted(haze, 0.18, img, 0.82, 0, img)
    return img


# ── Vessel drawing ────────────────────────────────────────────────────────────
def draw_civilian(img, cx, cy, length):
    hl, hw = length // 2, max(4, length // 7)
    pts = np.array([[cx-hl, cy-hw], [cx+hl-4, cy-hw],
                    [cx+hl, cy],    [cx+hl-4, cy+hw],
                    [cx-hl, cy+hw]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (30, 55, 110))
    sx = cx - hl // 4
    cv2.rectangle(img, (sx, cy-hw), (sx+hl//3, cy+hw), (195, 195, 195), -1)
    cv2.line(img, (cx, cy-hw-1), (cx, cy-hw-10), (180, 180, 180), 1)


def draw_warship(img, cx, cy, length):
    hl, hw = length // 2, max(9, length // 8)
    pts = np.array([[cx-hl, cy-hw], [cx+hl-6, cy-hw],
                    [cx+hl, cy],    [cx+hl-6, cy+hw],
                    [cx-hl, cy+hw]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (78, 78, 78))
    # Superstructure
    sx = cx - hl // 3
    cv2.rectangle(img, (sx, cy-hw-hw//2), (sx+hl//2, cy+hw//2), (98, 98, 98), -1)
    # Mast
    mx = cx - hl // 6
    cv2.line(img, (mx, cy-hw-hw//2), (mx, cy-hw-hw//2-24), (135, 135, 135), 2)
    cv2.ellipse(img, (mx, cy-hw-hw//2-24), (9, 4), 0, 0, 180, (155, 155, 155), 1)
    # Gun turret
    tx = cx + hl // 4
    cv2.circle(img, (tx, cy-hw//2), 5, (58, 58, 58), -1)
    cv2.line(img, (tx, cy-hw//2), (tx+15, cy-hw//2-5), (58, 58, 58), 2)
    cv2.putText(img, "F-511", (cx-hl//2, cy+hw+11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (155, 155, 155), 1)


# ── FOV overlay ───────────────────────────────────────────────────────────────
def draw_fov(img, cx, cy, mode: str, lock_frac: float = 1.0):
    """
    Draw FOV box centred exactly on (cx, cy).
    lock_frac: 0→1, used to animate box shrink during lock-on.
    mode: 'search' | 'acquired' | 'tracking'
    """
    fw = int(FOV_W2 * (1.0 + 0.5 * (1.0 - lock_frac)))  # shrinks as lock_frac→1
    fh = int(FOV_H2 * (1.0 + 0.5 * (1.0 - lock_frac)))

    x1, y1 = cx - fw, max(SKY_H, cy - fh)
    x2, y2 = cx + fw, min(H,     cy + fh)

    if mode == "tracking":
        fill, edge = (0, 60, 180), (30, 130, 255)
    elif mode == "acquired":
        fill, edge = (0, 150, 0),  (0, 235, 80)
    else:
        fill, edge = (0, 85, 35),  (0, 165, 60)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
    cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), edge, 2, cv2.LINE_AA)

    # Corner brackets
    bk = 13
    for (bx, by), (dx, dy) in [((x1,y1),(1,1)), ((x2,y1),(-1,1)),
                                 ((x2,y2),(-1,-1)), ((x1,y2),(1,-1))]:
        cv2.line(img, (bx, by), (bx+dx*bk, by),      edge, 2)
        cv2.line(img, (bx, by), (bx,      by+dy*bk), edge, 2)

    # Label above box
    lbl = "[ TRACKING ]" if mode == "tracking" else "[ SEARCH AREA ]"
    cv2.putText(img, lbl, (x1+3, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, edge, 1, cv2.LINE_AA)

    return (x1, y1, x2, y2)


def draw_crosshair(img, cx, cy):
    col, gap, sz = (30, 130, 255), 7, 22
    cv2.line(img, (cx-sz, cy), (cx-gap, cy), col, 1, cv2.LINE_AA)
    cv2.line(img, (cx+gap, cy), (cx+sz, cy),  col, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy-sz), (cx, cy-gap),  col, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy+gap), (cx, cy+sz),  col, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), gap, col, 1, cv2.LINE_AA)


# ── Vessels ───────────────────────────────────────────────────────────────────
# y positions are derived from realistic depression angles at ~150 m altitude
#   dep = atan(150/dist) → y = SKY_H + gy*SEA_H
#   CIVILIAN-A: dep ~12° → gy~0.12, y~328
#   CIVILIAN-B: dep ~28° → gy~0.33, y~421
#   WARSHIP:    dep ~46° → gy~0.55, y~519   ← inside row-2 scan band
VESSELS = [
    {"start": -60,   "end": W+80,  "y": SKY_H+int(SEA_H*0.12),
     "length": 42,  "mil": False, "speed": 1.6, "label": "CARGO-A"},
    {"start": W+70,  "end": -70,   "y": SKY_H+int(SEA_H*0.33),
     "length": 62,  "mil": False, "speed": 1.3, "label": "TANKER-B"},
    {"start": W+100, "end": -120,  "y": SKY_H+int(SEA_H*0.55),
     "length": 145, "mil": True,  "speed": 0.85, "label": "WARSHIP F-511"},
]

# ── State ─────────────────────────────────────────────────────────────────────
gimbal_gx: float = GX_LO    # current gimbal look direction (normalised)
gimbal_gy: float = GY_LO

tracking      = False
lock_timer    = 0          # frames elapsed since first detection
LOCK_FRAMES   = 12         # frames to complete lock-on animation
track_conf    = 0.0        # 0–100 confidence
warship_pos: tuple[int,int] | None = None

# ── Render ────────────────────────────────────────────────────────────────────
print(f"Generating {N_FRAMES} frames → {OUT_PATH} ...")
for fi in range(N_FRAMES):
    t_secs = fi / FPS
    sea = make_sea(fi)

    # ── Vessel positions ──────────────────────────────────────────────────────
    vessel_list = []
    warship_pos = None
    for v in VESSELS:
        progress = (fi * v["speed"]) / N_FRAMES
        cx = int(v["start"] + (v["end"] - v["start"]) * progress)
        cy = int(v["y"])
        if cx < -250 or cx > W + 250:
            continue
        vessel_list.append({**v, "cx": cx, "cy": cy})
        if v["mil"] and 0 <= cx <= W:
            warship_pos = (cx, cy)

    # ── Gimbal update ─────────────────────────────────────────────────────────
    if tracking and warship_pos is not None:
        # α-tracker: converge toward target centre
        tgt_gx, tgt_gy = px_to_gxy(*warship_pos)
        alpha = 0.14
        gimbal_gx += (tgt_gx - gimbal_gx) * alpha
        gimbal_gy += (tgt_gy - gimbal_gy) * alpha
        gimbal_gx = float(np.clip(gimbal_gx, 0, 1))
        gimbal_gy = float(np.clip(gimbal_gy, 0, 1))
        lock_timer += 1
    else:
        gimbal_gx, gimbal_gy = search_gxy(t_secs)

    # ── FOV centre (screen pixels) ────────────────────────────────────────────
    # This IS where the gimbal is pointing — no offset, no drift
    fov_cx, fov_cy = gxy_to_px(gimbal_gx, gimbal_gy)

    # ── Detection: is warship inside FOV? ─────────────────────────────────────
    for vp in vessel_list:
        inside = (abs(vp["cx"] - fov_cx) <= FOV_W2 and
                  abs(vp["cy"] - fov_cy) <= FOV_H2)
        vp["in_fov"] = inside
        if vp["mil"] and inside and not tracking:
            tracking = True

    # ── Display mode ──────────────────────────────────────────────────────────
    lock_frac = min(1.0, lock_timer / LOCK_FRAMES)   # 0→1 during lock animation
    if tracking:
        disp_mode = "tracking"
    elif any(vp.get("in_fov") for vp in vessel_list):
        disp_mode = "acquired"
    else:
        disp_mode = "search"

    # ── Draw FOV box ──────────────────────────────────────────────────────────
    fov_rect = draw_fov(sea, fov_cx, fov_cy, disp_mode, lock_frac)

    # ── Draw vessels ──────────────────────────────────────────────────────────
    for vp in vessel_list:
        if vp["mil"]:
            draw_warship(sea, vp["cx"], vp["cy"], vp["length"])
        else:
            draw_civilian(sea, vp["cx"], vp["cy"], vp["length"])

        if vp.get("in_fov"):
            hl = vp["length"] // 2
            hw = max(9, vp["length"] // 8) + 5
            box_col = (0, 50, 215) if vp["mil"] else (0, 195, 80)
            cv2.rectangle(sea,
                          (vp["cx"]-hl-4, vp["cy"]-hw-16),
                          (vp["cx"]+hl+4, vp["cy"]+hw+4), box_col, 2)
            cv2.putText(sea, vp["label"],
                        (vp["cx"]-hl, vp["cy"]-hw-19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, box_col, 1, cv2.LINE_AA)

    # ── Crosshair (tracking mode) ─────────────────────────────────────────────
    if tracking and lock_frac >= 1.0 and warship_pos:
        draw_crosshair(sea, warship_pos[0], warship_pos[1])

    # ── Confidence (tracking mode) ────────────────────────────────────────────
    if tracking and warship_pos:
        dx = (warship_pos[0] - fov_cx) / FOV_W2
        dy = (warship_pos[1] - fov_cy) / FOV_H2
        dist_norm = min(1.0, math.sqrt(dx*dx + dy*dy))
        instant = 100.0 * (1.0 - dist_norm) * lock_frac
        track_conf += (instant - track_conf) * 0.08

    # ── HUD ───────────────────────────────────────────────────────────────────
    pan_deg, tilt_deg = gxy_to_hud(gimbal_gx, gimbal_gy)

    hud_lines = [
        (f"PAN   {pan_deg:+6.1f}\u00b0",
         (30, 130, 255) if tracking else (170, 215, 255)),
        (f"TILT  {tilt_deg:6.1f}\u00b0",
         (30, 130, 255) if tracking else (170, 215, 255)),
        (f"ALT    150 m",  (170, 215, 255)),
        (f"MODE  {'TRACK ' if tracking else 'SEARCH'}",
         (30, 130, 255) if tracking else (0, 195, 100)),
    ]
    if tracking:
        hud_lines.append((f"CONF  {track_conf:5.1f}%",
                          (30, 130, 255)))
    for i, (txt, col) in enumerate(hud_lines):
        cv2.putText(sea, txt, (12, 22 + i * 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, col, 1, cv2.LINE_AA)

    # Status top-right
    if disp_mode == "tracking":
        st, sc = "TRACKING TARGET", (30, 130, 255)
    elif disp_mode == "acquired":
        st, sc = "TARGET ACQUIRED", (0, 235, 80)
    else:
        st, sc = "SEARCHING...", (0, 180, 220)
    (tw, _), _ = cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(sea, st, (W - tw - 12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, sc, 1, cv2.LINE_AA)

    # Timestamp / coordinates
    ts = f"AEGIS-001  |  {t_secs:05.1f}s  |  37\u00b055'06\"N  26\u00b020'24\"E"
    cv2.putText(sea, ts, (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 210, 255), 1, cv2.LINE_AA)

    writer.write(sea)
    if fi % 75 == 0:
        print(f"  {fi}/{N_FRAMES} ({t_secs:.1f}s)  PAN={pan_deg:+.1f}  "
              f"TILT={tilt_deg:.1f}  mode={disp_mode}", flush=True)

writer.release()
print(f"\nDone → {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")
