# -*- coding: utf-8 -*-
"""
AEGIS-Cloud - Maritime Surveillance Pipeline
=============================================
Drone-mounted YOLOv8 detection -> vessel classification -> PCA9685 servo tracking
Scans sea surface for vessels. Locks onto military contacts.

Usage:
  python run_live.py --camera 0          # USB webcam / drone camera
  python run_live.py --video path/to.mp4 # video file
  python run_live.py --gstreamer         # IMX219 CSI camera (Nano)
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── PCA9685 ──────────────────────────────────────────────────────────────────
try:
    from adafruit_servokit import ServoKit
    _kit = ServoKit(channels=16, address=0x40)
    _kit.frequency = 50
    SERVO_OK = True
    print("[SERVO] PCA9685 connected")
except Exception as e:
    SERVO_OK = False
    print(f"[SERVO] PCA9685 not found — angles will be logged only ({e})")

# ── YOLOv8 ───────────────────────────────────────────────────────────────────
# COCO-80 maritime remaps: surface vessel labels normalised for naval context
_VESSEL_ALIASES = {
    "boat":       "vessel",
    "ship":       "vessel",
    "sailboat":   "vessel",
    "surfboard":  "vessel",   # false-positive mitigation — treated as small boat
}

MODEL_OK = False
_model   = None
_backend = "Mock"

# Prefer ONNX Runtime — no PyTorch/CUDA dependency, faster on Nano CPU
try:
    import onnxruntime as _ort  # noqa: F401
    from edge.src.perception.vision_node import _OnnxModel
    _onnx_path = Path("edge/models/yolov8n.onnx")
    if not _onnx_path.exists():
        raise FileNotFoundError(f"{_onnx_path} not found")
    _model   = _OnnxModel(str(_onnx_path))
    MODEL_OK = True
    _backend = "ONNX"
    print(f"[MODEL] YOLOv8 ONNX loaded  providers={_model._sess.get_providers()}")
except Exception as _e_onnx:
    print(f"[MODEL] ONNX unavailable ({_e_onnx}) — trying PyTorch …")

# Fallback: ultralytics PyTorch
if not MODEL_OK:
    try:
        from ultralytics import YOLO
        _model_path = Path("edge/models/yolov8n.pt")
        if not _model_path.exists():
            _model_path.parent.mkdir(parents=True, exist_ok=True)
            print("[MODEL] Downloading yolov8n.pt ...")
        _model   = YOLO(str(_model_path))
        MODEL_OK = True
        _backend = "PyTorch"
        print("[MODEL] YOLOv8 PyTorch loaded")
    except Exception as e:
        print(f"[MODEL] YOLOv8 unavailable — mock detections active ({e})")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TARGET_CLASSES  = {"vessel", "boat", "ship"}   # Maritime surface contacts only
CONF_THRESHOLD  = 0.35                          # Lower threshold — distant vessels at horizon
KP, KD, DEADZONE = 0.4, 0.05, 0.03
SEARCH_SPEED    = 0.4   # rad/s — pan sweep speed in search mode
SEARCH_AMP      = 60.0  # degrees — half-amplitude of search sweep

# Colours (BGR)
C_GREEN  = (0,   220,   0)
C_RED    = (0,    30, 220)
C_YELLOW = (0,   220, 220)
C_CYAN   = (220, 220,   0)
C_WHITE  = (255, 255, 255)
C_DARK   = ( 20,  20,  20)

# ─────────────────────────────────────────────────────────────────────────────
# Servo state
# ─────────────────────────────────────────────────────────────────────────────
_pan_deg        = 90.0
_tilt_deg       = 90.0
_prev_pan_err   = 0.0
_prev_tilt_err  = 0.0
_last_track_t   = time.perf_counter()
_search_t0      = time.perf_counter()


def _write_servo(pan: float, tilt: float) -> None:
    pan  = max(0.0,  min(180.0, pan))
    tilt = max(45.0, min(135.0, tilt))
    if SERVO_OK:
        _kit.servo[0].angle = pan
        _kit.servo[1].angle = tilt


def servo_search() -> tuple:
    """Sinusoidal pan sweep while no target is visible."""
    global _pan_deg, _tilt_deg, _search_t0
    elapsed  = time.perf_counter() - _search_t0
    _pan_deg  = 90.0 + SEARCH_AMP * math.sin(SEARCH_SPEED * elapsed)
    _tilt_deg = 90.0
    _write_servo(_pan_deg, _tilt_deg)
    return _pan_deg, _tilt_deg


def servo_track(bbox: list) -> tuple:
    """PD controller drives pan/tilt toward bbox centre."""
    global _pan_deg, _tilt_deg, _prev_pan_err, _prev_tilt_err, _last_track_t, _search_t0
    now = time.perf_counter()
    dt  = max(now - _last_track_t, 0.001)
    _last_track_t = now
    _search_t0    = now   # reset search phase so sweep restarts smoothly

    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0

    pan_err  = cx - 0.5
    tilt_err = cy - 0.5

    if abs(pan_err) > DEADZONE:
        _pan_deg += KP * pan_err + KD * (pan_err - _prev_pan_err) / dt
        _pan_deg  = max(0.0, min(180.0, _pan_deg))
        _prev_pan_err = pan_err

    if abs(tilt_err) > DEADZONE:
        _tilt_deg += KP * tilt_err + KD * (tilt_err - _prev_tilt_err) / dt
        _tilt_deg  = max(45.0, min(135.0, _tilt_deg))
        _prev_tilt_err = tilt_err

    _write_servo(_pan_deg, _tilt_deg)
    return _pan_deg, _tilt_deg


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────
def _classify_vessel(bbox: list) -> str:
    """
    Heuristic vessel threat classification based on bounding-box size.
    Large, well-defined contacts → warship/patrol_boat.
    Small/distant contacts → vessel (unknown civilian).

    Moondream VLM will later refine this with visual hull analysis.
    """
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    area = bw * bh
    if area > 0.12:          # >12% of frame — large warship/frigate
        return "warship"
    elif area > 0.04:        # 4-12% — patrol boat / coast guard
        return "patrol_boat"
    return "vessel"          # Small/distant — civilian until confirmed otherwise


def detect(frame: np.ndarray, frame_id: int) -> list:
    """Returns list of (cls_name, confidence, [x1,y1,x2,y2] normalised)."""
    if MODEL_OK:
        results = _model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                raw_cls  = r.names[int(box.cls.item())].lower()
                cls_name = _VESSEL_ALIASES.get(raw_cls, raw_cls)
                if cls_name not in TARGET_CLASSES:
                    continue
                bbox = box.xyxyn.tolist()[0]
                cls_name = _classify_vessel(bbox)
                out.append((cls_name, float(box.conf.item()), bbox))
        return out
    # Mock: vessel crossing frame — simulates drone patrol over sea
    t  = frame_id % 300
    cx = 0.05 + t * 0.003
    mock_bbox = [cx - 0.08, 0.42, cx + 0.08, 0.58]
    return [(_classify_vessel(mock_bbox),
             round(0.75 + (frame_id % 5) * 0.02, 2),
             mock_bbox)]


# ─────────────────────────────────────────────────────────────────────────────
# Overlay drawing
# ─────────────────────────────────────────────────────────────────────────────
def _put(img, text, pos, scale=0.55, color=C_WHITE, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)


def draw_overlay(frame: np.ndarray,
                 detections: list,
                 mode: str,
                 pan_deg: float,
                 tilt_deg: float,
                 fps: float,
                 frame_id: int) -> np.ndarray:
    h, w = frame.shape[:2]
    cx_frame, cy_frame = w // 2, h // 2
    out = frame.copy()

    # ── Crosshair at frame centre ──────────────────────────────────────────
    cv2.line(out, (cx_frame - 25, cy_frame), (cx_frame + 25, cy_frame), C_CYAN, 1)
    cv2.line(out, (cx_frame, cy_frame - 25), (cx_frame, cy_frame + 25), C_CYAN, 1)
    cv2.circle(out, (cx_frame, cy_frame), 4, C_CYAN, 1)

    # ── Detection boxes ────────────────────────────────────────────────────
    MILITARY_CLASSES = {"warship", "patrol_boat"}
    best_det = None
    best_military = None
    for cls_name, conf, bbox in detections:
        x1 = int(bbox[0] * w); y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w); y2 = int(bbox[3] * h)
        is_military = cls_name in MILITARY_CLASSES
        color = C_RED if is_military else C_YELLOW
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Confidence bar beneath label
        bar_w = int((x2 - x1) * conf)
        cv2.rectangle(out, (x1, y2 + 2), (x1 + (x2 - x1), y2 + 8), C_DARK, -1)
        cv2.rectangle(out, (x1, y2 + 2), (x1 + bar_w, y2 + 8), color, -1)

        threat_tag = "!! MILITARY !!" if is_military else "CONTACT"
        label = f"{cls_name.upper()}  {conf*100:.0f}%  {threat_tag}"
        _put(out, label, (x1, y1 - 8), scale=0.5, color=color, thickness=2)

        if is_military and (best_military is None or conf > best_military[1]):
            best_military = (cls_name, conf, bbox)
        if best_det is None or conf > best_det[1]:
            best_det = (cls_name, conf, bbox)

    # Military contact takes priority as tracking target
    tracking_target = best_military if best_military else best_det

    # ── Line from crosshair to tracking target ────────────────────────────
    if tracking_target is not None:
        bbox = tracking_target[2]
        tx = int((bbox[0] + bbox[2]) / 2 * w)
        ty = int((bbox[1] + bbox[3]) / 2 * h)
        line_color = C_RED if tracking_target in (best_military,) and best_military else C_YELLOW
        cv2.line(out, (cx_frame, cy_frame), (tx, ty), line_color, 1, cv2.LINE_AA)
        cv2.circle(out, (tx, ty), 5, line_color, -1)

    # ── Top-left status panel ──────────────────────────────────────────────
    panel_h = 105
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (320, panel_h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    # Title line
    _put(out, "AEGIS  MARITIME SURVEILLANCE", (8, 17), scale=0.55, color=C_CYAN, thickness=2)

    military_active = best_military is not None
    mode_color = C_RED if military_active else (C_GREEN if mode == "TRACKING" else C_YELLOW)
    mode_label = "MILITARY LOCK" if military_active else mode
    _put(out, f"MODE : {mode_label}", (8, 37), scale=0.6, color=mode_color, thickness=2)
    _put(out, f"BRG  : {pan_deg:6.1f} deg", (8, 57), color=C_WHITE)
    _put(out, f"ELEV : {tilt_deg:6.1f} deg", (8, 73), color=C_WHITE)
    _put(out, f"SERVO: {'ACTIVE' if SERVO_OK else 'SIM'}", (8, 90), color=C_CYAN)

    # ── Top-right FPS & frame ──────────────────────────────────────────────
    _put(out, f"FPS {fps:4.1f}", (w - 100, 20), color=C_WHITE)
    _put(out, f"F#{frame_id}", (w - 100, 40), color=C_WHITE, scale=0.45)

    # ── Bottom bar: model & target info ───────────────────────────────────
    bar_y = h - 30
    overlay2 = out.copy()
    cv2.rectangle(overlay2, (0, bar_y - 5), (w, h), C_DARK, -1)
    cv2.addWeighted(overlay2, 0.6, out, 0.4, 0, out)

    model_str = f"YOLOv8n[{_backend}]" if MODEL_OK else "MOCK"
    active = tracking_target if tracking_target else best_det
    if active:
        cls_name, conf, bbox = active
        cx_t = (bbox[0] + bbox[2]) / 2
        cy_t = (bbox[1] + bbox[3]) / 2
        is_mil = cls_name in {"warship", "patrol_boat"}
        info_color = C_RED if is_mil else C_YELLOW
        threat = "MILITARY CONTACT" if is_mil else "UNKNOWN VESSEL"
        info = (f"{threat}: {cls_name.upper()}  conf={conf:.2f}  "
                f"brg={pan_deg:.1f}  [{model_str}]")
        _put(out, info, (8, h - 10), scale=0.48, color=info_color)
    else:
        _put(out, f"CONTACT: NONE  [{model_str}]  SCANNING SEA SURFACE...",
             (8, h - 10), scale=0.48, color=C_CYAN)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Video source
# ─────────────────────────────────────────────────────────────────────────────
def open_capture(args):
    if args.gstreamer:
        pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1920, height=1080, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=640, height=360, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=true"
        )
        cap   = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        label = "IMX219 CSI"
    elif args.camera is not None:
        cap   = cv2.VideoCapture(args.camera)
        label = f"Webcam {args.camera}"
    else:
        cap   = cv2.VideoCapture(str(args.video))
        label = str(args.video)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {label}")
        sys.exit(1)

    print(f"[CAMERA] Opened: {label}")
    return cap, label


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--camera",    type=int,  help="Webcam index (e.g. 0)")
    src.add_argument("--video",     type=Path,
                     default=Path("data/sim_samples/maritime_sim.mp4"))
    src.add_argument("--gstreamer", action="store_true")
    parser.add_argument("--headless", action="store_true",
                        help="Disable GUI window (no display needed)")
    args = parser.parse_args()

    cap, label = open_capture(args)

    print("\n" + "=" * 65)
    print("  AEGIS — Maritime Surveillance Pipeline")
    print(f"  Source    : {label}")
    print(f"  Model     : {'YOLOv8n [' + _backend + ']' if MODEL_OK else 'Mock detections'}")
    print(f"  Servo     : {'PCA9685 ACTIVE' if SERVO_OK else 'Simulation (log only)'}")
    print("  Targets   : vessel / patrol_boat / warship")
    print("  Priority  : MILITARY contacts lock immediately")
    print("  Stop      : press Q in the window or Ctrl+C")
    print("=" * 65 + "\n")

    frame_id  = 0
    fps       = 0.0
    fps_t     = time.perf_counter()
    fps_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            t0 = time.perf_counter()

            # ── Step 1: Detect ────────────────────────────────────────────
            detections = detect(frame, frame_id)

            # ── Step 2: Servo ─────────────────────────────────────────────
            # Military contacts (warship/patrol_boat) take tracking priority
            military_dets = [d for d in detections
                             if d[0] in {"warship", "patrol_boat"}]
            if military_dets:
                target = max(military_dets, key=lambda d: d[1])
                pan_deg, tilt_deg = servo_track(target[2])
                mode = "MILITARY LOCK"
            elif detections:
                target = max(detections, key=lambda d: d[1])
                pan_deg, tilt_deg = servo_track(target[2])
                mode = "TRACKING"
            else:
                pan_deg, tilt_deg = servo_search()
                mode = "SEARCHING"

            # ── Step 3: FPS ───────────────────────────────────────────────
            fps_frames += 1
            if fps_frames >= 10:
                fps = fps_frames / (time.perf_counter() - fps_t)
                fps_t      = time.perf_counter()
                fps_frames = 0

            # ── Step 4: Draw & display ────────────────────────────────────
            out = draw_overlay(frame, detections, mode,
                               pan_deg, tilt_deg, fps, frame_id)
            if not args.headless:
                cv2.imshow("AEGIS  |  Live Tracking", out)

            # ── Step 5: Terminal log ──────────────────────────────────────
            all_dets = military_dets if military_dets else detections
            if all_dets:
                best = max(all_dets, key=lambda d: d[1])
                mil_tag = " [!!MILITARY!!]" if best[0] in {"warship", "patrol_boat"} else ""
                print(
                    f"[F{frame_id:04d}] {mode:14s} | "
                    f"{best[0]:12s} {best[1]:.2f}{mil_tag} | "
                    f"brg={pan_deg:6.1f}  elev={tilt_deg:6.1f} | "
                    f"{(time.perf_counter()-t0)*1000:.0f}ms"
                )
            else:
                print(
                    f"[F{frame_id:04d}] {'SCANNING':14s} | "
                    f"brg={pan_deg:6.1f}  elev={tilt_deg:6.1f} | "
                    f"{(time.perf_counter()-t0)*1000:.0f}ms"
                )

            if not args.headless and cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("\n[STOPPED]")
    finally:
        _write_servo(90.0, 90.0)
        cap.release()
        cv2.destroyAllWindows()
        print("[DONE] Servo centred, camera released.")


if __name__ == "__main__":
    main()
