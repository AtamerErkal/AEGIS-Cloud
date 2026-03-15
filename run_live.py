"""
AEGIS-Cloud — Live Tracking Pipeline
======================================
Real video -> YOLOv8 detection -> PCA9685 servo tracking
Displays a full tactical overlay on the connected screen.

Usage:
  python run_live.py --camera 0          # USB webcam
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
try:
    from ultralytics import YOLO
    _model_path = Path("edge/models/yolov8n.pt")
    if not _model_path.exists():
        _model_path.parent.mkdir(parents=True, exist_ok=True)
        print("[MODEL] Downloading yolov8n.pt ...")
    _model = YOLO(str(_model_path))
    MODEL_OK = True
    print("[MODEL] YOLOv8 loaded")
except Exception as e:
    MODEL_OK = False
    _model = None
    print(f"[MODEL] YOLOv8 unavailable — mock detections active ({e})")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TARGET_CLASSES  = {"drone", "person", "vehicle", "car", "truck",
                   "bus", "motorcycle", "boat", "ship"}
CONF_THRESHOLD  = 0.45
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
def detect(frame: np.ndarray, frame_id: int) -> list:
    """Returns list of (cls_name, confidence, [x1,y1,x2,y2] normalised)."""
    if MODEL_OK:
        results = _model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls.item())].lower()
                if cls_name not in TARGET_CLASSES:
                    continue
                out.append((cls_name, float(box.conf.item()), box.xyxyn.tolist()[0]))
        return out
    # Mock: sliding drone
    t  = frame_id % 240
    cx = 0.10 + t * 0.0033
    return [("drone", round(0.82 + (frame_id % 5) * 0.02, 2),
             [cx - 0.06, 0.38, cx + 0.06, 0.62])]


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
    best_det = None
    for cls_name, conf, bbox in detections:
        x1 = int(bbox[0] * w); y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w); y2 = int(bbox[3] * h)
        color = C_RED if cls_name == "drone" else C_GREEN
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Confidence bar (40px wide, beneath label)
        bar_w = int((x2 - x1) * conf)
        cv2.rectangle(out, (x1, y2 + 2), (x1 + (x2 - x1), y2 + 8), C_DARK, -1)
        cv2.rectangle(out, (x1, y2 + 2), (x1 + bar_w, y2 + 8), color, -1)

        label = f"{cls_name.upper()}  {conf*100:.0f}%"
        _put(out, label, (x1, y1 - 8), scale=0.5, color=color, thickness=2)

        if best_det is None or conf > best_det[1]:
            best_det = (cls_name, conf, bbox)

    # ── Line from crosshair to target centre (tracking mode) ──────────────
    if best_det is not None:
        bbox = best_det[2]
        tx = int((bbox[0] + bbox[2]) / 2 * w)
        ty = int((bbox[1] + bbox[3]) / 2 * h)
        cv2.line(out, (cx_frame, cy_frame), (tx, ty), C_YELLOW, 1, cv2.LINE_AA)
        cv2.circle(out, (tx, ty), 5, C_YELLOW, -1)

    # ── Top-left status panel ──────────────────────────────────────────────
    panel_h = 90
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (300, panel_h), C_DARK, -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    mode_color = C_GREEN if mode == "TRACKING" else C_YELLOW
    _put(out, f"MODE : {mode}", (8, 20), scale=0.6, color=mode_color, thickness=2)
    _put(out, f"PAN  : {pan_deg:6.1f} deg", (8, 44), color=C_WHITE)
    _put(out, f"TILT : {tilt_deg:6.1f} deg", (8, 63), color=C_WHITE)
    _put(out, f"SERVO: {'ACTIVE' if SERVO_OK else 'SIM'}", (8, 82), color=C_CYAN)

    # ── Top-right FPS & frame ──────────────────────────────────────────────
    _put(out, f"FPS {fps:4.1f}", (w - 100, 20), color=C_WHITE)
    _put(out, f"F#{frame_id}", (w - 100, 40), color=C_WHITE, scale=0.45)

    # ── Bottom bar: model & target info ───────────────────────────────────
    bar_y = h - 30
    overlay2 = out.copy()
    cv2.rectangle(overlay2, (0, bar_y - 5), (w, h), C_DARK, -1)
    cv2.addWeighted(overlay2, 0.6, out, 0.4, 0, out)

    model_str = "YOLOv8n" if MODEL_OK else "MOCK"
    if best_det:
        cls_name, conf, bbox = best_det
        cx_t = (bbox[0] + bbox[2]) / 2
        cy_t = (bbox[1] + bbox[3]) / 2
        info = (f"TARGET: {cls_name.upper()}  conf={conf:.2f}  "
                f"cx={cx_t:.2f}  cy={cy_t:.2f}  [{model_str}]")
        _put(out, info, (8, h - 10), scale=0.48, color=C_YELLOW)
    else:
        _put(out, f"TARGET: NONE  [{model_str}]  SEARCHING...",
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
                     default=Path("data/sim_samples/drone_flyby.mp4"))
    src.add_argument("--gstreamer", action="store_true")
    args = parser.parse_args()

    cap, label = open_capture(args)

    print("\n" + "=" * 60)
    print("  AEGIS — Live Tracking Pipeline")
    print(f"  Source : {label}")
    print(f"  Model  : {'YOLOv8n' if MODEL_OK else 'Mock detections'}")
    print(f"  Servo  : {'PCA9685 ACTIVE' if SERVO_OK else 'Simulation (log only)'}")
    print("  Stop   : press Q in the window or Ctrl+C")
    print("=" * 60 + "\n")

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
            if detections:
                best = max(detections, key=lambda d: d[1])
                pan_deg, tilt_deg = servo_track(best[2])
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
            cv2.imshow("AEGIS  |  Live Tracking", out)

            # ── Step 5: Terminal log ──────────────────────────────────────
            if detections:
                best = max(detections, key=lambda d: d[1])
                print(
                    f"[F{frame_id:04d}] {mode:9s} | "
                    f"{best[0]:8s} {best[1]:.2f} | "
                    f"pan={pan_deg:6.1f}  tilt={tilt_deg:6.1f} | "
                    f"{(time.perf_counter()-t0)*1000:.0f}ms"
                )
            else:
                print(
                    f"[F{frame_id:04d}] {mode:9s} | "
                    f"pan={pan_deg:6.1f}  tilt={tilt_deg:6.1f} | "
                    f"{(time.perf_counter()-t0)*1000:.0f}ms"
                )

            if cv2.waitKey(1) & 0xFF == ord("q"):
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
