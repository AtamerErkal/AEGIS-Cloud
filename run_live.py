# -*- coding: utf-8 -*-
"""
AEGIS-Cloud - Maritime Surveillance Pipeline
=============================================
Drone-mounted YOLOv8 detection -> vessel classification -> PCA9685 servo tracking
Scans sea surface for vessels. Locks onto military contacts.

Usage:
  python run_live.py --camera 0                      # USB webcam / drone camera
  python run_live.py --video path/to.mp4             # video file (with GUI)
  python run_live.py --video path/to.mp4 --headless  # SSH / no display (logs only)
  python run_live.py --video path/to.mp4 --save out.mp4   # save annotated output
  python run_live.py --gstreamer                     # IMX219 CSI camera (Nano)

SSH example:
  ssh user@drone 'cd AEGIS-Cloud && python run_live.py \\
      --video data/videos/test_ship.mp4 \\
      --save data/videos/test_ship_out.mp4'
"""

import argparse
import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# Auto-detect headless: no DISPLAY env var → force headless mode
_HEADLESS_AUTO = (os.environ.get("DISPLAY", "") == "" and
                  os.environ.get("WAYLAND_DISPLAY", "") == "")

# ── PCA9685 ──────────────────────────────────────────────────────────────────
try:
    from adafruit_servokit import ServoKit
    _kit = ServoKit(channels=16, address=0x40)
    _kit.frequency = 50
    SERVO_OK = True
    print("[SERVO] PCA9685 connected")
except Exception as e:
    SERVO_OK = False
    print(f"[SERVO] PCA9685 not found - angles will be logged only ({e})")

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
_dnn_net = None   # cv2.dnn fallback

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
    except Exception as _e_pt:
        print(f"[MODEL] PyTorch unavailable ({_e_pt}) — trying cv2.dnn …")

# Fallback: OpenCV DNN (no extra packages — uses bundled ONNX runtime in cv2)
if not MODEL_OK:
    _dnn_onnx = Path("edge/models/yolov8n.onnx")
    if _dnn_onnx.exists():
        try:
            _dnn_net = cv2.dnn.readNetFromONNX(str(_dnn_onnx))
            _dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            _dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            MODEL_OK = True
            _backend = "cv2.dnn"
            print("[MODEL] YOLOv8 cv2.dnn loaded (no extra packages required)")
        except Exception as _e_dnn:
            print(f"[MODEL] cv2.dnn unavailable ({_e_dnn})")
    else:
        print("[MODEL] yolov8n.onnx not found — cv2.dnn skipped")

# ── Moondream VLM (optional — requires Ollama + moondream model) ──────────────
_vlm_lock  = threading.Lock()
_vlm_state = {
    "busy":        False,
    "description": "",
    "risk":        "",
    "latency_ms":  0.0,
}
# Best-frame window: collect detections for N seconds, send peak frame to VLM
_VLM_WINDOW_S = 5.0
_vlm_best = {"conf": 0.0, "frame": None, "bbox": None, "t0": time.perf_counter()}

VLM_OK    = False
_reasoning = None
try:
    from edge.src.perception.reasoning_node import ReasoningNode as _ReasoningNode
    from edge.src.perception.vision_node import VisionNode as _VisionNode
    _reasoning = _ReasoningNode()
    VLM_OK = True
    print("[VLM] Moondream ReasoningNode ready")
except Exception as _e_vlm:
    print(f"[VLM] Moondream unavailable ({_e_vlm}) — run with --no-vlm to silence")


def _vlm_worker(frame: np.ndarray, bbox: list, det_id: str) -> None:
    """Background thread: calls Moondream and updates _vlm_state."""
    result = _reasoning.describe(frame=frame, bbox=bbox, detection_id=det_id)
    risk   = _VisionNode._parse_risk_from_description(result.description)
    with _vlm_lock:
        _vlm_state["busy"]        = False
        _vlm_state["description"] = result.description
        _vlm_state["risk"]        = risk
        _vlm_state["latency_ms"]  = result.inference_time_ms
    print(f"\n[VLM] {result.description[:150]}")
    print(f"[VLM] Risk → {risk}  ({result.inference_time_ms:.0f}ms)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TARGET_CLASSES  = {"vessel", "boat", "ship"}   # Maritime surface contacts only
CONF_THRESHOLD  = 0.25                          # Low threshold — catches distant/partial vessels
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


_COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _detect_dnn(frame: np.ndarray) -> list:
    """Run YOLOv8 inference via cv2.dnn — no ultralytics/onnxruntime needed."""
    oh, ow = frame.shape[:2]
    scale  = 640.0 / max(oh, ow)
    nw, nh = int(ow * scale), int(oh * scale)

    # Letterbox into a 640×640 grey canvas
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))

    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (640, 640),
                                 swapRB=True, crop=False)
    _dnn_net.setInput(blob)
    pred = _dnn_net.forward()[0]  # (84, 8400)

    raw_boxes, raw_scores, raw_cls_ids = [], [], []
    for i in range(pred.shape[1]):
        col       = pred[:, i]
        cls_id    = int(np.argmax(col[4:]))
        conf      = float(col[4 + cls_id])
        if conf < CONF_THRESHOLD:
            continue
        cx, cy, bw, bh = col[:4]
        x1 = float(np.clip((cx - bw / 2) / nw, 0.0, 1.0))
        y1 = float(np.clip((cy - bh / 2) / nh, 0.0, 1.0))
        x2 = float(np.clip((cx + bw / 2) / nw, 0.0, 1.0))
        y2 = float(np.clip((cy + bh / 2) / nh, 0.0, 1.0))
        raw_boxes.append([x1, y1, x2, y2])
        raw_scores.append(conf)
        raw_cls_ids.append(cls_id)

    if not raw_boxes:
        return []

    # NMS (cv2 expects x,y,w,h in pixel space)
    nms_in = [[b[0] * ow, b[1] * oh,
               (b[2] - b[0]) * ow, (b[3] - b[1]) * oh] for b in raw_boxes]
    idxs = cv2.dnn.NMSBoxes(nms_in, raw_scores, CONF_THRESHOLD, 0.45)
    if len(idxs) == 0:
        return []

    out = []
    for idx in (idxs.flatten() if hasattr(idxs, "flatten") else idxs):
        raw_cls  = (_COCO80[raw_cls_ids[idx]]
                    if raw_cls_ids[idx] < len(_COCO80) else "unknown")
        cls_name = _VESSEL_ALIASES.get(raw_cls, raw_cls)
        if cls_name not in TARGET_CLASSES:
            continue
        bbox     = raw_boxes[idx]
        cls_name = _classify_vessel(bbox)
        out.append((cls_name, raw_scores[idx], bbox))
    return out


def detect(frame: np.ndarray, frame_id: int, mock: bool = False) -> list:
    """Returns list of (cls_name, confidence, [x1,y1,x2,y2] normalised)."""
    if MODEL_OK and _dnn_net is not None:
        return _detect_dnn(frame)
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
    if mock:
        # Mock: vessel crossing frame — simulates drone patrol over sea
        t  = frame_id % 300
        cx = 0.05 + t * 0.003
        mock_bbox = [cx - 0.08, 0.42, cx + 0.08, 0.58]
        return [(_classify_vessel(mock_bbox),
                 round(0.75 + (frame_id % 5) * 0.02, 2),
                 mock_bbox)]
    return []


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
                 frame_id: int,
                 vlm_state=None) -> np.ndarray:
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

    # ── Search sweep indicator (SEARCHING mode only) ──────────────────────
    if mode == "SEARCHING":
        # Map pan_deg (30..150) to x pixel — shows where camera is looking
        sweep_x = int(np.clip((pan_deg - 30.0) / 120.0, 0.0, 1.0) * w)
        cv2.line(out, (sweep_x, 0), (sweep_x, h), C_CYAN, 1, cv2.LINE_AA)
        # Shade the expected sweep band
        band_x1 = int(max(0,     (90.0 - SEARCH_AMP - 30.0) / 120.0 * w))
        band_x2 = int(min(w - 1, (90.0 + SEARCH_AMP - 30.0) / 120.0 * w))
        band_overlay = out.copy()
        cv2.rectangle(band_overlay, (band_x1, 0), (band_x2, h), C_CYAN, -1)
        cv2.addWeighted(band_overlay, 0.07, out, 0.93, 0, out)
        _put(out, "SCANNING", (sweep_x + 6, h // 2), scale=0.5, color=C_CYAN)

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

    # ── VLM status panel (bottom-left, above bottom bar) ──────────────────
    if vlm_state is not None:
        vlm_y = bar_y - 30
        vlm_overlay = out.copy()
        cv2.rectangle(vlm_overlay, (0, vlm_y - 18), (w, vlm_y + 4), C_DARK, -1)
        cv2.addWeighted(vlm_overlay, 0.65, out, 0.35, 0, out)

        if vlm_state.get("busy"):
            _put(out, "VLM: ANALYZING...", (8, vlm_y - 2),
                 scale=0.48, color=C_YELLOW)
        elif vlm_state.get("description"):
            risk  = vlm_state.get("risk", "")
            lat   = vlm_state.get("latency_ms", 0.0)
            desc  = vlm_state["description"][:90]
            risk_color = C_RED if risk == "Hostile" else (C_GREEN if risk == "Friendly" else C_YELLOW)
            prefix = f"VLM[{lat:.0f}ms] {risk}: " if risk else f"VLM[{lat:.0f}ms]: "
            _put(out, prefix + desc, (8, vlm_y - 2),
                 scale=0.45, color=risk_color)
        else:
            _put(out, "VLM: waiting for first detection window...",
                 (8, vlm_y - 2), scale=0.45, color=C_CYAN)

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
                        help="Disable GUI window (required for SSH / no display)")
    parser.add_argument("--save", type=Path, default=None,
                        help="Save annotated output video to this path (e.g. out.mp4)")
    parser.add_argument("--no-loop", action="store_true",
                        help="Stop at end of video instead of looping")
    parser.add_argument("--conf", type=float, default=None,
                        help="Detection confidence threshold (default 0.25)")
    parser.add_argument("--mock", action="store_true",
                        help="Force mock detections (useful when no model is available)")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Disable Moondream VLM reasoning (faster, no Ollama required)")
    args = parser.parse_args()

    # Allow overriding confidence at runtime
    if args.conf is not None:
        global CONF_THRESHOLD
        CONF_THRESHOLD = args.conf

    # Respect auto-detected headless (SSH without X11 forwarding)
    if _HEADLESS_AUTO:
        args.headless = True

    cap, label = open_capture(args)

    # ── Video info (for progress display) ─────────────────────────────────
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_file      = (args.video is not None and not args.gstreamer
                    and args.camera is None)

    # ── Optional output writer ─────────────────────────────────────────────
    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.save), fourcc, src_fps, (src_w, src_h))
        print(f"[SAVE] Output → {args.save}")

    print("\n" + "=" * 65)
    print("  AEGIS — Maritime Surveillance Pipeline")
    print(f"  Source    : {label}")
    if total_frames > 0:
        dur = total_frames / src_fps
        print(f"  Duration  : {dur:.1f}s  ({total_frames} frames @ {src_fps:.0f}fps)")
    _model_str = 'YOLOv8n [' + _backend + ']' if MODEL_OK else 'NO MODEL'
    if not MODEL_OK and args.mock:
        _model_str += ' + MOCK detections forced'
    elif not MODEL_OK:
        _model_str += ' — no detections (pass --mock to simulate)'
    print(f"  Model     : {_model_str}")
    print(f"  Servo     : {'PCA9685 ACTIVE' if SERVO_OK else 'Simulation (log only)'}")
    print(f"  Display   : {'headless' if args.headless else 'window'}")
    if args.save:
        print(f"  Saving    : {args.save}")
    print("  Targets   : vessel / patrol_boat / warship")
    print("  Stop      : press Q in window  or  Ctrl+C")
    print("=" * 65 + "\n")

    frame_id   = 0
    fps        = 0.0
    fps_t      = time.perf_counter()
    fps_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_file and args.no_loop:
                    print(f"\n[DONE] End of video ({frame_id} frames processed)")
                    break
                # Loop: rewind
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if is_file:
                    print("[LOOP] Rewinding video …")
                continue

            t0 = time.perf_counter()

            # ── Step 1: Detect ────────────────────────────────────────────
            detections = detect(frame, frame_id, mock=args.mock)

            # ── Step 1b: Best-frame selection → Moondream VLM ────────────
            # Collect the highest-confidence detection over a 5-second window.
            # When the window expires, send that frame to Moondream in the
            # background so the video loop is never blocked.
            if VLM_OK and not args.no_vlm and detections:
                best_conf = max(d[1] for d in detections)
                if best_conf > _vlm_best["conf"]:
                    _vlm_best["conf"]  = best_conf
                    _vlm_best["frame"] = frame.copy()
                    _vlm_best["bbox"]  = max(detections, key=lambda d: d[1])[2]

            if VLM_OK and not args.no_vlm:
                window_elapsed = time.perf_counter() - _vlm_best["t0"]
                if window_elapsed >= _VLM_WINDOW_S:
                    with _vlm_lock:
                        is_busy = _vlm_state["busy"]
                    if not is_busy and _vlm_best["frame"] is not None:
                        with _vlm_lock:
                            _vlm_state["busy"] = True
                        det_id = f"f{frame_id}_{int(time.time())}"
                        threading.Thread(
                            target=_vlm_worker,
                            args=(_vlm_best["frame"], _vlm_best["bbox"], det_id),
                            daemon=True,
                        ).start()
                        print(f"[VLM] Sending frame {frame_id} to Moondream…")
                    # Reset window regardless (don't flood VLM)
                    _vlm_best["conf"]  = 0.0
                    _vlm_best["frame"] = None
                    _vlm_best["bbox"]  = None
                    _vlm_best["t0"]    = time.perf_counter()

            # ── Step 2: Servo ─────────────────────────────────────────────
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

            # ── Step 4: Draw ──────────────────────────────────────────────
            _vlm_snapshot = None
            if VLM_OK and not args.no_vlm:
                with _vlm_lock:
                    _vlm_snapshot = dict(_vlm_state)
            out = draw_overlay(frame, detections, mode,
                               pan_deg, tilt_deg, fps, frame_id,
                               vlm_state=_vlm_snapshot)

            # ── Step 5: Display / save ────────────────────────────────────
            if not args.headless:
                cv2.imshow("AEGIS  |  Maritime Surveillance", out)
            if writer is not None:
                writer.write(out)

            # ── Step 6: Terminal log ──────────────────────────────────────
            elapsed_ms = (time.perf_counter() - t0) * 1000
            progress   = f"{frame_id}/{total_frames}" if total_frames > 0 else f"F{frame_id}"
            all_dets   = military_dets if military_dets else detections
            if all_dets:
                best    = max(all_dets, key=lambda d: d[1])
                mil_tag = " [!!MILITARY!!]" if best[0] in {"warship", "patrol_boat"} else ""
                print(
                    f"[{progress}] {mode:14s} | "
                    f"{best[0]:12s} {best[1]:.2f}{mil_tag} | "
                    f"brg={pan_deg:6.1f}  elev={tilt_deg:6.1f} | "
                    f"{elapsed_ms:.0f}ms"
                )
            else:
                # Only log search status every 25 frames to reduce noise
                if frame_id % 25 == 0:
                    print(
                        f"[{progress}] {'SCANNING':14s} | "
                        f"brg={pan_deg:6.1f}  elev={tilt_deg:6.1f} | "
                        f"{elapsed_ms:.0f}ms"
                    )

            if not args.headless and cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[STOPPED] Q pressed")
                break

            frame_id += 1

    except KeyboardInterrupt:
        print("\n[STOPPED] Ctrl+C")
    finally:
        _write_servo(90.0, 90.0)
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[SAVED] {args.save}")
        cv2.destroyAllWindows()
        print("[DONE] Servo centred, camera released.")


if __name__ == "__main__":
    main()
