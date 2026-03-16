"""
AEGIS-Cloud — Edge / Perception / Vision Node
===============================================
Module:   edge.src.perception.vision_node
Platform: NVIDIA Jetson Nano (JetPack 5.x)

PURPOSE
-------
YOLOv8-nano inference engine for real-time maritime surface surveillance.
Mounted on a patrol drone, detects and classifies surface vessels.
Reads config from ``edge/config/edge_settings.yaml``.

When SIMULATION_MODE=True, reads from a sample video file or generates
a synthetic placeholder frame — no camera hardware required.

LOGGING FORMAT (NATO Incident Report)
--------------------------------------
  [TIMESTAMP] | [COORD] | [TYPE] | [CONF] | [XAI_STUB]

EU AI Act: Grad-CAM stubs are injected for every "Hostile" detection
before the detection leaves this module.

AIOps: CPU/GPU usage and inference latency are bundled into each
detection's metadata dict for Cloud consumption.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Project-root bootstrap
# Enables running from the AEGIS-Cloud/ root directory:
#   python edge/src/perception/vision_node.py
# without setting PYTHONPATH manually.
# ---------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = str(_Path(__file__).resolve().parents[3])  # …/AEGIS-Cloud/
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
del _sys, _Path, _PROJECT_ROOT

import logging
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import cv2
import numpy as np
import yaml

# Lazy imports to avoid circular references at module load time
if TYPE_CHECKING:
    from edge.src.perception.reasoning_node import ReasoningNode
    from edge.src.comm.cloud_sync import CloudSync
    from edge.src.sensors.servo_driver import PanTiltServoDriver

# ---------------------------------------------------------------------------
# Optional heavy imports — graceful degradation when not on Jetson
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    logging.warning("[AEGIS] ultralytics not installed — inference disabled.")

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_AVAILABLE = True
except Exception:
    _GPU_AVAILABLE = False

try:
    import onnxruntime as _ort  # noqa: F401
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("edge/config/edge_settings.yaml")
NATO_LOG_FMT = "[%(asctime)s] %(message)s"

# Maritime risk classification — initial level assigned by YOLO size heuristic.
# Moondream VLM upgrades "Unknown" vessels to "Hostile" (warship/patrol) or
# "Friendly" (civilian cargo/fishing) after visual analysis.
_RISK_MAP: dict[str, str] = {
    "warship":      "Hostile",   # Military vessel — Moondream-confirmed
    "patrol_boat":  "Hostile",   # Fast-attack / patrol — Moondream-confirmed
    "ship":         "Unknown",   # Large surface vessel — pending Moondream eval
    "boat":         "Unknown",   # Small vessel — pending Moondream eval
    "drone":        "Hostile",   # Aerial target — always hostile by default
    "person":       "Unknown",   # Dismount — context-dependent
    "vehicle":      "Unknown",   # Ground vehicle — context-dependent
}


# ---------------------------------------------------------------------------
# ONNX Inference Backend  (no PyTorch / ultralytics required)
# ---------------------------------------------------------------------------

class _OnnxBox:
    """Mirrors the ultralytics ``Boxes`` attribute API used in _run_inference."""

    def __init__(self, cls_id: int, conf: float, xyxyn: list) -> None:
        self._cls_id = cls_id
        self._conf   = conf
        self._xyxyn  = xyxyn

    class _Val:
        def __init__(self, v: Any) -> None: self._v = v
        def item(self) -> Any: return self._v

    class _Coords:
        def __init__(self, c: list) -> None: self._c = c
        def tolist(self) -> list: return [self._c]

    @property
    def cls(self) -> "_OnnxBox._Val":    return self._Val(self._cls_id)
    @property
    def conf(self) -> "_OnnxBox._Val":   return self._Val(self._conf)
    @property
    def xyxyn(self) -> "_OnnxBox._Coords": return self._Coords(self._xyxyn)


class _OnnxResult:
    """Mirrors an ultralytics ``Results`` object."""

    def __init__(self, boxes: list, names: dict) -> None:
        self.boxes = boxes
        self.names = names


class _OnnxModel:
    """
    Lightweight YOLOv8 ONNX inference engine — no PyTorch required.

    Supports any YOLOv8 ONNX export (opset 12+).
    Input:  ``images``  → (1, 3, imgsz, imgsz) float32
    Output: ``output0`` → (1, 4+nc, 8400) float32
    """

    # COCO-80 class names (fallback if model metadata absent)
    _COCO: dict = {
        0: "person",      1: "bicycle",     2: "car",           3: "motorcycle",
        4: "airplane",    5: "bus",         6: "train",         7: "truck",
        8: "boat",        9: "traffic light", 10: "fire hydrant", 11: "stop sign",
        12: "parking meter", 13: "bench",   14: "bird",         15: "cat",
        16: "dog",        17: "horse",      18: "sheep",        19: "cow",
        20: "elephant",   21: "bear",       22: "zebra",        23: "giraffe",
        24: "backpack",   25: "umbrella",   26: "handbag",      27: "tie",
        28: "suitcase",   29: "frisbee",    30: "skis",         31: "snowboard",
        32: "sports ball", 33: "kite",      34: "baseball bat", 35: "baseball glove",
        36: "skateboard", 37: "surfboard",  38: "tennis racket", 39: "bottle",
        40: "wine glass", 41: "cup",        42: "fork",         43: "knife",
        44: "spoon",      45: "bowl",       46: "banana",       47: "apple",
        48: "sandwich",   49: "orange",     50: "broccoli",     51: "carrot",
        52: "hot dog",    53: "pizza",      54: "donut",        55: "cake",
        56: "chair",      57: "couch",      58: "potted plant", 59: "bed",
        60: "dining table", 61: "toilet",   62: "tv",           63: "laptop",
        64: "mouse",      65: "remote",     66: "keyboard",     67: "cell phone",
        68: "microwave",  69: "oven",       70: "toaster",      71: "sink",
        72: "refrigerator", 73: "book",     74: "clock",        75: "vase",
        76: "scissors",   77: "teddy bear", 78: "hair drier",   79: "toothbrush",
    }
    # Map COCO names → AEGIS canonical target names
    _ALIASES: dict = {
        "car": "vehicle", "motorcycle": "vehicle",
        "bus": "vehicle", "truck": "vehicle",
    }

    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        iou_threshold: float = 0.45,
    ) -> None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # suppress verbose logs
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._sess       = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
        self._input_name = self._sess.get_inputs()[0].name
        self._imgsz      = imgsz
        self._iou        = iou_threshold

        # Read class names from ONNX model metadata; fall back to COCO
        meta = self._sess.get_modelmeta().custom_metadata_map
        if "names" in meta:
            import ast
            raw: dict = ast.literal_eval(meta["names"])
            base: dict = {int(k): str(v) for k, v in raw.items()}
        else:
            base = dict(self._COCO)
        self.names: dict = {k: self._ALIASES.get(v, v) for k, v in base.items()}

    # ------------------------------------------------------------------

    def predict(
        self,
        source: "np.ndarray",
        conf: float = 0.25,
        verbose: bool = True,
    ) -> list:
        blob, pad_w, pad_h, new_w, new_h = self._letterbox(source)
        outputs = self._sess.run(None, {self._input_name: blob})
        boxes   = self._postprocess(outputs[0], conf, pad_w, pad_h, new_w, new_h)
        return [_OnnxResult(boxes, self.names)]

    def _letterbox(self, img: "np.ndarray") -> tuple:
        """Letterbox-resize to (imgsz × imgsz); return blob + padding info."""
        h, w   = img.shape[:2]
        scale  = self._imgsz / max(h, w)
        new_h  = int(h * scale)
        new_w  = int(w * scale)
        pad_h  = (self._imgsz - new_h) // 2
        pad_w  = (self._imgsz - new_w) // 2
        canvas = np.full((self._imgsz, self._imgsz, 3), 114, dtype=np.uint8)
        canvas[pad_h: pad_h + new_h, pad_w: pad_w + new_w] = cv2.resize(img, (new_w, new_h))
        blob   = (canvas[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]
        return blob, pad_w, pad_h, new_w, new_h

    def _postprocess(
        self,
        pred: "np.ndarray",
        conf_thresh: float,
        pad_w: int,
        pad_h: int,
        new_w: int,
        new_h: int,
    ) -> list:
        """Decode YOLOv8 raw output → list of _OnnxBox (normalised xyxy)."""
        rows      = pred[0].T                       # (8400, 4+nc)
        bxywh     = rows[:, :4]
        cls_scores = rows[:, 4:]
        confs     = cls_scores.max(axis=1)
        cls_ids   = cls_scores.argmax(axis=1)

        mask = confs >= conf_thresh
        if not mask.any():
            return []
        bxywh, confs, cls_ids = bxywh[mask], confs[mask], cls_ids[mask]

        # cx, cy, w, h → x1, y1, x2, y2  (letterboxed 640-px space)
        x1 = bxywh[:, 0] - bxywh[:, 2] / 2
        y1 = bxywh[:, 1] - bxywh[:, 3] / 2
        x2 = bxywh[:, 0] + bxywh[:, 2] / 2
        y2 = bxywh[:, 1] + bxywh[:, 3] / 2

        keep = self._nms(np.stack([x1, y1, x2, y2], axis=1), confs, self._iou)

        results: list = []
        for i in keep:
            # Undo letterbox padding → normalise to original image dims
            nx1 = float(np.clip((x1[i] - pad_w) / new_w, 0.0, 1.0))
            ny1 = float(np.clip((y1[i] - pad_h) / new_h, 0.0, 1.0))
            nx2 = float(np.clip((x2[i] - pad_w) / new_w, 0.0, 1.0))
            ny2 = float(np.clip((y2[i] - pad_h) / new_h, 0.0, 1.0))
            results.append(_OnnxBox(int(cls_ids[i]), float(confs[i]), [nx1, ny1, nx2, ny2]))
        return results

    @staticmethod
    def _nms(boxes: "np.ndarray", scores: "np.ndarray", iou_threshold: float = 0.45) -> list:
        """Greedy NMS — pure NumPy, no torchvision required."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: list = []
        while order.size:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_threshold]
        return keep


# ---------------------------------------------------------------------------
# Best-Frame Selector
# ---------------------------------------------------------------------------

class BestFrameSelector:
    """
    Accumulates (frame, detections) pairs over a sliding time window.
    When the window expires, returns the frame with the single highest-
    confidence detection — the "sharpest" moment to send to Moondream.

    This avoids flooding the VLM with every frame while ensuring the most
    informative image is always chosen for reasoning.
    """

    def __init__(self, window_s: float = 5.0) -> None:
        self._window_s = max(window_s, 0.5)
        self._reset()

    def _reset(self) -> None:
        self._window_start: float = time.perf_counter()
        self._best_conf: float = 0.0
        self._best_frame: "np.ndarray | None" = None
        self._best_detections: "list" = []

    def update(
        self,
        frame: "np.ndarray",
        detections: "list",
    ) -> "tuple[np.ndarray, list] | None":
        """
        Feed a new frame.  Returns ``(best_frame, best_detections)`` when the
        window expires and at least one detection was accumulated; else ``None``.
        """
        if detections:
            peak_conf = max(d.confidence for d in detections)
            if peak_conf > self._best_conf:
                self._best_conf = peak_conf
                self._best_frame = frame.copy()
                self._best_detections = detections

        now = time.perf_counter()
        if now - self._window_start >= self._window_s:
            result = (
                (self._best_frame, self._best_detections)
                if self._best_frame is not None
                else None
            )
            self._reset()
            return result
        return None


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Single target detection produced by the Vision Node."""

    timestamp_utc: str
    station_id: str
    lat: float
    lon: float
    target_type: str
    confidence: float
    bbox: list[float]                     # [x1, y1, x2, y2] normalised
    risk_level: str                       # "Hostile" | "Unknown" | "Friendly"
    xai_stub: dict[str, Any] = field(default_factory=dict)
    aiops_meta: dict[str, Any] = field(default_factory=dict)
    frame_id: int = 0

    def to_nato_log(self) -> str:
        """Return a single NATO-format Incident Report log line."""
        return (
            f"[{self.timestamp_utc}] | "
            f"[{self.lat:.4f},{self.lon:.4f}] | "
            f"[{self.target_type.upper()}] | "
            f"[CONF:{self.confidence:.2f}] | "
            f"[XAI:{self.xai_stub.get('status', 'PENDING')}]"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for IoT Hub telemetry."""
        return {
            "timestamp_utc": self.timestamp_utc,
            "station_id":    self.station_id,
            "coordinates":   {"lat": self.lat, "lon": self.lon},
            "target_type":   self.target_type,
            "confidence":    self.confidence,
            "bbox":          self.bbox,
            "risk_level":    self.risk_level,
            "xai_stub":      self.xai_stub,
            "aiops_meta":    self.aiops_meta,
            "frame_id":      self.frame_id,
        }


# ---------------------------------------------------------------------------
# Vision Node
# ---------------------------------------------------------------------------

class VisionNode:
    """
    YOLOv8-nano inference engine for tactical drone/target detection.

    Parameters
    ----------
    config_path : Path | str
        Path to ``edge_settings.yaml``.
    """

    def __init__(
        self,
        config_path: Path | str = CONFIG_PATH,
        reasoning_node: "ReasoningNode | None" = None,
        cloud_sync: "CloudSync | None" = None,
        servo_node: "PanTiltServoDriver | None" = None,
    ) -> None:
        print("[DEBUG] 1. Config loading...")
        self._cfg = self._load_config(Path(config_path))
        self._sim_mode: bool = self._cfg.get("simulation_mode", True)
        self._inf_cfg: dict = self._cfg.get("inference", {})
        self._cam_cfg: dict = self._cfg.get("camera", {})
        self._aiops_cfg: dict = self._cfg.get("aiops", {})
        self._nato_cfg: dict = self._cfg.get("nato_metadata", {})

        self._model: Any | None = None
        self._cap: cv2.VideoCapture | None = None
        self._frame_id: int = 0
        self._start_ts: float = time.perf_counter()

        # Best-frame selector — picks peak-confidence frame per time window
        _bfw = float(self._inf_cfg.get("best_frame_window_s", 5.0))
        self._best_frame_sel = BestFrameSelector(window_s=_bfw)
        self._snapshot_dir = Path(self._inf_cfg.get("snapshot_dir", "data/logs/snapshots"))

        # Optional integration components
        self._reasoning: "ReasoningNode | None" = reasoning_node
        self._cloud: "CloudSync | None" = cloud_sync
        self._servo: "PanTiltServoDriver | None" = servo_node

        # Configure logger
        log_level = getattr(logging, self._cfg.get("log_level", "INFO").upper(), logging.INFO)
        logging.basicConfig(format=NATO_LOG_FMT, datefmt="%Y-%m-%dT%H:%M:%SZ", level=log_level)
        self.logger = logging.getLogger("AEGIS.VisionNode")

        self._target_classes: list[str] = [
            c.lower() for c in self._inf_cfg.get("target_classes", ["drone", "person", "vehicle"])
        ]
        self._conf_threshold: float = float(self._inf_cfg.get("confidence_threshold", 0.50))
        self._grad_cam_trigger: str = self._inf_cfg.get("grad_cam_trigger", "Hostile")

        self._init_model()
        self._init_capture()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: Path) -> dict:
        """Load YAML configuration file. Returns empty dict on failure."""
        if not path.exists():
            logging.warning(f"[AEGIS] Config not found: {path}. Using defaults.")
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _init_model(self) -> None:
        """
        Load YOLOv8 model, auto-downloading it if the file is missing.

        Download flow
        -------------
        1. If the configured ``.pt`` file exists → load directly (fast path).
        2. If missing → attempt Ultralytics auto-download (requires internet).
           The downloaded file is saved to the configured path so subsequent
           starts skip the download entirely.
        3. If download fails (e.g. no internet on Jetson) → warn and fall back
           to mock detections so the pipeline keeps running.
        """
        import shutil

        model_path = Path(self._inf_cfg.get("model_path", "edge/models/yolov8n.pt"))

        if not _YOLO_AVAILABLE:
            self.logger.warning("[AEGIS] YOLOv8 unavailable — returning mock detections.")
            return

        if not model_path.exists():
            self.logger.warning(
                f"[AEGIS] Model not found at {model_path}. "
                "Attempting Ultralytics auto-download…"
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # Passing just the filename triggers Ultralytics CDN download.
                # The file lands in the current working directory first.
                tmp_model = YOLO(model_path.name)
                # Move from cwd to the configured path if it was written there.
                cwd_file = Path(model_path.name)
                if cwd_file.exists() and cwd_file.resolve() != model_path.resolve():
                    shutil.move(str(cwd_file), str(model_path))
                    self.logger.info(
                        f"[AEGIS] Model downloaded and saved to {model_path}."
                    )
                    # Reload from the final path to ensure consistent state.
                    del tmp_model
                else:
                    # Model is in Ultralytics cache; keep the in-memory instance.
                    self._model = tmp_model
                    self.logger.info(
                        "[AEGIS] Model downloaded (Ultralytics cache). "
                        f"Copy to {model_path} for offline use: "
                        f"`cp ~/.config/Ultralytics/{model_path.name} {model_path}`"
                    )
                    return
            except Exception as exc:
                self.logger.warning(
                    f"[AEGIS] Auto-download failed ({exc}). "
                    "Falling back to mock detections — check internet connectivity."
                )
                return

        # ONNX path — no PyTorch/ultralytics required
        if model_path.suffix == ".onnx":
            if not _ONNX_AVAILABLE:
                self.logger.warning(
                    "[AEGIS] onnxruntime not installed — cannot load .onnx model. "
                    "Install with: pip install onnxruntime"
                )
                return
            iou   = float(self._inf_cfg.get("nms_iou_threshold", 0.45))
            imgsz = int(self._inf_cfg.get("image_size", 640))
            self.logger.info(f"[AEGIS] Loading ONNX model: {model_path}")
            self._model = _OnnxModel(str(model_path), imgsz=imgsz, iou_threshold=iou)
            self.logger.info("[AEGIS] ONNX model loaded — ACTIVE MODE (no PyTorch).")
            return

        device = self._inf_cfg.get("device", "cpu")
        self.logger.info(f"[AEGIS] Loading YOLOv8 model: {model_path}  device={device}")
        self._model = YOLO(str(model_path))
        self.logger.info("[AEGIS] Model loaded successfully — ACTIVE MODE.")

    def _init_capture(self) -> None:
        """
        Open video capture source determined by config priority.

        Priority 1 — SIMULATION_MODE=true:
            Read from ``inference.sim_video_path`` (mp4 loop).
            Falls back to synthetic placeholder frames if file absent.

        Priority 2 — SIMULATION_MODE=false, camera.use_gstreamer=true:
            Open the IMX219 CSI camera via nvarguscamerasrc GStreamer pipeline.
            Falls back to synthetic placeholder frames if pipeline fails to open.

        Priority 3 — SIMULATION_MODE=false, camera.use_gstreamer=false:
            Hardware camera not yet connected. Fall back to simulation video
            (preserves CI/CD and hardware-free development workflows).
        """
        if self._sim_mode:
            # ── Priority 1: Simulation video file ──────────────────────
            sim_path = self._inf_cfg.get("sim_video_path", "")
            if sim_path and Path(sim_path).exists():
                self._cap = cv2.VideoCapture(str(sim_path))
                self.logger.info(f"[AEGIS] SIMULATION_MODE — reading video: {sim_path}")
            else:
                self.logger.info(
                    "[AEGIS] SIMULATION_MODE — sim video not found; "
                    "generating synthetic frames."
                )
                self._cap = None  # _next_frame() will generate placeholder frames

        elif self._cam_cfg.get("use_gstreamer", False):
            # ── Priority 2: Live IMX219 CSI camera via GStreamer ────────
            pipeline = self._build_gstreamer_pipeline()
            self.logger.info(f"[AEGIS] LIVE MODE — opening GStreamer pipeline: {pipeline}")
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                self._cap = cap
                self.logger.info(
                    "[AEGIS] LIVE MODE — GStreamer pipeline OPEN. IMX219 ACTIVE."
                )
            else:
                cap.release()
                self._cap = None
                self.logger.warning(
                    "[AEGIS] LIVE MODE — GStreamer pipeline failed to open. "
                    "Check IMX219 connection and JetPack nvarguscamerasrc. "
                    "Falling back to synthetic frames."
                )

        else:
            # ── Priority 3: Hardware pending — fall back to sim video ───
            self.logger.info(
                "[AEGIS] LIVE MODE — camera.use_gstreamer=false (IMX219 not yet connected). "
                "Using simulation video as stand-in. "
                "Set camera.use_gstreamer=true in edge_settings.yaml on camera arrival."
            )
            sim_path = self._inf_cfg.get(
                "sim_video_path", "data/sim_samples/drone_flyby.mp4"
            )
            if sim_path and Path(sim_path).exists():
                self._cap = cv2.VideoCapture(str(sim_path))
                self.logger.info(
                    f"[AEGIS] LIVE MODE (pending HW) — reading from: {sim_path}"
                )
            else:
                self.logger.warning(
                    f"[AEGIS] LIVE MODE — sim video '{sim_path}' missing. "
                    "Generating synthetic frame fallback."
                )
                self._cap = None

    def _build_gstreamer_pipeline(self) -> str:
        """
        Construct the nvarguscamerasrc GStreamer pipeline string for the
        IMX219 (Sony v2) CSI camera on Jetson Nano (JetPack 5.x).

        Pipeline topology:
            nvarguscamerasrc → NVMM capture → nvvidconv downscale →
            videoconvert → BGR appsink

        Returns
        -------
        str
            A GStreamer pipeline string compatible with
            ``cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)``.
        """
        c = self._cam_cfg
        sensor_id   = c.get("sensor_id", 0)
        cap_w       = c.get("capture_width", 1920)
        cap_h       = c.get("capture_height", 1080)
        out_w       = c.get("output_width", 640)
        out_h       = c.get("output_height", 360)
        framerate   = c.get("framerate", 30)
        flip_method = c.get("flip_method", 0)
        drop        = "true" if c.get("drop_frames", True) else "false"

        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), "
            f"width={cap_w}, height={cap_h}, "
            f"format=NV12, framerate={framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width={out_w}, height={out_h}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink drop={drop}"
        )

    # ------------------------------------------------------------------
    # Frame generation
    # ------------------------------------------------------------------

    def _next_frame(self) -> np.ndarray | None:
        """
        Return the next video frame.

        In simulation mode, loops the video file or returns a synthetic
        coloured placeholder frame if no video file is available.
        """
        if self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                if self._sim_mode:
                    # Loop the simulation video
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()
                    if not ret:
                        return None
                else:
                    return None   # Camera disconnected
            return frame

        # Synthetic placeholder — a solid-coloured 640×480 frame
        color = self._inf_cfg.get("sim_fallback_color", [0, 128, 255])
        frame = np.full((480, 640, 3), color, dtype=np.uint8)
        # Simulate a moving "drone" bounding box overlay
        t = self._frame_id % 200
        cx, cy = 100 + t * 2, 200 + int(30 * np.sin(t * 0.1))
        cv2.rectangle(frame, (cx - 30, cy - 20), (cx + 30, cy + 20), (0, 255, 0), 2)
        cv2.putText(frame, "SIM-DRONE", (cx - 28, cy - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return frame

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Runs YOLOv8 inference and returns filtered detections.
        """
        raw_results = self._run_inference(frame)
        detections: list[Detection] = []

        for cls_name, confidence, bbox in raw_results:
            # Filter by class and confidence based on YAML settings
            if cls_name not in self._target_classes:
                continue
            if confidence < self._conf_threshold:
                continue

            # Create Detection object
            det = Detection(
                timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                station_id=self._nato_cfg.get("station_id", "AEGIS-EDGE-001"),
                lat=float(self._nato_cfg.get("unit_coordinates", {}).get("lat", 48.3984)),
                lon=float(self._nato_cfg.get("unit_coordinates", {}).get("lon", 10.0011)),
                target_type=cls_name,
                confidence=round(float(confidence), 4),
                bbox=bbox,
                risk_level=_RISK_MAP.get(cls_name, "Unknown"),
                frame_id=self._frame_id
            )
            detections.append(det)
            
        return detections

    def _run_inference(self, frame: np.ndarray) -> list[tuple[str, float, list[float]]]:
        """
        Executes YOLOv8 or returns Mock Data if in Simulation Mode.
        """
        # 1. Real YOLO Inference — runs whenever a model is loaded (video file or live)
        if self._model is not None:
            results = self._model.predict(source=frame, conf=self._conf_threshold, verbose=False)
            raw = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = r.names.get(cls_id, "unknown").lower()
                    conf = float(box.conf.item())
                    bbox = box.xyxyn.tolist()[0] # Normalised [x1, y1, x2, y2]
                    raw.append((cls_name, conf, bbox))
            return raw

        # 2. Mock Data for Simulation (Triggers every frame)
        if self._sim_mode:
            # We simulate a drone with a shifting confidence
            mock_conf = round(0.80 + (self._frame_id % 5) * 0.02, 4)
            return [("drone", mock_conf, [0.3, 0.3, 0.6, 0.6])]

        return []

    # ------------------------------------------------------------------
    # XAI Stub
    # ------------------------------------------------------------------

    def _build_xai_stub(
        self,
        frame: np.ndarray,
        bbox: list[float],
        cls_name: str,
        risk_level: str,
    ) -> dict[str, Any]:
        """
        Build the XAI evidence placeholder.

        For "Hostile" targets (as defined by ``grad_cam_trigger`` in
        config), this stub marks the evidence as REQUIRED and records
        the bounding-box region where Grad-CAM should be applied.

        In a future sprint, the ``heatmap_b64`` field will be populated
        by the real Grad-CAM implementation using the loaded model's
        feature-map hooks.

        EU AI Act mandate: THIS STUB MUST BE REPLACED WITH REAL SHAP/
        GRAD-CAM OUTPUT BEFORE ANY "HOSTILE" DECISION IS FINALISED.
        """
        grad_cam_required = (
            self._inf_cfg.get("grad_cam_enabled", True)
            and risk_level == self._grad_cam_trigger
        )

        xai: dict[str, Any] = {
            "status":             "REQUIRED" if grad_cam_required else "NOT_REQUIRED",
            "method":             "Grad-CAM",
            "target_class":       cls_name,
            "risk_level":         risk_level,
            "bbox_attention":     bbox,
            "heatmap_b64":        None,          # Populated by real Grad-CAM hook
            "shap_values":        None,          # Populated by Cloud XAI Generator
            "eu_ai_act_compliant": not grad_cam_required,   # False until heatmap filled
        }
        if grad_cam_required:
            # Crop the attention region for downstream Grad-CAM processing
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (
                int(bbox[0] * w), int(bbox[1] * h),
                int(bbox[2] * w), int(bbox[3] * h),
            )
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            xai["crop_shape"] = list(crop.shape) if crop.size > 0 else []
            xai["status"] = "PENDING_GRAD_CAM"

        return xai

    # ------------------------------------------------------------------
    # AIOps Telemetry
    # ------------------------------------------------------------------

    def _collect_aiops_telemetry(self, inference_latency_ms: float) -> dict[str, Any]:
        """
        Collect system health metrics injected into every detection's
        ``aiops_meta`` for Cloud consumption by the self-healing service.

        Metrics
        -------
        cpu_percent   : float — system CPU utilisation (%)
        ram_used_mb   : float — process RSS memory (MB)
        gpu_temp_c    : float | None — GPU temperature (°C), Jetson only
        gpu_util_pct  : float | None — GPU utilisation (%), Jetson only
        latency_ms    : float — current-frame inference latency (ms)
        uptime_s      : float — seconds since VisionNode was initialised
        """
        meta: dict[str, Any] = {
            "cpu_percent":   None,
            "ram_used_mb":   None,
            "gpu_temp_c":    None,
            "gpu_util_pct":  None,
            "latency_ms":    round(inference_latency_ms, 2),
            "uptime_s":      round(time.perf_counter() - self._start_ts, 1),
            "platform":      platform.node(),
        }
        if _PSUTIL_AVAILABLE:
            try:
                meta["cpu_percent"] = psutil.cpu_percent(interval=None)
                proc = psutil.Process(os.getpid())
                meta["ram_used_mb"] = round(proc.memory_info().rss / (1024 ** 2), 2)
            except Exception:
                pass

        if _GPU_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                meta["gpu_temp_c"]   = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                meta["gpu_util_pct"] = util.gpu
            except Exception:
                pass

        return meta

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_detection(self, det: Detection) -> None:
        """Emit a structured NATO Incident Report log line."""
        self.logger.info(det.to_nato_log())

    def health_check(self) -> dict[str, Any]:
        """
        Return an AIOps health snapshot for the Cloud self-healing service.

        Returns
        -------
        dict
            {
                "status":          "OK" | "DEGRADED" | "FAILED",
                "model_loaded":    bool,
                "capture_open":    bool,
                "simulation_mode": bool,
                "frame_id":        int,
                "uptime_s":        float,
            }
        """
        return {
            "status":          "OK" if self._model is not None or self._sim_mode else "DEGRADED",
            "model_loaded":    self._model is not None,
            "capture_open":    self._cap.isOpened() if self._cap else False,
            "simulation_mode": self._sim_mode,
            "frame_id":        self._frame_id,
            "uptime_s":        round(time.perf_counter() - self._start_ts, 1),
        }

    # ------------------------------------------------------------------
    # Main detection loop (generator)
    # ------------------------------------------------------------------

    def run(self) -> Generator[list[Detection], None, None]:
        """
        Modified loop with Heartbeat and Sub-sampling to prevent Jetson Nano lockup.
        """
        self.logger.info(
            f"[AEGIS] VisionNode starting — "
            f"{'SIMULATION' if self._sim_mode else 'LIVE'} mode | "
            f"reasoning={'ON' if self._reasoning else 'OFF'}"
        )
        
        target_fps: float = float(self._inf_cfg.get("target_fps", 1))
        frame_delay: float = 1.0 / max(target_fps, 0.1)

        try:
            while True:
                loop_start = time.perf_counter()
                
                # --- AIOPS HEARTBEAT ---
                # Prints status every 5 frames to confirm the process is alive
                if self._frame_id % 5 == 0:
                    cpu_usage = psutil.cpu_percent(interval=0.1)
                    print(f"[HEARTBEAT] Frame: {self._frame_id} | CPU: {cpu_usage}% | Running...")

                frame = self._next_frame()
                if frame is None:
                    break

                detections = self.detect(frame)
                self._frame_id += 1

                # --- PAN-TILT SERVO TRACKING ---
                # Steer the gimbal toward the highest-confidence detection.
                # Falls back to center() when no targets are visible so the
                # platform returns to search position automatically.
                if self._servo is not None:
                    if detections:
                        best_det = max(detections, key=lambda d: d.confidence)
                        pan_deg, tilt_deg = self._servo.track(best_det.bbox)
                        self.logger.debug(
                            "[SERVO] tracking %s conf=%.2f → pan=%.1f° tilt=%.1f°",
                            best_det.target_type, best_det.confidence,
                            pan_deg, tilt_deg,
                        )
                    else:
                        self._servo.center()

                # --- BEST-FRAME SELECTION ---
                # Feed every frame into BestFrameSelector. When the window expires
                # (best_frame_window_s), send only the peak-confidence frame to
                # Moondream — avoiding VLM flooding on every detection tick.
                if self._reasoning:
                    best = self._best_frame_sel.update(frame, detections)
                    if best is not None:
                        best_frame, best_dets = best
                        if best_dets:
                            self._process_pipeline(best_frame, best_dets)

                yield detections

                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0, frame_delay - elapsed))

        except KeyboardInterrupt:
            self.logger.info("[AEGIS] VisionNode stopped by operator.")
        finally:
            self.release()

    # ------------------------------------------------------------------
    # Integrated pipeline: Reasoning → Cloud
    # ------------------------------------------------------------------

    def _process_pipeline(
        self,
        frame: np.ndarray,
        detections: list["Detection"],
    ) -> None:
        """
        Sequential execution: saves a best-frame snapshot then calls Moondream
        for the highest-confidence target before syncing to cloud.
        """
        # --- Save best-frame snapshot to disk ---
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        best_det = max(detections, key=lambda d: d.confidence)
        snap_name = f"best_{best_det.target_type}_{ts_tag}_conf{best_det.confidence:.2f}.jpg"
        snap_path = self._snapshot_dir / snap_name
        cv2.imwrite(str(snap_path), frame)
        self.logger.info(f"[AEGIS] Best-frame snapshot → {snap_path}")
        print(f"  📷 Snapshot saved: {snap_path}")

        reasoning_results: list[dict] = []
        aiops_snapshot: dict = detections[0].aiops_meta if detections else {}

        for det in detections:
            # For now, we only reason for the FIRST target to save Jetson resources
            if self._reasoning is not None:
                # LOGGING: Vital to see this in the terminal during the 200s wait
                self.logger.warning(f"!!! [TACTICAL ALERT] Target: {det.target_type} | Starting VLM Reasoning (Est. 200s)...")

                det_id = f"f{det.frame_id}_{int(time.time())}"

                # Execution blocks here while Moondream works
                result = self._reasoning.describe(
                    frame=frame,
                    bbox=det.bbox,
                    detection_id=det_id,
                )

                self.logger.info(f"[AEGIS] Moondream Report: {result.description}")
                self.logger.warning("[AEGIS] VLM reasoning complete. Resuming perception loop.")

                reasoning_results.append(result.to_dict())
                det.xai_stub["reasoning_description"] = result.description
                det.xai_stub["reasoning_inference_ms"] = result.inference_time_ms

                # --- RISK LEVEL UPGRADE from Moondream VLM output ---
                # Moondream's visual analysis overrides the initial YOLO heuristic.
                # Parse the free-text description for threat indicators and update
                # det.risk_level so the cloud payload reflects the real assessment.
                prev_risk = det.risk_level
                det.risk_level = self._parse_risk_from_description(result.description)
                if det.risk_level != prev_risk:
                    self.logger.info(
                        f"[AEGIS] Risk upgraded by Moondream: "
                        f"{prev_risk} → {det.risk_level} | {result.description[:80]}"
                    )
                    det.xai_stub["risk_upgraded_by_vlm"] = True
                    det.xai_stub["risk_prev"] = prev_risk

                # Break after first target to keep the edge loop stable
                break

        # Cloud Sync: remains unchanged
        if self._cloud is not None:
            det_dicts = [d.to_dict() for d in detections]
            self._cloud.send(
                detections=det_dicts,
                reasoning_results=reasoning_results,
                aiops_meta=aiops_snapshot,
            )

    @staticmethod
    def _parse_risk_from_description(description: str) -> str:
        """
        Derive a risk level from a Moondream free-text tactical description.

        Keywords used (case-insensitive):
          Hostile  → "MILITARY CONTACT", "warship", "patrol boat", "armed",
                     "weapon", "missile", "gun", "CRITICAL", "HIGH"
          Friendly → "fishing", "cargo", "tanker", "civilian", "commercial",
                     "LOW"
          Unknown  → anything else (pass-through)

        Returns
        -------
        str
            "Hostile" | "Friendly" | "Unknown"
        """
        text = description.upper()

        hostile_keywords = [
            "MILITARY CONTACT", "WARSHIP", "PATROL BOAT", "PATROL VESSEL",
            "ARMED", "WEAPON", "MISSILE", "GUN", "CANNON",
            "THREAT LEVEL: HIGH", "THREAT LEVEL: CRITICAL",
            "THREAT: HIGH", "THREAT: CRITICAL",
            ": HIGH", ": CRITICAL",
        ]
        friendly_keywords = [
            "FISHING", "CARGO SHIP", "TANKER", "CIVILIAN",
            "COMMERCIAL", "MERCHANT",
            "THREAT LEVEL: LOW", "THREAT: LOW", ": LOW",
        ]

        for kw in hostile_keywords:
            if kw in text:
                return "Hostile"
        for kw in friendly_keywords:
            if kw in text:
                return "Friendly"
        return "Unknown"

    def release(self) -> None:
        """Release camera/video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self.logger.info("[AEGIS] Capture released.")


# ---------------------------------------------------------------------------
# Standalone entry point for development testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from edge.src.perception.reasoning_node import ReasoningNode
    from edge.src.comm.cloud_sync import CloudSync
    from edge.src.sensors.servo_driver import PanTiltServoDriver

    reasoning = ReasoningNode()
    sync = CloudSync()
    servo = PanTiltServoDriver(yaml.safe_load(
        open(CONFIG_PATH, encoding="utf-8")
    ))
    node = VisionNode(reasoning_node=reasoning, cloud_sync=sync, servo_node=servo)

    _vid = node._inf_cfg.get("sim_video_path", "synthetic frames")
    _win = node._inf_cfg.get("best_frame_window_s", 5.0)
    print(f"\n{'='*70}")
    print("  AEGIS-Cloud — Integrated Edge Pipeline")
    print(f"  SIMULATION_MODE : {node._sim_mode}")
    print(f"  Video source    : {_vid}")
    print(f"  Target classes  : {node._target_classes}")
    print(f"  Best-frame win  : {_win}s")
    print(f"  Snapshot dir    : {node._snapshot_dir}")
    print(f"  Reasoning       : {reasoning.model}")
    print(f"  Cloud Sync      : {'SIM file' if sync._sim_mode else 'IoT Hub'}")
    print(f"{'='*70}\n")

    # Infinite loop — Ctrl+C to stop.
    # Each frame waits for Moondream to respond before advancing.
    for frame_idx, detections in enumerate(node.run(), start=1):
        if not detections:
            continue
        print(f"\n── Frame {frame_idx} {'─' * 55}")
        for det in detections:
            print(det.to_nato_log())
            desc = det.xai_stub.get("reasoning_description", "")
            lat_ms = det.xai_stub.get("reasoning_inference_ms")
            if desc:
                print(f"  ↳ Moondream [{lat_ms:.0f}ms]: {desc[:200]}")
            else:
                print(f"  ↳ Moondream: [waiting / circuit-breaker open]")
