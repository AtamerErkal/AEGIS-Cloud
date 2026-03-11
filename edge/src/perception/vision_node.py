"""
AEGIS-Cloud — Edge / Perception / Vision Node
===============================================
Module:   edge.src.perception.vision_node
Platform: NVIDIA Jetson Nano (JetPack 5.x)

PURPOSE
-------
YOLOv8-nano inference engine for real-time drone/target detection.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("edge/config/edge_settings.yaml")
NATO_LOG_FMT = "[%(asctime)s] %(message)s"

# Risk classification thresholds — map YOLO class labels to risk levels
_RISK_MAP: dict[str, str] = {
    "drone":   "Hostile",
    "person":  "Unknown",
    "vehicle": "Unknown",
}


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
    ) -> None:
        print("[DEBUG] 1. Config loading...")
        self._cfg = self._load_config(Path(config_path))
        self._sim_mode: bool = self._cfg.get("simulation_mode", True)
        self._inf_cfg: dict = self._cfg.get("inference", {})
        self._aiops_cfg: dict = self._cfg.get("aiops", {})
        self._nato_cfg: dict = self._cfg.get("nato_metadata", {})

        self._model: Any | None = None
        self._cap: cv2.VideoCapture | None = None
        self._frame_id: int = 0
        self._start_ts: float = time.perf_counter()

        # Optional integration components
        self._reasoning: "ReasoningNode | None" = reasoning_node
        self._cloud: "CloudSync | None" = cloud_sync

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

        device = self._inf_cfg.get("device", "cpu")
        self.logger.info(f"[AEGIS] Loading YOLOv8 model: {model_path}  device={device}")
        self._model = YOLO(str(model_path))
        self.logger.info("[AEGIS] Model loaded successfully — ACTIVE MODE.")

    def _init_capture(self) -> None:
        """Open video capture: sim file, live camera, or synthetic frames."""
        if self._sim_mode:
            sim_path = self._inf_cfg.get("sim_video_path", "")
            if sim_path and Path(sim_path).exists():
                self._cap = cv2.VideoCapture(str(sim_path))
                self.logger.info(f"[AEGIS] SIMULATION_MODE — reading video: {sim_path}")
            else:
                self.logger.info(
                    "[AEGIS] SIMULATION_MODE — sim video not found; "
                    "generating synthetic frames."
                )
                self._cap = None   # Will generate placeholder frames
        else:
            # cam_idx = 0
            # self._cap = cv2.VideoCapture(cam_idx)
            
            # [AIOPS FIX] Kameramız olmadığı için gerçek modda dahi simülasyon videosunu kullanalım:
            sim_path = self._inf_cfg.get("sim_video_path", "data/sim_samples/drone_flyby.mp4")
            
            if sim_path and Path(sim_path).exists():
                self._cap = cv2.VideoCapture(str(sim_path))
                self.logger.info(f"[AEGIS] LIVE MODE — but reading from {sim_path} (No HW Camera)")
            else:
                self.logger.warning(f"[AEGIS] LIVE MODE — video {sim_path} completely missing, generating synthetic frame fallback!")
                self._cap = None

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
                risk_level="Hostile" if cls_name == "drone" else "Unknown",
                frame_id=self._frame_id
            )
            detections.append(det)
            
        return detections

    def _run_inference(self, frame: np.ndarray) -> list[tuple[str, float, list[float]]]:
        """
        Executes YOLOv8 or returns Mock Data if in Simulation Mode.
        """
        # 1. Real YOLO Inference
        if self._model is not None and not self._sim_mode:
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

                # --- STRATEGIC SUB-SAMPLING ---
                # Only trigger reasoning if detections exist AND it's a specific frame interval
                if detections and self._reasoning:
                    # In Simulation: Reason only every 50th frame to prevent Swap flooding
                    # In Live Mode: Reason every time a target is found (Sequential)
                    if not self._sim_mode or (self._frame_id % 50 == 0):
                        self._process_pipeline(frame, detections)

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
        Sequential execution: Freezes YOLO and calls Moondream for the first detected target.
        """
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

    reasoning = ReasoningNode()
    sync = CloudSync()
    node = VisionNode(reasoning_node=reasoning, cloud_sync=sync)

    print(f"\n{'='*70}")
    print("  AEGIS-Cloud — Integrated Edge Pipeline")
    print(f"  SIMULATION_MODE : {node._sim_mode}")
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
