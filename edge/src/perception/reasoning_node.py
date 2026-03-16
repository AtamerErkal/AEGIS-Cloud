"""
AEGIS-Cloud — Edge / Perception / Reasoning Node
==================================================
Module:   edge.src.perception.reasoning_node
Platform: NVIDIA Jetson Nano (JetPack 5.x)

Ollama/Moondream wrapper that generates a tactical scene description from a
detection crop.  Communicates with the local Ollama server via the
``ollama`` Python library (with a ``requests`` fallback).

Model: moondream (1.6B) — resource-optimised for Jetson Nano 4GB.
Pull:  ollama pull moondream

When SIMULATION_MODE=True and Ollama is unreachable, returns a cached
synthetic response so the rest of the pipeline keeps running.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Project-root bootstrap
# Enables running from the AEGIS-Cloud/ root directory:
#   python edge/src/perception/reasoning_node.py
# without setting PYTHONPATH manually.
# ---------------------------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = str(_Path(__file__).resolve().parents[3])  # …/AEGIS-Cloud/
if _PROJECT_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJECT_ROOT)
del _sys, _Path, _PROJECT_ROOT

import base64
import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Optional ollama library — fall back to requests if absent
# ---------------------------------------------------------------------------
try:
    import ollama as _ollama_lib
    _OLLAMA_LIB = True
except ImportError:
    _OLLAMA_LIB = False

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional direct Moondream — HuggingFace transformers + PyTorch CUDA
# Bypasses Ollama entirely; uses Jetson Nano's GPU via JetPack PyTorch.
# Install: pip install transformers pillow
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModelForCausalLM as _AutoModel
    from transformers import AutoTokenizer as _AutoTokenizer
    from PIL import Image as _PIL_Image
    import torch as _torch
    _DIRECT_AVAILABLE = True
except ImportError:
    _DIRECT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("edge/config/edge_settings.yaml")

_TACTICAL_PROMPT = (
    "Naval vessel image. One line only: "
    "TYPE:[warship/patrol/cargo/fishing/unknown] "
    "THREAT:[LOW/MEDIUM/HIGH/CRITICAL] "
    "NOTES:[hull colour, armament if visible]. "
    "If military add MILITARY CONTACT."
)

_SIM_RESPONSE = (
    "SIM | MILITARY CONTACT. Grey-hulled patrol vessel, approx 40-60 m length. "
    "Radar mast visible. No visible armament. Hull markings unclear. "
    "Range: estimated 800-1200 m. "
    "Threat level: HIGH. Recommend tracking and reporting to command."
)

# Default crop size — overridden by reasoning.image_resize_px in YAML
_DEFAULT_IMAGE_RESIZE_PX = 336


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ReasoningResult:
    """Output produced by the Reasoning Node for a single detection."""

    detection_id: str                  # Correlates to Detection.frame_id + timestamp
    description: str                   # Moondream tactical description (XAI payload field)
    inference_time_ms: float           # AIOps: Moondream wall-clock latency
    model_used: str                    # e.g. "moondream"
    prompt: str                        # Prompt sent to Moondream for XAI audit trail
    simulation_mode: bool = False
    error: str | None = None
    aiops_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id":       self.detection_id,
            "description":        self.description,
            "inference_time_ms":  round(self.inference_time_ms, 2),
            "model_used":         self.model_used,
            "prompt":             self.prompt,
            "simulation_mode":    self.simulation_mode,
            "error":              self.error,
            "aiops_meta":         self.aiops_meta,
        }


# ---------------------------------------------------------------------------
# Reasoning Node
# ---------------------------------------------------------------------------

class ReasoningNode:
    """
    Moondream multi-modal reasoning wrapper.

    Sends a detection crop + tactical prompt to the local Ollama server
    and returns a structured :class:`ReasoningResult`.

    Model: moondream (1.6B) — optimised for Jetson Nano 4GB edge deployment.
    Deploy: ``ollama pull moondream``

    Circuit-Breaker with Cool-down
    --------------------------------
    After ``max_retries`` consecutive Ollama failures the node enters
    DEGRADED mode and returns pass-through results instantly, preventing
    pipeline stalls on a flapping Ollama process.

    After ``cooldown_seconds`` the breaker automatically makes one probe
    call.  On success it closes (ACTIVE); on failure it re-arms the timer
    so the pipeline is never permanently blocked without a manual reset.

    Parameters
    ----------
    config_path : Path | str
        Path to ``edge_settings.yaml``.
    """

    def __init__(self, config_path: Path | str = CONFIG_PATH) -> None:
        self._cfg = self._load_config(Path(config_path))
        self._sim_mode: bool = self._cfg.get("simulation_mode", True)
        self._rsn_cfg: dict = self._cfg.get("reasoning", {})

        self.endpoint: str = self._rsn_cfg.get("ollama_endpoint", "http://localhost:11434")
        self.model: str = self._rsn_cfg.get("model_name", "moondream")
        self.timeout: int = int(self._rsn_cfg.get("timeout_seconds", 60))
        self.max_retries: int = int(self._rsn_cfg.get("max_retries", 3))
        self._cooldown_seconds: float = float(
            self._rsn_cfg.get("cooldown_seconds", 120)
        )
        self._image_resize_px: int = int(
            self._rsn_cfg.get("image_resize_px", _DEFAULT_IMAGE_RESIZE_PX)
        )
        # Moondream inference options (temperature, token limit)
        self._model_options: dict = self._rsn_cfg.get(
            "options", {"temperature": 0.1, "num_predict": 200}
        )

        self._consecutive_failures: int = 0
        self._degraded: bool = False        # Circuit-breaker flag
        self._degraded_since: float | None = None  # perf_counter timestamp when opened

        # Direct mode: use HuggingFace transformers + CUDA instead of Ollama
        self._direct_mode: bool = self._rsn_cfg.get("direct_mode", False)
        self._md_model = None   # lazy-loaded on first _call_direct()
        self._md_tok   = None

        self.logger = logging.getLogger("AEGIS.ReasoningNode")
        mode_str = "DIRECT/CUDA" if self._direct_mode else f"Ollama({self.endpoint})"
        self.logger.info(
            f"[AEGIS] ReasoningNode initialised — "
            f"mode={mode_str}  model={self.model}  "
            f"sim={self._sim_mode}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe(
        self,
        frame: np.ndarray,
        bbox: list[float],
        detection_id: str,
    ) -> ReasoningResult:
        """
        Generate a tactical description for the detection crop.

        Parameters
        ----------
        frame : np.ndarray
            Full BGR frame from the Vision Node.
        bbox : list[float]
            Normalised bounding box ``[x1, y1, x2, y2]``.
        detection_id : str
            Correlation ID (``"frame{id}_{timestamp}"``).

        Returns
        -------
        ReasoningResult
            Populated with Moondream description, inference time, and
            AIOps metadata.  The ``description`` field is always a non-empty
            string so the XAI payload contract is never violated.
        """
        if self._degraded:
            # ── Cool-down probe ─────────────────────────────────────────────
            # Once cooldown_seconds have elapsed, tentatively close the
            # breaker and fall through to a real Ollama attempt.  If that
            # attempt fails, the breaker re-opens and the timer resets.
            if self._degraded_since is not None:
                elapsed = time.perf_counter() - self._degraded_since
                if elapsed >= self._cooldown_seconds:
                    self.logger.info(
                        f"[AEGIS] Circuit-breaker PROBING after "
                        f"{elapsed:.0f}s cool-down — attempting Ollama call…"
                    )
                    self._degraded = False          # Tentatively close
                    self._consecutive_failures = 0  # Fresh slate for the probe
                    # Fall through to normal attempt below
                else:
                    return self._degraded_result(detection_id)
            else:
                return self._degraded_result(detection_id)

        crop_bgr = self._get_crop(frame, bbox)
        crop_b64 = self._bgr_to_b64(crop_bgr)
        t0 = time.perf_counter()

        try:
            if self._direct_mode:
                description = self._call_direct(crop_bgr)
            else:
                description = self._call_ollama(crop_b64)
            self._consecutive_failures = 0          # Reset circuit-breaker
        except Exception as exc:
            self._consecutive_failures += 1
            self.logger.warning(
                f"[AEGIS] Ollama call failed ({self._consecutive_failures}/"
                f"{self.max_retries}): {exc}"
            )
            if self._consecutive_failures >= self.max_retries:
                self._degraded = True
                self._degraded_since = time.perf_counter()  # Start cool-down clock
                self.logger.error(
                    f"[AEGIS] Circuit-breaker OPEN — ReasoningNode degraded to pass-through. "
                    f"Will auto-probe after {self._cooldown_seconds:.0f}s."
                )
            return ReasoningResult(
                detection_id=detection_id,
                description="[REASONING UNAVAILABLE — circuit-breaker open]",
                inference_time_ms=(time.perf_counter() - t0) * 1000,
                model_used=self.model,
                prompt=_TACTICAL_PROMPT,
                simulation_mode=self._sim_mode,
                error=str(exc),
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.logger.info(
            f"[AEGIS][ReasoningNode] id={detection_id}  "
            f"latency={latency_ms:.1f}ms  model={self.model}"
        )

        return ReasoningResult(
            detection_id=detection_id,
            description=description,
            inference_time_ms=latency_ms,
            model_used=self.model,
            prompt=_TACTICAL_PROMPT,
            simulation_mode=self._sim_mode,
            aiops_meta={"consecutive_failures": self._consecutive_failures},
        )

    # ------------------------------------------------------------------
    # Ollama communication
    # ------------------------------------------------------------------

    def _call_ollama(self, image_b64: str) -> str:
        """
        Send the cropped image + tactical prompt to the Ollama Moondream model.

        Call order
        ----------
        1. ``requests`` HTTP POST — preferred; socket-level timeout is guaranteed.
        2. ``ollama`` Python library — fallback; wrapped in a daemon thread so
           ``self.timeout`` is enforced even on a slow Jetson cold-start.
        3. SIM cached response — when no network lib is available.
        """
        self.logger.info(
            f"[AEGIS][ReasoningNode] Calling Moondream (timeout={self.timeout}s) — "
            "waiting for inference…"
        )

        messages = [
            {
                "role": "user",
                "content": _TACTICAL_PROMPT,
                "images": [image_b64],
            }
        ]

        # --- 1. requests HTTP path (hard timeout at socket level) ---
        if _REQUESTS_AVAILABLE:
            url = f"{self.endpoint}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": self._model_options,
            }
            resp = _requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()

        # --- 2. ollama library fallback (thread-based timeout) ---
        if _OLLAMA_LIB:
            def _lib_call() -> str:
                resp = _ollama_lib.chat(
                    model=self.model,
                    messages=messages,
                    options=self._model_options,
                )
                return resp["message"]["content"].strip()

            _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = _executor.submit(_lib_call)
            try:
                result = future.result(timeout=self.timeout)
                _executor.shutdown(wait=False)   # clean up without blocking
                return result
            except concurrent.futures.TimeoutError:
                _executor.shutdown(wait=False)   # abandon thread — do NOT wait
                raise TimeoutError(
                    f"Ollama library call timed out after {self.timeout}s "
                    "(Moondream cold-start on Jetson?)"
                )

        # --- 3. Simulation fallback ---
        if self._sim_mode:
            self.logger.warning(
                "[AEGIS] No Ollama client available — using SIM cached response."
            )
            return _SIM_RESPONSE

        raise RuntimeError(
            "Cannot reach Ollama: neither 'requests' nor 'ollama' library "
            "is installed, and SIMULATION_MODE is False."
        )


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _get_crop(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """Crop + resize the bbox region from frame. Returns BGR ndarray."""
        h, w = frame.shape[:2]
        x1, y1 = int(bbox[0] * w), int(bbox[1] * h)
        x2, y2 = int(bbox[2] * w), int(bbox[3] * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frame
        px = self._image_resize_px
        return cv2.resize(crop, (px, px), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _bgr_to_b64(crop_bgr: np.ndarray) -> str:
        """Encode a BGR crop as base64 PNG for the Ollama image API."""
        ok, buf = cv2.imencode(".png", crop_bgr)
        if not ok:
            raise RuntimeError("cv2.imencode failed — cannot prepare image for Moondream.")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # Keep old name as alias so external callers don't break
    def _encode_crop(self, frame: np.ndarray, bbox: list) -> str:
        return self._bgr_to_b64(self._get_crop(frame, bbox))

    def _call_direct(self, crop_bgr: np.ndarray) -> str:
        """
        Run Moondream2 inference directly via HuggingFace transformers + CUDA.

        Bypasses Ollama entirely — uses Jetson Nano's GPU through JetPack PyTorch.
        Model is lazy-loaded and cached on the first call (~1.7 GB download once).

        Install deps:  pip install transformers pillow
        """
        if not _DIRECT_AVAILABLE:
            raise RuntimeError(
                "Direct mode requires: pip install transformers pillow\n"
                "torch is pre-installed on JetPack — no extra step needed."
            )

        # ── Lazy load (once per process) ──────────────────────────────────
        if self._md_model is None:
            _MODEL_ID  = "vikhyatk/moondream2"
            _REVISION  = "2024-05-08"
            device     = "cuda" if _torch.cuda.is_available() else "cpu"
            dtype      = _torch.float16 if device == "cuda" else _torch.float32
            self.logger.info(
                f"[AEGIS] Loading Moondream2 on {device} "
                f"(first call — may download ~1.7 GB)…"
            )
            self._md_tok = _AutoTokenizer.from_pretrained(
                _MODEL_ID, revision=_REVISION, trust_remote_code=True
            )
            self._md_model = _AutoModel.from_pretrained(
                _MODEL_ID, revision=_REVISION, trust_remote_code=True,
                torch_dtype=dtype, low_cpu_mem_usage=True,
            ).to(device).eval()
            self.logger.info(f"[AEGIS] Moondream2 READY on {device}")

        # ── Inference ─────────────────────────────────────────────────────
        pil_img = _PIL_Image.fromarray(crop_bgr[:, :, ::-1])   # BGR → RGB
        with _torch.no_grad():
            try:
                # moondream2 API post-2024
                result = self._md_model.query(pil_img, _TACTICAL_PROMPT)
                if isinstance(result, dict):
                    return result.get("answer", str(result)).strip()
                return str(result).strip()
            except AttributeError:
                # Older revision fallback
                enc = self._md_model.encode_image(pil_img)
                return self._md_model.answer_question(
                    enc, _TACTICAL_PROMPT, self._md_tok
                ).strip()

    def _degraded_result(self, detection_id: str) -> ReasoningResult:
        """Return a pass-through result when the circuit-breaker is open."""
        return ReasoningResult(
            detection_id=detection_id,
            description="[PASS-THROUGH — ReasoningNode circuit-breaker open]",
            inference_time_ms=0.0,
            model_used=self.model,
            prompt=_TACTICAL_PROMPT,
            simulation_mode=self._sim_mode,
            error="Circuit-breaker open after repeated Ollama failures.",
        )

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit-breaker (e.g. operator command via C2D)."""
        self._consecutive_failures = 0
        self._degraded = False
        self.logger.info("[AEGIS] ReasoningNode circuit-breaker RESET by operator.")

    def health_check(self) -> dict[str, Any]:
        """AIOps health snapshot consumed by the Cloud self-healing service."""
        ollama_reachable = False
        if _REQUESTS_AVAILABLE:
            try:
                r = _requests.get(f"{self.endpoint}/api/tags", timeout=3)
                ollama_reachable = r.status_code == 200
            except Exception:
                pass
        return {
            "status":                "DEGRADED" if self._degraded else "OK",
            "ollama_reachable":      ollama_reachable,
            "model":                 self.model,
            "consecutive_failures":  self._consecutive_failures,
            "circuit_breaker_open":  self._degraded,
            "cooldown_remaining_s":  max(
                0.0,
                self._cooldown_seconds - (time.perf_counter() - self._degraded_since)
            ) if self._degraded and self._degraded_since else 0.0,
            "simulation_mode":       self._sim_mode,
        }
