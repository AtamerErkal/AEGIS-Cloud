"""
AEGIS-Cloud — Edge / Comm / Cloud Sync
========================================
Module:   edge.src.comm.cloud_sync
Platform: NVIDIA Jetson Nano (JetPack 5.x)

Bidirectional Azure IoT Hub gateway.

D2C (Device → Cloud):  Sends composite NATO payloads (YOLO + Moondream +
                        telemetry) to Azure IoT Hub.
C2D (Cloud → Device):  Receives self-healing commands (motor speed,
                        recalibrate, config overrides).

When SIMULATION_MODE=True all outgoing payloads are written to
``data/logs/cloud_payload.json`` (overwritten each call so the file
always shows the latest transmission).  C2D commands are read from
``data/sim_samples/c2d_commands.json`` if present.

Reasoning model: Moondream (1.6B) via Ollama — the ``description`` field
in every reasoning entry of the D2C payload is populated directly from
Moondream's output.  The ``model_used`` field reflects the active model
tag so the Azure Function can log model provenance without code changes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Optional Azure IoT Hub SDK
# ---------------------------------------------------------------------------
try:
    from azure.iot.device import IoTHubDeviceClient, Message
    _IOT_SDK_AVAILABLE = True
except ImportError:
    _IOT_SDK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("edge/config/edge_settings.yaml")
_SIM_PAYLOAD_PATH = Path("data/logs/cloud_payload.json")
_SIM_LOG_PATH     = Path("data/logs/cloud_sync_history.jsonl")


# ---------------------------------------------------------------------------
# Cloud Sync
# ---------------------------------------------------------------------------

class CloudSync:
    """
    Azure IoT Hub bidirectional gateway.

    Parameters
    ----------
    config_path : Path | str
        Path to ``edge_settings.yaml``.
    """

    def __init__(self, config_path: Path | str = CONFIG_PATH) -> None:
        self._cfg = self._load_config(Path(config_path))
        self._sim_mode: bool = self._cfg.get("simulation_mode", True)
        self._sync_cfg: dict = self._cfg.get("cloud_sync", {})
        self._nato_cfg: dict = self._cfg.get("nato_metadata", {})
        # Read active reasoning model from config for telemetry provenance logging
        self._reasoning_model: str = (
            self._cfg.get("reasoning", {}).get("model_name", "moondream")
        )

        self._client: Any | None = None
        self._message_count: int = 0
        self._start_ts: float = time.perf_counter()

        self.logger = logging.getLogger("AEGIS.CloudSync")

        # Ensure log directories exist
        _SIM_PAYLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SIM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not self._sim_mode:
            self._connect()
        else:
            self.logger.info(
                f"[AEGIS] CloudSync SIMULATION_MODE — payloads → {_SIM_PAYLOAD_PATH}"
            )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Establish IoT Hub connection using env-var connection string."""
        if not _IOT_SDK_AVAILABLE:
            self.logger.error(
                "[AEGIS] azure-iot-device not installed. "
                "Run: pip install azure-iot-device"
            )
            return

        env_key = self._sync_cfg.get("iot_hub_connection_env", "AEGIS_IOT_HUB_CONN_STR")
        conn_str = os.environ.get(env_key, "")
        if not conn_str:
            self.logger.error(
                f"[AEGIS] IoT Hub connection string not found in env var '{env_key}'. "
                "Falling back to simulation mode."
            )
            self._sim_mode = True
            return

        max_attempts = int(self._sync_cfg.get("retry_max_attempts", 5))
        backoff = float(self._sync_cfg.get("retry_backoff_base_seconds", 2))

        for attempt in range(1, max_attempts + 1):
            try:
                self._client = IoTHubDeviceClient.create_from_connection_string(conn_str)
                self._client.connect()
                self.logger.info("[AEGIS] CloudSync connected to Azure IoT Hub.")
                return
            except Exception as exc:
                wait = backoff ** attempt
                self.logger.warning(
                    f"[AEGIS] IoT Hub connect attempt {attempt}/{max_attempts} "
                    f"failed: {exc}. Retrying in {wait:.0f}s…"
                )
                time.sleep(wait)

        self.logger.error("[AEGIS] IoT Hub: all retries exhausted — falling back to sim mode.")
        self._sim_mode = True

    def disconnect(self) -> None:
        """Gracefully close the IoT Hub connection."""
        if self._client is not None:
            try:
                self._client.disconnect()
                self.logger.info("[AEGIS] CloudSync disconnected.")
            except Exception as exc:
                self.logger.warning(f"[AEGIS] Disconnect error: {exc}")

    # ------------------------------------------------------------------
    # D2C — Device-to-Cloud telemetry
    # ------------------------------------------------------------------

    def send(
        self,
        detections: list[dict],
        reasoning_results: list[dict] | None = None,
        aiops_meta: dict | None = None,
    ) -> bool:
        """
        Build and transmit (or log) a composite NATO payload.

        Parameters
        ----------
        detections : list[dict]
            Serialised :class:`Detection` objects from the Vision Node.
        reasoning_results : list[dict] | None
            Serialised :class:`ReasoningResult` objects from the Reasoning Node.
            The ``description`` field of each entry is populated directly from
            Moondream's output (XAI requirement).
        aiops_meta : dict | None
            Hardware telemetry from the latest Vision Node AIOps collection.

        Returns
        -------
        bool
            ``True`` if the payload was sent/written successfully.
        """
        payload = self._build_payload(detections, reasoning_results or [], aiops_meta or {})
        self._message_count += 1

        if self._sim_mode:
            return self._sim_send(payload)
        return self._iot_send(payload)

    def _build_payload(
        self,
        detections: list[dict],
        reasoning_results: list[dict],
        aiops_meta: dict,
    ) -> dict[str, Any]:
        """
        Assemble the composite D2C payload in the NATO Incident Report format.

        The schema is STABLE regardless of reasoning availability — all
        fields are always present so the downstream Azure Function can rely
        on a fixed contract without defensive None-checks.

        Payload schema (v1.1)
        ---------------------
        {
          "schema_version":       "1.1",
          "message_id":           int,
          "timestamp_utc":        str,          // ISO-8601 ms precision + Z
          "nato_metadata":        {
              "station_id":           str,
              "unit_designation":     str,
              "classification_level": str,
              "reporting_format":     str,
              "coordinates":          {lat, lon, alt_m}
          },
          "detections": [
            {
              ... (all Detection.to_dict() fields) ...
            }
          ],
          "reasoning": [
            {
              "detection_id":         str,
              "status":               "OK" | "PASS_THROUGH" | "UNAVAILABLE",
              "description":          str,       // Moondream output — always a string
              "model_used":           str,       // e.g. "moondream"
              "inference_time_ms":    float,
              "simulation_mode":      bool,
              "error":                str | null
            }
          ],
          "hardware_telemetry":    {
              "cpu_percent":   float | null,
              "ram_used_mb":   float | null,
              "gpu_temp_c":    float | null,
              "gpu_util_pct":  float | null,
              "latency_ms":    float,
              "uptime_s":      float
          },
          "azure_function_hints": {
              "trigger_classification":  str,    // Highest-risk label in batch
              "hostile_count":           int,
              "unknown_count":           int,
              "reasoning_available":     bool,
              "priority":                "HIGH" | "MEDIUM" | "LOW",
              "human_review_required":   bool
          }
        }
        """
        coords = self._nato_cfg.get("unit_coordinates", {})
        normalised_reasoning = [self._normalise_reasoning(r) for r in reasoning_results]

        # ── Azure Function routing hints ─────────────────────────────────
        risk_levels = [d.get("risk_level", "Unknown") for d in detections]
        hostile_count = risk_levels.count("Hostile")
        unknown_count = risk_levels.count("Unknown")
        reasoning_ok = any(r["status"] == "OK" for r in normalised_reasoning)

        if hostile_count > 0:
            trigger_classification = "Hostile"
            priority = "HIGH"
        elif unknown_count > 0:
            trigger_classification = "Unknown"
            priority = "MEDIUM"
        else:
            trigger_classification = "Friendly"
            priority = "LOW"

        # Human review is mandatory for any Hostile detection (EU AI Act)
        human_review_required = hostile_count > 0

        return {
            "schema_version": "1.1",
            "message_id":     self._message_count,
            "timestamp_utc":  datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3] + "Z",
            "nato_metadata": {
                "station_id":           self._nato_cfg.get("station_id", "AEGIS-EDGE-001"),
                "unit_designation":     self._nato_cfg.get("unit_designation", "DIEHL-GARMR-UNIT-1"),
                "classification_level": self._nato_cfg.get(
                    "classification_level", "UNCLASSIFIED//FOR TRAINING ONLY"
                ),
                "reporting_format":     self._nato_cfg.get("reporting_format", "STANAG-4586-STUB"),
                "coordinates": {
                    "lat":   coords.get("lat", 0.0),
                    "lon":   coords.get("lon", 0.0),
                    "alt_m": coords.get("alt_m", 0.0),
                },
            },
            "detections":           detections,
            "reasoning":            normalised_reasoning,
            "hardware_telemetry":   self._normalise_telemetry(aiops_meta),
            "azure_function_hints": {
                "trigger_classification": trigger_classification,
                "hostile_count":          hostile_count,
                "unknown_count":          unknown_count,
                "reasoning_available":    reasoning_ok,
                "priority":               priority,
                "human_review_required":  human_review_required,
            },
        }

    @staticmethod
    def _normalise_reasoning(raw: dict) -> dict[str, Any]:
        """
        Guarantee a fixed-schema reasoning entry regardless of whether
        Moondream was available, degraded, or in PASS-THROUGH mode.

        The ``description`` field is always the raw Moondream text output
        when status is ``OK``, satisfying the XAI requirement that the JSON
        payload ``description`` is populated by the model's response.

        The Azure Function can always key off ``status`` without
        needing to inspect the ``description`` string for sentinel values.

        Status values
        -------------
        ``OK``           — Moondream produced a real tactical description.
        ``PASS_THROUGH`` — Circuit-breaker degraded; no description available.
        ``UNAVAILABLE``  — Reasoning was not wired up for this detection.
        """
        desc: str = raw.get("description", "") or ""
        error: str | None = raw.get("error")

        if error or "PASS-THROUGH" in desc or "UNAVAILABLE" in desc:
            status = "PASS_THROUGH"
            # Sanitise description so the Azure Function never receives
            # internal sentinel strings — replace with a stable empty value.
            clean_desc = ""
        else:
            status = "OK"
            clean_desc = desc   # Moondream's verbatim tactical output

        return {
            "detection_id":      raw.get("detection_id", ""),
            "status":            status,
            "description":       clean_desc,   # XAI: Moondream output propagated here
            "model_used":        raw.get("model_used", "moondream"),
            "inference_time_ms": raw.get("inference_time_ms", 0.0),
            "simulation_mode":   raw.get("simulation_mode", True),
            "error":             error,
        }

    @staticmethod
    def _normalise_telemetry(raw: dict) -> dict[str, Any]:
        """
        Ensure hardware_telemetry always contains the full set of expected
        fields (defaulting to None for unavailable metrics) so the Azure
        Function schema validator never encounters a missing key.
        """
        return {
            "cpu_percent":  raw.get("cpu_percent"),
            "ram_used_mb":  raw.get("ram_used_mb"),
            "gpu_temp_c":   raw.get("gpu_temp_c"),
            "gpu_util_pct": raw.get("gpu_util_pct"),
            "latency_ms":   raw.get("latency_ms", 0.0),
            "uptime_s":     raw.get("uptime_s", 0.0),
            "platform":     raw.get("platform", ""),
        }

    def _sim_send(self, payload: dict) -> bool:
        """
        Write the payload to two locations:

        1. ``data/logs/cloud_payload.json``         — always the latest message
           (overwritten each call, easy to inspect in VS Code).
        2. ``data/logs/cloud_sync_history.jsonl``   — rolling append-only log
           of message summaries for trend analysis.

        The payload written is the exact same JSON the real IoT Hub path
        would transmit, so it is directly usable by the Azure Functions
        local emulator (``func start``) or Postman.
        """
        try:
            with _SIM_PAYLOAD_PATH.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)

            # Rolling history — compact one-liner per message
            hints = payload.get("azure_function_hints", {})
            summary = {
                "message_id":              payload["message_id"],
                "timestamp_utc":           payload["timestamp_utc"],
                "station_id":              payload["nato_metadata"]["station_id"],
                "detection_count":         len(payload.get("detections", [])),
                "priority":                hints.get("priority", "UNKNOWN"),
                "trigger_classification":  hints.get("trigger_classification", "Unknown"),
                "human_review_required":   hints.get("human_review_required", False),
                "reasoning_available":     hints.get("reasoning_available", False),
                "reasoning_model":         self._reasoning_model,
            }
            with _SIM_LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary) + "\n")

            self.logger.info(
                f"[AEGIS][CloudSync] SIM D2C msg#{payload['message_id']} "
                f"priority={hints.get('priority','?')} "
                f"class={hints.get('trigger_classification','?')} "
                f"→ {_SIM_PAYLOAD_PATH}"
            )
            return True
        except Exception as exc:
            self.logger.error(f"[AEGIS][CloudSync] SIM write failed: {exc}")
            return False

    def _iot_send(self, payload: dict) -> bool:
        """Send payload to Azure IoT Hub as a JSON D2C message."""
        if self._client is None:
            self.logger.error("[AEGIS][CloudSync] No IoT client — message dropped.")
            return False
        try:
            raw = json.dumps(payload)
            msg = Message(raw)
            msg.content_encoding = "utf-8"
            msg.content_type = "application/json"
            msg.custom_properties["station_id"] = (
                payload["nato_metadata"]["station_id"]
            )
            msg.custom_properties["detection_count"] = str(
                len(payload.get("detections", []))
            )
            self._client.send_message(msg)
            self.logger.info(
                f"[AEGIS][CloudSync] D2C msg#{payload['message_id']} sent "
                f"({len(raw)} bytes)"
            )
            return True
        except Exception as exc:
            self.logger.error(f"[AEGIS][CloudSync] D2C send failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # C2D — Cloud-to-Device command reception
    # ------------------------------------------------------------------

    def receive_command(self) -> dict | None:
        """
        Poll for the next C2D command.

        In simulation mode, reads from ``data/sim_samples/c2d_commands.json``
        (if present and non-empty) and pops the first command.

        Returns None if no command is pending.
        """
        if self._sim_mode:
            return self._sim_receive()
        return self._iot_receive()

    def _sim_receive(self) -> dict | None:
        """Read and pop the first command from the sim commands file."""
        sim_cmd_path = Path(
            self._sync_cfg.get("sim_commands_path", "data/sim_samples/c2d_commands.json")
        )
        if not sim_cmd_path.exists():
            return None
        try:
            with sim_cmd_path.open("r", encoding="utf-8") as fh:
                commands: list = json.load(fh)
            if not commands:
                return None
            cmd = commands.pop(0)
            with sim_cmd_path.open("w", encoding="utf-8") as fh:
                json.dump(commands, fh, indent=2)
            self.logger.info(f"[AEGIS][CloudSync] SIM C2D received: {cmd}")
            return cmd
        except Exception as exc:
            self.logger.warning(f"[AEGIS][CloudSync] SIM C2D read error: {exc}")
            return None

    def _iot_receive(self) -> dict | None:
        """Non-blocking poll for a C2D message from IoT Hub."""
        if self._client is None:
            return None
        try:
            msg = self._client.receive_message(timeout=0.1)
            if msg is None:
                return None
            data = json.loads(msg.data.decode("utf-8"))
            self.logger.info(f"[AEGIS][CloudSync] C2D received: {data}")
            return data
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def health_check(self) -> dict[str, Any]:
        """AIOps health snapshot — consumed by the Cloud self-healing service."""
        return {
            "status":          "OK" if (self._sim_mode or self._client is not None) else "FAILED",
            "simulation_mode": self._sim_mode,
            "iot_connected":   self._client is not None,
            "messages_sent":   self._message_count,
            "uptime_s":        round(time.perf_counter() - self._start_ts, 1),
            "reasoning_model": self._reasoning_model,
        }
