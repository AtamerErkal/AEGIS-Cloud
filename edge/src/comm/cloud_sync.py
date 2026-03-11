"""
AEGIS-Cloud — Edge / Comm / Cloud Sync
========================================
Module:    edge.src.comm.cloud_sync
Platform:  NVIDIA Jetson Nano

Bidirectional Azure IoT Hub gateway.
- D2C: Transmits composite NATO payloads (YOLO + Moondream + Telemetry).
- C2D: Receives self-healing commands and configuration overrides from Cloud.
- XAI: Propagates verbatim Moondream reports for Explainable AI compliance.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
import os

load_dotenv() # This automatically looks for the .env file
conn_str = os.getenv("AEGIS_IOT_HUB_CONN_STR")

import yaml

# ---------------------------------------------------------------------------
# Azure IoT Hub SDK Integration
# ---------------------------------------------------------------------------
try:
    from azure.iot.device import IoTHubDeviceClient, Message
    _IOT_SDK_AVAILABLE = True
except ImportError:
    _IOT_SDK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants & Persistence Paths
# ---------------------------------------------------------------------------
CONFIG_PATH = Path("edge/config/edge_settings.yaml")
_SIM_PAYLOAD_PATH = Path("data/logs/cloud_payload.json")
_SIM_LOG_PATH     = Path("data/logs/cloud_sync_history.jsonl")

class CloudSync:
    """
    Azure IoT Hub bidirectional gateway with NATO STANAG-4586 schema support.
    """

    def __init__(self, config_path: Path | str = CONFIG_PATH) -> None:
        self._cfg = self._load_config(Path(config_path))
        self._sim_mode: bool = self._cfg.get("simulation_mode", True)
        self._sync_cfg: dict = self._cfg.get("cloud_sync", {})
        self._nato_cfg: dict = self._cfg.get("nato_metadata", {})
        
        # Provenance tracking: capture the active reasoning model from config
        self._reasoning_model: str = (
            self._cfg.get("reasoning", {}).get("model_name", "moondream")
        )

        self._client: IoTHubDeviceClient | None = None
        self._message_count: int = 0
        self._start_ts: float = time.perf_counter()

        self.logger = logging.getLogger("AEGIS.CloudSync")

        # Ensure directory structure for local persistence
        _SIM_PAYLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SIM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not self._sim_mode:
            self._connect()
        else:
            self.logger.info(
                f"[AEGIS] CloudSync in SIMULATION_MODE — output: {_SIM_PAYLOAD_PATH}"
            )

    # ------------------------------------------------------------------
    # Connection Management (Reliable Connectivity)
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Establish connection with exponential backoff retry logic."""
        if not _IOT_SDK_AVAILABLE:
            self.logger.error("[AEGIS] azure-iot-device SDK not found. Defaulting to SIM.")
            self._sim_mode = True
            return

        # Force load .env from the project root since cwd might differ
        from dotenv import load_dotenv
        env_path = Path.home() / "AEGIS-Cloud" / ".env"
        load_dotenv(env_path)

        env_key = self._sync_cfg.get("iot_hub_connection_env", "AEGIS_IOT_HUB_CONN_STR")
        conn_str = os.environ.get(env_key, "")
        
        if not conn_str:
            self.logger.error(f"[AEGIS] Env-var '{env_key}' is empty. Cannot connect to Cloud.")
            self._sim_mode = True
            return

        # [AIOPS FIX] If we found a connection string, force sim_mode to False
        self._sim_mode = False

        max_attempts = int(self._sync_cfg.get("retry_max_attempts", 5))
        backoff = float(self._sync_cfg.get("retry_backoff_base_seconds", 2))

        for attempt in range(1, max_attempts + 1):
            try:
                self._client = IoTHubDeviceClient.create_from_connection_string(conn_str)
                self._client.connect()
                self.logger.info("[AEGIS] CloudSync: Successfully authenticated with Azure IoT Hub.")
                return
            except Exception as exc:
                wait = backoff ** attempt
                self.logger.warning(
                    f"[AEGIS] IoT Hub connection attempt {attempt}/{max_attempts} failed: {exc}. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)

        self.logger.error("[AEGIS] IoT Hub: All retries exhausted. System locked in SIMULATION_MODE.")
        self._sim_mode = True

    def disconnect(self) -> None:
        """Gracefully close the hardware-cloud bridge."""
        if self._client:
            try:
                self._client.disconnect()
                self.logger.info("[AEGIS] CloudSync: Gateway closed.")
            except Exception as e:
                self.logger.debug(f"Disconnect error: {e}")

    # ------------------------------------------------------------------
    # D2C — Device-to-Cloud (Telemetry Transmission)
    # ------------------------------------------------------------------

    def send(self, detections: list[dict], reasoning_results: list[dict] | None = None, aiops_meta: dict | None = None) -> bool:
        """
        Builds and transmits a composite NATO STANAG-4586 compliant payload.
        """
        payload = self._build_payload(detections, reasoning_results or [], aiops_meta or {})
        self._message_count += 1

        if self._sim_mode:
            return self._sim_send(payload)
        return self._iot_send(payload)

    def _build_payload(self, detections: list[dict], reasoning_results: list[dict], aiops_meta: dict) -> dict[str, Any]:
        """
        Assembles a v1.1 composite payload.
        Ensures a stable schema for downstream Azure Function consumption.
        """
        coords = self._nato_cfg.get("unit_coordinates", {})
        normalised_reasoning = [self._normalise_reasoning(r) for r in reasoning_results]

        # Triage logic for Cloud routing
        risk_levels = [d.get("risk_level", "Unknown") for d in detections]
        hostile_count = risk_levels.count("Hostile")
        priority = "HIGH" if hostile_count > 0 else "LOW"

        return {
            "schema_version": "1.1",
            "message_id": self._message_count,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "nato_metadata": {
                "station_id": self._nato_cfg.get("station_id", "AEGIS-EDGE-001"),
                "unit_designation": self._nato_cfg.get("unit_designation", "DIEHL-GARMR-1"),
                "classification": "UNCLASSIFIED//FOR TRAINING ONLY",
                "coordinates": coords
            },
            "detections": detections,
            "reasoning": normalised_reasoning,
            "hardware_telemetry": self._normalise_telemetry(aiops_meta),
            "azure_function_hints": {
                "priority": priority,
                "hostile_count": hostile_count,
                "reasoning_available": any(r["status"] == "OK" for r in normalised_reasoning),
                "human_review_required": hostile_count > 0 # Mandatory for EU AI Act compliance
            }
        }

    def _iot_send(self, payload: dict) -> bool:
        """Serializes and sends the JSON message to Azure IoT Hub with routing tags."""
        try:
            raw_json = json.dumps(payload)
            msg = Message(raw_json)
            msg.content_encoding = "utf-8"
            msg.content_type = "application/json"
            
            # Custom properties for Azure Stream Analytics hot-path routing
            msg.custom_properties["priority"] = payload["azure_function_hints"]["priority"]
            msg.custom_properties["station_id"] = payload["nato_metadata"]["station_id"]
            
            self._client.send_message(msg)
            self.logger.info(f"[AEGIS][CloudSync] D2C msg#{payload['message_id']} sent to IoT Hub.")
            return True
        except Exception as exc:
            self.logger.error(f"[AEGIS][CloudSync] D2C Transmission Failed: {exc}")
            return False

    def _sim_send(self, payload: dict) -> bool:
        """Mock transmission: persists the exact JSON payload to local disk."""
        try:
            with _SIM_PAYLOAD_PATH.open("w") as f:
                json.dump(payload, f, indent=2)
            
            # History log (jsonl format)
            with _SIM_LOG_PATH.open("a") as fl:
                summary = {"msg_id": payload["message_id"], "ts": payload["timestamp_utc"]}
                fl.write(json.dumps(summary) + "\n")
                
            self.logger.info(f"[AEGIS][CloudSync] SIM D2C msg#{payload['message_id']} saved to logs.")
            return True
        except Exception as exc:
            self.logger.error(f"[AEGIS][CloudSync] SIM Log Write Error: {exc}")
            return False

    # ------------------------------------------------------------------
    # C2D — Cloud-to-Device (Control Logic)
    # ------------------------------------------------------------------

    def receive_command(self) -> dict | None:
        """Polls for inbound commands (Self-healing / Remote config)."""
        if self._sim_mode:
            return self._sim_receive()
        
        try:
            # Poll with 1s timeout to prevent perception loop blockage
            msg = self._client.receive_message(timeout=1)
            if msg:
                cmd = json.loads(msg.data.decode("utf-8"))
                self.logger.info(f"[AEGIS] Cloud Command Received: {cmd}")
                return cmd
        except Exception:
            return None
        return None

    def _sim_receive(self) -> dict | None:
        """Reads mock commands from local JSON buffer."""
        sim_cmd_path = Path("data/sim_samples/c2d_commands.json")
        if not sim_cmd_path.exists(): return None
        try:
            with sim_cmd_path.open("r") as f:
                cmds = json.load(f)
                if not cmds: return None
                cmd = cmds.pop(0)
            # Update file after popping the command
            with sim_cmd_path.open("w") as f:
                json.dump(cmds, f, indent=2)
            return cmd
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Data Normalization (Schema Integrity)
    # ------------------------------------------------------------------

    def _normalise_reasoning(self, raw: dict) -> dict:
        """Ensure verbatim Moondream output is preserved for XAI compliance."""
        desc = raw.get("description", "")
        # Filter internal sentinel strings from Moondream
        status = "OK" if (desc and "PASS-THROUGH" not in desc) else "UNAVAILABLE"
        
        return {
            "detection_id": raw.get("detection_id"),
            "status": status,
            "description": desc if status == "OK" else "",
            "model_used": raw.get("model_used", self._reasoning_model),
            "inference_time_ms": raw.get("inference_time_ms", 0.0)
        }

    def _normalise_telemetry(self, raw: dict) -> dict:
        """Guarantee a full set of hardware metrics for cloud analysis."""
        return {
            "cpu_percent": raw.get("cpu_percent"),
            "gpu_temp_c": raw.get("gpu_temp_c"),
            "ram_used_mb": raw.get("ram_used_mb"),
            "latency_ms": raw.get("latency_ms", 0.0),
            "uptime_s": round(time.perf_counter() - self._start_ts, 1)
        }

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists(): return {}
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}