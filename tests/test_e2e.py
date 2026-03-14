"""
AEGIS-Cloud — End-to-End Integration Test
==========================================
Validates the full data flow in simulation mode:

  Edge (sim) ──D2C──▶ CloudSync ──▶ [IoT Hub stub]
                                           ▓
                              ThreatEvaluator (LLM parser)
                                           ▓
                              RAGContext (doctrinal retrieval)
                                           ▓
                              SelfHealingService (C2D command)

Run from project root:
    python tests/test_e2e.py
"""

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(name)s — %(message)s")

# Project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from edge.src.comm.cloud_sync import CloudSync
from cloud.integrations.rag_context import RAGContext
from cloud.ops.self_healing_service import SelfHealingService
from cloud.functions.threat_evaluator.threat_evaluator import _parse_llm_json

_PAYLOAD_PATH = Path("data/logs/cloud_payload.json")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"


# ── Test 1: CloudSync D2C (simulation) ────────────────────────────────────

def test_cloud_sync_sim():
    print("\n[TEST 1] CloudSync — simulation D2C transmission")
    sync = CloudSync()

    detections = [{
        "target_type": "drone",
        "confidence": 0.92,
        "risk_level": "Hostile",
        "bbox": [0.3, 0.25, 0.55, 0.45],
    }]
    reasoning = [{
        "detection_id": "test-frame-1",
        "description": "A small quadcopter approaching from the north at low altitude.",
        "model_used": "moondream",
        "inference_time_ms": 1240.0,
    }]
    aiops = {
        "cpu_percent": 2.4,
        "gpu_temp_c": 78.0,
        "ram_used_mb": 3200.0,
        "latency_ms": 28.0,
    }

    result = sync.send(detections, reasoning, aiops)
    assert result, "CloudSync.send() returned False"

    assert _PAYLOAD_PATH.exists(), "cloud_payload.json not created"
    payload = json.loads(_PAYLOAD_PATH.read_text())
    assert payload["schema_version"] == "1.1"
    assert payload["azure_function_hints"]["priority"] == "HIGH"
    print(f"  {PASS} D2C payload written — msg_id={payload['message_id']} priority=HIGH")


# ── Test 2: LLM JSON Parser ───────────────────────────────────────────────

def test_parse_llm_json():
    print("\n[TEST 2] ThreatEvaluator — _parse_llm_json()")

    # Clean JSON
    r = _parse_llm_json(
        '{"threat_score": 85, "action_recommendation": "Jamming", '
        '"tactical_summary": "High-priority hostile drone.", "confidence": 0.91}'
    )
    assert r["threat_score"] == 85
    assert r["action_recommendation"] == "Jamming"
    print(f"  {PASS} Clean JSON parsed correctly")

    # Markdown-wrapped (common LLM output pattern)
    r = _parse_llm_json(
        '```json\n{"threat_score": 60, "action_recommendation": "Observation", '
        '"tactical_summary": "Unknown UAV. Monitoring.", "confidence": 0.65}\n```'
    )
    assert r["action_recommendation"] == "Observation"
    print(f"  {PASS} Markdown-wrapped JSON parsed correctly")

    # Unparseable fallback
    r = _parse_llm_json("I cannot provide a valid assessment at this time.")
    assert r.get("parse_error") is True
    print(f"  {PASS} Unparseable text triggers fallback correctly")


# ── Test 3: RAG Context ───────────────────────────────────────────────────

def test_rag_context():
    print("\n[TEST 3] RAGContext — simulation retrieval")
    rag = RAGContext()

    results = rag.retrieve("hostile drone counter-UAS engagement rules")
    assert isinstance(results, list)
    assert len(results) > 0, "No results returned"
    assert "content" in results[0]
    assert "relevance_score" in results[0]
    assert "source" in results[0]
    print(f"  {PASS} {len(results)} doctrinal results retrieved")
    print(f"  {PASS} Top result: score={results[0]['relevance_score']} "
          f"src={results[0]['source']}")

    # Signature retrieval
    sig_results = rag.retrieve_by_signature({
        "rf_freq_mhz": 2400, "visual_class": "quadcopter", "wingspan_cm": 30
    })
    assert isinstance(sig_results, list)
    print(f"  {PASS} Signature retrieval returned {len(sig_results)} results")


# ── Test 4: Self-Healing Service ──────────────────────────────────────────

def test_self_healing():
    print("\n[TEST 4] SelfHealingService — telemetry analysis + C2D command")

    config = {
        "motor": {
            "speed_min_rpm": 200,
            "speed_max_rpm": 8000,
            "vibration_safe_threshold_g": 2.5,
        },
        "aiops": {"anomaly_alert_threshold": 3},
    }
    svc = SelfHealingService(config)

    # Normal telemetry → no action
    normal = [{"vibration_rms_g": 1.2, "motor_rpm": 5000, "latency_ms": 25.0}] * 5
    action = svc.analyse_telemetry(normal)
    assert action is None, "Normal telemetry should not trigger action"
    print(f"  {PASS} Normal telemetry: no action (correct)")

    # Vibration spike → corrective action
    anomalous = [{"vibration_rms_g": 3.8, "motor_rpm": 6000, "latency_ms": 30.0}] * 5
    action = svc.analyse_telemetry(anomalous)
    assert action is not None, "Anomalous telemetry should trigger action"
    assert action["trigger"] == "vibration_anomaly"
    assert action["corrective_rpm"] < 6000, "Corrective RPM should be lower"
    print(f"  {PASS} Vibration anomaly: {action['measured_value_g']}g → "
          f"RPM {action['current_rpm']} → {action['corrective_rpm']}")

    # Latency spike → safe-mode RPM
    svc2 = SelfHealingService(config)
    slow = [{"vibration_rms_g": 0.5, "motor_rpm": 5000, "latency_ms": 750.0}] * 5
    action2 = svc2.analyse_telemetry(slow)
    assert action2 is not None
    assert action2["trigger"] == "latency_spike"
    assert action2["corrective_rpm"] == 200  # min RPM safe-mode
    print(f"  {PASS} Latency spike: {action2['measured_value_ms']}ms → "
          f"safe-mode RPM={action2['corrective_rpm']}")

    # Issue command → HITL gate (default: human_approval_required=True)
    cmd = svc.issue_command(action)
    assert cmd["status"] == "pending_approval"
    assert cmd["command_id"].startswith("cmd-")
    print(f"  {PASS} Command {cmd['command_id']} held for HITL approval "
          f"(EU AI Act Article 14)")

    # Audit trail written
    from cloud.ops.self_healing_service import _AUDIT_TRAIL_PATH
    assert _AUDIT_TRAIL_PATH.exists(), "Audit trail not written"
    print(f"  {PASS} Audit trail updated: {_AUDIT_TRAIL_PATH}")


# ── Test 5: Payload Schema Validation ─────────────────────────────────────

def test_payload_schema():
    print("\n[TEST 5] Payload schema validation (NATO STANAG-4586 v1.1)")

    if not _PAYLOAD_PATH.exists():
        print(f"  ⚠  No payload found at {_PAYLOAD_PATH}, run test_cloud_sync_sim first")
        return

    payload = json.loads(_PAYLOAD_PATH.read_text())

    required_fields = [
        "schema_version", "message_id", "timestamp_utc",
        "nato_metadata", "detections", "reasoning",
        "hardware_telemetry", "azure_function_hints",
    ]
    missing = [f for f in required_fields if f not in payload]
    assert not missing, f"Missing required fields: {missing}"

    hints = payload["azure_function_hints"]
    assert "priority" in hints
    assert "human_review_required" in hints
    assert "hostile_count" in hints

    nato = payload["nato_metadata"]
    assert "station_id" in nato
    assert "classification" in nato

    print(f"  {PASS} All required NATO fields present")
    print(f"  {PASS} priority={hints['priority']} | "
          f"hostile_count={hints['hostile_count']} | "
          f"HITL={hints['human_review_required']}")
    print(f"  {PASS} station_id={nato['station_id']}")


# ── Runner ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 56)
    print("  AEGIS-Cloud — End-to-End Integration Tests")
    print("=" * 56)

    tests = [
        test_cloud_sync_sim,
        test_parse_llm_json,
        test_rag_context,
        test_self_healing,
        test_payload_schema,
    ]

    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  {FAIL} FAILED: {e}")
            import traceback; traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 56}")
    print(f"  Results: {passed}/{len(tests)} passed", end="")
    if failed:
        print(f"  |  {failed} FAILED")
        sys.exit(1)
    else:
        print("  — All tests passed ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
