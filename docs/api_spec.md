# AEGIS-Cloud — API Specification

> **Version**: 0.1.0-skeleton
> **Date**: 2026-03-10
> **Status**: Draft — interfaces defined, implementations pending

---

## 1. Overview

This document defines the internal and external API contracts for the
AEGIS-Cloud system.  It covers:

1. **Edge → Cloud** (D2C Telemetry Messages via Azure IoT Hub)
2. **Cloud → Edge** (C2D Command Messages via Azure IoT Hub)
3. **Cloud → VANGUARD** (External Intelligence API)
4. **Cloud → Azure AI Search** (RAG Query API)
5. **Internal Module Interfaces** (Python inter-module contracts)

---

## 2. D2C Telemetry Message Schema (Edge → Cloud)

```json
{
  "message_type": "detection" | "telemetry" | "health",
  "correlation_id": "uuid-v4",
  "timestamp_utc": "ISO-8601",
  "source_device": "aegis-edge-001",

  "detection": {
    "bbox": [x1, y1, x2, y2],
    "class_id": 0,
    "class_name": "drone",
    "confidence": 0.87,
    "grad_cam_ref": "sha256:...",
    "description": "Small quadcopter, 4-rotor, no visible payload"
  },

  "platform_state": {
    "orientation": [0.5, -1.2, 45.0],
    "acceleration": [0.01, 0.02, 9.81],
    "range_mm": 1250,
    "vibration_rms": 0.8,
    "provenance": "sha256:..."
  },

  "health": {
    "vision_fps": 28.5,
    "inference_latency_ms": 12.3,
    "sensor_health": { "imu": "OK", "tof": "OK" },
    "ollama_status": "healthy"
  }
}
```

---

## 3. C2D Command Message Schema (Cloud → Edge)

```json
{
  "command_type": "motor_adjust" | "recalibrate" | "safe_mode" | "config_update",
  "command_id": "uuid-v4",
  "timestamp_utc": "ISO-8601",
  "issued_by": "self_healing_service",
  "human_approved": true,

  "payload": {
    "target_rpm": 4500,
    "vibration_threshold_override": 3.0,
    "justification": "Vibration RMS exceeded safe threshold (2.5g → 3.1g)"
  }
}
```

---

## 4. VANGUARD API Contract (Cloud → External)

### POST `/api/v1/classify`

**Request:**
```json
{
  "image_b64": "base64-encoded-crop",
  "detection_meta": {
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.87,
    "class_id": 0
  },
  "context": "RAG-retrieved doctrinal context string"
}
```

**Response:**
```json
{
  "classification": "Friendly" | "Unknown" | "Hostile",
  "confidence": 0.92,
  "shap_values": {
    "top_features": [
      {"feature": "rotor_count", "attribution": 0.34},
      {"feature": "size_ratio", "attribution": 0.28}
    ],
    "base_value": 0.5
  },
  "grad_cam_b64": "base64-encoded-heatmap-png",
  "model_version": "vanguard-v2.1.0"
}
```

### GET `/api/v1/health`

**Response:**
```json
{
  "status": "healthy",
  "model_version": "vanguard-v2.1.0",
  "uptime_seconds": 86400
}
```

---

## 5. Azure AI Search Query Schema (Cloud → RAG)

### Hybrid Search (Keyword + Vector)

```json
{
  "search": "quadcopter hostile payload",
  "filter": "doc_type eq 'incident_report'",
  "top": 5,
  "queryType": "semantic",
  "semanticConfiguration": "aegis-semantic-config",
  "vectorQueries": [
    {
      "kind": "text",
      "text": "small drone carrying suspicious payload",
      "fields": "embedding"
    }
  ]
}
```

---

## 6. Internal Module Interfaces

| Producer              | Consumer               | Data Contract                     |
|-----------------------|------------------------|-----------------------------------|
| `VisionNode`          | `ReasoningNode`        | `Detection(bbox, class_id, conf, grad_cam)` |
| `ReasoningNode`       | `CloudSync`            | `EnrichedDetection(detection + description + xai)` |
| `FusionEngine`        | `CloudSync`            | `PlatformState` dataclass         |
| `CloudSync`           | `ThreatEvaluator`      | D2C telemetry JSON (via IoT Hub)  |
| `ThreatEvaluator`     | `SelfHealingService`   | `ThreatAssessment` dict           |
| `SelfHealingService`  | `CloudSync` (C2D)      | C2D command JSON                  |
| `CloudSync` (C2D)     | `FusionEngine`         | Motor/calibration command dict    |
| `XAIGenerator`        | `audit_trail.log`      | XAI evidence package JSON         |

---

*This API specification will be refined as functional code is implemented
during the 4-day sprint.*
