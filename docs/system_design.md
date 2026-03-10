# AEGIS-Cloud — System Design Document

> **Project**: AEGIS-Cloud — Autonomous Edge Guardian Intelligence System
> **Client**: Diehl Defence GmbH & Co. KG
> **Application**: Counter-UAS / GARMR Integration
> **Version**: 0.1.0-skeleton
> **Date**: 2026-03-10

---

## 1. Executive Summary

AEGIS-Cloud is a **hybrid AIOps framework** that combines Edge-based
tactical perception with Cloud-based strategic reasoning for autonomous
Counter-Unmanned Aircraft System (C-UAS) operations.  The system is
designed for deployment on the **Diehl Defence GARMR** platform, with
the AEGIS architecture providing the AI perception and decision-making
backbone.

The system is split into three operational domains:

| Domain    | Runtime           | Role                                         |
|-----------|-------------------|----------------------------------------------|
| **Edge**  | Jetson Nano       | Real-time detection, scene reasoning, sensors |
| **Cloud** | Azure Functions   | Strategic threat evaluation, self-healing     |
| **MLOps** | GitHub + OpenShift | CI/CD, compliance, data versioning           |

---

## 2. Mission Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: DETECTION (Edge — Vision Node)                             │
│  YOLOv8-nano detects potential UAS targets in real-time video.      │
│  Each detection includes a Grad-CAM saliency map (XAI evidence).   │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 2: LOCAL REASONING (Edge — Reasoning Node)                    │
│  LLaVA generates a natural-language scene description.              │
│  Circuit-breaker ensures graceful degradation if Ollama fails.      │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 3: SENSOR FUSION (Edge — Fusion Engine)                       │
│  IMU + ToF data fused into PlatformState (orientation, range, etc.) │
│  Provenance hash links fused output to raw sensor readings.         │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 4: TELEMETRY UPLOAD (Edge — Cloud Sync)                       │
│  D2C message sent to Azure IoT Hub with detection + telemetry.      │
│  NATO Incident Report format.                                       │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 5: STRATEGIC EVALUATION (Cloud — Threat Evaluator)            │
│  LangChain agent queries RAG context + VANGUARD API.                │
│  Assembles full XAI evidence package (SHAP + Grad-CAM + RAG).       │
│  Classification: Friendly / Unknown / Hostile.                      │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 6: HUMAN-IN-THE-LOOP (Cloud — Compliance)                     │
│  "Hostile" classifications require HITL approval before actuation.  │
│  Full evidence package logged to audit_trail.log.                   │
├─────────────────────────────────────────────────────────────────────┤
│  STEP 7: SELF-HEALING (Cloud — Self-Healing Service)                 │
│  Analyses Edge telemetry for anomalies (vibration, sensor health).  │
│  Issues C2D motor-speed adjustment commands back to Edge.           │
│  Closes the AIOps feedback loop.                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Architecture Diagram

*See `architecture_diagram.drawio` for the interactive diagram (to be
created in draw.io).  The diagram should contain three swim-lanes:*

1. **Edge Lane** — Vision Node → Reasoning Node → Fusion Engine → Cloud Sync
2. **Cloud Lane** — Threat Evaluator → VANGUARD API → RAG Context → Self-Healing
3. **MLOps Lane** — CI/CD Pipeline → Container Registry → DVC → Compliance

---

## 4. Key Design Decisions

### 4.1 VANGUARD as External API
The VANGUARD threat classification model is maintained as a **separate
project**.  AEGIS-Cloud accesses it exclusively through
`vanguard_api_client.py`.  This demonstrates multi-project lifecycle
management and enables independent model retraining.

### 4.2 Simulation-First Development
Every hardware-dependent module checks the `SIMULATION_MODE` flag from
`edge/config/edge_settings.yaml`.  This enables:
- Full pipeline testing in CI/CD without physical sensors
- Developer onboarding without Jetson Nano hardware
- Reproducible integration tests using DVC-tracked sim data

### 4.3 AIOps Closed-Loop Architecture
The system implements a true closed-loop:
```
Edge Sensors → Cloud Reasoning → Cloud Self-Healing → Edge Motors
```
This transforms AEGIS from passive monitoring to active self-healing.

### 4.4 EU AI Act–First Compliance
Compliance is not an afterthought — the `mlops/compliance/` layer is
a first-class component with:
- `xai_generator.py` — Mandatory SHAP/Grad-CAM before Hostile decisions
- `audit_trail.log` — Immutable NATO-format decision log (10-year retention)
- `risk_assessment.md` — Living risk assessment document (Article 9)

---

## 5. Technology Stack

| Component         | Technology                                |
|-------------------|-------------------------------------------|
| Edge Inference    | YOLOv8-nano (Ultralytics) + TensorRT      |
| Edge Reasoning    | Ollama + LLaVA 13B                        |
| Edge Sensors      | ICM-20948 (IMU) + VL6180X (ToF)           |
| Edge Comms        | Azure IoT Hub SDK (Python)                |
| Cloud Reasoning   | LangChain + Azure Functions               |
| Cloud Intelligence| VANGUARD API (external) + Azure AI Search  |
| Cloud Ops         | Self-Healing Service (custom)             |
| Containers        | Podman (OCI-compliant)                    |
| Orchestration     | Kubernetes / OpenShift (Minikube for sim)  |
| CI/CD             | GitHub Actions                            |
| Data Versioning   | DVC + Azure Blob Storage                  |
| Compliance        | Custom XAI generator + audit trail        |
| Dashboards        | Power BI (via Cosmos DB)                  |

---

*This document is maintained as a living design narrative throughout the
AEGIS-Cloud project lifecycle.*
