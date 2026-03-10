# AEGIS-Cloud — Risk Assessment (EU AI Act)

> **Classification**: HIGH-RISK AI System (Annex III, Category 1)
> **Regulation**: EU Artificial Intelligence Act (Regulation 2024/1689)
> **System**: AEGIS-Cloud — Autonomous Edge Guardian Intelligence System
> **Operator**: Diehl Defence GmbH & Co. KG
> **Version**: 0.1.0-skeleton
> **Last Updated**: 2026-03-10

---

## 1. System Description

AEGIS-Cloud is a hybrid AI system designed for **Counter-Unmanned
Aircraft System (C-UAS)** operations. It combines Edge-based tactical
perception (YOLOv8 drone detection, LLaVA scene reasoning) with
Cloud-based strategic threat evaluation (LangChain agent consuming the
VANGUARD classification model).

The system operates in a **closed-loop AIOps architecture** where sensor
telemetry flows from Edge to Cloud, and self-healing motor-speed commands
flow from Cloud back to Edge.

---

## 2. Risk Classification

| Criterion                          | Assessment                         |
|------------------------------------|-------------------------------------|
| **Annex III Category**             | 1 — Safety component of a product  |
| **Risk Level**                     | HIGH                                |
| **Intended Purpose**               | Autonomous threat detection/classification for defence |
| **Deployment Context**             | Military / critical infrastructure  |
| **Autonomy Level**                 | Semi-autonomous (HITL required for Hostile actuation) |

---

## 3. Identified Risks

### 3.1 False Positive — Hostile Classification
- **Description**: System incorrectly classifies a friendly/civilian drone as hostile.
- **Severity**: CRITICAL
- **Mitigation**: Mandatory XAI evidence package (SHAP + Grad-CAM) + Human-in-the-Loop approval before any Hostile actuation.

### 3.2 False Negative — Missed Hostile Target
- **Description**: System fails to detect or correctly classify a hostile drone.
- **Severity**: CRITICAL
- **Mitigation**: Multi-model redundancy (YOLO + LLaVA + VANGUARD), confidence threshold gating, continuous model retraining via DVC pipeline.

### 3.3 Self-Healing Actuator Misfire
- **Description**: Cloud issues incorrect motor-speed command causing hardware damage.
- **Severity**: HIGH
- **Mitigation**: RPM safety bounds enforced in `self_healing_service.py`, vibration threshold gating, HITL approval for commands exceeding safety limits.

### 3.4 VANGUARD API Unavailability
- **Description**: External VANGUARD model API becomes unreachable.
- **Severity**: MEDIUM
- **Mitigation**: Circuit-breaker pattern in `vanguard_api_client.py`, graceful degradation to "Unknown" classification (never guesses "Hostile").

### 3.5 Data Poisoning / Model Drift
- **Description**: Training data contamination or distributional shift degrades model accuracy.
- **Severity**: HIGH
- **Mitigation**: DVC data versioning, model performance monitoring via AIOps telemetry, automated drift detection alerts.

---

## 4. Transparency Obligations (Article 13)

| Obligation                                | Implementation                          |
|-------------------------------------------|-----------------------------------------|
| Interpretable outputs                     | `xai_generator.py` — SHAP + Grad-CAM   |
| Operator documentation                    | `docs/system_design.md`                 |
| Performance metrics display               | Power BI dashboards via Cosmos DB       |
| Limitations disclosure                    | This document, Section 3                |

---

## 5. Human Oversight (Article 14)

| Control                                   | Implementation                          |
|-------------------------------------------|-----------------------------------------|
| HITL for Hostile decisions                | `threat_evaluator.py` → `audit_trail.log` |
| Manual override capability                | C2D command channel via `cloud_sync.py` |
| System shutdown                           | Operator dashboard kill-switch          |
| Audit trail access                        | `mlops/compliance/audit_trail.log`      |

---

## 6. Record-Keeping (Article 12)

- All decisions logged in `audit_trail.log` with NATO Incident Report format.
- Retention period: **minimum 10 years**.
- Logs stored in Azure Blob Storage with immutability policy.
- DVC tracks all training data and model versions for reproducibility.

---

## 7. Review Schedule

| Review Type          | Frequency      | Responsible           |
|----------------------|----------------|-----------------------|
| Risk re-assessment   | Quarterly      | AI Safety Officer     |
| Model performance    | Monthly        | MLOps Team            |
| Audit trail review   | Monthly        | Compliance Officer    |
| Penetration testing  | Bi-annually    | Security Team         |

---

*This document is a living artefact maintained throughout the system
lifecycle per EU AI Act Article 9 requirements.*
