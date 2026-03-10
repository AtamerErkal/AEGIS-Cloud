# 🛡️ AEGIS-Cloud

**Autonomous Edge Guardian Intelligence System**

> A hybrid AIOps framework for Counter-UAS operations, designed for Diehl Defence.

---

## 🏗️ Architecture

```
┌──────────────────┐     D2C Telemetry      ┌──────────────────────┐
│   EDGE (Jetson)  │ ──────────────────────▶ │   CLOUD (Azure)      │
│                  │                         │                      │
│  ┌─ Vision Node  │                         │  ┌─ Threat Evaluator │
│  ├─ Reasoning    │     C2D Commands        │  ├─ VANGUARD Client  │
│  ├─ Fusion Eng.  │ ◀────────────────────── │  ├─ RAG Context      │
│  └─ Cloud Sync   │                         │  └─ Self-Healing     │
└──────────────────┘                         └──────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    │   MLOps (CI/CD)    │
                    │                    │
                    │  ┌─ Podman         │
                    │  ├─ OpenShift      │
                    │  ├─ GitHub Actions │
                    │  ├─ DVC            │
                    │  └─ EU AI Act      │
                    └────────────────────┘
```

## 📁 Project Structure

```
AEGIS-Cloud/
├── edge/                           # Jetson Nano — Tactical Perception
│   ├── src/
│   │   ├── perception/
│   │   │   ├── vision_node.py      # YOLOv8 drone detection
│   │   │   └── reasoning_node.py   # Ollama/LLaVA scene reasoning
│   │   ├── sensors/
│   │   │   ├── fusion_engine.py    # IMU + ToF sensor fusion
│   │   │   └── driver_interface.py # Hardware abstraction layer
│   │   └── comm/
│   │       └── cloud_sync.py       # Azure IoT Hub gateway
│   └── config/
│       └── edge_settings.yaml      # Simulation flags & thresholds
│
├── cloud/                          # Azure — Strategic Reasoning
│   ├── functions/
│   │   └── threat_evaluator/       # LangChain agent (Azure Function)
│   ├── integrations/
│   │   ├── vanguard_api_client.py  # VANGUARD model connector
│   │   └── rag_context.py          # Azure AI Search RAG engine
│   └── ops/
│       └── self_healing_service.py # AIOps closed-loop motor control
│
├── mlops/                          # Governance & Automation
│   ├── deploy/                     # Container & orchestration manifests
│   ├── pipelines/                  # CI/CD & data versioning
│   └── compliance/                 # EU AI Act audit & XAI
│
├── data/
│   ├── sim_samples/                # Synthetic test data
│   └── logs/                       # Runtime telemetry buffer
│
└── docs/
    ├── system_design.md            # Architecture narrative
    └── api_spec.md                 # Interface contracts
```

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/diehl-defence/AEGIS-Cloud.git
cd AEGIS-Cloud

# 2. Run in Simulation Mode (no hardware required)
export AEGIS_SIMULATION_MODE=true

# 3. Build Edge container
podman build -f mlops/deploy/Podmanfile.edge -t aegis-edge:latest .

# 4. Build Cloud container
podman build -f mlops/deploy/Podmanfile.cloud -t aegis-cloud:latest .

# 5. Deploy to local Minikube
minikube start
kubectl apply -f mlops/deploy/openshift-deploy.yaml
```

## 🔑 Design Principles

| Principle                  | Implementation                                       |
|----------------------------|------------------------------------------------------|
| **Simulation-First**       | `SIMULATION_MODE` flag in every hardware module      |
| **VANGUARD Separation**    | External API via `vanguard_api_client.py` only       |
| **AIOps Closed-Loop**      | Edge → Cloud → Self-Healing → Edge feedback loop     |
| **EU AI Act Compliance**   | XAI generator + immutable audit trail + HITL gating  |
| **NATO-Standard Logging**  | Structured Incident Report format throughout         |

## 📖 Documentation

- [System Design](docs/system_design.md) — Full architecture narrative
- [API Specification](docs/api_spec.md) — Interface contracts
- [Risk Assessment](mlops/compliance/risk_assessment.md) — EU AI Act compliance

---

**⚠️ SKELETON BUILD** — This repository contains architectural scaffolding only.
No functional code has been implemented yet.  See `docs/system_design.md` for
the complete mission flow and sprint plan.
