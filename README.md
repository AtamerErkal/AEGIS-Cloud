# 🛡️ AEGIS-Cloud
**Autonomous Edge Guardian Intelligence System**

![Edge AI](https://img.shields.io/badge/Domain-Edge%20AI-blue)
![TensorRT](https://img.shields.io/badge/Hardware_Acceleration-TensorRT%20FP16-76B900)
![YOLOv8](https://img.shields.io/badge/Perception-YOLOv8-FF9900)
![Moondream](https://img.shields.io/badge/Reasoning-Moondream%20VLM-8A2BE2)
![C++](https://img.shields.io/badge/Compilation-C%2B%2B-00599C)

> A heterogeneous AIOps framework designed for autonomous Counter-UAS and UAV operations. Engineered for resource-constrained Edge AI environments (SWaP-C), featuring hardware-compiled perception and tactical Vision-Language reasoning.

---

## 🏗️ Architecture

```text
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

## 🚀 Key Engineering & Optimizations (Edge AI)

Deploying modern perception and reasoning architectures on legacy edge devices (e.g., NVIDIA Jetson Nano, 4GB RAM) requires aggressive resource management and hardware-level compilation.

### 1. Hardware Compilation (ONNX -> TensorRT)
To overcome severe CPU bottlenecks and thermal constraints, the YOLOv8 perception pipeline was decoupled from Python/PyTorch dependencies:
* **Host-Target Separation:** Exported the target detection model to `.onnx` on the host machine.
* **C++ Edge Compilation:** Utilized NVIDIA's native `trtexec` binary to compile the model directly into a `.engine` (TensorRT) format with **FP16 precision** on the edge device.

### 2. AIOps & Resource Benchmarks
*The integration of TensorRT successfully offloaded the entire vision pipeline to the 128-core Maxwell GPU.*

| Metric | PyTorch Baseline (.pt) | TensorRT Optimized (.engine) | Improvement |
| :--- | :--- | :--- | :--- |
| **CPU Usage** | ~82.0% | **0.0% - 2.4%** | CPU fully recovered for Cloud Sync |
| **SoC Temperature** | Rapid overheating | **Stable (~80°C limit)** | Thermal throttling avoided |
| **Latency** | > 300ms | **< 30ms** | Real-time tracking enabled |

### 3. VLM Circuit-Breaker & Tactical Sub-sampling
Running a Vision-Language Model (Moondream) on 4GB RAM edge devices inherently relies on Swap memory. To maintain system integrity:
* Implemented a **Circuit-Breaker Pattern**: The VLM acts as an asynchronous sidecar. If reasoning latency exceeds tactical limits, the perception loop continues unimpeded.
* **Tactical Sub-sampling:** The VLM is not triggered per frame. It only activates for `CONF > 0.80` targets classified strictly as `Hostile`.
* **Output Pruning:** Token generation is hard-capped (`num_predict: 50`) to prevent memory lockups and ensure concise tactical reporting.

## 🛠️ System Telemetry (Hardware Verification)
The system features a built-in AIOps telemetry node to monitor hardware constraints in real-time (critical for fanless deployment):

```log
── Frame 49 ───────────────────────────────────────────────────────
[HEARTBEAT] Frame: 49 | CPU: 2.4% | Running...
[2026-03-11T11:36:26Z] !!! [TACTICAL ALERT] Target: drone | Starting VLM Reasoning...
[2026-03-11T11:40:46Z] [AEGIS] Moondream Report: The image shows a large orange square with a green border, and the text "DRONE"...
[2026-03-11T11:40:46Z] [AEGIS][CloudSync] SIM D2C msg#0 priority=HIGH class=Hostile → data/logs/cloud_payload.json

── Frame 50 ───────────────────────────────────────────────────────
[HEARTBEAT] Frame: 50 | CPU: 0.0% | Running...
```

## 📁 Project Structure

```text
AEGIS-Cloud/
├── edge/                           # Target Device (e.g. Jetson)
│   ├── src/
│   │   ├── perception/
│   │   │   ├── vision_node.py      # TensorRT YOLOv8 inference
│   │   │   └── reasoning_node.py   # Moondream VLM tactical analysis
│   │   ├── sensors/
│   │   │   ├── fusion_engine.py    # IMU + ToF abstraction
│   │   │   └── driver_interface.py # Hardware API
│   │   └── comm/
│   │       └── cloud_sync.py       # Azure IoT gateway
│   └── config/
│       └── edge_settings.yaml      # Tactical sub-sampling thresholds
│
├── cloud/                          # Azure — Strategic Reasoning
│   ├── functions/
│   │   └── threat_evaluator/       # LangChain agent
│   └── integrations/
│       └── vanguard_api_client.py  # Strategic AI connector
│
└── mlops/                          # Governance & Automation
    ├── deploy/                     # Container manifests (Podman/Minikube)
    └── compliance/                 # EU AI Act audit logs
```

## ⚙️ Quick Start (Edge Deployment)
*Ensure NVIDIA JetPack SDK is installed and configured on the target hardware.*

**1. Clone the repository (Shallow):**
```bash
git clone --depth 1 https://github.com/AtamerErkal/AEGIS-Cloud.git
cd AEGIS-Cloud
```

**2. Run Native TensorRT Compilation:**
```bash
# Compile ONNX to TensorRT on the Edge device
/usr/src/tensorrt/bin/trtexec --onnx=edge/models/yolov8n.onnx --saveEngine=edge/models/yolov8n.engine --fp16
```

**3. Launch the Pipeline:**
```bash
# Ensure edge_settings.yaml uses the .engine model path
python edge/src/perception/vision_node.py
```

## 🔑 Design Principles

| Principle                  | Implementation                                               |
|----------------------------|--------------------------------------------------------------|
| **SWaP-C Optimization** | Host-Target cross-compilation & aggressive Memory Pruning    |
| **AIOps Closed-Loop** | Edge → Cloud → Self-Healing → Edge feedback loop             |
| **EU AI Act Compliance** | XAI generator + immutable audit trail + HITL gating          |
| **NATO-Standard Logging**  | Structured Incident Report format throughout                 |
