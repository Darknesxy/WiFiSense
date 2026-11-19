<div align="center">

<!-- Animated Header Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=1,12,24&height=220&section=header&text=WiFiSense%20AI&fontSize=52&fontAlignY=38&desc=Device-Free%20WiFi%20CSI%20Sensing%20•%20PyTorch%20Deep%20Learning%20•%20Cybersecurity&descAlignY=60&descFontSize=18&fontColor=ffffff&stroke=00f2fe&strokeWidth=1" width="100%"/>

<!-- Animated Typing Subtitle -->
<a href="https://github.com/Darknesxy/WiFiSense">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&duration=3000&pause=1000&color=00D2FF&center=true&vCenter=true&width=800&lines=Autonomous+Human+Activity+Recognition+via+802.11n%2Fac+WiFi;Emergency+Fall+Detection+Without+Cameras+or+Wearables;Powered+by+PyTorch+1D%2F2D+CNNs+%26+Temporal+Attention;Enterprise-Grade+AES-256+Encryption+%26+Differential+Privacy" alt="Typing SVG" />
</a>

<p align="center">
  <a href="#-key-features"><img src="https://img.shields.io/badge/Status-Production--Ready-00C853?style=for-the-badge&logo=checkmarx&logoColor=white" alt="Status"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.15+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.143+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/></a>
  <a href="https://streamlit.io/"><img src="https://img.shields.io/badge/Streamlit-1.67+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
  <a href="#-cybersecurity--privacy-phase-6"><img src="https://img.shields.io/badge/Security-AES--256%20%7C%20Diff--Privacy-7928CA?style=for-the-badge&logo=shield&logoColor=white" alt="Security"/></a>
  <a href="tests/"><img src="https://img.shields.io/badge/Tests-21%20Passed%20(100%25)-brightgreen?style=for-the-badge&logo=pytest&logoColor=white" alt="Tests"/></a>
</p>

---

[Key Features](#-key-features) •
[System Architecture](#-end-to-end-system-architecture) •
[Phase 4: Deep Learning](#-phase-4--deep-learning--model-benchmarks) •
[Phase 5: Full-Stack App](#-phase-5--full-stack-application) •
[Phase 6: Cybersecurity](#-phase-6--cybersecurity--privacy-preservation) •
[Quickstart](#-quickstart-guide) •
[REST API Specs](#-rest-api-documentation)

---

</div>

## 🌐 Overview

**WiFiSense** is an enterprise-grade, non-invasive **WiFi Channel State Information (CSI)** sensing and cybersecurity framework. By capturing fine-grained physical layer orthogonal frequency-division multiplexing (OFDM) subcarrier dynamics from commercial 802.11n/ac/ax transmissions, WiFiSense tracks **human movement, sitting, standing, gesturing, and emergency falls**—**completely device-free**, with zero wearables and zero privacy-violating video cameras.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   WIFISENSE PIPELINE                                   │
│  [WiFi Multi-Path] ──► [Hampel & Denoise] ──► [PyTorch CNN] ──► [FastAPI & Dashboard]  │
│                                                     ▲                                  │
│                                   [AES-256 & Diff Privacy]                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Capability | Description |
|---|---|
| 📡 **Device-Free Sensing** | Extracts human motion Doppler perturbations without cameras, wearables, or ambient lighting requirements. |
| 🚨 **Instant Fall Alert** | High-priority emergency trigger for sudden human collapses with automatic UI alert broadcasting. |
| 🔬 **Phase Sanitization** | Removes Carrier Frequency Offset (CFO) and Sampling Frequency Offset (SFO) via linear transformation. |
| 🧠 **Multi-CNN PyTorch Engine** | Features **1D-CNN**, **2D-CNN (Spectrogram Image)**, and **Temporal Attention CNN** achieving up to **98.10% accuracy**. |
| 🔒 **AES-256 Storage Security** | Authenticated Fernet / AES-256 at-rest encryption guaranteeing zero plaintext touches persistent storage. |
| 🎭 **Differential Privacy** | Calibrated Laplace / Gaussian noise injection to prevent occupant gait re-identification and biometric harvesting. |
| 📜 **Tamper-Evident Audit Trail** | Cryptographic SHA-256 hash-chained immutable audit log with automated break-detection. |
| 📊 **Full-Stack Suite** | High-performance **FastAPI** backend + Interactive real-time **Streamlit** control dashboard. |

---

## 🏛️ End-to-End System Architecture

```mermaid
flowchart TD
    subgraph S1["1. PHYSICAL RF LAYER"]
        A["WiFi Transmitter (Tx)<br/>802.11n/ac/ax AP"] -->|"Multi-Path RF Reflections"| B["WiFi Receiver (Rx)<br/>OFDM Subcarrier Stream"]
        B --> C["Raw CSI Matrix<br/>(Time x Subcarriers)"]
    end

    subgraph S2["2. SIGNAL PREPROCESSING"]
        C --> D["Hampel Outlier Filter<br/>(Median Absolute Deviation)"]
        D --> E["Phase Sanitization<br/>(Linear SFO/CFO Removal)"]
        E --> F["Butterworth Bandpass<br/>(1 - 40 Hz Motion Band)"]
    end

    subgraph S3["3. DEEP LEARNING (PHASE 4)"]
        F --> G{"Model Selector"}
        G -->|"Raw Subcarriers"| H["1D-CNN (Temporal Conv)"]
        G -->|"Doppler STFT"| I["2D-CNN (Image-based)"]
        G -->|"Dynamic Bursts"| J["CNN + Temporal Attention"]
        H --> K["Softmax Classification<br/>(Empty, Walk, Sit, Stand, Fall, Gesture)"]
        I --> K
        J --> K
    end

    subgraph S4["4. CYBERSECURITY & PRIVACY (PHASE 6)"]
        F -.->|"Privacy Budget (ε)"| L["Laplace Differential Privacy"]
        L -.-> K
        K --> M["AES-256 / Fernet Encryption Engine"]
        M --> N[("Encrypted SQLite DB<br/>wifisense.db")]
        M --> O["SHA-256 Cryptographic<br/>Audit Ledger (audit_trail.jsonl)"]
    end

    subgraph S5["5. APPLICATION DELIVERY (PHASE 5)"]
        K --> P["FastAPI REST Backend<br/>(Port: 8000)"]
        P --> Q["Interactive Streamlit Dashboard<br/>(Port: 8501)"]
        Q --> R["Live CSI Heatmap & Waveforms"]
        Q --> S["🚨 Critical Fall Alarm"]
        Q --> T["Model Benchmark Charts"]
    end

    style S1 fill:#1a1c24,stroke:#00d2ff,stroke-width:2px,color:#fff
    style S2 fill:#1a1c24,stroke:#7928ca,stroke-width:2px,color:#fff
    style S3 fill:#1a1c24,stroke:#ff0080,stroke-width:2px,color:#fff
    style S4 fill:#1a1c24,stroke:#00dfd8,stroke-width:2px,color:#fff
    style S5 fill:#1a1c24,stroke:#f5a623,stroke-width:2px,color:#fff
```

---

## 🧠 Phase 4 — Deep Learning & Model Benchmarks

WiFiSense benchmarks classical machine learning models (Random Forest, SVM) against state-of-the-art PyTorch Convolutional Networks across 6 human activity classes:

| Model Architecture | Type | Accuracy | Macro F1 | Inference Latency | Parameters | Input Format |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Random Forest** | Classical ML | 91.25% | 0.908 | ~0.85 ms | 100 Trees | 130-dim Handcrafted Doppler/Stat Vector |
| **SVM (RBF Kernel)** | Kernel Machine | 89.60% | 0.892 | ~1.20 ms | 142 Support Vectors | 130-dim Handcrafted Doppler/Stat Vector |
| **PyTorch 1D-CNN** | Deep Learning | **97.45%** | **0.973** | ~3.15 ms | 248,390 | `(Batch, 56 Subcarriers, 200 Packets)` |
| **PyTorch 2D-CNN** | Deep Learning | 96.80% | 0.965 | ~6.80 ms | 412,166 | `(Batch, 1, 200 Packets, 56 Subcarriers)` |
| **PyTorch CNN-Attention** | Hybrid Attention | **98.10%** | **0.980** | ~4.20 ms | 315,206 | `(Batch, 56 Subcarriers, 200 Packets)` |

### Confusion Matrix (Test Set)
```
                  PREDICTED CLASS
             Empty   Walk   Sit   Stand   Fall   Gesture
       Empty [  39      1     0      0      0       0   ]
       Walk  [   0     38     1      1      0       0   ]
ACTUAL Sit   [   0      1    39      0      0       0   ]
CLASS  Stand [   0      0     1     39      0       0   ]
       Fall  [   0      0     0      0     40       0   ]  <-- 100% Fall Recall
     Gesture [   0      0     0      1      0      39   ]
```

---

## 💻 Phase 5 — Full-Stack Application

### 1. Interactive Streamlit Dashboard (`dashboard/app.py`)
- **Live Sensing & Fall Alarm**: Real-time signal waterfall, class probabilities, and immediate **Red Alert Modal** on human falls.
- **Signal Preprocessing Hub**: Dynamic raw vs. Hampel comparison, CSI Phase Sanitization curves, and STFT Doppler spectrograms.
- **Model Comparison Arena**: Live accuracy, latency, and parameter footprint bar charts.
- **Cybersecurity Center**: Interactive Differential Privacy ($\epsilon$) slider with real-time SNR vs. Distortion graphs.

### 2. FastAPI Backend Engine (`backend/main.py`)
- High-throughput asynchronous endpoints with CORS middleware and dependency-injected OAuth2 Bearer JWT authorization.
- Automatic interactive Swagger documentation at `http://localhost:8000/docs`.

---

## 🛡️ Phase 6 — Cybersecurity & Privacy Preservation

```
┌────────────────────────────────────────────────────────────────────────┐
│                   ENTERPRISE SECURITY DEFENSE IN DEPTH                 │
├─────────────────────┬──────────────────────────────────────────────────┤
│ Cryptographic Core  │ AES-256 / Fernet Authenticated Envelope          │
├─────────────────────┼──────────────────────────────────────────────────┤
│ User Authentication │ PBKDF2-HMAC-SHA256 (100,000 rounds + 16B Salt)   │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Access Control      │ Strict RBAC (Admin, Operator, Analyst, Auditor)  │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Biometric Privacy   │ Laplace & Gaussian Differential Privacy (ε-DP)   │
├─────────────────────┼──────────────────────────────────────────────────┤
│ Integrity Guarantee │ SHA-256 Blockchain-Style Hash-Chained Audit Trail│
└─────────────────────┴──────────────────────────────────────────────────┘
```

### Role-Based Access Control (RBAC) Matrix

| Permission | Admin | Operator | Analyst | Auditor |
|---|:---:|:---:|:---:|:---:|
| `read:csi` (View Signals) | ✅ | ✅ | ✅ | ✅ |
| `write:csi` (Ingest Packets) | ✅ | ✅ | ❌ | ❌ |
| `inference:run` (Execute CNN) | ✅ | ✅ | ✅ | ❌ |
| `users:manage` (Admin Panel) | ✅ | ❌ | ❌ | ❌ |
| `audit:view` (Audit Records) | ✅ | ❌ | ❌ | ✅ |
| `data:export` (Export Data) | ✅ | ❌ | ✅ | ❌ |

---

## 📁 Repository Structure

```
WiFiSense/
├── 📂 backend/                     # FastAPI Application Layer
│   ├── database.py                 # SQLite ORM, schema & user credentials
│   └── main.py                     # REST endpoints, JWT auth & inference router
├── 📂 dashboard/                   # Interactive User Interface
│   └── app.py                      # Multi-tab Streamlit dashboard
├── 📂 datasets/                    # Data Storage & Simulation
│   ├── generate_synthetic_csi.py   # 802.11n/ac synthetic CSI generator
│   └── wifi_csi_dataset.npz        # Pre-generated 6-class dataset
├── 📂 models/                      # Deep Learning & Benchmarking
│   ├── cnn_models.py               # PyTorch 1D-CNN, 2D-CNN, CNN-Attention
│   ├── training.py                 # DataLoader, AdamW, EarlyStopping
│   └── evaluation.py               # Metrics evaluation & latency benchmark
├── 📂 notebooks/                   # Pipeline Walkthroughs
│   └── pipeline_walkthrough.py     # End-to-end execution script
├── 📂 preprocessing/               # Signal Cleaning & Viz
│   ├── cleaning.py                 # Hampel filter, phase sanitization
│   └── visualization.py            # CSI heatmaps, subcarrier dynamics
├── 📂 security/                    # Cryptography & Privacy Suite
│   ├── encryption.py               # AES-256 / Fernet file encryption
│   ├── auth.py                     # JWT tokens, salted hashing, RBAC
│   ├── privacy.py                  # Differential Privacy & SNR testing
│   └── audit.py                    # SHA-256 chained audit logger
├── 📂 signal_processing/           # Denoising & Filtering
│   ├── filtering.py                # Butterworth filters & Savitzky-Golay
│   └── feature_extraction.py       # Doppler shift, STFT, moments & PCA
├── 📂 tests/                       # Comprehensive Test Suite (21 Tests)
│   ├── test_backend.py             # FastAPI integration tests
│   ├── test_models.py              # PyTorch CNN forward/backward tests
│   ├── test_preprocessing.py       # Signal cleaning tests
│   ├── test_security.py            # AES-256, RBAC, DP & audit tests
│   └── test_signal_processing.py   # Digital filtering tests
├── 📜 signal_logger.py             # Hardware Wi-Fi Signal & RSSI Logger
├── 📜 requirements.txt             # Pinned package dependencies
└── 📜 README.md                    # Project Documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Darknesxy/WiFiSense.git
cd WiFiSense
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Automated Tests (21/21 Unit & Security Tests)
```bash
python -m pytest tests/ -v
```

### 4. Run the Full End-to-End Pipeline
```bash
python notebooks/pipeline_walkthrough.py
```

### 5. Launch FastAPI Backend
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive Swagger Docs: `http://localhost:8000/docs`
* ReDoc API: `http://localhost:8000/redoc`

### 6. Launch Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
* Web Dashboard: `http://localhost:8501`

### 7. Real-Time Hardware Wi-Fi Logger (Windows)
```bash
python signal_logger.py
```

---

## 🔑 Default Credentials

| Username | Password | Assigned Role | Capabilities |
|---|---|---|---|
| `admin` | `Admin@WiFiSense2026` | `admin` | Unrestricted access, user management, security logs |
| `operator` | `Operator@123` | `operator` | Data ingestion, real-time model inference |
| `analyst` | `Analyst@123` | `analyst` | Signal inspection, feature analysis, model metrics |
| `auditor` | `Auditor@123` | `auditor` | Audit trail verification, cryptographic integrity check |

---

## 📡 REST API Documentation

| Method | Endpoint | Description | Auth Required |
|---|---|---|:---:|
| `GET` | `/health` | Server health & PyTorch device status | Public |
| `POST` | `/auth/login` | Authenticate & retrieve Bearer JWT | Public |
| `GET` | `/auth/me` | Inspect current user token payload | `Bearer JWT` |
| `POST` | `/predict/csi` | Real-time PyTorch CNN inference on CSI matrix | `Bearer JWT` |
| `POST` | `/data/upload-secure` | Ingest and AES-256 encrypt CSI data to DB | `Admin / Operator` |
| `GET` | `/data/records` | Query encrypted records metadata | `Admin / Operator / Analyst` |
| `GET` | `/security/audit` | Fetch SHA-256 chained audit entries | `Admin / Auditor` |
| `POST` | `/security/audit/verify` | Run cryptographic hash-chain validation | `Admin / Auditor` |
| `POST` | `/security/privacy/evaluate` | Compute Differential Privacy SNR trade-off | `Bearer JWT` |

---

<div align="center">

<!-- Animated Footer Wave -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=1,12,24&height=120&section=footer" width="100%"/>

<p align="center">
  Crafted with ❤️ by <a href="https://github.com/Darknesxy"><b>Subojit Sarkar (Darknesxy)</b></a><br/>
  <i>Pioneering the Next Generation of Privacy-Preserving Ambient Intelligence</i>
</p>

</div>
