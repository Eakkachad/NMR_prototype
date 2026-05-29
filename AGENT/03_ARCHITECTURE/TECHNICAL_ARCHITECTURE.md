# Technical Architecture & Developer Handbook: Hybrid Physics-Aware NMR Spectroscopy AI Platform

This document serves as the comprehensive engineering guide, developer handbook, and technical reference for the **TRL 5+ Hybrid Physics-Aware NMR Spectroscopy AI Platform** built for the BDI Young Innovator Hackathon 2026.

---

## 🏛️ 1. Platform Directory Structure & File Inventory

The workspace is organized into a clean, decoupled architecture. The **FastAPI Python backend** handles scientific computations and deep learning inference, while the **Astro frontend** delivers a premium web dashboard experience.

```
d:/competetetion/BDI KKU/
├── AGENT/
│   └── 04_DATA/
│       └── scripts/
│           ├── app_server.py             # FastAPI Python API REST backend server (Port 8000)
│           ├── run_poc.py                 # Master batch CLI runner & evaluation pipeline
│           ├── generate_synthetic_nmr.py  # Module 1: Physics-based Lorentzian simulator
│           ├── models_core.py             # Module 2 & 3: SequenceAwareEncoder, SpectrumDecoder, LocalizedPatchEBM
│           └── database_matcher.py        # Module 4: Local HMDB fastdtw library matcher
│
├── frontend/                              # Astro Web Application (Port 4321)
│   ├── src/
│   │   ├── pages/
│   │   │   └── index.astro                # Main clinical dashboard page & custom design system
│   │   └── assets/, layouts/, components/ # Component directory trees
│   ├── astro.config.mjs
│   ├── package.json
│   └── pnpm-lock.yaml
│
├── anomalous_peaks_feedback.json          # Active learning telemetry trace log (JSON)
├── clinical_report.json                   # Headless hospital EMR payload output (JSON)
└── TECHNICAL_ARCHITECTURE.md              # This developer reference handbook
```

---

## 🧬 2. The 4-Stage Hybrid Physics-Aware AI Pipeline

The platform is designed to process **4,000-dimensional 1H NMR signals**, correcting chemical shift drifts and eliminating unphysical "ghost peak" artifacts using a generative energy boundary.

```
Raw Input 1D Signal [Batch, 1, 4000]
     │
     ▼ (Stage 1)
SequenceAwareEncoder ──► Latent Dense Embeddings [Batch, 512]
     │
     ▼ (Stage 2)
Neural ODE Solver ──► Continuous Shift Alignment Trajectory [Batch, 512]
     │
     ▼ (Stage 3)
SpectrumDecoder ──► Reconstructed Signal [Batch, 1, 4000]
     │
     ├─► LocalizedPatchEBM ──► Physical Energy Score [Batch, 1] (Threshold Alarm)
     │
     ▼ (Stage 4)
LocalDatabaseMatcher ──► fastdtw Alignment (abs lambda) ──► Hybrid Match Confidence
```

### Stage 1: Generative Dimensionality Reduction (Feature Selection)
* **Target File**: `models_core.py` -> `SequenceAwareEncoder`
* **Mathematical Function**: Compresses the raw 1D spectroscopic grid into a low-dimensional dense latent space. It purposefully avoids standard U-Net style skip connections from the encoder layers to the decoder to prevent leaking positional physical drift errors.
* **Input Tensor Shape**: `[Batch_Size, 1, 4000]`
* **Output Tensor Shape**: `[Batch_Size, 512]` (Feature activations)
* **Neural Layers**: `Conv1d(1 -> 16, kernel=15)` -> `GroupNorm` -> `GELU` -> `Conv1d(16 -> 64, kernel=7)` -> `GroupNorm` -> `GELU` -> `AdaptiveAvgPool1d(128)` -> `Linear(8192 -> 512)`.

### Stage 2: Continuous Alignment & Drift Correction (Pattern Recognition)
* **Target File**: `pipeline.py` -> `LatentSpaceODESolver`
* **Mathematical Function**: Solves continuous state-space trajectories over time steps simulating a Neural ODE to correct chemical shift offsets caused by environmental temperature/pH variations.
  $$\frac{dh(t)}{dt} = f_{\theta}(h(t), t)$$
* **Input/Output Embeddings**: `[Batch_Size, 512]`
* **Solver Algorithm**: 4-step Euler Integration loop (`dt = 0.1`) computing continuous gradient fields parameterizing latent alignments.

### Stage 3: Physics-Constrained Deconvolution & Verification
* **Target File**: `models_core.py` -> `LocalizedPatchEBM` & `SpectrumDecoder`
* **Mathematical Function**: Maps latent embeddings back to standard 1D space using transpositions. Slices the reconstructed 4,000-point signal into 3 distinct operational biochemical zones, calculating local structural energy scores. If unphysical ghost peaks are present (e.g. violating spin-spin coupling height ratios), the Energy-Based Model assigns a high global score and suppresses the peak:
  $$E_{\text{global}} = 0.4 \times E_{\text{aliphatic}} (0.5\text{-}3.0\text{ ppm}) + 0.4 \times E_{\text{carbohydrate}} (3.0\text{-}5.5\text{ ppm}) + 0.2 \times E_{\text{aromatic}} (5.5\text{-}9.0\text{ ppm})$$
* **Input Shape**: `[Batch_Size, 512]`
* **Decoded Output Shape**: `[Batch_Size, 1, 4000]`
* **Global Energy Shape**: `[Batch_Size, 1]` (Gauge Alarm triggers above threshold, e.g. `1.1`).

### Stage 4: Local Database Matching & Hybrid Scorer
* **Target File**: `database_matcher.py` -> `LocalDatabaseMatcher`
* **Mathematical Function**: Aligns the reconstructed spectrum against mock Human Metabolome Database (HMDB) libraries using constrained Dynamic Time Warping (DTW) with a radius of 10.
* **Metric Formula**: Computes a weighted, multi-dimensional clinical confidence profile:
  $$\text{Confidence} = 0.45 \times \text{PeakAssignmentScore} (0.85) + 0.35 \times \text{DTW\_Similarity} + 0.20 \times \sigma(-\text{ebm\_score})$$
* **Output Payload**: Structured candidate name, match confidence, and EBM physics score.

---

## ⚡ 3. FastAPI Python Backend Service Endpoints

The FastAPI server provides live endpoint routing, enabling immediate CORS-enabled browser requests.

### 1. `GET /api/sample`
* **Objective**: Simulates a clinical 1H NMR sample at 4,000 points resolution.
* **Parameters**: `sample_type` (Plant_Extract_A/B/C), `noise_level` (float), `shift_drift` (float), `add_ghost` (boolean).
* **Return Payload JSON**:
  ```json
  {
      "sample_type": "Plant_Extract_A",
      "raw_spectrum": [0.015, 0.024, ...],  // 4000 floats
      "ppm_axis": [10.0, 9.997, ...],       // 4000 floats
      "drifted_ppm": [10.015, 10.012, ...], // 4000 floats
      "annotations": [
          { "compound_name": "Sucrose", "ppm_position": 5.41, "relative_intensity": 1.8 }
      ]
  }
  ```

### 2. `POST /api/analyze`
* **Objective**: Feeds the 4,000-point spectrum through the deep learning encoder, continuous ODE integration, and EBM physics gate.
* **Payload Request JSON**:
  ```json
  {
      "raw_spectrum": [...],
      "label": "Plant_Extract_A",
      "ebm_threshold": 1.1,
      "shift_drift": 0.015,
      "add_ghost": true
  }
  ```
* **Return Payload JSON**:
  ```json
  {
      "telemetry": { "diagnostic_reports": [...] },
      "latent_features": [...],             // 64 dims for front-end activation plotting
      "trajectory_steps": [[...], ...],      // 5 integration steps trajectory
      "diagnostics": {
          "sample_id": "NMR-SAMPLE-2026-X99",
          "predicted_compound_class": "Plant_Extract_A",
          "raw_energy_score": 1.542,
          "cleaned_energy_score": 0.403,
          "ghost_peak_detected": true,
          "ebm_validation": "ANOMALY_CLEARED",
          "biomarkers": [...]
      }
  }
  ```

### 3. `POST /api/telemetry`
* **Objective**: Automatically logs anomalous peak structures to `anomalous_peaks_feedback.json` for active learning telemetry.

---

## 🎨 4. Astro Premium Vanilla CSS Frontend Specifications

The frontend is built on top of the Astro framework using custom stylesheets to ensure rapid loading, complete control, and maximum interactivity.

* **Design System**: Built entirely with Vanilla CSS inside `index.astro` using custom CSS variables (Slate backdrop, Outline/Space Grotesk typography, Emerald/Amber warning themes).
* **Glassmorphic Grid**: Renders a left-hand control sidebar and scorecard metrics, a dual-pane Plotly spectroscopic workspace, and expandable tech ML grading charts.
* **Plotly.js CDN script**: Utilizes the CDN distribution to completely bypass Vite packaging overhead, enabling zero-lag client-side plotting and custom downsampling (4000 inputs down to 2000 points for canvas rendering efficiency).
* **Active Learning Callback**: Hooks into analysis responses: if a ghost peak is flagged, it outputs a warning alert card, corrects the signal, and automatically hits `/api/telemetry` silently in the background.

---

## 🛠️ 5. Key Bug Resolutions & Engineering Tweaks

During integration, three critical issues were resolved to establish TRL 5+ robustness:

### Bug 1: SpectrumDecoder Dimension Mismatch
* **Symptom**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x512 and 4096x4000)` inside the output layer of `SpectrumDecoder`.
* **Root Cause**: The transpose convolutions in `reconstruct` produced a sequence length of `512` points. However, the final linear layer was defined in the blueprint as `nn.Linear(4096, output_dim)`.
* **Fix**: Aligned the dimensions by modifying the linear layer to `nn.Linear(512, output_dim)`.

### Bug 2: fastdtw Scipy Euclidean Vector Check
* **Symptom**: `ValueError: Input vector should be 1-D` inside scipy's `euclidean` distance utility.
* **Root Cause**: When executing DTW over 1D time-series, each point index passed to the distance function is a scalar float. Scipy's `euclidean` function expects a 1-D array vector of features, throwing a validation exception on scalars.
* **Fix**: Replaced `dist=euclidean` with a high-performance absolute difference lambda `dist=lambda a, b: abs(a - b)`. This natively supports scalar values and executes alignment significantly faster.

### Bug 3: Astro esbuild Curly Braces JSX Escape
* **Symptom**: `Transform failed: ERROR: Expected "}" but found ":"` in `index.astro` during dev compiling or building.
* **Root Cause**: Astro interprets standard curly braces `{ ... }` inside raw HTML markup as JSX expressions. Renders like `<pre>{ "status": "Pending" }</pre>` cause the esbuild parser to fail when encountering the JSON colon.
* **Fix**: Replaced the curly braces with HTML character entities `&#123;` and `&#125;` to render the literal characters without triggering JSX compiler parsing.

---

## 🚀 6. Developer Walkthrough & Operational Guide

Follow these steps to run, test, and develop the platform locally:

### 1. Boot the FastAPI Python Backend Service
Open a terminal in the root folder and execute:
```powershell
python AGENT/04_DATA/scripts/app_server.py
```
* **Verify**: Confirm console prints: `Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)`.

### 2. Boot the Astro Dev Server
Open a second terminal in the `frontend/` folder and execute:
```powershell
pnpm run dev
```
* **Verify**: Open `http://localhost:4321/` in your browser. Confirm that the glassmorphic clinical workstation loads, and the Plotly graphs display the Sugarcane extract spectrum.

### 3. Verify Active Learning Anomaly Telemetry
* Turn on the **"Inject Anomalous Ghost Peak"** checkbox in the sidebar.
* Click **"⚡ Run AI Diagnostic Analysis"**.
* Confirm that the **EBM circular gauge** lights up red and reads `ANOMALY`.
* Open `anomalous_peaks_feedback.json` in the root folder. Confirm a new active learning trace entry has been appended containing timestamp, raw energy, and anomalous peak details.

### 4. Build for Production Deployment
To package the Astro client into an optimized, static folder:
```powershell
cd frontend
pnpm run build
```
The compiled assets will be placed inside `frontend/dist/` ready to be served by any CDN or static server.
