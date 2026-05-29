# BDI KKU NMR Astro Frontend Workstation
=====================================

This folder contains the **high-performance single-page web client** for the AI NMR Spectroscopy Workstation, built with the modern Astro framework. 

It communicates directly with the GPU-accelerated FastAPI backend on port 8000 and features premium styling, interactive zoomable Plotly charts, and a client-side laboratory file ingestion engine.

---

## 📁 Directory Structure

```
frontend/
├── .astro/                 # Astro build cache and metadata
├── public/                 # Static assets (icons, layouts)
├── src/                    # Web Application Source
│   ├── assets/             # Images and local styles
│   ├── components/         # Reusable UI card structures
│   ├── layouts/            # Base HTML grid skeletons
│   └── pages/              
│       └── index.astro     # Core Single-Page App (HTML + CSS + JS)
├── package.json            # PNPM dependencies manifest
└── pnpm-lock.yaml          # PNPM Lockfile
```

---

## ⚡ Core UI Technologies & Features

1. **Forced Premium Light Theme**: Designed specifically for clinical utility with Slate-Gray layouts, crisp white glass cards, and amber/emerald warning badges to maximize readability under hospital diagnostic lighting.
2. **Unified Single-Page Flow**: Combines patient profile selection, file uploads, real-time biomarker integral abundance tables, zoomable overlay spectrometry canvases, and stacked reference waterfalls in a single, seamless, non-tabbed workflow.
3. **Advanced Client-Side File Ingestion**:
   - Reads standard laboratory CSV/TXT spectral coordinate files on-the-fly via the browser `FileReader` API.
   - Automatically detects delimiters (comma, tab, semicolon, whitespace) and filters out text headers dynamically.
   - Performs high-performance linear interpolation (`np.interp` equivalent) to sort and align raw data onto the model's standard 4,000 points grid (0.0 to 10.0 ppm) in < 2ms.
   - Automatically calculates a real-time **Pearson Correlation Coefficient** to match the uploaded spectrum against standard metabolic profiles, loading the appropriate biomarker references and PubChem links instantly.
4. **Interactive Spectroscopic Workspace**: Powered by Plotly.js to render dual-pane spectroscopy plots, enabling real-time zooming in defined chemical zones (Aliphatic, Carbohydrate, Aromatic) and stacked library comparisons.
5. **Technical Expander Panel**: Keeps the clinician's workspace perfectly tidy by tucking away deep technical telemetries (PyTorch 512-dim latent activations, Euler integration trajectories, EBM gauge dials, ML accuracy comparisons, and EMR hospital JSON EMR payloads) inside a collapsed accordion.

---

## 🚀 Getting Started

### 1. Installation of PNPM
If pnpm is not installed on your system, install it via npm globally:
```powershell
npm install -g pnpm
```

### 2. Install Project Dependencies
Run the install command inside the frontend directory:
```powershell
pnpm install
```

### 3. Run the Development Server
Launch the local dev server on http://localhost:4321:
```powershell
pnpm run dev
```

### 4. Build for Production
Create the optimized production build:
```powershell
pnpm run build
```

---

*Verified Production-Grade Web UI Client – Powered by Astro & Plotly.js.*
