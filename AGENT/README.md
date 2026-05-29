# BDI KKU NMR AI Clinical Context & Development Protocols
=======================================================

This folder contains the **clinical context**, **scientific research notes**, **system architecture blueprints**, **datasets**, and **agentic development protocols** for the NMR AI diagnostic workspace.

---

## 📁 Folder Directory

```
AGENT/
├── 00_MASTER_BRIEF.md         # Master brief outlining clinical goals
├── 01_CONTEXT/                # Competition rules, parameters, & guidelines
├── 02_RESEARCH/               # Scientific literature, HMDB/BMRB benchmarks
├── 03_ARCHITECTURE/           # Blueprints for continuous 4-stage convolutional AI flow
├── 04_DATA/                   # Datasets, dictionary schemas, and core scripts (FastAPI/Streamlit)
│   └── (See 04_DATA/README.md for detailed data briefings)
├── 05_DELIVERABLES/           # Final handbooks and Thailand manuals
├── 06_AGENT_PROTOCOL/         # Active learning checklists, tasks, and task boards
├── 07_QA/                     # Automatic and manual verification scripts
└── README.md                  # This documentation file
```

---

## ⚙️ Engineering & Development Methodology

Our clinical engineering workflow adheres to rigorous scientific and programming standards:
1. **01_CONTEXT**: Establishes competition rules, ensuring that all deep learning pipelines meet the specified Data Contracts and EHR compliance guidelines.
2. **02_RESEARCH**: Anchors the pipeline's compound library on authentic clinical chemical shifts sourced from international databases (HMDB/BMRB).
3. **03_ARCHITECTURE**: Visualizes and outlines the 4-stage PyTorch pipeline (1D-Conv sequence encoder, continuous latent ODE alignment, spectrum reconstruction decoder, and localized EBM physics checks).
4. **04_DATA**: Houses raw spectrometer data, field definitions, and the actual PyTorch/FastAPI codebase. (For more details, see [04_DATA README](04_DATA/README.md)).
5. **05_DELIVERABLES**: Packages Thailand user handbooks and diagnostic workstation manuals.
6. **06_AGENT_PROTOCOL & 07_QA**: Tracks active checklists, tasks, and test results, maintaining absolute repository cleanliness and rigorous validation standards.
