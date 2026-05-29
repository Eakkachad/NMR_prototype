# BDI KKU Clinical NMR Datasets & Analytical Core
================================================

This directory houses the laboratory dataset schemas, metadata tables, parsed dictionary fields, and core deep learning pipeline scripts.

---

## 📁 Folder Structure

```
AGENT/04_DATA/
├── parsed/                    # Intermediate JSON metadata files
├── raw/                       # Original NMR spectrometer outputs (.csv/.txt)
├── scripts/                   # Core Python, PyTorch, & Web API scripts (FastAPI/Streamlit)
│   └── (See scripts/README.md for detailed script briefings)
├── data_dictionary.md         # Detailed description of clinical data fields
└── README.md                  # This documentation file
```

---

## 📊 Core Data Concepts

### 1. Raw Spectrometer Outputs (`raw/`)
Contains high-resolution NMR spectroscopy datasets. The standard spectrometer output is represented as two-column numeric coordinates:
- **`ppm` (Chemical Shift)**: Sourced from 0.0 to 10.0 ppm, indicating the magnetic resonance frequency offset of nuclear spins.
- **`intensity`**: Real-valued amplitude of absorption, representing the quantitative abundance of chemical compounds.

### 2. Intermediate Parsed Tables (`parsed/`)
Holds processed, downsampled, or metadata-synchronized representations of NMR signals, optimized for fast training and REST payloads.

### 3. Data Dictionary (`data_dictionary.md`)
A structured markdown catalog describing:
- Variables, ranges, data types, and physical units (e.g. chemical shifts in ppm, relative integrals, EBM energy scores).
- Schema data contracts for electronic medical record (EMR/EHR) clinical integration.

---

## 🚀 Execution Guide
For detailed instructions on running the PyTorch pipelines, REST API servers, or visual dashboards, please refer to the script-specific documentation at:
👉 **[Core Scripts README (scripts/README.md)](scripts/README.md)**
