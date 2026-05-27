# Available Code Repositories

## Tier 1: Directly Usable (has code + relevant to our pipeline)

### NMRQNet — Quantitative Metabolite Analysis
- **URL:** https://github.com/LiuzLab/NMRQNet
- **What:** CRNN (CNN + GRU) that reconstructs spectra and quantifies 38 plasma metabolites
- **Language:** Python
- **Use in our POC:** Reference implementation for spectral classification
- **Status:** 🟢 Active
- **Notes:** Resolves overlapping peaks implicitly through reconstruction

### DEEP Picker — Peak Picking & Deconvolution
- **URL:** https://github.com/lidawei1975/deep
- **What:** Deep NN for 2D/1D NMR peak picking, Voigt fitting, spectral reconstruction
- **Language:** C++/Python
- **Use in our POC:** Preprocessing step — extract clean peak lists from raw spectra
- **Status:** 🟢 Active (published Nature Communications 2021, updated 2023)
- **Notes:** Trained on synthetic spin-dynamics simulations

### SPA-STOCSY — Metabolite Clustering
- **URL:** https://github.com/LiuzLab/SPA-STOCSY
- **What:** Spatial correlation clustering to identify metabolite peak groups
- **Language:** Python/R
- **Use in our POC:** Unknown compound discovery via correlated peak clustering
- **Status:** 🟢 Active
- **Notes:** Outperforms SRV, comparable to Chenomx without operator bias

### Latent ODE — Continuous Time Series Modeling
- **URL:** https://github.com/YuliaRubanova/latent_ode
- **What:** PyTorch implementation of Latent ODEs for irregularly-sampled time series
- **Language:** Python (PyTorch)
- **Use in our POC:** Foundation for Stage 2 (continuous drift correction)
- **Status:** 🟡 Archived (2019) but still functional
- **Notes:** General framework — would need adaptation for NMR

## Tier 2: Reference Only (pattern or concept borrowing)

### SPIN-ODE — Physics-Constrained Neural ODE
- **URL:** https://github.com/pvvq/SPIN-ODE
- **What:** Physics-constrained Neural ODE for chemical reaction rate estimation
- **Language:** Python (PyTorch)
- **Use in our POC:** Borrow the physics-constraint pattern for EBM loss function
- **Status:** 🟢

### NMRNet — Chemical Shift Prediction
- **URL:** https://github.com/Colin-Jay/NMRNet
- **What:** Benchmark framework for DL-based NMR chemical shift prediction
- **Language:** Python
- **Use in our POC:** Reference for model architecture choices
- **Status:** 🟢

### Computational Metabolomics Review
- **URL:** https://github.com/enveda/computational-metabolomics-review
- **What:** Curated list of tools and databases for metabolomics (2021-2025)
- **Language:** Markdown
- **Use in our POC:** Find additional tools/references for the proposal
- **Status:** 🟢

## Key Python Libraries Needed

```
# Core ML
scikit-learn          # PLS-DA, SVM, Random Forest, cross-validation
numpy                 # Array operations
pandas                # Data manipulation

# Deep Learning (if exploring advanced models)
torch                 # PyTorch
torchdiffeq           # Neural ODE solver

# NMR-Specific
nmrglue               # NMR data processing
pdfplumber             # PDF data extraction
tabula-py              # Alternative PDF table extraction

# Visualization
matplotlib             # Basic plots
plotly                 # Interactive spectra
seaborn               # Statistical plots

# Dashboard
streamlit              # Or gradio — quick web UI for POC demo
```
