# Consolidated Literature Matrix

> Deduplicated from `research01.md` and `research02.md`. Single source of truth for all referenced papers.

## Theme 1: Deep Learning for NMR — Transformer & Attention Architectures

| Paper | Year | Architecture | Input | Performance | Code Available? | Relevance to POC |
|-------|------|-------------|-------|-------------|-----------------|-------------------|
| **NMRformer** (Zhou et al.) | 2025 | Transformer encoder + self-attention + peak-height ratios | 1D ¹H NMR (cells & biofluids) | >88% peak assignment, >80% metabolite ID | ❌ No public repo | HIGH — best accuracy reference |
| **NMRTrans** (Yang et al.) | 2026 | Set Transformer (unordered peak sets) | ¹H & ¹³C (NMRSpec corpus) | Top-10 acc: 61.15% (+17.82 pts) | ✅ Hugging Face (arXiv:2602.10158) | MEDIUM — novel concept |
| **FlavorFormer** (Liao et al.) | 2025 | Hybrid CNN + Transformer | ¹H NMR flavor mixtures | Outperforms prior methods | ❌ No public repo | MEDIUM — mixture analysis |
| **MolDeTr** (Schmid et al.) | 2026 | Detection Transformer (DETR-like) | 1D ¹H NMR (synthetic training) | Median err: 0.89 Hz (shifts), 93.5% proton acc | ❌ ChemRxiv preprint only | LOW — overkill for POC |
| **NMRMind** | 2025 | Transformer for structure elucidation | Multidimensional NMR | Not detailed | ❌ | LOW |
| **NMRQNet** (LiuzLab) | 2023 | CRNN (CNN + GRU) | 1D ¹H NMR plasma | 38 metabolites quantified | ✅ `LiuzLab/NMRQNet` | HIGH — has code, quantitative |
| **DEEP Phaser** | 2024-25 | Tandem Vision Transformer | Solution NMR | Automated phasing | ⚠️ Web server only | LOW for POC |
| **K-M3AID** | 2024 | Contrastive learning (graph+spectra) | NMR spectra | Zero-shot retrieval | ❌ | LOW |
| **DeepSAT** | 2023 | Neural network | ¹H-¹³C HSQC | Scaffold prediction | ❌ | LOW (requires 2D HSQC) |
| **ANN-DA / ANNDL-DA** | 2024 | ANN for 2D peak picking | 2D NMR | ~90% peak selection accuracy | ❌ | MEDIUM |

## Theme 2: Neural ODEs & Continuous Models

| Paper | Year | Architecture | Application | NMR-specific? | Code Available? | Relevance to POC |
|-------|------|-------------|-------------|---------------|-----------------|-------------------|
| **Latent ODEs** (Rubanova et al.) | 2019 | Latent ODE + ODE-RNN encoder | General irregular time series | ❌ | ✅ `YuliaRubanova/latent_ode` | HIGH — foundation code |
| **Latent Space Energy-based Neural ODEs** | 2024 | Energy prior + Latent ODE | General sequences | ❌ | ⚠️ arXiv only | HIGH — closest to our concept |
| **Path-minimizing Latent ODEs** | 2024 | Latent ODE + path penalty | General dynamical systems | ❌ | ❌ | LOW |
| **ChromAlignNet** | 2025 | Deep Learning alignment | GC-MS peak alignment | ❌ (GC-MS not NMR) | ❌ | MEDIUM — transferable concept |
| **SPIN-ODE** | 2025 | Physics-constrained Neural ODE | Chemical reaction rates | ❌ (kinetics) | ✅ `pvvq/SPIN-ODE` | MEDIUM — physics constraint pattern |
| **BINODEs** | 2026 | Biochemically Informed Neural ODE | Mechanistic ODE models | ❌ | ❌ | LOW |
| **Autoencoders + Neural ODE (astrochem)** | 2023 | AE + Latent ODE solver | Astrochemical reactions | ❌ | ❌ | MEDIUM — blueprint pattern |

> [!WARNING]
> **No published work applies Neural ODEs directly to NMR peak alignment.** This is a genuine research gap. Our proposal should frame this as novel, not proven.

## Theme 3: Energy-Based Models & Physics-Informed ML

| Paper | Year | Architecture | Application | Ghost Peak Suppression? | Code Available? | Relevance to POC |
|-------|------|-------------|-------------|------------------------|-----------------|-------------------|
| **DEEP Picker** (Li et al.) | 2021 | 8-layer CNN | 2D NMR peak picking | ✅ Deconvolves overlaps | ✅ `lidawei1975/deep` | HIGH — mature tool |
| **DEEP Picker1D & Voigt Fitter1D** | 2023 | DNN extension | 1D NMR peak/Voigt fitting | ✅ | ✅ Same repo | HIGH — directly relevant |
| **EB-gMCR** | 2024-25 | Energy-based generative MCR | Spectral unmixing | ✅ Removes ghost signals | ❌ OpenReview only | HIGH — core concept |
| **MIST-CF** | 2023 | EBM + Spectrum Transformer | MS/MS formula ranking | ✅ Suppresses false positives | ❌ | MEDIUM — MS not NMR |
| **FitNMR** | 2020 | Analytical peak modeling | 1D NMR | ✅ Overlap resolution | ❌ | LOW |
| **PINN for Laplace NMR** | 2024 | Physics-Informed NN | 2D Laplace NMR reconstruction | ✅ Reduces artifacts | ❌ | MEDIUM |

## Theme 4: Unknown Compound Discovery & Database Matching

| Paper | Year | Algorithm | Application | Code Available? | Relevance to POC |
|-------|------|-----------|-------------|-----------------|-------------------|
| **SPA-STOCSY** (LiuzLab) | 2023 | Spatial clustering + STOCSY | NMR metabolite clustering | ✅ `LiuzLab/SPA-STOCSY` | HIGH — has code |
| **SUMMIT Motif** | 2024-25 | Motif extraction from 2D/3D NMR | De novo identification | ❌ | MEDIUM |
| **DBsimilarity** | 2023 | Similarity network | PubChem/HMDB matching | ❌ | LOW |
| **NMR-TS** | 2020-24 | ML + DFT spectra | Candidate molecule matching | ❌ | LOW |

## Priority for POC Implementation

Based on code availability and direct relevance:

1. 🔴 **NMRQNet** — Has code, does quantitative metabolite analysis
2. 🔴 **DEEP Picker** — Has code, does peak picking and deconvolution
3. 🔴 **SPA-STOCSY** — Has code, does metabolite clustering
4. 🟠 **Latent ODE** (Rubanova) — Has code, foundation for advanced layer
5. 🟠 **SPIN-ODE** — Has code, physics constraint pattern
6. 🟡 **NMRTrans** — Has Hugging Face data, novel concept for proposal
