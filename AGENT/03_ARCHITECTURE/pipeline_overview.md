# Pipeline Overview — Canonical Architecture (v1.0)

> **This is the SINGLE official pipeline design.** Do NOT reference information.md §5 or preplane.md §4 directly — they contain duplicated, inconsistent versions. Use this document.

## Architecture Diagram (Text)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                       │
│  Raw NMR Data (PDF / CSV) ──→ PDF Parser ──→ 2D Matrix (N × 20,000+)   │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   PREPROCESSING LAYER                                    │
│  ┌──────────────┐    ┌─────────────────┐    ┌────────────────────────┐  │
│  │ Spectral     │    │ PQN             │    │ Quality Control        │  │
│  │ Binning      │───→│ Normalization   │───→│ (Outlier removal,     │  │
│  │ (20K → 500)  │    │                 │    │  baseline correction)  │  │
│  └──────────────┘    └─────────────────┘    └────────────────────────┘  │
└──────────────────────────────────┬───────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                    ▼                             ▼
┌─────────────────────────────┐   ┌──────────────────────────────────────┐
│     CORE LAYER (Baseline)   │   │     ADVANCED LAYER (Innovation)      │
│     ═══════════════════     │   │     ═══════════════════════════      │
│                             │   │                                      │
│  Feature Selection:         │   │  Stage 1: Latent Projection          │
│  ├─ PLS-DA → VIP scores    │   │  ├─ Autoencoder / Transformer        │
│  └─ Select top features    │   │  └─ 20K dims → 64-dim latent space   │
│                             │   │                                      │
│  Classification:            │   │  Stage 2: Continuous Alignment       │
│  ├─ SVM (RBF kernel)       │   │  ├─ Latent Neural ODE                │
│  ├─ Random Forest           │   │  └─ dh/dt = f(h(t), t, θ)           │
│  └─ Cross-validation        │   │                                      │
│                             │   │  Stage 3: Physics-Constrained ID     │
│  Output:                    │   │  ├─ Energy-Based Model               │
│  ├─ Compound class label    │   │  ├─ Spin-coupling constraints        │
│  ├─ Confidence score        │   │  └─ Ghost peak suppression           │
│  └─ Feature importance map  │   │                                      │
│                             │   │  Output:                             │
│  Status: 🟢 GUARANTEED      │   │  ├─ Refined compound labels          │
│                             │   │  ├─ Alignment quality metrics        │
└─────────────────────────────┘   │  └─ Anomaly/OOD detection            │
                                   │                                      │
                                   │  Status: 🟡 EXPERIMENTAL (NOVEL)     │
                                   └──────────────────────────────────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER (Dashboard)                            │
│  ┌──────────────┐   ┌─────────────────┐   ┌────────────────────────┐   │
│  │ Spectral     │   │ Compound        │   │ Report                 │   │
│  │ Visualization│   │ Classification  │   │ Generation             │   │
│  │ (Interactive)│   │ Results + Conf. │   │ (PDF / Screenshot)     │   │
│  └──────────────┘   └─────────────────┘   └────────────────────────┘   │
│                                                                         │
│  Target User: Medical personnel (non-expert in NMR)                     │
│  Interface: Streamlit / Gradio web app                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

## Design Principles

1. **Dual-track architecture:** Core Layer always works, Advanced Layer is bonus
2. **POC-first:** Core Layer is built first and can stand alone for the demo
3. **Medical-personnel UX:** Output must be interpretable without NMR expertise
4. **Modular:** Each stage is an independent module that can be swapped/upgraded
5. **Honest framing:** Advanced Layer is labeled "experimental/novel" in all documentation

## Stage Specifications

See individual files for detailed specifications:
- [baseline_model.md](baseline_model.md) — PLS-DA, SVM, Random Forest
- [stage1_feature_extraction.md](stage1_feature_extraction.md) — Dimensionality reduction
- [stage2_alignment.md](stage2_alignment.md) — Neural ODE drift correction
- [stage3_deconvolution.md](stage3_deconvolution.md) — EBM physics constraints
