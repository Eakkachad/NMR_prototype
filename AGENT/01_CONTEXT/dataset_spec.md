# Dataset Specification — NMR Spectrum Data

## Data Identity

| Field | Value |
|-------|-------|
| **Title** | NMR Spectrum Dataset for Chemical Pattern Detection |
| **Track** | Phenome |
| **Domain** | Metabolomics, Compound Profiling |
| **Source** | National Phenome Institute (สถาบันฟีโนมแห่งชาติ) |
| **Status** | ⚠️ STILL BEING COLLECTED — sample PDF available |

## Data Format

| Property | Detail |
|----------|--------|
| **File format** | PDF (sample: `Domain_1_processed_NMR_spectrum.pdf`) |
| **Layout** | 2D Matrix: Samples × Signals |
| **Features per sample** | 20,000+ columns |
| **Feature meaning** | Chemical shift positions on NMR spectrum (ppm) |
| **Sample profiles** | Plant Extract A, B, and C across multiple subjects |
| **Total records** | Not specified (null in official specs) |
| **Sample size (N)** | Unknown — critical gap |

## Clarification: "2D" Terminology

> [!WARNING]
> The dataset description says "2D NMR" but this is **ambiguous**:
> - The README describes a "2D matrix" meaning Samples × Signals — this is a **table of 1D spectra**
> - True 2D NMR (HSQC, COSY, TOCSY) would have two frequency axes per sample
>
> **Working assumption:** The data is **1D ¹H NMR spectra** arranged in a 2D matrix (rows = samples, columns = ppm positions). This should be verified when the dataset is finalized.

## What the Data Represents

Each row is one biological sample (e.g., a plant extract).
Each column is a signal intensity at a specific chemical shift (ppm position).
The values indicate the presence and concentration of metabolites at that position.

**Example compounds detectable:** Glucose, amino acids, organic acids, lipids — typical metabolomics targets.

## Known Challenges

1. **High dimensionality:** 20,000+ features vs. small N (curse of dimensionality)
2. **Peak shifting:** Same compound appears at slightly different ppm across samples due to pH, temperature, instrument drift
3. **Signal overlap:** Multiple metabolites share similar chemical shifts
4. **Baseline noise:** Many features are just noise or solvent signals
5. **PDF format:** Data needs to be extracted from PDF before processing
6. **Ghost peaks:** Automated tools may generate false-positive signals
7. **Expert dependency:** Traditional interpretation requires trained spectroscopists

## Data Extraction Strategy

Since the dataset is not yet finalized:
1. **Phase 1 (now):** Build pipeline with **synthetic NMR data** to prove feasibility
2. **Phase 2 (when available):** Parse the PDF using `pdfplumber` or `tabula-py`
3. **Phase 3:** Validate pipeline with real data

## Reference Databases for Compound Matching

| Database | URL | Purpose |
|----------|-----|---------|
| HMDB | hmdb.ca | Human Metabolome Database |
| PubChem | pubchem.ncbi.nlm.nih.gov | Chemical compound reference |
| BMRB | bmrb.io | Biological Magnetic Resonance Bank |
