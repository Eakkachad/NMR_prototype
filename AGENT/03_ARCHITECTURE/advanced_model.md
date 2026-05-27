# Advanced Model — Innovation Layer Specification

> Priority: 🟡 EXPERIMENTAL — Build only after baseline is working. This is for Novelty points (15 pts).

## Overview

The advanced layer introduces two novel components that have **no prior application in NMR spectroscopy**:
1. **Latent Neural ODE** — for continuous chemical shift drift correction
2. **Energy-Based Model (EBM)** — for physics-constrained deconvolution and ghost peak suppression

## Stage 1: Feature Extraction (Autoencoder)

### Purpose
Compress 20,000+ dimensional raw spectra (or ~500 bins) into a compact latent representation (~32-64 dimensions).

### Architecture
```
Input (500 bins) → Encoder [FC→ReLU→FC→ReLU→FC] → Latent (64) → Decoder [FC→ReLU→FC→ReLU→FC] → Reconstruction (500)
```

### Loss
Reconstruction loss (MSE) + optional KL divergence (if using VAE variant)

---

## Stage 2: Continuous Alignment via Latent Neural ODE

### The Innovation

> [!WARNING]
> **THIS IS NOVEL RESEARCH.** No published work applies Neural ODEs to NMR peak alignment. Frame accordingly.

### Concept
Instead of treating the ppm axis as discrete columns, model the spectral signal as a continuous function. Chemical shift drift (caused by pH, temperature, instrument variation) is modeled as a smooth transformation in latent space:

$$\frac{dh(t)}{dt} = f_\theta(h(t), t)$$

Where:
- $h(t)$ is the latent state at ppm position $t$
- $f_\theta$ is a neural network parameterizing the dynamics
- The ODE is solved with `torchdiffeq.odeint`

### Why It Should Work (Theoretical Justification)
1. Chemical shift drift is CONTINUOUS and SMOOTH — ideal for ODE modeling
2. Latent ODEs are proven for irregularly-sampled time series (Rubanova et al., 2019)
3. NMR spectra along the ppm axis are mathematically analogous to time series

### Implementation Reference
Base code: `YuliaRubanova/latent_ode` (PyTorch)

### Key Dependencies
```python
import torchdiffeq  # pip install torchdiffeq
```

---

## Stage 3: Energy-Based Deconvolution

### Concept
An EBM assigns low energy to physically plausible compound signals and high energy to artifacts (ghost peaks).

### Energy Function
$$E(x) = E_{reconstruction}(x) + \lambda \cdot E_{physics}(x)$$

Where:
- $E_{reconstruction}$: how well the signal can be reconstructed from known compound templates
- $E_{physics}$: penalty for violating physical constraints (e.g., wrong spin-coupling ratios, impossible peak-height ratios)

### Physical Constraints to Encode
1. **Multiplicity rules:** Each compound has specific multiplet patterns (singlet, doublet, triplet, etc.)
2. **J-coupling constants:** Fixed for each molecular bond type
3. **Peak-height ratios:** Follow binomial distribution for coupled spins
4. **Chemical shift ranges:** Each compound type appears in known ppm ranges

### Inspiration
- EB-gMCR: energy-based component selection for spectral unmixing
- MIST-CF: energy function ranking for chemical formula assignment

---

## Feasibility Assessment for POC

| Component | 7-Day Feasibility | Fallback |
|-----------|-------------------|----------|
| Autoencoder | ✅ Achievable | Use PCA instead |
| Neural ODE | ⚠️ Challenging | Skip — use standard binning alignment |
| EBM | ❌ Unlikely for working demo | Conceptual only in proposal |

### Recommended POC Strategy
1. Build the Autoencoder (2-3 hours)
2. Attempt a simple Neural ODE integration (4-6 hours)
3. If Neural ODE shows ANY improvement → include in demo with caveats
4. If Neural ODE fails → document the attempt, present as "future work" in proposal
5. EBM → describe in proposal only, don't code

---

## For the Proposal (How to Frame This)

> "We identify a previously unexplored application of Neural ODEs to NMR spectral alignment. While Neural ODEs have been validated for continuous-time modeling in irregularly-sampled time series (Rubanova et al., 2019) and physics-constrained chemical kinetics (SPIN-ODE, 2025), their application to chemical shift drift correction represents a novel contribution. Our POC demonstrates the feasibility of this approach in a controlled setting, with the full pipeline designed for iterative refinement as validation data becomes available."
