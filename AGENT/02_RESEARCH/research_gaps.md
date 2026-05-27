# Research Gaps — Honest Assessment

## Confirmed Gaps

### Gap 1: Neural ODEs for NMR Peak Alignment — NO PRIOR WORK EXISTS
- **Evidence:** research01.md explicitly states: "no published work was found that directly applies Neural ODEs or Latent ODEs to chemical-shift drift correction or peak alignment in NMR spectroscopy"
- **Implication for proposal:** This is SIMULTANEOUSLY our biggest novelty claim AND our biggest feasibility risk
- **Recommendation:** Frame as "We identify a clear gap in the literature and propose Neural ODEs as a novel solution for continuous spectral alignment. While this application is unprecedented, the underlying framework (Rubanova et al., 2019) is well-validated for continuous-time modeling, and NMR chemical shift drift is mathematically analogous to time-series drift."
- **DO NOT:** Claim this is a "proven" or "established" technique

### Gap 2: EBMs for NMR Ghost Peak Detection — ONLY CONCEPTUAL
- **Evidence:** EB-gMCR demonstrates the concept for general spectral unmixing, not specifically NMR peak fitting
- **Closest work:** DEEP Picker uses DNNs (not EBMs) for peak deconvolution
- **Recommendation:** Say "inspired by EB-gMCR's success in spectral unmixing" rather than "applying the proven EB-gMCR approach"

### Gap 3: Code Availability for Key Papers
- **NMRformer:** No public code
- **FlavorFormer:** No public code
- **MolDeTr:** No public code (preprint only)
- **EB-gMCR:** OpenReview paper only, no code
- **MIST-CF:** No code specified
- **Implication:** We cannot directly use these architectures. Our POC must rely on the repos that DO have code (NMRQNet, DEEP Picker, SPA-STOCSY, Latent ODE)

### Gap 4: Validation on Plant Extract NMR
- **Evidence:** Most literature focuses on biofluid NMR (serum, plasma, urine). Our data is plant extracts.
- **Implication:** Model performance benchmarks from literature may not directly transfer. Plant metabolomes have different compound profiles (e.g., flavonoids, terpenes, alkaloids).

## What This Means for the POC

1. **The baseline (PLS-DA + SVM) is the guaranteed deliverable.** It works, it's proven, it's fast.
2. **Neural ODE and EBM are "innovation bonus" items.** If they work, great. If not, the baseline still scores well.
3. **Honesty wins.** Judges will respect a team that acknowledges gaps and proposes solutions, more than a team that overclaims.
