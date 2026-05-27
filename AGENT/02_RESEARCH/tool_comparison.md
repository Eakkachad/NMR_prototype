# Tool Comparison — Existing NMR Analysis Software

## Head-to-Head Comparison

| Tool | Algorithm | Speed | Accuracy | Scalability | Automation | Ghost Peak Risk | Cost | Verdict |
|------|-----------|-------|----------|-------------|------------|-----------------|------|---------|
| **Chenomx** | Human-guided template matching | ❌ Hours/days per sample | ✅ Extremely high (expert-driven) | ❌ Unscalable | ❌ Manual | ❌ Human bias | 💰 Commercial license | Gold standard but not viable for screening |
| **Bayesil** | Sequential Monte Carlo (SMC) | ✅ 2-7 min/sample | ✅ Good for supported fluids | ❌ Rigid (serum/plasma only) | ✅ Fully automated | ⚠️ Limited to known templates | 🆓 Free (web-based) | Good but too rigid for general use |
| **BATMAN** | MCMC Bayesian deconvolution | ❌ Extremely slow | ✅ Strong statistical foundation | ❌ Fails at 20,000+ features | ✅ Automated | ✅ Models shift drifts | 🆓 R package | Theoretically sound, computationally unviable |
| **ASICS** | LASSO / Penalized Linear Regression | ✅ Exceptionally fast | ⚠️ OK for simple spectra | ✅ Highly scalable | ✅ Automated | ❌ HIGH — ghost peaks, misidentification | 🆓 R package | Fast but unreliable in complex mixtures |

## Where Our Pipeline Fits

| Capability | Chenomx | Bayesil | BATMAN | ASICS | **Our POC** |
|------------|---------|---------|--------|-------|-------------|
| Speed | ❌ | ✅ | ❌ | ✅ | ✅ (seconds) |
| Automation | ❌ | ✅ | ✅ | ✅ | ✅ |
| Scalability | ❌ | ❌ | ❌ | ✅ | ✅ |
| Ghost Peak Suppression | ❌ | ⚠️ | ✅ | ❌ | ✅ (EBM) |
| Drift Correction | ❌ | ❌ | ⚠️ | ❌ | ✅ (Neural ODE) |
| No Expert Required | ❌ | ✅ | ✅ | ✅ | ✅ |
| Flexible Input | ❌ | ❌ | ⚠️ | ✅ | ✅ |

## Key Messaging for Proposal

> "Existing tools force a trade-off between accuracy and scalability. Chenomx is accurate but unscalable. ASICS is scalable but unreliable. Our hybrid pipeline bridges this gap by combining deep learning feature extraction (scalability) with physics-constrained verification (accuracy), while remaining fully automated and accessible to non-expert medical personnel."
