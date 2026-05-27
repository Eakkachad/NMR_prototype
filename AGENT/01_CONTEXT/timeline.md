# 7-Day Sprint Timeline

## Overview

| Day | Date | Focus | Weight | Deliverable |
|-----|------|-------|--------|-------------|
| 1 | TBD | Data + Baseline Setup | 🔴 Critical | Synthetic data + working PLS-DA/SVM baseline |
| 2 | TBD | Pipeline Hardening | 🔴 Critical | End-to-end pipeline: raw data → prediction |
| 3 | TBD | Advanced Model (Exploration) | 🟡 Medium | Neural ODE / EBM integration attempt |
| 4 | TBD | POC Dashboard | 🔴 Critical | User-facing demo for medical personnel |
| 5 | TBD | Proposal Writing (Page 1) | 🔴 Critical | Problem, Solution, Impact sections |
| 6 | TBD | Proposal Writing (Page 2) | 🔴 Critical | Tech stack, Workflow diagram, Novelty, Self-assessment |
| 7 | TBD | QA + Submission | 🟠 High | Final review, cross-check, package, submit |

---

## Day 1: Data + Baseline Setup

### Tasks
- [ ] Generate synthetic NMR data mimicking 2D matrix format (N samples × 20,000 features)
- [ ] Simulate 3 compound classes (Plant Extract A, B, C) with distinct spectral signatures
- [ ] Implement spectral binning (reduce 20,000 → ~500-1000 bins)
- [ ] Implement PQN normalization
- [ ] Run PLS-DA → extract VIP scores → identify top features
- [ ] Train SVM (RBF kernel) on binned data
- [ ] Evaluate: accuracy, F1, confusion matrix

### Exit Criteria
✅ Baseline model achieves >70% accuracy on synthetic data  
✅ EDA notebook with plotted spectra and feature importance

---

## Day 2: Pipeline Hardening

### Tasks
- [ ] Refactor code into modular pipeline (preprocessing → feature selection → classification)
- [ ] Add Random Forest as alternative classifier
- [ ] Implement cross-validation (5-fold or LOO if N is small)
- [ ] Add model comparison table
- [ ] Write PDF parser script (for when real data arrives)
- [ ] Test parser on sample PDF

### Exit Criteria
✅ Pipeline runs end-to-end in a single script  
✅ Cross-validation results documented  
✅ PDF parser handles at least the sample file

---

## Day 3: Advanced Model (Exploration)

### Tasks
- [ ] Set up `torchdiffeq` for Neural ODE
- [ ] Build a simple Autoencoder for latent space projection
- [ ] Attempt Latent ODE on the latent representation
- [ ] If Neural ODE works → demonstrate alignment improvement
- [ ] If Neural ODE fails → document the attempt and fallback to baseline
- [ ] Explore EBM concepts (even theoretical is fine)

### Exit Criteria
✅ At least an ATTEMPT at Neural ODE is documented  
✅ Clear decision: does it add value or do we stay with baseline?  
✅ Decision logged in `06_AGENT_PROTOCOL/decision_log.md`

---

## Day 4: POC Dashboard

### Tasks
- [ ] Design simple web dashboard or Streamlit/Gradio app
- [ ] Features: upload NMR data → preprocess → classify → show results
- [ ] Add spectral visualization (plotly/matplotlib)
- [ ] Add feature importance visualization
- [ ] Add compound match results with confidence scores
- [ ] Ensure medical personnel can use it without NMR expertise

### Exit Criteria
✅ Working demo that a non-expert can operate  
✅ Visual output that makes biological sense  
✅ Screenshot/recording for proposal inclusion

---

## Day 5: Proposal Page 1

### Tasks
- [ ] Write Header section (track, project name, slogan, target audience)
- [ ] Write Problem Statement
- [ ] Write Solution Approach
- [ ] Write Expected Impact (hackathon, 12-24 month, success metrics)
- [ ] Internal review

### Exit Criteria
✅ Page 1 complete and reviewed

---

## Day 6: Proposal Page 2

### Tasks
- [ ] Write Tech Stack section
- [ ] Create Workflow Diagram
- [ ] Write Innovation & Feasibility section
- [ ] Complete Self-Assessment table with justifications
- [ ] Internal review

### Exit Criteria
✅ Page 2 complete and reviewed  
✅ Self-assessment scores add up to ≤100

---

## Day 7: QA + Submission

### Tasks
- [ ] Cross-reference proposal claims against POC capabilities
- [ ] Verify all required sections are present
- [ ] Proofread for grammar and consistency
- [ ] Check file format requirements
- [ ] Package all deliverables
- [ ] SUBMIT

### Exit Criteria
✅ All deliverables submitted before deadline  
✅ Consistency check passed  

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Dataset not available in time | Use synthetic data — already planned for Day 1 |
| Neural ODE doesn't work | Baseline (PLS-DA + SVM) is always the guaranteed fallback |
| Proposal too technical for judges | Focus on clinical impact narrative + clear diagrams |
| Time runs out on Day 3 | Day 3 is explicitly "exploration" — non-critical for submission |
