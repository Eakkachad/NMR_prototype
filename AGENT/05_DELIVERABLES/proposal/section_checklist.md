# Proposal Draft — Section Checklist

## Page 1: The Vision & Impact

### Header
- [ ] Track: Phenome (Track 1)
- [ ] Project Name: _TBD — needs catchy, memorable name_
- [ ] Main Idea/Slogan: _TBD_
- [ ] Target Audience: Medical personnel, clinical laboratories, metabolomics researchers

### Problem Statement (โจทย์ปัญหา)
- [ ] Core challenge: 20,000+ features of overlapping NMR signals
- [ ] Significance: Manual interpretation takes hours/days per sample
- [ ] Current gap: No automated, scalable, ghost-peak-free solution exists
- [ ] PDF constraint mentioned

### Solution Approach (แนวทางการแก้ปัญหา)
- [ ] High-level: Automated ML pipeline for compound classification
- [ ] Hybrid architecture: Baseline (PLS-DA + SVM) + Innovation (Neural ODE + EBM)
- [ ] End-user focus: Dashboard for medical personnel

### Expected Impact
- [ ] Hackathon impact: Working POC demo classifying 3 compound types
- [ ] 12-24 month impact: Deployable web service for hospital metabolomics screening
- [ ] Success Metric 1: "Classify compounds with >85% accuracy on validation set"
- [ ] Success Metric 2: "Reduce NMR interpretation time from 4 hours to < 5 minutes"
- [ ] Success Metric 3: "Enable non-specialist operators to perform compound screening"

---

## Page 2: The Execution & Evaluation

### Tech Stack
- [ ] AI/ML: scikit-learn (PLS-DA, SVM, RF), PyTorch (Neural ODE, Autoencoder)
- [ ] Libraries: torchdiffeq, numpy, pandas, pdfplumber
- [ ] Visualization: plotly, matplotlib
- [ ] Platform: Streamlit web dashboard
- [ ] Deployment: Docker container, deployable to cloud or local
- [ ] Security: Encrypted data handling, no patient data transmitted externally

### Workflow Diagram
- [ ] Visual flowchart: PDF → Parser → Preprocessing → ML → Dashboard
- [ ] Must be clear and visually appealing
- [ ] Include both Core and Advanced paths

### Innovation & Feasibility
- [ ] Innovation: Neural ODE for NMR alignment (novel application)
- [ ] Innovation: EBM for ghost peak suppression (novel for NMR)
- [ ] Feasibility: Baseline uses proven, well-documented libraries
- [ ] Feasibility: Team has Python + ML experience
- [ ] Risks: Dataset timing, advanced model complexity

### Self-Assessment (คะแนน)
| Criterion | Max | Self-Score | Justification |
|-----------|-----|------------|---------------|
| Feasibility | 30 | _TBD_ | Working POC demo with baseline ML |
| Problem Match & Dataset | 25 | _TBD_ | Direct match to Phenome track objectives |
| Sustainable Impact | 20 | _TBD_ | Clinical screening acceleration + cost reduction |
| Novelty | 15 | _TBD_ | Neural ODE + EBM — first application to NMR |
| Implementation Plan | 10 | _TBD_ | 7-day sprint with clear milestones |
| **TOTAL** | **100** | **_TBD_** | |

---

## Status

| Section | Status | Assignee |
|---------|--------|----------|
| Header | ⬜ Not started | TBD |
| Problem Statement | ⬜ Not started | TBD |
| Solution Approach | ⬜ Not started | TBD |
| Expected Impact | ⬜ Not started | TBD |
| Tech Stack | ⬜ Not started | TBD |
| Workflow Diagram | ⬜ Not started | TBD |
| Innovation & Feasibility | ⬜ Not started | TBD |
| Self-Assessment | ⬜ Not started | TBD |
