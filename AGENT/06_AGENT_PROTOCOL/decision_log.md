# Decision Log

> Record all significant decisions here. This prevents future agents from re-debating settled questions.

---

## Decision #001 — Project Framing
**Date:** 2026-05-27  
**Decided By:** Initial setup agent + user  
**Decision:** This is a POC (Proof of Concept), not a production system  
**Rationale:** 7-day timeline, 2-person team, dataset not finalized  
**Impact:** Scope all work to "minimum viable demo" level  
**Reversible:** No  

---

## Decision #002 — Dual-Track Architecture
**Date:** 2026-05-27  
**Decided By:** Initial setup agent  
**Decision:** Use dual-track approach — Core Layer (baseline, guaranteed) + Advanced Layer (innovation, optional)  
**Rationale:** Core Layer scores Feasibility (30 pts), Advanced Layer scores Novelty (15 pts). Core Layer must work even if Advanced fails.  
**Impact:** Day 1-2 build Core; Day 3 attempt Advanced; Day 4+ build dashboard + proposal  
**Reversible:** Yes, but not recommended  

---

## Decision #003 — Neural ODE Framing
**Date:** 2026-05-27  
**Decided By:** Initial setup agent  
**Decision:** Neural ODE for NMR alignment must be framed as "novel proposed approach" not "proven technique"  
**Rationale:** research01.md confirms no prior work exists. Overclaiming risks credibility with judges.  
**Impact:** All proposal text and agent outputs must use hedge language  
**Reversible:** No (this is a factual constraint)  

---

## Decision #004 — Synthetic Data First
**Date:** 2026-05-27  
**Decided By:** Initial setup agent + user  
**Decision:** Build pipeline with synthetic NMR data since real dataset is not yet available  
**Rationale:** Can't wait for dataset — need working POC for Feasibility criterion  
**Impact:** Synthetic data generator must produce realistic spectra for 3 compound classes  
**Reversible:** Yes — swap in real data when available  

---

## Decision #005 — English Primary Language
**Date:** 2026-05-27  
**Decided By:** User  
**Decision:** Work primarily in English to save tokens  
**Rationale:** User preference for efficiency  
**Impact:** All code, comments, agent communication in English. Thai only for final submission if required.  
**Reversible:** Yes  

---

## Decision #006 — Compute Target
**Date:** 2026-05-27  
**Decided By:** User  
**Decision:** RTX 4060 (local) or Google Colab  
**Rationale:** Available hardware  
**Impact:** No heavy compute — baseline models only, Neural ODE must be lightweight  
**Reversible:** Yes  

---

_Add new decisions below this line._

---

## Decision #007 — High-fidelity Visual POC Web App and PyTorch Core Established
**Date:** 2026-05-27  
**Decided By:** Antigravity (Gemini 3.5 Flash) + User  
**Decision:** Selected Streamlit as the primary interactive showcase framework and developed an ASCII-safe CLI script.  
**Rationale:** Streamlit allows for rich, ultra-fast Plotly overlays demonstrating Latent ODE alignment and EBM physics suppression. The ASCII runner ensures 100% robust hospital pipeline simulation on Windows without code page encoding failures.  
**Impact:** A stunning space-dark themed diagnostic hub serves at port 8501, and headless LIS runs output standard clinical JSON diagnostic files.  
**Reversible:** Yes  
