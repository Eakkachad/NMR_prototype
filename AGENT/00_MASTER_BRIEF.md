# MASTER BRIEF — BDI KKU NMR Metabolomics POC

> **READ THIS FILE FIRST.** This is the single source of truth for any AI agent joining this project.

---

## What Is This Project?

A **Proof of Concept (POC)** for the **BDI YOUNG INNOVATOR HACKATHON 2026** (Phenome Track).
The goal is to build an **automated ML pipeline** that classifies chemical compounds from 2D NMR spectral signals, designed for **ease of use by medical personnel**.

## Competition Format

| Item | Detail |
|------|--------|
| **Event** | BDI Young Innovator Hackathon 2026 |
| **Track** | Track 1: Phenome |
| **Timeline** | 7 days total |
| **Team Size** | 2 people |
| **Deliverables** | 2-page proposal + POC prototype |
| **Language** | English (primary), Thai acceptable for submission |
| **Compute** | RTX 4060 local or Google Colab |

## Scoring Rubric (100 pts total)

| Criterion | Points | Key Question |
|-----------|--------|--------------|
| Feasibility | 30 | Can you build a Prototype/PoC within 2 months? |
| Problem Match & Dataset | 25 | Does the project match the problem? How is data handled? |
| Sustainable Impact | 20 | Long-term, equitable change? |
| Novelty | 15 | Why is this new and not a copy? |
| Implementation Plan | 10 | Concrete execution plan? |

## The Data

- **Type:** 2D NMR spectra (Samples × Signals matrix, 20,000+ features per sample)
- **Source:** National Phenome Institute
- **Status:** ⚠️ DATASET IS STILL BEING COLLECTED — not yet finalized
- **Sample PDF:** `../nmr-pattern/Domain_1_processed_NMR_spectrum.pdf`
- **Labels:** Plant Extract A, B, C (compound classification targets)

## Technical Architecture (3-Stage Hybrid)

1. **Stage 1 — Feature Extraction:** Transformer/CNN → compress 20,000+ dims to latent space
2. **Stage 2 — Alignment:** Latent Neural ODE → continuous drift correction
3. **Stage 3 — Deconvolution:** Energy-Based Model → physics-constrained classification

**Baseline (guaranteed demo):** PLS-DA + SVM on binned/PQN-normalized spectra
**Advanced (innovation):** Neural ODE + EBM (novel — no prior NMR application exists)

## Key Directories

| Path | Purpose |
|------|---------|
| `AGENT/01_CONTEXT/` | Competition rules, rubric, team info, timeline |
| `AGENT/02_RESEARCH/` | Consolidated literature, tool comparisons, code repos |
| `AGENT/03_ARCHITECTURE/` | Canonical pipeline design (single source of truth) |
| `AGENT/04_DATA/` | Raw PDF, parsed data, processing scripts |
| `AGENT/05_DELIVERABLES/` | Proposal drafts, video storyboard, submission checklist |
| `AGENT/06_AGENT_PROTOCOL/` | Agent onboarding, task board, standards, handoffs |
| `AGENT/07_QA/` | Error log, consistency checks, review feedback |

## Critical Warnings

1. **Neural ODE for NMR is NOVEL** — no published work exists. Frame as "proposed" not "proven"
2. **This is a POC** — allocate 60% effort to proposal, 40% to working demo
3. **Dataset not finalized** — design pipeline to work with synthetic/sample data first
4. **Medical personnel are end users** — UX and explainability matter more than raw accuracy

## Agent Instructions

1. Read `06_AGENT_PROTOCOL/agent_onboarding.md` for detailed working instructions
2. Check `06_AGENT_PROTOCOL/task_board.md` for current task status
3. Log all decisions in `06_AGENT_PROTOCOL/decision_log.md`
4. Follow `06_AGENT_PROTOCOL/coding_standards.md` for any code
5. Follow `06_AGENT_PROTOCOL/writing_standards.md` for any proposal text
