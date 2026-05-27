# Scoring Rubric — Strategic Analysis

## Official Rubric (100 points)

### 1. Feasibility — 30 Points (HIGHEST WEIGHT)

**Judge's Question:** "Can you build a Prototype/PoC within 2 months? Explain your team's technical capacity."

**Strategy to Maximize:**
- ✅ Show a WORKING demo, even if simple (PLS-DA + SVM baseline)
- ✅ List specific Python libraries with version numbers
- ✅ Reference open-source repos we can actually use (NMRQNet, DEEP Picker, SPA-STOCSY)
- ✅ Show a clear timeline with milestones
- ✅ Demonstrate team technical skills concretely
- ❌ AVOID: Claiming Neural ODE will be production-ready (it won't be)
- ❌ AVOID: Over-promising on advanced features

**Target Score:** 25-28 / 30

---

### 2. Problem Match & Dataset — 25 Points

**Judge's Question:** "Does your project perfectly match the problem statement? How will you source or handle the required Dataset?"

**Strategy to Maximize:**
- ✅ Map every competition objective to our pipeline stage
- ✅ Demonstrate deep understanding of NMR data structure
- ✅ Acknowledge the PDF parsing challenge and show a solution
- ✅ Explain how 20,000+ features are handled (binning, PQN)
- ✅ Show understanding of peak shifting and overlap
- ✅ Explicitly address the "dataset still being collected" situation with synthetic data strategy

**Target Score:** 22-25 / 25

---

### 3. Sustainable Impact — 20 Points

**Judge's Question:** "How will your project create long-term, equitable change?"

**Strategy to Maximize:**
- ✅ Focus on **medical personnel usability** — this is our differentiator
- ✅ Concrete metric: "Reduce NMR interpretation from 4 hours to 5 minutes"
- ✅ Concrete metric: "Enable non-specialist clinics to perform metabolomics screening"
- ✅ Scale argument: "Deploy as web dashboard accessible by any hospital"
- ✅ Cost argument: "Eliminates need for specialized spectroscopist at each site"
- ✅ Clinical use case: Early metabolic disease screening (diabetes, inborn errors of metabolism)

**Target Score:** 16-18 / 20

---

### 4. Novelty — 15 Points

**Judge's Question:** "Why is this genuinely new and not just a copy?"

**Strategy to Maximize:**
- ✅ Neural ODE for NMR alignment is GENUINELY novel (no prior work exists)
- ✅ EBM for ghost peak suppression is novel in NMR context
- ✅ The combination of Transformer + Latent ODE + EBM is unprecedented
- ✅ Frame as: "We identified a research gap and propose a novel bridge"
- ⚠️ CRITICAL: Must clearly label as "proposed novel approach" not "proven technique"

**Target Score:** 12-14 / 15

---

### 5. Implementation Plan — 10 Points

**Judge's Question:** "Does your team have a concrete plan to execute within the hackathon?"

**Strategy to Maximize:**
- ✅ Day-by-day sprint plan with clear deliverables
- ✅ Assign specific tasks to each team member
- ✅ Risk mitigation: "If advanced model fails, baseline guarantees a demo"
- ✅ Show dependency graph: what blocks what

**Target Score:** 8-9 / 10

---

## Total Target: 83-94 / 100

## Priority Ranking for Time Allocation

1. 🔴 **Feasibility (30 pts)** — Build the working baseline demo FIRST
2. 🟠 **Problem Match (25 pts)** — Write the data understanding section with depth
3. 🟡 **Sustainable Impact (20 pts)** — Concrete numbers and clinical use case
4. 🔵 **Novelty (15 pts)** — Explain Neural ODE innovation carefully
5. 🟢 **Implementation Plan (10 pts)** — Fill in the team sprint table
