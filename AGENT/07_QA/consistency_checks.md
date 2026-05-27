# Consistency Checks — Cross-Reference Validation

> Run these checks before final submission to ensure all documents agree.

## Check 1: Architecture Consistency
- [ ] `00_MASTER_BRIEF.md` architecture matches `03_ARCHITECTURE/pipeline_overview.md`
- [ ] Proposal draft architecture matches `03_ARCHITECTURE/pipeline_overview.md`
- [ ] Dashboard code implements what the architecture describes
- [ ] No references to old architecture in `information.md` or `preplane.md` leak into proposal

## Check 2: Scoring Alignment
- [ ] Proposal covers ALL 5 criteria from `01_CONTEXT/scoring_rubric.md`
- [ ] Self-assessment scores are justified with specific evidence
- [ ] Self-assessment total ≤ 100
- [ ] Each score is realistic (not all perfect scores — looks dishonest)

## Check 3: Literature Accuracy
- [ ] All cited papers exist and years are correct
- [ ] No paper is cited as "proven for NMR" unless it actually was applied to NMR
- [ ] EB-gMCR year is consistent across all documents
- [ ] NMRTrans citation is properly noted as 2026 (preprint or accepted)

## Check 4: Technical Claims vs POC Capability
- [ ] If proposal says "classifies with >85% accuracy" → POC must show this on synthetic data
- [ ] If proposal says "reduces time to 5 minutes" → POC must demonstrate speed
- [ ] If proposal mentions Neural ODE → it's labeled as "proposed/novel"
- [ ] No claim in the proposal that the POC cannot back up

## Check 5: Data Consistency
- [ ] `01_CONTEXT/dataset_spec.md` agrees with what the data scripts produce
- [ ] Number of features matches across all references (20,000+)
- [ ] Number of compound classes matches (3: A, B, C)
- [ ] Binned feature count matches preprocessing output (~500)

## Check 6: Proposal Format
- [ ] Exactly 2 pages (no more, no less)
- [ ] Page 1 has: Header, Problem, Solution, Impact
- [ ] Page 2 has: Tech Stack, Workflow Diagram, Innovation & Feasibility, Self-Assessment
- [ ] All sections from `proposalform.md` are addressed

## Check 7: Agent Protocol Integrity
- [ ] `task_board.md` is up to date
- [ ] `decision_log.md` captures all major decisions
- [ ] No contradictions between agent protocol files

---

## Validation Results

| Check | Status | Notes | Date |
|-------|--------|-------|------|
| 1. Architecture | ⬜ Pending | | |
| 2. Scoring | ⬜ Pending | | |
| 3. Literature | ⬜ Pending | | |
| 4. Claims vs POC | ⬜ Pending | | |
| 5. Data | ⬜ Pending | | |
| 6. Format | ⬜ Pending | | |
| 7. Protocol | ⬜ Pending | | |
