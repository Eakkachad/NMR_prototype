# Error Log — Known Issues

> Track all discovered errors and their resolution status.

---

## ERR-001: EB-gMCR Publication Year Inconsistency
**Found in:** preplane.md (2024), research01.md (2025), research02.md (2024?)  
**Severity:** Low (cosmetic)  
**Impact:** Citation accuracy in proposal  
**Resolution:** ⬜ Verify actual publication year and standardize  

## ERR-002: "2D NMR" Terminology Misuse
**Found in:** information.md, README.md  
**Severity:** Medium  
**Impact:** Could confuse judges or suggest wrong pipeline (2D HSQC vs 1D ¹H in matrix)  
**Resolution:** ⬜ Use "2D data matrix of 1D NMR spectra" or "Sample × Signal matrix" consistently  

## ERR-003: Neural ODE Overclaiming
**Found in:** information.md §5, preplane.md §4  
**Severity:** HIGH  
**Impact:** Credibility risk with judges  
**Resolution:** ✅ Decided to frame as "novel proposed approach" (Decision #003)  

## ERR-004: Duplicate Architecture Descriptions
**Found in:** information.md §5 and preplane.md §4  
**Severity:** Medium  
**Impact:** Confusion for agents — which version is canonical?  
**Resolution:** ✅ Created canonical version at AGENT/03_ARCHITECTURE/pipeline_overview.md  

## ERR-005: Missing Team Members
**Found in:** information.md §178  
**Severity:** Medium  
**Impact:** Cannot write Team Synergy section (10 pts)  
**Resolution:** ⬜ User must provide team member names and skills  

## ERR-006: No Sample Size (N) Specified
**Found in:** All documents  
**Severity:** Medium  
**Impact:** Cannot determine appropriate ML model complexity or cross-validation strategy  
**Resolution:** ⬜ Will be resolved when dataset arrives. Using N=30 synthetic (10 per class) for now  

## ERR-007: NMRTrans Future Date
**Found in:** preplane.md L153 — "Yang et al., 2026"  
**Severity:** Low  
**Impact:** May be a preprint or accepted paper with future date  
**Resolution:** ⬜ Verify — if preprint, cite as "(Yang et al., 2026, preprint)"  

## ERR-008: Filename Typo
**Found in:** `preplane.md` — likely intended as `preplan.md`  
**Severity:** Low  
**Impact:** None (file is readable)  
**Resolution:** ⬜ Optional — rename if desired  

---

_Add new errors below this line._
