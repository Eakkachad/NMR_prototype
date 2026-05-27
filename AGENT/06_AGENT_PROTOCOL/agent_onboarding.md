# Agent Onboarding — Read This First

> **You are an AI agent joining the BDI KKU NMR Metabolomics competition project.**
> Follow this document exactly. It ensures continuity across agent sessions and model changes.

## Step 1: Understand the Project (2 minutes)

1. Read `../00_MASTER_BRIEF.md` — this gives you the full context in one page
2. Check `task_board.md` — see what's been done and what needs doing
3. Check `decision_log.md` — understand past decisions and their rationale

## Step 2: Understand Your Task

The user will tell you what to work on. Map their request to the correct area:

| Request Type | Go To |
|-------------|-------|
| "Write code" / "Build pipeline" | `../03_ARCHITECTURE/` for specs, `../04_DATA/scripts/` for code |
| "Write proposal text" | `../05_DELIVERABLES/proposal/` for drafts, `writing_standards.md` for style |
| "Research something" | `../02_RESEARCH/` for existing knowledge, add new findings there |
| "Fix a bug" | Check `../07_QA/error_log.md` for known issues |
| "What's the status?" | `task_board.md` has the current state |

## Step 3: Follow Standards

- **Code:** Follow `coding_standards.md`
- **Writing:** Follow `writing_standards.md`
- **Decisions:** Log in `decision_log.md`
- **Handoffs:** Use `handoff_template.md` when your session ends

## Step 4: Update Status

When you complete work:
1. Update `task_board.md` — move tasks from TODO → DOING → DONE
2. Log any decisions in `decision_log.md`
3. If your session is ending, fill in `handoff_template.md`

## Critical Rules

1. **NEVER** claim Neural ODE for NMR is a proven technique — it's novel
2. **ALWAYS** prioritize the baseline (PLS-DA + SVM) over the advanced model
3. **This is a POC** — don't over-engineer. Simple and working > complex and broken
4. **Medical personnel are end users** — every output must be interpretable
5. **Write in English** to save tokens (except Thai submission materials)
6. **Dataset is not finalized** — use synthetic data for development
7. **2-page proposal maximum** — every word must earn its space
