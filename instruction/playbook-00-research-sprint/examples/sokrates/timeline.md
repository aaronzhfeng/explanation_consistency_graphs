# SOKRATES Timeline

> Actual timeline from idea to acceptance — measured in **focused work hours**

---

## Overview

| Phase | Hours | Calendar Time |
|-------|-------|---------------|
| Ideation & Design | 4h | Day 1 |
| Implementation | 8h | Days 2-3 |
| Training & Debugging | 12h | Days 4-7 (+ GPU wait time) |
| Paper Writing | 10h | Days 8-12 |
| Submission & Polish | 4h | Days 13-14 |
| **Total Focused Work** | **~38h** | ~2 weeks calendar |

> **Note**: Calendar time includes waiting for training runs, sleep, and non-project time. Actual concentrated human+AI work was ~38 hours.

---

## Detailed Timeline (Hours)

### Phase 0: Ideation & Design (4 hours)

| Hour | Activity | Output |
|------|----------|--------|
| 0:00-0:30 | Human: "OaK for AAAI Bridge" | Direction set |
| 0:30-1:30 | AI: Research design | `01_title_and_abstract.md` |
| 1:30-2:30 | AI: Full project plan | `02_paper_plan.md`, `03_project_plan.md` |
| 2:30-3:30 | AI: OaK connection analysis | `04_oak_connection_notes.md` |
| 3:30-4:00 | AI: Technical specification | `05_technical_spec.md` |

**Phase 0 Total: 4 hours**

---

### Phase 1: Implementation (8 hours)

| Hour | Activity | Output |
|------|----------|--------|
| 4:00-5:00 | AI: Core data structures | `src/data/structures.py` |
| 5:00-6:00 | AI: Data loading + optionizer | `src/data/optionizer.py`, `loader.py` |
| 6:00-7:00 | AI: Solver implementations | `src/solvers/*.py` |
| 7:00-8:00 | AI: Training modules | `src/training/*.py` |
| 8:00-9:00 | AI: Inference + evaluation | `src/inference/*.py`, `src/evaluation/*.py` |
| 9:00-10:00 | AI: Training scripts | `scripts/train_sft.py`, `scripts/train_dpo.py` |
| 10:00-11:00 | AI: Config files + tests | `configs/*.yaml`, `tests/*.py` |
| 11:00-12:00 | Human: Environment setup + data download | Ready to train |

**Phase 1 Total: 8 hours**

---

### Phase 2: Training & Debugging (12 hours)

| Hour | Activity | Output |
|------|----------|--------|
| 12:00-12:30 | Human: Launch SFT training | Training started |
| — | *GPU wait: ~50 min* | — |
| 12:30-13:00 | Human: Test generation → garbage output | Bug discovered |
| 13:00-14:00 | AI: Root cause analysis | Found 4 bugs |
| 14:00-15:00 | AI: Fix data loader | `TrainingTextWrapper` |
| 15:00-16:00 | AI: Fix optionizer | Natural language thoughts |
| 16:00-16:30 | AI: Document debugging | `14_debugging_session.md` |
| 16:30-17:00 | Human: Re-run SFT | Fixed training |
| — | *GPU wait: ~50 min* | — |
| 17:00-18:00 | Human: DPO iter 1 + test | 44.7% step validity |
| 18:00-19:00 | AI: Multi-GPU optimization | vLLM parallelization |
| 19:00-20:00 | Human: DPO iter 2 + iter 3 | 91.8% step validity |
| — | *GPU wait: ~2 hours* | — |
| 20:00-21:00 | Human: FOLIO transfer eval | 45.3% → 53.2% |
| 21:00-22:00 | AI: Ablation design + run | Ablation results |
| 22:00-23:00 | Human: Upload to HuggingFace | Models + data public |
| 23:00-24:00 | AI: Results analysis | Tables, insights |

**Phase 2 Total: 12 hours** (+ ~4h GPU wait time)

---

### Phase 3: Paper Writing (10 hours)

| Hour | Activity | Output |
|------|----------|--------|
| 24:00-26:00 | AI: First draft | `paper/sokrates.tex` v1 |
| 26:00-27:00 | AI: Generate figures | TikZ architecture, traces |
| 27:00-28:00 | AI: Generate tables | Main results, ablations |
| 28:00-29:00 | Human: Review + feedback | Revision notes |
| 29:00-30:00 | AI: Revision 1 (structure) | v2 |
| 30:00-31:00 | AI: Revision 2 (claims) | v3 |
| 31:00-32:00 | AI: Revision 3 (polish) | v4 |
| 32:00-33:00 | AI: Add citations | 6 new references |
| 33:00-34:00 | AI: Fix "33× → 44×" error | Corrected claims |

**Phase 3 Total: 10 hours**

---

### Phase 4: Submission & Polish (4 hours)

| Hour | Activity | Output |
|------|----------|--------|
| 34:00-34:30 | AI: Create single-file LaTeX | `sokrates_single.tex` |
| 34:30-35:00 | AI: Validate all references | 29 refs verified |
| 35:00-35:30 | AI: Fix figure placement | Figure 1 on page 2 |
| 35:30-36:00 | AI: Pre-submission checklist | All flags cleared |
| 36:00-36:30 | Human: Create OpenReview account | Portal ready |
| 36:30-37:00 | AI: Write abstract + TL;DR | `A_submit.md` |
| 37:00-37:30 | AI: Fix TL;DR char limit | Under 250 chars |
| 37:30-38:00 | Human: Upload + submit | **Submitted** |

**Phase 4 Total: 4 hours**

---

## Time Breakdown by Actor

| Actor | Hours | % |
|-------|-------|---|
| AI Agent (design, code, writing) | 28h | 74% |
| Human (decisions, running, reviewing) | 10h | 26% |
| **Total Focused Work** | **38h** | 100% |

### Additional Time (not counted)
| Category | Time |
|----------|------|
| GPU training/inference wait | ~6h |
| Sleep/breaks between sessions | ~12 days |
| HuggingFace upload wait | ~1h |

---

## Hourly Breakdown by Category

| Category | Hours | Notes |
|----------|-------|-------|
| **Design & Planning** | 4h | Project plan, technical spec |
| **Implementation** | 8h | All source code |
| **Debugging** | 4h | Finding and fixing 4 bugs |
| **Experiments** | 4h | Running, monitoring, analyzing |
| **Paper Writing** | 10h | Draft + 4 revisions |
| **Submission Prep** | 4h | Formatting, validation |
| **Human Decision Points** | 4h | Reviews, approvals, key choices |

---

## Speed Comparison

| Task | Traditional | AI-Assisted | Speedup |
|------|-------------|-------------|---------|
| Literature review | 20h | 2h | 10× |
| Implementation | 60h | 8h | 7.5× |
| Debugging | 20h | 4h | 5× |
| Paper first draft | 40h | 3h | 13× |
| Paper revisions | 20h | 7h | 3× |
| **Total** | **160h** | **38h** | **4.2×** |

---

## Key Observations

### What Took Longer Than Expected
| Task | Expected | Actual | Why |
|------|----------|--------|-----|
| Debugging | 1h | 4h | Data pipeline bugs compound |
| Multi-GPU setup | 0.5h | 2h | DDP conflicts, work splitting |
| Reference validation | 0.5h | 1.5h | Many year/venue errors |

### What Was Faster Than Expected
| Task | Expected | Actual | Why |
|------|----------|--------|-----|
| Implementation | 16h | 8h | AI generated clean code |
| First paper draft | 8h | 2h | AI writes fast |
| Figure creation | 4h | 1h | TikZ from description |

---

## Reproducibility Estimate

For a similar project (neuro-symbolic AI, clear benchmarks):

| Experience Level | Estimated Hours |
|------------------|-----------------|
| Expert + AI | 30-40h |
| Intermediate + AI | 50-60h |
| Beginner + AI | 80-100h |
| Expert, no AI | 120-160h |

The AI compresses most tasks by 3-10×, with the biggest gains in:
1. **Boilerplate code** (10×)
2. **First drafts** (10×)
3. **Documentation** (8×)
4. **Debugging with context** (5×)
