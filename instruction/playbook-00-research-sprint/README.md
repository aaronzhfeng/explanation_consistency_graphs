<div align="center">

# 🏃 Speedrun Paper

**A template for AI-assisted research paper production**

*From idea to accepted paper in ~38 hours of focused work*

</div>

---

## What Is This?

This repo documents a methodology for rapidly producing research papers using AI agents (LLMs) as collaborative researchers. It's based on the [SOKRATES](https://github.com/aaronzhfeng/sokrates-oak) project, which went from idea to **accepted AAAI workshop paper** in ~5 weeks with AI doing most of the work.

## The Core Insight

> **AI agents can handle the entire research pipeline** — from literature review to implementation to paper writing — when given clear direction and iterative feedback.

The human role shifts from "doing research" to "directing research":
- Provide high-level goals
- Supply compute resources
- Make key decisions at branching points
- Review and approve outputs

---

## The Speedrun Phases

| Phase | Hours | Human Input | AI Agent Output |
|-------|-------|-------------|-----------------|
| **0. Ideation** | 1-2h | Topic + venue | Research design, novelty framing |
| **1. Design** | 2-4h | Approval | Full project plan, architecture |
| **2. Implementation** | 6-10h | "Build it" | All source code, tests |
| **3. Training/Experiments** | 8-15h | Compute + "Run it" | Results, debugging, optimization |
| **4. Analysis** | 2-4h | — | Tables, figures, insights |
| **5. Writing** | 8-12h | Revisions | Full paper draft |
| **6. Submission** | 2-4h | "Submit" | Formatting, checks, final polish |

**Total: 30-50 hours of focused work** (calendar time depends on GPU wait, sleep, etc.)

---

## Repository Structure

```
speedrun-paper/
├── README.md                    # This file
├── templates/
│   ├── 00_ideation.md          # Prompt template for ideation phase
│   ├── 01_project_plan.md      # Project plan template
│   ├── 02_technical_spec.md    # Technical specification template
│   ├── 03_session_log.md       # Development log template
│   ├── 04_paper_outline.md     # Paper structure template
│   ├── 05_venue_selection.md   # How to pick the right venue
│   └── 06_prompt_patterns.md   # Effective prompting for research
├── checklists/
│   ├── pre_implementation.md   # Before coding
│   ├── pre_training.md         # Before experiments
│   ├── pre_submission.md       # Before submitting
│   └── debugging.md            # When things break
├── examples/
│   └── sokrates/               # The SOKRATES case study
│       ├── timeline.md         # Hour-by-hour breakdown
│       ├── prompts.md          # Key prompts used
│       ├── lessons.md          # What we learned
│       ├── failures.md         # What didn't work
│       ├── costs.md            # Resource & cost breakdown
│       ├── decision_tree.md    # When human intervenes
│       ├── before_after.md     # AI draft vs final versions
│       ├── tables_figures.md   # LaTeX iterations & best practices
│       └── assets/             # Actual LaTeX source files
│           ├── sokrates_single.pdf
│           ├── figures/*.tex   # TikZ diagrams
│           └── tables/*.tex    # Result tables
└── workflows/
    ├── cursor_rules.md         # Cursor IDE configuration
    └── git_workflow.md         # Version control patterns
```

---

## Quick Start

### 1. Fork this repo

```bash
git clone https://github.com/YOUR_USERNAME/speedrun-paper.git
cd speedrun-paper
```

### 2. Start with ideation

Open `templates/00_ideation.md` and fill in:
- Target venue
- Rough topic area
- Available resources (compute, time)

### 3. Prompt the AI agent

```
I want to submit a paper to [VENUE] on [TOPIC].
I have [X weeks] and access to [HARDWARE].
Design a research project that is:
- Novel enough for acceptance
- Implementable in the timeframe
- Within my compute budget

Use the structure in templates/01_project_plan.md
```

### 4. Iterate through phases

Follow the checklists, update the session log, and let the AI handle the heavy lifting.

---

## Key Principles

### 1. **Documentation is Everything**

The AI can only help if it has context. Maintain:
- Session logs (what was tried, what worked)
- Technical specs (architecture decisions)
- Paper drafts (even rough ones)

### 2. **Verify Early, Verify Often**

Don't wait until the end to test. After each phase:
- Test the code
- Generate sample outputs
- Check metrics make sense

### 3. **Principled Trade-offs**

When time runs out, make documented trade-offs:
- Reduce dataset size (but keep it statistically significant)
- Skip ablations (can add post-acceptance)
- Simplify experiments (focus on core contribution)

### 4. **The AI Does the Work, You Make Decisions**

At each branch point, the AI proposes options. You choose:
- Which baseline to prioritize
- Whether to pursue a failed experiment
- When to stop iterating

---

## The SOKRATES Case Study

### Timeline (Focused Hours)

| Phase | Hours | What Happened |
|-------|-------|---------------|
| **Ideation & Design** | 4h | Research plan, technical spec, OaK framing |
| **Implementation** | 8h | All 50+ source files generated |
| **Training & Debug** | 12h | SFT + 3 DPO + find/fix 4 bugs |
| **Paper Writing** | 10h | First draft + 4 revision cycles |
| **Submission** | 4h | Formatting, validation, upload |
| **Total** | **38h** | (~2 weeks calendar time) |

### What AI Did

- Generated 50+ Python source files (8h)
- Found and fixed 4 critical bugs (4h)
- Wrote 738-line LaTeX paper (10h)
- Created all figures (TikZ)
- Validated 29 references

### What Human Did

- Provided initial direction ("OaK for AAAI")
- Ran training commands
- Made key decisions at branch points
- Clicked "Submit"

### Result

**Accepted to AAAI-26 Bridge Workshop** 🎉

---

## Recommended Tools

| Tool | Purpose |
|------|---------|
| **Cursor IDE** | AI-integrated development environment |
| **Claude/GPT-4** | Research design, writing |
| **vLLM** | Fast inference for experiments |
| **Weights & Biases** | Experiment tracking |
| **Overleaf** | Collaborative LaTeX (optional) |
| **HuggingFace** | Model/dataset hosting |

---

## Common Failure Modes

| Problem | Solution |
|---------|----------|
| AI generates plausible but wrong code | Test frequently, inspect outputs |
| Training loss good but model broken | Verify data loading, test generation |
| Paper claims don't match results | Re-check numbers before submission |
| Running out of time | Make documented trade-offs |
| Scope creep | Stick to core contribution |

---

## FAQ

### Q: Isn't this just "AI writes my paper"?

**A:** No. The human provides:
- Research direction and taste
- Resource allocation decisions
- Quality control and verification
- Domain expertise for judgment calls

The AI handles execution, but the human shapes the research.

### Q: Will reviewers detect AI-written papers?

**A:** The goal isn't to hide AI involvement. The goal is to produce *good research faster*. The ideas, experiments, and contributions are real — the AI just accelerates the process.

### Q: What if the AI makes mistakes?

**A:** It will. The methodology includes verification at each step. Bugs get caught because you test early and often.

### Q: Can this work for any field?

**A:** Best suited for:
- ML/AI research (where AI agents understand the domain)
- Projects with clear metrics and benchmarks
- Implementation-heavy work

Less suited for:
- Theoretical work requiring novel proofs
- Empirical work requiring human subjects
- Fields with specialized domain knowledge

---

## Contributing

This is a living document. If you use this methodology:
1. Open an issue with your experience
2. Submit PRs with improvements
3. Add your project to the examples folder

---

## Citation

If this helps your research:

```bibtex
@misc{speedrun-paper,
  author = {Feng, Zhaoxiang},
  title = {Speedrun Paper: AI-Assisted Research Production},
  year = {2025},
  url = {https://github.com/aaronzhfeng/speedrun-paper}
}
```

---

## License

MIT License — use freely, attribute if you can.

