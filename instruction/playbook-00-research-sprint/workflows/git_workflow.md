# Git Workflow for Speedrun Papers

> Version control patterns for rapid development

---

## Branch Strategy

For a speedrun, keep it simple:

```
main ← everything goes here
```

Why? Time is precious. Don't waste it on merge conflicts.

---

## Commit Patterns

### Commit Often
```bash
# After each working session
git add .
git commit -m "Session [DATE]: [SUMMARY]"
```

### Meaningful Messages
```bash
# Good
git commit -m "Fix data loader to use training_text field"
git commit -m "Add multi-GPU support for trace generation"
git commit -m "Complete paper draft v1"

# Bad
git commit -m "fix"
git commit -m "updates"
git commit -m "wip"
```

### Phase Markers
```bash
# Mark major milestones
git tag -a "sft-complete" -m "SFT training complete"
git tag -a "dpo-complete" -m "DPO training complete"
git tag -a "paper-v1" -m "First complete paper draft"
git tag -a "submitted" -m "Submitted to venue"
```

---

## What to Track

### Always Track
- Source code (`src/`, `scripts/`)
- Configuration files (`configs/`)
- Documentation (`docs/`, `paper/`)
- Requirements (`requirements.txt`, `pyproject.toml`)

### Never Track (add to .gitignore)
```gitignore
# Model outputs
outputs/
checkpoints/
*.pt
*.bin
*.safetensors

# Data (use HuggingFace or download scripts)
data/raw/
data/processed/

# Build artifacts
__pycache__/
*.pyc
build/
dist/
*.egg-info/

# IDE
.vscode/
.idea/
.cursor/

# Logs
*.log
wandb/

# Large files
*.pdf  # except paper, use LFS or separate
```

---

## Large Files Strategy

### Option 1: Git LFS
```bash
git lfs install
git lfs track "*.pdf"
git lfs track "data/*.jsonl"
```

### Option 2: External Storage
- Store models on HuggingFace
- Store data on HuggingFace Datasets
- Include download scripts

### Option 3: Separate Repos
- Code repo (public)
- Data repo (private/gated)
- Model repo (HuggingFace)

---

## Pre-Submission Backup

Before submitting, create a complete snapshot:

```bash
# Tag the submission state
git tag -a "v1-submitted" -m "Submitted to [VENUE] on [DATE]"
git push origin v1-submitted

# Create backup archive
git archive --format=zip HEAD > backup-[DATE].zip
```

---

## Recovery Patterns

### Undo Last Commit (keep changes)
```bash
git reset --soft HEAD~1
```

### Recover Deleted File
```bash
git checkout HEAD -- path/to/file.py
```

### See What Changed
```bash
git diff HEAD~5 -- src/  # Last 5 commits in src/
git log --oneline -10     # Last 10 commits
```

### Bisect to Find Bug
```bash
git bisect start
git bisect bad              # Current is broken
git bisect good abc123      # This commit was good
# Git will help find the breaking commit
```

---

## Collaboration (if applicable)

For multi-person speedruns:

```bash
# Pull before working
git pull origin main

# Push often
git push origin main

# If conflict
git stash
git pull origin main
git stash pop
# Resolve conflicts, then commit
```

---

## Typical Speedrun Git Log

```
* a1b2c3d (HEAD -> main, tag: submitted) Final submission prep
* d4e5f6g Fix TL;DR character limit
* g7h8i9j Paper revision v3 - polish
* j0k1l2m Paper revision v2 - reviewer objections
* m3n4o5p Paper revision v1 - claims vs evidence
* p6q7r8s Complete paper draft v1
* s9t0u1v DPO iteration 3 complete (91.8% step validity)
* v2w3x4y DPO iteration 2 complete (83.5% step validity)
* y5z6a7b DPO iteration 1 complete (44.7% step validity)
* b8c9d0e (tag: sft-complete) SFT training complete
* e1f2g3h Fix data loader bug - use training_text
* h4i5j6k Fix optionizer - natural language thoughts
* k7l8m9n Debug session - identify 4 bugs
* n0o1p2q Initial implementation complete
* q3r4s5t Project setup and design docs
* t6u7v8w Initial commit
```

