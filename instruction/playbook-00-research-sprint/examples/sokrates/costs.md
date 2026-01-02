# SOKRATES: Cost & Resource Breakdown

> What did this actually cost?

---

## Compute Costs

### GPU Usage

| Phase | GPUs | Hours | GPU-Hours |
|-------|------|-------|-----------|
| SFT Training | 2× B200 | 0.8h | 1.6 |
| DPO Iter 1 | 6× B200 | 0.5h | 3.0 |
| DPO Iter 2 | 6× B200 | 0.5h | 3.0 |
| DPO Iter 3 | 6× B200 | 0.5h | 3.0 |
| Trace Generation | 6× B200 | 4.0h | 24.0 |
| Evaluation | 1× B200 | 1.0h | 1.0 |
| **Total** | | | **35.6 GPU-hours** |

### Cost Estimate (Cloud Pricing)

| GPU Type | $/hour | Our GPU-hours | Cost |
|----------|--------|---------------|------|
| B200 (if rented) | ~$8-12 | 35.6 | ~$350-430 |
| A100 80GB equivalent | ~$3-4 | 50 (scaled) | ~$150-200 |
| H100 equivalent | ~$4-5 | 40 (scaled) | ~$160-200 |

**Our actual cost: $0** (university compute cluster)

### What You'd Need Minimally

| Budget | Setup | Feasibility |
|--------|-------|-------------|
| **$0** | Colab Pro ($10/mo) + smaller model | Possible but slow |
| **$50-100** | Lambda/RunPod spot instances | Comfortable |
| **$200-400** | Full cloud training | Fast, no constraints |
| **$0** | University/company cluster | Ideal if available |

---

## AI API Costs

### If Using API-Based LLM (Claude/GPT-4)

| Task | Tokens (est.) | Cost @ $15/M |
|------|---------------|--------------|
| Research design | 50K | $0.75 |
| Code generation | 200K | $3.00 |
| Debugging sessions | 100K | $1.50 |
| Paper writing | 150K | $2.25 |
| Revisions | 100K | $1.50 |
| **Total** | **600K** | **~$9** |

### Cursor IDE

| Plan | Cost | What You Get |
|------|------|--------------|
| Pro | $20/mo | Unlimited fast requests |
| Free | $0 | Limited requests |

**Our cost: $20** (Cursor Pro monthly)

---

## Total Project Cost

| Category | Cost |
|----------|------|
| Compute (GPU) | $0 (cluster) or ~$200-400 (cloud) |
| AI Assistant | $20 (Cursor Pro) |
| HuggingFace | $0 (free tier) |
| OpenReview | $0 |
| **Total** | **$20 - $420** |

---

## Time Investment

### Focused Work Hours

| Actor | Hours | Hourly Value* | "Cost" |
|-------|-------|---------------|--------|
| Human | 10h | $50-100/h | $500-1000 |
| AI Agent | 28h | $0 (sunk) | $0 |

*Opportunity cost of researcher time

### Total "Investment"

| Component | Value |
|-----------|-------|
| Compute | $0-400 |
| Tools | $20 |
| Human time | $500-1000 |
| **Total** | **$520-1420** |

Compare to: Traditional research paper often takes 200+ hours → $10K+ in researcher time alone.

---

## What We'd Do Differently for Cost Optimization

### If Compute-Constrained

1. **Use smaller model** (3B instead of 8B) — 4× less compute
2. **Reduce DPO iterations** (2 instead of 3) — 33% less
3. **Smaller dataset** (5K instead of 14K) — 65% less
4. **Single GPU training** — Slower but works

### If Time-Constrained

1. **Skip ablations** — Add post-acceptance
2. **Use existing baselines** — Don't reimplement
3. **Parallel everything** — More GPUs = faster

### If Budget-Constrained

1. **Google Colab Pro** ($10/mo) — For small experiments
2. **Spot instances** — 70% cheaper than on-demand
3. **Off-peak hours** — Some providers cheaper at night

---

## Resource Requirements Summary

### Minimum Viable Setup

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 24GB (RTX 4090) | 80GB (A100/H100) |
| GPU Count | 1 | 4-8 |
| RAM | 32GB | 128GB |
| Storage | 50GB SSD | 200GB NVMe |
| AI Assistant | Free tier | Cursor Pro |

### Our Setup

| Resource | Specification |
|----------|---------------|
| GPUs | 6× NVIDIA B200 (183GB each) |
| RAM | 256GB |
| Storage | 1TB NVMe |
| AI Assistant | Cursor Pro (Claude) |

---

## ROI Analysis

| Metric | Traditional | AI-Assisted | Savings |
|--------|-------------|-------------|---------|
| Human hours | 160h | 38h | 122h |
| Calendar time | 3-4 months | 2 weeks | 2.5-3.5 months |
| Compute cost | Same | Same | $0 |
| AI tool cost | $0 | $20 | -$20 |
| **Net savings** | — | — | **~$6,000 in researcher time** |

*Assuming $50/h researcher cost: 122h × $50 = $6,100 saved*

