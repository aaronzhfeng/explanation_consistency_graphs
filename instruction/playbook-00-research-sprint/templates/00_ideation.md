# Phase 0: Ideation Template

> Fill this out before prompting the AI agent

---

## 1. Target Venue

| Field | Value |
|-------|-------|
| **Venue name** | |
| **Deadline** | |
| **Page limit** | |
| **Review type** | Single-blind / Double-blind / Open |
| **Acceptance rate** | ~X% (estimate) |
| **URL** | |

---

## 2. Topic Direction

### What area are you interested in?
<!-- e.g., "neuro-symbolic AI", "LLM reasoning", "reinforcement learning" -->


### What's the general problem you want to address?
<!-- One sentence -->


### Do you have a specific method/approach in mind?
<!-- Optional — AI can help generate this -->


---

## 3. Available Resources

| Resource | Details |
|----------|---------|
| **Time** | X weeks until deadline |
| **Compute** | GPUs available (type, count) |
| **Data** | Datasets you have access to |
| **Existing code** | Any codebase you can build on |
| **Collaborators** | Who's working on this |

---

## 4. Constraints

### Must-haves
<!-- e.g., "Must run on 1 GPU", "Must use public datasets" -->
- 
- 

### Nice-to-haves
<!-- e.g., "Would be great to compare against X", "Ideally open-source" -->
- 
- 

### Dealbreakers
<!-- e.g., "Can't require API access", "No human studies" -->
- 
- 

---

## 5. Prior Work (if known)

### Papers you want to build on
<!-- List 2-3 key papers -->
1. 
2. 
3. 

### Methods you want to compare against
<!-- Baselines -->
1. 
2. 

---

## 6. Success Criteria

### Minimum viable paper
<!-- What's the bare minimum for submission? -->


### Ideal outcome
<!-- What would make this a great paper? -->


---

## Prompt for AI Agent

Once filled, use this prompt:

```
I want to submit a paper to [VENUE] by [DEADLINE].

Topic area: [TOPIC]
Available resources: [RESOURCES]
Time: [X] weeks

Constraints:
- [CONSTRAINT 1]
- [CONSTRAINT 2]

Design a research project that:
1. Has clear novelty (what's new vs prior work)
2. Is implementable in the timeframe
3. Has measurable success criteria
4. Fits the venue's scope

Output:
1. Title (with clever acronym if applicable)
2. One-paragraph abstract
3. 3-5 key contributions/novelties
4. Proposed experiments and baselines
5. Risk assessment and backup plans
6. Rough timeline
```

