# brainstorm-03-llm-explainability

A curated literature bank for **Explanation-Consistency Graphs (ECG)** — targeting ACL 2026 Theme Track.

## Target Venue

- **Conference:** ACL 2026
- **Theme:** Explainability of NLP Models
- **Deadline:** January 5, 2026 (ARR submission)
- **Conference Dates:** July 2-7, 2026, San Diego, CA

## Selected Project: ECG

**Explanation-Consistency Graphs for Fast Training-Data Debugging**

Use LLM-generated explanations to build an instance graph of explanation agreement/contradiction, and flag nodes that are explanation-inconsistent as likely label errors or artifact-driven examples.

- **Proposal:** [`proposals/02_ecg.md`](proposals/02_ecg.md)
- **GitHub:** https://github.com/aaronzhfeng/explanation_consistency_graphs
- **Compute:** ~20 H100 hours

## Status

| Phase | Status |
|-------|--------|
| Proposals | ✅ Complete (VARIF + ECG) |
| Direction selected | ✅ ECG |
| Literature search | ✅ 103 papers curated |
| Implementation | 🔲 Ready to start |

---

## Folder Structure

```
brainstorm-03-llm-explainability/
├── README.md                  ← this file
├── topic.yml                  ← metadata
├── topic_brief.md             ← research direction
├── proposals/                 ← research proposals (VARIF, ECG)
│   ├── 01_varif.md           ← evaluation protocol proposal
│   └── 02_ecg.md             ← data debugging proposal ★
├── prompts/                   ← brainstorm + literature prompts
├── raw_llm_outputs/           ← LLM search results (6 files)
├── inbox.md                   ← raw paper dump (116 entries)
├── literature.md              ← curated index (103 papers, 8 categories)
├── literature/                ← PDFs
└── literature_readmes/        ← paper summaries
```

---

## Related

- [`brainstorm-00-core`](../brainstorm-00-core/) — pipeline and prompts
- [`brainstorm-01-agentic-ai`](../brainstorm-01-agentic-ai/) — agents, benchmarks
- [`brainstorm-02-protein-dl`](../brainstorm-02-protein-dl/) — geometric trust, protein modeling

