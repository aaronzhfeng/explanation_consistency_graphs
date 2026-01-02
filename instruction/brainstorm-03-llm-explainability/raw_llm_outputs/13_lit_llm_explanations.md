# Literature Search: LLM-Generated Explanations

**Prompt:** `10_lit_llm_explanations.md`  
**Date:**  
**Source:** GPT-5.2 Pro

---

<!-- Paste results below -->


## 1) Structured output generation (JSON/XML/schema adherence)

1. **Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning** — Saibo Geng, Martin Josifoski, Maxime Peyrard, Robert West (2023, **EMNLP 2023**).
   **arXiv:** 2305.13971
   **Key contribution:** Uses formal grammars (incl. *input-dependent grammars*) to constrain decoding so outputs are *guaranteed* to match a target structure; demonstrates strong gains on structured tasks (IE, entity disambiguation, parsing) without finetuning—useful for enforcing parseable “explanations-as-JSON.”
   **Code:** GitHub repo available.

2. **Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation** — Luca Beurer-Kellner, Marc Fischer, Martin Vechev (2024, arXiv).
   **arXiv:** 2403.06988
   **Key contribution:** Proposes **DOMINO**, a subword-aligned constrained decoding approach designed to reduce overhead and avoid accuracy loss from token–constraint misalignment—relevant if you want strict JSON/schema constraints without brittle prompt-only formatting.
   **Code:** Not clearly indicated in the arXiv record.

3. **FOFO: A Benchmark to Evaluate LLMs’ Format-Following Capability** — Congying Xia, Chen Xing, Jiangshu Du, Xinyi Yang, Yihao Feng, Ran Xu, Wenpeng Yin, Caiming Xiong (2024, **ACL 2024**).
   **arXiv:** 2402.18667
   **Key contribution:** Introduces a benchmark specifically for complex format-following; evaluates open and closed models and shows open models lag substantially on strict adherence—useful for selecting/finetuning an open model to reliably emit JSON explanations.
   **Code/data:** Released (GitHub).

4. **SLOT: Structuring the Output of Large Language Models** — Darren Yow-Bang Wang, Zhengyuan Shen, Soumya Smruti Mishra, Zhichao Xu, Yifei Teng, Haibo Ding (2025, **EMNLP 2025 Industry track** / arXiv).
   **arXiv:** 2505.04016
   **Key contribution:** Instead of relying only on constrained decoding, uses a **lightweight post-processing model** to transform “nearly structured” LLM outputs into schema-valid structured outputs; reports strong schema accuracy and discusses using open models (e.g., Mistral / Llama variants).
   **Code:** Not clearly indicated in the arXiv record.

5. **Generating Structured Outputs from Language Models: Benchmark and Studies** — Saibo Geng, Hudson Cooper, Michał Moskal (2025, arXiv).
   **arXiv:** 2501.10868
   **Key contribution:** Benchmarks constrained decoding approaches around **JSON Schema**, evaluating (i) constraint compliance efficiency, (ii) constraint coverage, and (iii) output quality—useful for choosing a decoding strategy for JSON rationales (and understanding quality trade-offs).
   **Code:** Not clearly indicated in the arXiv record.

---

## 2) LLM-as-judge / LLM-as-annotator (labeling + verification workflows)

6. **Human-LLM Collaborative Annotation Through Effective Verification of LLM Labels** — Xinru Wang, Hannah Kim, Sajjadur Rahman, *et al.* (2024, **CHI 2024**).
   **DOI:** 10.1145/3613904.3641960
   **Key contribution:** A multi-step pipeline where LLMs produce labels **and explanations**, a verifier scores label quality using multiple signals, and humans focus on low-verification items—highly aligned with “use explanations to detect mislabeled training data.”
   **Code:** Not clearly indicated on the venue landing page.

7. **AnnoLLM: Making Large Language Models to Be Better Crowdsourced Annotators** — Xingwei He, Zhenghao Lin, Yeyun Gong, A-Long Jin, Hang Zhang, Chen Lin, Jian Jiao, Siu Ming Yiu, Nan Duan, Weizhu Chen (2023, arXiv).
   **arXiv:** 2303.16854
   **Key contribution:** “Explain-then-annotate”: prompts LLMs to generate explanations for labels and reuses those explanations in few-shot prompts to improve annotation quality—directly relevant if you want structured rationales as part of an annotation/verification loop.
   **Code:** Not clearly indicated in the arXiv record.

8. **Making Large Language Models as Active Annotators** — Rui Zhang, *et al.* (2023, arXiv).
   **arXiv:** 2310.19596
   **Key contribution:** Introduces **LLMaAA**, putting LLM annotators into an active-learning loop to select what to annotate efficiently; relevant as a control baseline for “LLM-generated label verification” pipelines (even if explanations are not the main object).
   **Code:** Not clearly indicated in the arXiv record.

9. **A Survey on LLM-as-a-Judge** — Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, Jian Guo (2024, arXiv).
   **arXiv:** 2411.15594
   **Key contribution:** Systematizes “LLM-as-judge” reliability issues (consistency, bias, scenario adaptation) and proposes evaluation methodologies + a benchmark—useful context if you use an LLM to adjudicate contradictions between (instance, rationale, label) triples.
   **Code:** Not clearly indicated in the arXiv record.

---

## 3) Rationale extraction vs generation, explanation verification, and faithfulness

10. **Can Large Language Models Explain Themselves? A Study of LLM-Generated Self-Explanations** — Shiyuan Huang, Siddarth Mamidanna, Shreedhar Jangam, Yilun Zhou, Leilani H. Gilpin (2023, arXiv).
    **arXiv:** 2310.11207
    **Key contribution:** Systematic study of eliciting *feature-attribution-style* self-explanations and evaluating them against faithfulness-style metrics and traditional explanation methods—good grounding for extractive span rationales vs free-form generated rationales.
    **Code:** Not clearly indicated in the arXiv record.

11. **Are self-explanations from Large Language Models faithful?** — Andreas Madsen, Sarath Chandar, Siva Reddy (2024, **Findings of ACL 2024**).
    **arXiv:** 2401.07927
    **Key contribution:** Uses **self-consistency checks** to test whether different self-explanation types (counterfactual, feature attribution, redaction) are faithful; shows faithfulness depends on explanation type/model/task and evaluates open models (e.g., Llama2/Mistral/Falcon noted in the paper).
    **Code:** Not clearly indicated in the venue/arXiv records.

12. **Faithfulness vs. Plausibility: On the (Un)Reliability of Explanations from Large Language Models** — Chirag Agarwal, Sree Harsha Tanneru, Himabindu Lakkaraju (2024, arXiv).
    **arXiv:** 2402.04614
    **Key contribution:** Analyzes the gap between plausible explanations and explanations faithful to the model’s true decision process; helpful framing for why explanation-based mislabeled-data detection needs verification (e.g., NLI checks / perturbation tests) rather than trusting rationales.
    **Code:** Not indicated.

13. **Evaluating the Reliability of Self-Explanations in Large Language Models** — Korbinian Randl, John Pavlopoulos, Aron Henriksson, Tony Lindgren (2024, arXiv).
    **arXiv:** 2407.14487
    **Key contribution:** Evaluates extractive vs counterfactual self-explanations and reports that plausibility can diverge from faithfulness; argues counterfactual explanations can be more verifiable with tailored prompts—useful for building contradiction signals.
    **Code:** Not clearly indicated in the arXiv record.

14. **Evaluating Human Alignment and Model Faithfulness of LLM Rationale** — Sohail Fayyaz, Nassir Navab, Xiangyu Zou, Yue Zhang (2024, arXiv).
    **arXiv:** 2407.00219
    **Key contribution:** Compares human-perceived quality vs model-faithfulness for rationales, highlighting that human alignment does not guarantee faithfulness; directly relevant if you use explanation graphs/NLI to filter misleading rationales.
    **Code:** Not clearly indicated in the arXiv record. 

15. **On Measuring Faithfulness or Self-consistency of Natural Language Explanations** — Lucia Parcalabescu, *et al.* (2024, **ACL 2024**).
    **ACL Anthology:** 2024.acl-long.329
    **Key contribution:** Focuses on the challenge of assessing whether LLM explanations match underlying decision behavior; discusses/uses input perturbation–style testing as a route to faithfulness measurement—fits directly with “verify explanations before using them for data cleaning.”
    **Code:** Not clearly indicated in the ACL record.

---

## 4) Explanation uncertainty and calibration (confidence + rationale)

16. **Quantifying Uncertainty in Natural Language Explanations of Large Language Models** — Sree Harsha Tanneru, Chirag Agarwal, Himabindu Lakkaraju (2024, **AISTATS 2024 (PMLR)**).
    **Key contribution:** Studies uncertainty specifically in *natural-language explanations*; useful if you want to treat rationale confidence as a calibrated signal for contradiction detection or graph edge weighting.
    **Code:** Not clearly indicated in the proceedings record.

17. **SaySelf: Teaching LLMs to Express Confidence with Self-Reflective Rationales** — Tianyang Xu, Shujin Wu, Shizhe Diao, Xiaoze Liu, Xingyao Wang, Yangyi Chen, Jing Gao (2024, **EMNLP 2024**).
    **arXiv:** 2405.20974
    **Key contribution:** Training framework to improve fine-grained confidence calibration and generate *self-reflective rationales* that explain uncertainty (via inconsistency across sampled reasoning chains + SFT/RL); directly relevant to your “predicted label + confidence + rationale” JSON fields.
    **Code:** GitHub repo available.

---

## 5) Prompting/finetuning for more consistent or higher-quality rationales

18. **Towards Consistent Natural-Language Explanations via Explanation-Consistency Finetuning** — Yifan Chen, Guilin Xu, Zixuan Wu, Lei Zhang, Chaojun Xiao, Junfeng Guo, Qun Liu, Xiangpeng Wei (2024, arXiv; also appears in **COLING 2025**).
    **arXiv:** 2401.13986
    **Key contribution:** Proposes explanation-consistency finetuning to make explanations more stable/consistent; relevant if you need deterministic, parseable rationales for downstream graph construction and contradiction checks.
    **Code:** GitHub repo available.

19. **Tailoring Self-Rationalizers with Multi-Reward Distillation** — Sainbayar Ramnath, Preksha Nema, Dan Iter, Yiming Yang, Jason D. Williams, Marcus Collins, Jennifer B. Hamann, Christoforos Tzortzis, Sahin Cem Geyik (2024, **ICLR 2024**).
    **Key contribution:** Distills/optimizes self-rationalizing behavior using multiple reward signals to improve rationale properties; relevant if you want to tune an open model to generate rationales that are more useful for verification (e.g., less hallucinated evidence).
    **Code:** Not clearly indicated in the OpenReview record.

---

## 6) Explanation verification with explicit faithfulness measurement frameworks

20. **Walk the Talk? Measuring the Faithfulness of Large Language Model Explanations** — Katie Matton, Robert Osazuwa Ness, John Guttag, Emre Kıcıman (2025, **ICLR 2025**).
    **arXiv:** 2504.14150
    **Key contribution:** Defines faithfulness in terms of which **concepts** explanations claim are influential vs which truly are, and estimates causal influence via counterfactual concept interventions + Bayesian hierarchical modeling; strong fit for “explanation verification” before using rationales to flag label noise.
    **Code:** Not clearly indicated in the arXiv/ICLR records.
