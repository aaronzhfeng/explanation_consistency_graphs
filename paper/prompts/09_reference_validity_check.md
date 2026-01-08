# Reference Validity Check

## Task

Verify that all citations in this paper are valid, accurate, and correctly attributed. The goal is to **detect fake/hallucinated citations, incorrect attributions, and bibliographic errors**.

---

## Files Provided

- `main.tex` - LaTeX source with citations
- `ecg.bib` - BibTeX bibliography file

---

## Checklist to Verify

### 1. Citation Existence

For each cited work, verify:
- [ ] **The paper exists**: Is this a real published work?
- [ ] **Authors are correct**: Do the listed authors match the actual paper?
- [ ] **Year is correct**: Is the publication year accurate?
- [ ] **Venue is correct**: Is it published where claimed (conference/journal)?
- [ ] **Title is accurate**: Does the title match the actual paper?

### 2. Citation Key Consistency

- [ ] All `\citep{}` and `\citet{}` keys in `main.tex` have corresponding entries in `ecg.bib`
- [ ] No unused bib entries (optional but good practice)
- [ ] Citation keys are consistent (no typos like `smith2020` vs `smith2020a`)

### 3. Attribution Accuracy

For claims like "X et al. showed Y", verify:
- [ ] The cited paper actually makes that claim
- [ ] The claim is not misrepresented or oversimplified
- [ ] The citation is to the original source (not secondary)

### 4. BibTeX Entry Quality

For each bib entry, check:
- [ ] **Required fields present**: author, title, year, venue
- [ ] **No placeholder text**: No "TODO", "???", or template text
- [ ] **Consistent formatting**: Names, titles formatted consistently
- [ ] **DOI/URL present**: Where available
- [ ] **Correct entry type**: @article, @inproceedings, @misc used appropriately

### 5. Common Red Flags

Watch for these signs of potentially fake citations:
- [ ] ArXiv papers cited as peer-reviewed publications
- [ ] Papers from future years
- [ ] Authors with unusual name patterns
- [ ] Venues that don't exist
- [ ] Very round page numbers (e.g., pages 1-10, 100-110)
- [ ] Missing venue/publisher information
- [ ] Suspiciously perfect fit to claims made

### 6. Self-Citation Check

- [ ] Self-citations properly anonymized for review
- [ ] No excessive self-citation
- [ ] Self-citations are relevant, not gratuitous

---

## Specific Citations to Verify

Please verify these key citations that support major claims:

| Citation Key | Claim in Paper | Verify |
|--------------|----------------|--------|
| `northcutt2021pervasive` | "3-6% of labels are incorrect" | Real stat from paper? |
| `northcutt2021confident` | Confident learning method | Method described correctly? |
| `pleiss2020identifying` | AUM method | Method described correctly? |
| `poliak2018hypothesis` | "NLI solved with hypothesis only" | Accurate claim? |
| `gururangan2018annotation` | Annotation artifacts | Accurate attribution? |
| `bahri2020deep` | kNN noisy label detection | Method exists? |
| `kim2023neural` | Relation graph formulation | Paper exists? |
| `disalvo2025wann` | WANN reliability weighting | Paper exists? Year correct? |
| `lertvittayakumjorn2021explanation` | Explanation debugging survey | Paper exists? |
| `geng2023grammar` | Grammar-constrained decoding | Paper exists? |

---

## Output Format

### 1. Verification Summary
Overall assessment of citation quality

### 2. Confirmed Valid Citations
List of citations verified as accurate

### 3. Potentially Problematic Citations
Citations that could not be verified or appear incorrect:
- Citation key
- Issue found
- Suggested fix (if known)

### 4. Missing Information
Bib entries with incomplete information

### 5. Attribution Issues
Cases where claims don't match cited sources

### 6. Formatting Issues
BibTeX formatting problems

### 7. Recommendations
Specific fixes needed

---

## Notes

- ArXiv preprints are acceptable but should be marked as such
- Some 2024/2025 papers may be recent and harder to verify
- When in doubt, flag for manual verification
- Focus especially on citations supporting key claims

