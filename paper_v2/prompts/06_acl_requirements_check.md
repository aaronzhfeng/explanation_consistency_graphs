# ACL Rolling Review Submission Requirements Check

## Task

Review this paper submission against ACL Rolling Review (ARR) requirements. Identify any violations or issues that could lead to desk rejection or reviewer complaints.

**Reference**: https://aclrollingreview.org/cfp

---

## Files Provided

1. **Our paper** (to be checked):
   - `main.pdf` - Compiled paper
   - `main.tex` - LaTeX source

2. **ACL 2023 template** (reference for formatting):
   - `acl2023.pdf` - Template example
   - `acl2023.tex` - Template source

---

## Checklist to Verify

### 1. Formatting Requirements

- [ ] **Page limit**: Long papers ≤8 pages, short papers ≤4 pages (excluding references, limitations, ethics, acknowledgements, appendices)
- [ ] **Template**: Uses official ACL 2023 style file (`\usepackage{ACL2023}`)
- [ ] **Font size**: 11pt (`\documentclass[11pt]{article}`)
- [ ] **Margins and spacing**: Default template settings preserved
- [ ] **Two-column format**: Standard ACL layout
- [ ] **References**: Proper use of `natbib` with `acl_natbib.bst`
- [ ] **PDF output**: `\pdfoutput=1` in first 5 lines

### 2. Anonymity Requirements (Review Mode)

- [ ] **Review option**: `\usepackage[review]{ACL2023}` is used
- [ ] **No author names**: Author field is anonymous
- [ ] **No self-citations revealing identity**: Check for "we previously showed" or "our prior work" patterns
- [ ] **No acknowledgements that reveal identity**: Should be empty or placeholder
- [ ] **No institution names in affiliations**
- [ ] **No links to identifying resources**: GitHub repos, personal websites, etc.
- [ ] **Anonymous supplementary materials**: If any

### 3. Required Sections

- [ ] **Abstract**: Present and within word limits
- [ ] **Limitations section**: MANDATORY - papers without this will be desk-rejected
- [ ] **Ethics statement**: Present (recommended)
- [ ] **References**: Complete and properly formatted

### 4. Content Requirements

- [ ] **Substantial and original**: Not previously published
- [ ] **Complete**: All results should be final (no placeholders like "TBD")
- [ ] **Figures and tables**: Readable, properly captioned, referenced in text
- [ ] **Equations**: Properly numbered and referenced

### 5. Responsible NLP Checklist Considerations

- [ ] **Limitations clearly stated**
- [ ] **Reproducibility**: Sufficient implementation details
- [ ] **Dataset documentation**: If using/creating datasets
- [ ] **Computational resources**: Reported
- [ ] **Potential biases**: Acknowledged if relevant

### 6. Common Rejection Triggers

Check for these issues that commonly cause desk rejection:
- [ ] Missing limitations section
- [ ] Exceeding page limits
- [ ] Broken anonymity
- [ ] Incomplete or placeholder content
- [ ] Wrong template or formatting
- [ ] Self-plagiarism without disclosure

---

## Output Format

Please structure your response as:

### 1. Compliance Summary
Brief overall assessment: PASS / ISSUES FOUND / CRITICAL VIOLATIONS

### 2. Critical Issues (if any)
Issues that would cause desk rejection

### 3. Formatting Issues
Deviations from template requirements

### 4. Anonymity Issues
Any potential de-anonymization concerns

### 5. Content Completeness
Missing or incomplete sections

### 6. Recommendations
Specific fixes needed before submission

---

## Additional Notes

- Compare our `main.tex` against `acl2023.tex` to verify formatting compliance
- Check that custom packages don't break template formatting
- Verify line numbering is enabled for review (`[review]` option)
- Ensure all cross-references resolve correctly

