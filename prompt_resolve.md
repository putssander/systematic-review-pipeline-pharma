You are screening papers for a systematic review.

I will provide a paper as:
DOI:
Publication year:
Title:

Your job:
1. Resolve the DOI to the official article page or PubMed page.
2. Confirm that the DOI matches the provided title.
3. Confirm that the publication year matches the provided year.
4. Confirm the final resolved article URL.
5. Screen the paper using the criteria below.
6. Use short direct quotes as evidence, with citations/links.
7. Output in the exact format below.

Eligibility criteria:
- Written in English.
- DDI, ADE, ADR, or medication-related harm detection is an outcome.
- Applies an AI-based method, including NLP, ML, deep learning, transformers, or LLMs.
- Uses a real-world clinical data source, such as EHRs, clinical notes, discharge summaries, hospital records, claims linked to EHR, or n2c2-style clinical records.
- Original research, not a review, editorial, commentary, protocol, or meta-analysis.
- Peer-reviewed full text, not a preprint, conference abstract, supplement abstract, poster, erratum, correction, or abstract-only item.

Important screening rules:
- Clinical topic does not automatically mean clinical data source.
- Biomedical literature, PubMed abstracts, DrugBank, FAERS, social media, web forums, product labels, SPCs, package inserts, and scientific literature tables are not real-world clinical care data sources unless the study also uses patient-level clinical records.
- Reviews should be excluded but marked “save for snowballing.”
- If the paper is included, do not create a failed-criteria table. Just provide concise eligibility evidence.
- If the paper is excluded, list only failed criteria in the table.
- Keep quotes short and quote only what is needed as evidence.
- Be transparent if the full text is inaccessible or if evidence is unclear.

Required output format:

**Paper:** [Authors/year], *[confirmed title]*

**DOI confirmation:** [Confirmed / Not confirmed / Unclear] — DOI [DOI] resolves to [journal/source] and matches/does not match the supplied title.

**Publication year confirmation:** [Confirmed / Not confirmed / Unclear] — supplied year [year] matches/does not match the article publication year [confirmed year].

**Resolved URL:** [official resolved URL or PubMed URL]

**Decision:** **Include / Exclude / Exclude, but save for snowballing / Unclear**

**Exclusion reasons:** [Only list failed criteria in one line. If included: None.]

**Primary exclusion reason:** [Main reason, or N/A if included.]

If excluded, use this table:

| Failed criterion | Evidence quote | Interpretation |
|---|---|---|
| [criterion] | “[short quote]” | [brief explanation] |

If included, use this instead:

**Eligibility evidence:** [One concise paragraph with short quotes confirming outcome, AI method, real-world clinical data source, original research, peer-reviewed full text, and publication year.]

Now screen this paper:

DOI: [paste DOI here]
Publication year: [paste year here]
Title: [paste title here]