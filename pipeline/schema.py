"""The single source of truth for the screening protocol.

Both the OpenAI and the HuggingFace backends screen against exactly the same
criteria, JSON schema, and prompt, so their outputs are directly comparable.
This consolidates prompt_resolve.md, prompt_url.md, and the original notebook.
"""
from __future__ import annotations

# Human-readable eligibility criteria (kept verbatim in the prompt).
ELIGIBILITY = [
    "Written in English.",
    "DDI, ADE, ADR, or medication-related harm (MRH) detection is an outcome.",
    "Applies an AI-based method: NLP, ML, deep learning, transformers, or LLMs "
    "(purely manual / rule-only string matching without a learned model is borderline "
    "-> mark 'Unclear' and flag for human review).",
    "Uses a real-world clinical data source generated during clinical care: EHRs, "
    "clinical notes, discharge summaries, medication orders, hospital records, "
    "claims linked to EHR, or n2c2-style clinical records.",
    "Original research, not a review, editorial, commentary, protocol, or meta-analysis.",
    "Peer-reviewed full text, not a preprint, conference/supplement abstract, poster, "
    "erratum, or correction.",
]

# Enum labels used both in the schema and when flattening to spreadsheet columns.
CRITERIA_NAMES = [
    "Written in English",
    "DDI/ADE/ADR/MRH detection outcome",
    "AI-based method",
    "Real-world clinical data source",
    "Original research",
    "Peer-reviewed full text",
]

# JSON schema handed to the model (OpenAI strict json_schema; also embedded as
# text for local models that cannot enforce a schema).
SCREENING_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["Include", "Exclude", "Exclude, but save for snowballing", "Unclear"],
        },
        "exclusion_reasons": {
            "type": "string",
            "description": "One line, failed criteria only, separated by '; '. 'None' if included.",
        },
        "primary_exclusion_reason": {
            "type": "string",
            "description": "Single main reason. 'N/A' if included.",
        },
        "criteria": {
            "type": "object",
            "properties": {
                "written_in_english": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "detection_outcome": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "ai_based_method": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "real_world_clinical_data_source": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "original_research": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
                "peer_reviewed_full_text": {"type": "string", "enum": ["Yes", "No", "Unclear"]},
            },
            "required": [
                "written_in_english", "detection_outcome", "ai_based_method",
                "real_world_clinical_data_source", "original_research",
                "peer_reviewed_full_text",
            ],
            "additionalProperties": False,
        },
        "failed_criteria": {
            "type": "array",
            "description": "Only failed/unclear criteria. Empty if included.",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string", "enum": CRITERIA_NAMES},
                    "evidence_quote": {"type": "string"},
                    "interpretation": {"type": "string"},
                },
                "required": ["criterion", "evidence_quote", "interpretation"],
                "additionalProperties": False,
            },
        },
        "eligibility_evidence": {
            "type": "string",
            "description": "If included: one concise paragraph with short quotes confirming "
                           "each criterion. If excluded: ''.",
        },
        "screening_note": {"type": "string"},
        "needs_human_review": {"type": "boolean"},
    },
    "required": [
        "decision", "exclusion_reasons", "primary_exclusion_reason", "criteria",
        "failed_criteria", "eligibility_evidence", "screening_note", "needs_human_review",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a careful systematic-review screening assistant. "
    "You screen one paper at a time against fixed eligibility criteria and return "
    "ONLY a JSON object matching the requested schema. No prose outside the JSON."
)

_RULES = """
Screening rules:
- Clinical topic does NOT imply a clinical data source. PubMed/MEDLINE abstracts,
  biomedical literature, DDIExtraction-2013, DrugBank, FAERS, product/drug labels,
  SPCs, package inserts, regulatory labels, social media, blogs, web forums, and
  scientific-literature tables are NOT real-world clinical care data sources unless
  the study also uses patient-level clinical records for the detection task.
- Reviews / meta-analyses fail 'Original research' -> decision "Exclude, but save for snowballing".
- Conference/supplement abstracts, posters, errata and corrections fail 'Peer-reviewed full text'.
- `exclusion_reasons`: only failed criteria, one line, '; '-separated. 'None' if included.
- `failed_criteria`: only failed/unclear criteria, never passing ones.
- Evidence quotes must be SHORT (<= 25 words) and copied verbatim from the source
  material below. If no exact quote exists, use
  'No direct quote available in provided source material.' and set needs_human_review=true.
- If the source material is insufficient, use decision "Unclear" and needs_human_review=true.
- Be transparent when the full text is inaccessible.
""".strip()


def build_prompt(src: dict) -> str:
    """Assemble the user prompt from a source-material dict (see step3)."""
    criteria = "\n".join(f"{i}. {c}" for i, c in enumerate(ELIGIBILITY, 1))
    return f"""Screen this paper for a systematic review on AI-based DDI/ADE/ADR/MRH
detection using real-world clinical data.

Eligibility criteria:
{criteria}

{_RULES}

Return JSON with exactly these keys: decision, exclusion_reasons,
primary_exclusion_reason, criteria{{written_in_english, detection_outcome,
ai_based_method, real_world_clinical_data_source, original_research,
peer_reviewed_full_text}}, failed_criteria[{{criterion, evidence_quote,
interpretation}}], eligibility_evidence, screening_note, needs_human_review.

Source material:
TITLE: {src.get('title','')}
YEAR: {src.get('year','')}
AUTHORS: {src.get('authors','')}
DOI: {src.get('doi','')}
RESOLVED_URL: {src.get('resolved_url','')}
ABSTRACT: {src.get('abstract','')}
FULL_TEXT_EXCERPT: {src.get('fulltext','')}
""".strip()
