"""Systematic-review screening pipeline.

Steps:
  0  step0_load       xlsx  -> data/step0_papers.jsonl
  1  step1_resolve    DOI   -> canonical URL + metadata validation
  2  step2_fetch_text URL   -> raw article text (HTML; PDF text-layer; OCR fallback)
  3  step3_screen     text  -> screening decision (per backend)
     merge_results    backends -> combined xlsx with agreement flags
"""
__all__ = ["config", "schema", "util", "llm_client"]
