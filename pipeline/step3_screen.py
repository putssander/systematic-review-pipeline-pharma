"""Step 3 - screen each paper with a chosen backend and write include/reason + detail.

Reads the Step 0-2 artifacts, assembles grounded source material (sheet abstract +
fetched full-text excerpt + resolved URL), calls the backend, and writes one record
per paper to data/step3_<backend>.jsonl. Runs are resumable (--only-missing).

    python -m pipeline.step3_screen --backend gpt-5.5
    python -m pipeline.step3_screen --backend qwen3-8b
    python -m pipeline.step3_screen --backend gpt-5.5 --limit 2   # cheap smoke test
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from tqdm.auto import tqdm

from . import config, llm_client
from .schema import SYSTEM_PROMPT, build_prompt
from .util import clean_text, index_by, read_jsonl, write_jsonl

# decision -> boolean include column (matches the sheet's True/False convention)
INCLUDE_MAP = {
    "Include": True,
    "Exclude": False,
    "Exclude, but save for snowballing": False,
    "Unclear": None,
}


def build_source(paper: dict, resolved: dict, text: str) -> dict:
    return {
        "title": paper.get("title") or paper.get("short_title", ""),
        "year": paper.get("year", ""),
        "authors": paper.get("authors", ""),
        "doi": resolved.get("doi") or paper.get("doi", ""),
        "resolved_url": resolved.get("resolved_url") or paper.get("link", ""),
        "abstract": clean_text(paper.get("abstract"))[: config.MAX_ABSTRACT_CHARS],
        "fulltext": clean_text(text)[: config.MAX_FETCH_CHARS],
    }


def flatten(rid: str, result: dict, src: dict, backend: str) -> dict:
    crit = result.get("criteria", {}) or {}
    failed = result.get("failed_criteria", []) or []
    decision = result.get("decision", "Unclear")
    include = INCLUDE_MAP.get(decision)
    # The 'reason' cell mirrors your sheet: exclusion reasons, or eligibility evidence.
    if include:
        reason = result.get("eligibility_evidence") or "Included."
    else:
        reason = result.get("exclusion_reasons") or result.get("primary_exclusion_reason") or ""
    return {
        "record_id": rid,
        "backend": backend,
        "decision": decision,
        "include": include,
        "reason": reason,
        "primary_exclusion_reason": result.get("primary_exclusion_reason", ""),
        "c_english": crit.get("written_in_english", "Unclear"),
        "c_outcome": crit.get("detection_outcome", "Unclear"),
        "c_ai_method": crit.get("ai_based_method", "Unclear"),
        "c_clinical_data": crit.get("real_world_clinical_data_source", "Unclear"),
        "c_original": crit.get("original_research", "Unclear"),
        "c_peer_reviewed": crit.get("peer_reviewed_full_text", "Unclear"),
        "failed_criteria": "; ".join(
            f"{f.get('criterion','')}: \"{f.get('evidence_quote','')}\" ({f.get('interpretation','')})"
            for f in failed) or "None",
        "eligibility_evidence": result.get("eligibility_evidence", ""),
        "screening_note": result.get("screening_note", ""),
        "needs_human_review": bool(result.get("needs_human_review", False)),
        "source_url": src.get("resolved_url", ""),
        "model": config.BACKENDS[backend]["model"],
        "screened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error": "",
    }


def error_record(rid: str, backend: str, exc: Exception, src: dict) -> dict:
    return {
        "record_id": rid, "backend": backend, "decision": "Unclear", "include": None,
        "reason": "API/processing error", "primary_exclusion_reason": "API/processing error",
        "c_english": "Unclear", "c_outcome": "Unclear", "c_ai_method": "Unclear",
        "c_clinical_data": "Unclear", "c_original": "Unclear", "c_peer_reviewed": "Unclear",
        "failed_criteria": "None", "eligibility_evidence": "",
        "screening_note": "Human review required: backend call failed.",
        "needs_human_review": True, "source_url": src.get("resolved_url", ""),
        "model": config.BACKENDS[backend]["model"],
        "screened_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error": f"{type(exc).__name__}: {exc}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen papers with one backend.")
    ap.add_argument("--backend", required=True, choices=list(config.BACKENDS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--only-missing", action="store_true",
                    help="resume: keep existing results, only screen new record_ids")
    args = ap.parse_args()

    papers = read_jsonl(config.PAPERS_JSONL)
    if not papers:
        raise SystemExit("Run step0_load first.")
    resolved = index_by(read_jsonl(config.RESOLVED_JSONL), "record_id")
    text_idx = index_by(read_jsonl(config.TEXT_INDEX_JSONL), "record_id")

    out_path = config.step3_jsonl(args.backend)
    existing = index_by(read_jsonl(out_path), "record_id") if args.only_missing else {}

    if args.limit:
        papers = papers[: args.limit]

    results = dict(existing)  # preserve prior rows
    paper_order = [str(p["record_id"]) for p in read_jsonl(config.PAPERS_JSONL)]
    todo = [p for p in papers if not (args.only_missing and str(p["record_id"]) in existing)]
    print(f"Backend={args.backend}  model={config.BACKENDS[args.backend]['model']}  "
          f"to screen: {len(todo)}  (kept: {len(existing)})", flush=True)

    bar = tqdm(todo, desc=f"Step 3: screening ({args.backend})", unit="paper")
    for paper in bar:
        rid = str(paper["record_id"])
        bar.set_postfix_str(f"rid={rid}")
        r = resolved.get(rid, {})
        text = ""
        tf = (text_idx.get(rid) or {}).get("text_file")
        if tf and Path(tf).exists():
            text = Path(tf).read_text(encoding="utf-8")
        src = build_source(paper, r, text)
        prompt = build_prompt(src)
        try:
            result = llm_client.screen(args.backend, SYSTEM_PROMPT, prompt)
            rec = flatten(rid, result, src, args.backend)
            tag = f"{rec['decision']:<32} include={rec['include']}"
        except Exception as e:
            rec = error_record(rid, args.backend, e, src)
            tag = f"ERROR {e}"
        results[rid] = rec
        # write after every row so long runs are crash-safe (ordered like the sheet)
        write_jsonl(out_path, [results[k] for k in paper_order if k in results])
        tqdm.write(f"rid={rid}  {tag}")
        time.sleep(args.sleep)

    print(f"\nWrote {len(results)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
