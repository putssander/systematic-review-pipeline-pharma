"""Step 1 - resolve each DOI to the correct canonical article URL, and flag
metadata that disagrees with the spreadsheet.

This is the step that catches the root cause of several bad decisions in
'25 papers SP.xlsx': the `link` column contains wrong/duplicated URLs (e.g. three
different papers all pointing at the same ScienceDirect article), so the model was
sometimes screening the wrong paper.

For each paper we call Crossref (free, no key) to get:
  - the authoritative title, publication year, work type
  - the publisher landing-page URL (resource.primary.URL) and the doi.org URL
We then compare against the sheet and emit a `resolved_url` plus mismatch flags.

    python -m pipeline.step1_resolve
    python -m pipeline.step1_resolve --limit 5      # quick test
"""
from __future__ import annotations

import argparse
import time
import urllib.parse
import urllib.request
import json

from tqdm.auto import tqdm

from . import config
from .util import (clean_text, normalize_doi, read_jsonl, title_similarity,
                   write_jsonl)

TITLE_MATCH_THRESHOLD = 0.6
# Crossref work types that are NOT original research articles.
NON_ARTICLE_TYPES = {
    "journal-issue", "book", "book-chapter", "proceedings", "dataset",
    "reference-book", "report", "standard", "peer-review", "grant",
}
# Hosts that legitimately point at the right article even if they differ from the
# publisher landing page, so a host difference alone is not a red flag.
AGGREGATOR_HOSTS = {
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov",
    "doi.org", "dx.doi.org", "europepmc.org", "semanticscholar.org",
}


def _crossref(doi: str) -> dict | None:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=config.REQUEST_TIMEOUT) as r:
            return json.load(r)["message"]
    except Exception:
        return None


def _years_from(msg: dict) -> list[int]:
    """All plausible publication years (epub and print often differ by one year)."""
    years = []
    for key in ("published", "published-print", "published-online", "issued", "created"):
        parts = (msg.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            years.append(int(parts[0][0]))
    # de-duplicate, keep order
    seen, out = set(), []
    for y in years:
        if y not in seen:
            seen.add(y)
            out.append(y)
    return out


def resolve_one(rec: dict) -> dict:
    doi = normalize_doi(rec.get("doi"))
    link = clean_text(rec.get("link"))
    sheet_title = rec.get("title") or rec.get("short_title") or ""
    sheet_year = clean_text(rec.get("year"))

    out = {
        "record_id": rec.get("record_id"),
        "row": rec.get("row"),
        "doi": doi,
        "sheet_link": link,
        "resolved_url": link,          # fallback = whatever was in the sheet
        "resolved_via": "sheet_link" if link else "none",
        "crossref_title": "",
        "crossref_year": None,
        "crossref_type": "",
        "crossref_years": [],
        "title_similarity": None,
        "title_match": None,
        "year_match": None,
        "sheet_link_host": _host(link),
        "resolved_host": "",
        "link_is_aggregator": _host(link) in AGGREGATOR_HOSTS if link else None,
        "doi_valid": bool(doi),
        "flags": [],
        "notes": "",
    }

    if not doi:
        if not link:
            out["flags"].append("no_doi_no_link")
        else:
            out["notes"] = "No DOI; using sheet link as-is (verify manually)."
            out["flags"].append("no_doi")
        return out

    msg = _crossref(doi)
    if not msg:
        out["flags"].append("crossref_lookup_failed")
        out["notes"] = "DOI did not resolve via Crossref; verify DOI."
        out["doi_valid"] = False
        return out

    cr_title = clean_text((msg.get("title") or [""])[0])
    cr_years = _years_from(msg)
    cr_type = clean_text(msg.get("type"))
    primary = (msg.get("resource") or {}).get("primary", {}).get("URL")
    doi_url = msg.get("URL") or f"https://doi.org/{doi}"

    out["crossref_title"] = cr_title
    out["crossref_years"] = cr_years
    out["crossref_year"] = cr_years[0] if cr_years else None
    out["crossref_type"] = cr_type
    out["resolved_url"] = primary or doi_url
    out["resolved_via"] = "crossref_primary" if primary else "doi.org"
    out["resolved_host"] = _host(primary or doi_url)

    sim = title_similarity(sheet_title, cr_title)
    out["title_similarity"] = sim
    out["title_match"] = sim >= TITLE_MATCH_THRESHOLD

    if sheet_year and cr_years:
        # tolerate epub-vs-print by matching against ANY reported year (+-1)
        candidates = set(cr_years) | {y + 1 for y in cr_years} | {y - 1 for y in cr_years}
        out["year_match"] = int(sheet_year) in candidates if sheet_year.isdigit() else None

    # ---- flags (only actionable ones) ------------------------------------
    if out["title_match"] is False:
        out["flags"].append("title_mismatch")
    if out["year_match"] is False:
        out["flags"].append("year_mismatch")
    if cr_type in NON_ARTICLE_TYPES:
        out["flags"].append(f"non_article_type:{cr_type}")

    if out["flags"]:
        out["notes"] = ("Auto-resolved via Crossref; sheet link replaced. "
                        "Review flags before trusting the old decision.")
    else:
        out["notes"] = "Resolved and consistent with sheet metadata."
    return out


def _host(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _flag_duplicate_links(resolved: list[dict]) -> None:
    """Flag the same sheet link reused across different papers (a clear data error:
    e.g. three different DOIs all carrying the same ScienceDirect URL)."""
    from collections import defaultdict

    groups: dict[str, list] = defaultdict(list)
    for r in resolved:
        link = clean_text(r.get("sheet_link")).lower()
        if link:
            groups[link].append(r)
    for link, rows in groups.items():
        if len(rows) > 1:
            others = [str(x["record_id"]) for x in rows]
            for r in rows:
                r["flags"].append("duplicate_link_shared_with:" + ",".join(
                    o for o in others if o != str(r["record_id"])))
                r["notes"] = ("Sheet link is shared by multiple papers -> almost "
                              "certainly wrong. Use resolved_url instead. " + r.get("notes", ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve DOIs -> canonical URLs (Crossref).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.3, help="politeness delay between calls")
    args = ap.parse_args()

    papers = read_jsonl(config.PAPERS_JSONL)
    if not papers:
        raise SystemExit("No papers found. Run: python -m pipeline.step0_load")
    if args.limit:
        papers = papers[: args.limit]

    resolved = []
    bar = tqdm(papers, desc="Step 1: resolving DOIs", unit="paper")
    for rec in bar:
        r = resolve_one(rec)
        resolved.append(r)
        bar.set_postfix_str(f"rid={r['record_id']}")
        flag = ",".join(r["flags"]) or "ok"
        tqdm.write(f"rid={r['record_id']}  {r['resolved_via']:16s}  {flag}")
        time.sleep(args.sleep)

    _flag_duplicate_links(resolved)
    write_jsonl(config.RESOLVED_JSONL, resolved)
    flagged = [r for r in resolved if r["flags"]]
    print(f"\nWrote {len(resolved)} -> {config.RESOLVED_JSONL}")
    print(f"Flagged for review: {len(flagged)}")
    for r in flagged:
        print(f"  rid={r['record_id']}: {', '.join(r['flags'])}")


if __name__ == "__main__":
    main()
