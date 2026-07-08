"""Step 2 - retrieve the raw article text for each paper.

Strategy per paper:
  1. Fetch the resolved URL from Step 1 (falls back to the sheet link).
  2. HTML  -> strip boilerplate, keep visible text.
  3. PDF   -> pymupdf text layer; if scanned, GLM-OCR fallback (see ocr.py).
  4. On any failure, record it. The abstract from the sheet is still available to
     Step 3, so a blocked publisher page never stops screening.

Text is written to data/text/<record_id>.txt and indexed in step2_text_index.jsonl.

    python -m pipeline.step2_fetch_text
    python -m pipeline.step2_fetch_text --limit 5 --no-ocr
"""
from __future__ import annotations

import argparse
import time

import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from . import config, ocr
from .util import clean_text, index_by, read_jsonl, write_jsonl


def _html_to_text(html: str, max_chars: int) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    return clean_text(soup.get_text(" "))[:max_chars]


def fetch_text(url: str, allow_ocr: bool = True) -> dict:
    """Fetch one URL -> {text, method, http_status, source_url}."""
    url = clean_text(url)
    result = {"source_url": url, "text": "", "method": "none", "http_status": None}
    if not url.startswith(("http://", "https://")):
        result["method"] = "no_url"
        return result
    try:
        r = requests.get(url, headers={"User-Agent": config.USER_AGENT},
                         timeout=config.REQUEST_TIMEOUT, allow_redirects=True)
        result["http_status"] = r.status_code
        result["source_url"] = r.url
        if not r.ok:
            result["method"] = "http_error"
            result["text"] = f"[HTTP {r.status_code} when fetching {url}]"
            return result

        ctype = r.headers.get("content-type", "").lower()
        is_pdf = "pdf" in ctype or r.url.lower().endswith(".pdf")
        if is_pdf:
            text, method = ocr.pdf_to_text(r.content, allow_ocr=allow_ocr)
            result["text"] = text[: config.MAX_FETCH_CHARS]
            result["method"] = method
        else:
            result["text"] = _html_to_text(r.text, config.MAX_FETCH_CHARS)
            result["method"] = "html"
    except Exception as e:
        result["method"] = "fetch_error"
        result["text"] = f"[Fetch failed: {type(e).__name__}: {e}]"
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch raw article text per paper.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--no-ocr", action="store_true", help="disable GLM-OCR fallback")
    ap.add_argument("--only-missing", action="store_true",
                    help="skip papers that already have a non-empty text file")
    args = ap.parse_args()

    resolved = read_jsonl(config.RESOLVED_JSONL)
    if not resolved:
        raise SystemExit("Run step1_resolve first.")
    papers = index_by(read_jsonl(config.PAPERS_JSONL), "record_id")
    if args.limit:
        resolved = resolved[: args.limit]

    index = []
    bar = tqdm(resolved, desc="Step 2: fetching text", unit="paper")
    for r in bar:
        rid = str(r["record_id"])
        bar.set_postfix_str(f"rid={rid}")
        txt_path = config.TEXT_DIR / f"{rid}.txt"
        if args.only_missing and txt_path.exists() and txt_path.stat().st_size > 200:
            index.append({"record_id": rid, "text_file": str(txt_path), "skipped": True})
            tqdm.write(f"rid={rid}  skipped (exists)")
            continue

        # Prefer the Step-1 resolved URL; fall back to the sheet link.
        url = r.get("resolved_url") or (papers.get(rid, {}) or {}).get("link", "")
        res = fetch_text(url, allow_ocr=not args.no_ocr)
        txt_path.write_text(res["text"], encoding="utf-8")

        entry = {
            "record_id": rid,
            "requested_url": url,
            "source_url": res["source_url"],
            "method": res["method"],
            "http_status": res["http_status"],
            "char_count": len(res["text"]),
            "text_file": str(txt_path),
        }
        index.append(entry)
        tqdm.write(f"rid={rid}  {res['method']:12s}  {entry['char_count']:>6} chars  {url[:60]}")
        time.sleep(args.sleep)

    write_jsonl(config.TEXT_INDEX_JSONL, index)
    from collections import Counter
    methods = Counter(e.get("method", "skipped") for e in index)
    print(f"\nWrote {len(index)} -> {config.TEXT_INDEX_JSONL}")
    print("Methods:", dict(methods))


if __name__ == "__main__":
    main()
