"""Download full-text PDFs for the 'blocked' papers — run this where you HAVE access.

Colab and most cloud IPs are blocked by paywalled publishers (MDPI, OUP, Wiley,
ASCO, IEEE, Elsevier…), so Step 2 falls back to the sheet abstract for them. Run this
module on a machine with institutional/library access (e.g. your university network):
it reads the worklist of blocked papers, downloads each one's PDF into
``manual_pdfs/<record_id>.pdf``, and lists whatever it couldn't get so you can grab
those by hand. Then zip ``manual_pdfs/`` and bring it back to the main notebook —
Step 2 (``--only-missing``) will use the full text.

    python -m pipeline.download_pdfs                      # uses data/missing_fulltext.csv
    python -m pipeline.download_pdfs --worklist mine.csv  # an explicit worklist (record_id,resolved_url)
    python -m pipeline.download_pdfs --all                # every paper, not just the blocked ones

The worklist is written by the main notebook via ``report.write_worklist()``. Only two
columns are required: ``record_id`` and ``resolved_url`` (``short_title`` is used for
nicer logs if present).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from urllib.parse import urljoin

from . import config
from .util import read_jsonl

# A browser-like UA — some publishers refuse the plain Crossref UA for full-text.
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def _worklist_from_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _worklist_from_artifacts(do_all: bool) -> list[dict]:
    """Rebuild the target list from the pipeline's own JSONL, so this can run without a
    hand-made CSV. `do_all` targets every paper; otherwise only the blocked/thin ones."""
    papers = {str(p["record_id"]): p for p in read_jsonl(config.PAPERS_JSONL)}
    resolved = {str(r["record_id"]): r for r in read_jsonl(config.RESOLVED_JSONL)}
    idx = {str(e["record_id"]): e for e in read_jsonl(config.TEXT_INDEX_JSONL)}
    from .report import MIN_FULLTEXT_CHARS, _FAILED_METHODS
    rows = []
    for rid, p in papers.items():
        e = idx.get(rid, {})
        blocked = (e.get("char_count", 0) < MIN_FULLTEXT_CHARS
                   or e.get("method", "none") in _FAILED_METHODS)
        if not do_all and idx and not blocked:
            continue
        r = resolved.get(rid, {})
        rows.append({"record_id": rid,
                     "short_title": p.get("short_title") or p.get("title", ""),
                     "resolved_url": r.get("resolved_url") or p.get("link", "")})
    return rows


def find_pdf_url(html: str, base_url: str) -> str | None:
    """Find a PDF link on a landing page. Prefers the `citation_pdf_url` meta tag that
    most publishers emit (the de-facto standard), then any obvious .pdf anchor."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    meta = soup.find("meta", attrs={"name": "citation_pdf_url"})
    if meta and meta.get("content"):
        return urljoin(base_url, meta["content"])
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().split("?")[0].endswith(".pdf") or "/pdf" in href.lower():
            return urljoin(base_url, href)
    return None


def download_pdf(session, url: str) -> tuple[bytes | None, str]:
    """Return (pdf_bytes, note). Tries the URL directly, then its citation_pdf_url."""
    if not url or not url.startswith(("http://", "https://")):
        return None, "no url"
    try:
        r = session.get(url, timeout=config.REQUEST_TIMEOUT, allow_redirects=True)
    except Exception as e:
        return None, f"{type(e).__name__}"
    if not r.ok:
        return None, f"HTTP {r.status_code}"
    ctype = r.headers.get("content-type", "").lower()
    if "pdf" in ctype or r.content[:5] == b"%PDF-":
        return r.content, "direct"
    # A landing page — look for the real PDF link and fetch it.
    pdf_url = find_pdf_url(r.text, r.url)
    if not pdf_url:
        return None, "no pdf link on page (login required?)"
    try:
        r2 = session.get(pdf_url, timeout=config.REQUEST_TIMEOUT, allow_redirects=True)
    except Exception as e:
        return None, f"pdf link: {type(e).__name__}"
    if r2.ok and (r2.content[:5] == b"%PDF-" or "pdf" in r2.headers.get("content-type", "").lower()):
        return r2.content, "via citation_pdf_url"
    return None, "pdf link did not return a PDF (login required?)"


def run(worklist: str | None = None, do_all: bool = False, out_dir: Path | None = None):
    """Download PDFs for the worklist into out_dir. Returns (n_ok, n_fail, still_missing)."""
    import requests

    out_dir = Path(out_dir) if out_dir else config.PDF_DROP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if worklist:
        rows = _worklist_from_csv(Path(worklist))
    elif config.MISSING_CSV.exists() and not do_all:
        rows = _worklist_from_csv(config.MISSING_CSV)
        print(f"worklist: {config.MISSING_CSV}")
    else:
        rows = _worklist_from_artifacts(do_all)
        print("worklist: derived from pipeline artifacts "
              f"({'all papers' if do_all else 'blocked/thin only'})")

    if not rows:
        print("Nothing to download — no worklist and no blocked papers found. "
              "Run Steps 0-2 first, or pass --all.")
        return 0, 0, []

    session = requests.Session()
    session.headers.update({"User-Agent": BROWSER_UA})

    from tqdm.auto import tqdm
    ok, missing = 0, []
    for row in tqdm(rows, desc="Downloading PDFs", unit="pdf"):
        rid = str(row["record_id"])
        url = row.get("resolved_url", "")
        title = (row.get("short_title") or "")[:50]
        data, note = download_pdf(session, url)
        if data:
            (out_dir / f"{rid}.pdf").write_bytes(data)
            ok += 1
            tqdm.write(f"rid={rid}  OK  ({note}, {len(data)//1024} KB)  {title}")
        else:
            missing.append({"record_id": rid, "short_title": row.get("short_title", ""),
                            "resolved_url": url, "reason": note})
            tqdm.write(f"rid={rid}  --  {note}  {url[:60]}")

    if missing:
        miss_csv = out_dir / "_still_missing.csv"
        with open(miss_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["record_id", "short_title", "resolved_url", "reason"])
            w.writeheader()
            w.writerows(missing)
        print(f"\n{ok} PDF(s) saved to {out_dir}/  |  {len(missing)} still missing "
              f"(see {miss_csv} — open those links in a browser and save as <record_id>.pdf)")
    else:
        print(f"\nAll {ok} PDF(s) saved to {out_dir}/ — zip it and upload to the main notebook.")
    return ok, len(missing), missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Download full-text PDFs for blocked papers.")
    ap.add_argument("--worklist", help="CSV with record_id,resolved_url (default: data/missing_fulltext.csv)")
    ap.add_argument("--all", action="store_true", help="download every paper, not just the blocked ones")
    ap.add_argument("--out", help="output folder (default: manual_pdfs/)")
    args = ap.parse_args()
    run(worklist=args.worklist, do_all=args.all, out_dir=args.out)


if __name__ == "__main__":
    main()
