"""PDF -> text, with a text-layer-first strategy and a FREE local OCR fallback.

Most publisher PDFs have a real text layer, which pymupdf extracts instantly and
for free. Only scanned / image-only PDFs need OCR. For those we call **glm-ocr on
Ollama** (https://ollama.com/library/glm-ocr) - it is free, runs locally, needs no
API key, and speaks the OpenAI-compatible API.

Setup (one time):
    ollama pull glm-ocr
    ollama serve            # usually already running on http://localhost:11434

Override the endpoint/model with OCR_BASE_URL / OCR_MODEL if needed (see config.py).
The whole OCR call is isolated here so you can swap providers without touching the
pipeline.
"""
from __future__ import annotations

import sys

from . import config

# OCR is a LAST RESORT: only run it when the PDF has essentially NO extractable
# text (a genuinely scanned/image-only PDF). Real text PDFs yield thousands of
# characters per page, so these low thresholds cleanly separate the two.
MIN_CHARS_PER_PAGE = 20
MIN_CHARS_TOTAL = 200


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def extract_pdf_text_layer(pdf_bytes: bytes) -> tuple[str, int]:
    """Return (text, n_pages) from the PDF's embedded text layer using pymupdf."""
    import fitz  # pymupdf, lazy

    parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        n_pages = doc.page_count
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts).strip(), n_pages


def needs_ocr(text: str, n_pages: int) -> bool:
    """True only when the PDF is effectively textless (scanned/image-only)."""
    n = len((text or "").strip())
    if n_pages <= 0:
        return n < MIN_CHARS_TOTAL
    return n < MIN_CHARS_TOTAL and n < MIN_CHARS_PER_PAGE * n_pages


def _pdf_pages_to_png(pdf_bytes: bytes, max_pages: int, dpi: int) -> list[bytes]:
    import fitz

    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            images.append(page.get_pixmap(dpi=dpi).tobytes("png"))
    return images


def _ollama_reachable(base_url: str) -> bool:
    import urllib.request

    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root  # -> http://host:port
    try:
        urllib.request.urlopen(root + "/api/tags", timeout=3)
        return True
    except Exception:
        return False


def glm_ocr_pdf(pdf_bytes: bytes, max_pages: int | None = None) -> str:
    """OCR a PDF with local glm-ocr on Ollama (free, keyless).

    Raises RuntimeError if the server is unreachable, so the caller can fall back
    gracefully (e.g. keep using the abstract).
    """
    import base64

    cfg = config.OCR
    if not _ollama_reachable(cfg["base_url"]):
        raise RuntimeError(
            f"OCR server not reachable at {cfg['base_url']}. "
            "Start it with `ollama serve` and `ollama pull {}`.".format(cfg["model"]))

    from openai import OpenAI  # lazy

    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    pages = _pdf_pages_to_png(pdf_bytes, max_pages or cfg["max_pages"], cfg["dpi"])
    _log(f"    OCR (glm-ocr): {len(pages)} page(s) to transcribe...")

    out_pages = []
    for i, png in enumerate(pages, 1):
        b64 = base64.b64encode(png).decode()
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe all text from this page "
                     "verbatim in reading order. Output plain text only, no commentary."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            temperature=0,
        )
        out_pages.append(resp.choices[0].message.content or "")
        _log(f"    OCR page {i}/{len(pages)} done ({len(out_pages[-1])} chars)")
    return "\n".join(out_pages).strip()


def pdf_to_text(pdf_bytes: bytes, allow_ocr: bool = True) -> tuple[str, str]:
    """Return (text, method): 'pdf-text', 'pdf-ocr', 'pdf-textonly', or 'pdf-failed'."""
    try:
        text, n_pages = extract_pdf_text_layer(pdf_bytes)
    except Exception as e:
        text, n_pages = f"[text-layer extraction failed: {type(e).__name__}: {e}]", 0
    if text and not needs_ocr(text, n_pages):
        return text, "pdf-text"
    if not allow_ocr:
        return text, "pdf-textonly"
    # Scanned or empty -> try local OCR, but never hard-fail the pipeline.
    try:
        ocr_text = glm_ocr_pdf(pdf_bytes)
        if ocr_text:
            return ocr_text, "pdf-ocr"
    except Exception as e:
        _log(f"    OCR skipped: {type(e).__name__}: {e}")
        return (text or f"[OCR unavailable: {type(e).__name__}: {e}]"), "pdf-failed"
    return (text or ""), "pdf-failed"
