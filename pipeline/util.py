"""Small shared helpers: text cleaning, JSON extraction, fuzzy matching, jsonl IO."""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "clean_text", "normalize_doi", "title_similarity", "extract_json",
    "read_jsonl", "write_jsonl", "index_by",
]


def clean_text(text: Any) -> str:
    """Collapse whitespace; None/NaN -> ''."""
    if text is None:
        return ""
    s = str(text)
    if s.lower() == "nan":
        return ""
    return re.sub(r"\s+", " ", s).strip()


def normalize_doi(raw: Any) -> str:
    """Turn 'https://dx.doi.org/10.x/y', 'doi:10.x/y', '10.X/Y' -> lowercase bare DOI."""
    s = clean_text(raw).lower()
    if not s:
        return ""
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    m = re.search(r"10\.\d{4,9}/\S+", s)
    return m.group(0).rstrip(".,;)") if m else ""


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", clean_text(t).lower())


def title_similarity(a: str, b: str) -> float:
    """0..1 similarity combining char-ratio and token-overlap (Jaccard)."""
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    jacc = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return round(max(ratio, jacc), 3)


def extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object from raw model output.

    Handles ```json fences, <think>...</think> preambles (qwen3), and trailing prose
    by scanning for the first balanced {...} block.
    """
    if not text:
        raise ValueError("empty model output")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
        start = text.find("{", start + 1)
    raise ValueError("no valid JSON object found in model output")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def index_by(records: Iterable[dict], key: str) -> dict[str, dict]:
    return {str(r.get(key)): r for r in records}
