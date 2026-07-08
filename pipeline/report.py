"""Thin reporting helpers so the notebook can stay (almost) code-free.

Every function returns a pandas DataFrame or a small dict; the notebook just
displays them. Nothing here calls a model or the network - it only reads the
JSONL artifacts written by the steps.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .merge_results import build_table
from .util import read_jsonl


def papers_df() -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(config.PAPERS_JSONL))
    cols = [c for c in ["record_id", "year", "short_title", "doi", "link",
                        "human_include", "human_reason"] if c in df.columns]
    return df[cols]


def resolved_df() -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(config.RESOLVED_JSONL))
    if df.empty:
        return df
    df["flags"] = df["flags"].apply(lambda f: "; ".join(f) if isinstance(f, list) else f)
    cols = ["record_id", "doi", "title_similarity", "title_match", "year_match",
            "crossref_type", "resolved_via", "resolved_url", "flags"]
    return df[[c for c in cols if c in df.columns]]


def flagged_df() -> pd.DataFrame:
    """Only the Step-1 papers with data-quality problems (wrong link, etc.)."""
    df = resolved_df()
    return df[df["flags"].astype(bool) & (df["flags"] != "")].reset_index(drop=True)


def text_df() -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(config.TEXT_INDEX_JSONL))
    cols = [c for c in ["record_id", "method", "char_count", "http_status",
                        "source_url"] if c in df.columns]
    return df[cols] if not df.empty else df


def screen_df(backend: str) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(config.step3_jsonl(backend)))


def comparison_df(backends: list[str] | None = None) -> pd.DataFrame:
    backends = [b for b in (backends or config.DEFAULT_BACKENDS)
                if config.step3_jsonl(b).exists()]
    return build_table(backends)


def disagreements(backends: list[str] | None = None) -> pd.DataFrame:
    """Rows to eyeball first: model-vs-model or model-vs-human conflicts + flags."""
    df = comparison_df(backends)
    return df[df["review_priority"] != "ok"].reset_index(drop=True)


def summary(backends: list[str] | None = None) -> pd.DataFrame:
    """One-line-per-priority tally, the quickest health check."""
    df = comparison_df(backends)
    return (df["review_priority"].value_counts()
            .rename_axis("review_priority").reset_index(name="papers"))


def decision_counts(backends: list[str] | None = None) -> pd.DataFrame:
    """Decision distribution per backend, side by side with the human column."""
    backends = [b for b in (backends or config.DEFAULT_BACKENDS)
                if config.step3_jsonl(b).exists()]
    frames = {}
    for b in backends:
        s = screen_df(b)
        if not s.empty:
            frames[b] = s["decision"].value_counts()
    out = pd.DataFrame(frames).fillna(0).astype(int)
    return out
