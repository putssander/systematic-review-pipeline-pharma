"""Central configuration: paths, .env loading, and the model-backend registry.

Everything that a person might want to change lives here so the notebook and the
step scripts stay thin. Backends are provider-agnostic: GPT-5.5 goes through the
OpenAI API, while a local qwen3-8b can run either through an OpenAI-compatible
server (Ollama / LM Studio) or through HuggingFace transformers (great on Colab).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
INPUT_XLSX = ROOT / "25 papers SP.xlsx"
SHEET_NAME = "Blad1"

DATA_DIR = ROOT / "data"
TEXT_DIR = DATA_DIR / "text"
PAPERS_JSONL = DATA_DIR / "step0_papers.jsonl"
RESOLVED_JSONL = DATA_DIR / "step1_resolved.jsonl"
TEXT_INDEX_JSONL = DATA_DIR / "step2_text_index.jsonl"
OUTPUT_XLSX = ROOT / "25 papers SP_screened.xlsx"

for _d in (DATA_DIR, TEXT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Contact e-mail sent to the Crossref "polite pool" for faster, reliable service.
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "putssander@gmail.com")


def load_dotenv_if_present() -> None:
    """Load ROOT/.env into os.environ without overwriting existing values."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # optional dependency

        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass
    # Minimal fallback parser so the pipeline works without python-dotenv.
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


load_dotenv_if_present()

# --------------------------------------------------------------------------- #
# Backend registry
# --------------------------------------------------------------------------- #
# kind == "openai"  -> uses the OpenAI Python SDK (real OpenAI, or any
#                      OpenAI-compatible server via base_url: Ollama/LM Studio/vLLM)
# kind == "hf"      -> uses HuggingFace transformers locally (ideal on a Colab GPU)
#
# Override any field with an env var so you never have to edit code to retarget.
BACKENDS: dict[str, dict] = {
    "gpt-5.5": {
        "kind": "openai",
        "model": os.getenv("GPT_MODEL", "gpt-5.5"),
        "base_url": os.getenv("OPENAI_BASE_URL") or None,  # None -> api.openai.com
        "api_key_env": "OPENAI_API_KEY",
        # The Responses API + strict json_schema is the OpenAI-native path.
        # Set to "chat" for OpenAI-compatible servers that only speak chat.completions.
        "api_style": os.getenv("GPT_API_STYLE", "responses"),
    },
    # Generic local-qwen slot. The ACTUAL model is chosen by the notebook preset
    # (any Ollama tag: qwen3:8b, qwen3.5:27b, qwen3.6:27b, ...) via QWEN_MODEL, so
    # this one backend can be any qwen without renaming the comparison column.
    "qwen": {
        # Default to HuggingFace so it "just works" on Colab with a GPU.
        # Set QWEN_BACKEND=openai to instead hit Ollama/LM Studio locally.
        "kind": os.getenv("QWEN_BACKEND", "hf"),
        "model": os.getenv("QWEN_MODEL", "qwen3:8b"),
        # Used only when QWEN_BACKEND=openai (Ollama default port shown).
        "base_url": os.getenv("QWEN_BASE_URL", "http://localhost:11434/v1"),
        "api_key_env": "QWEN_API_KEY",  # Ollama ignores it; any value works.
        "api_style": "chat",
        # Qwen3 defaults to a long <think> preamble that is slow and looks like a
        # hang; the '/no_think' soft switch disables it for fast, clean JSON.
        # Set QWEN_NO_THINK=0 to keep reasoning on.
        "no_think": os.getenv("QWEN_NO_THINK", "1") == "1",
        # HF generation knobs (only used when kind == "hf").
        "hf_max_new_tokens": int(os.getenv("QWEN_MAX_NEW_TOKENS", "2048")),
        "hf_dtype": os.getenv("QWEN_DTYPE", "auto"),
        "hf_load_in_4bit": os.getenv("QWEN_4BIT", "0") == "1",
    },
}

DEFAULT_BACKENDS = ["gpt-5.5", "qwen"]

# Screening / fetching knobs
MAX_ABSTRACT_CHARS = 8000
MAX_FETCH_CHARS = 16000
REQUEST_TIMEOUT = 20
USER_AGENT = "screening-pipeline/1.0 (+https://doi.org; mailto:%s)" % CROSSREF_MAILTO

# OCR for scanned PDFs. Default = FREE, keyless, local glm-ocr on Ollama
# (https://ollama.com/library/glm-ocr): `ollama pull glm-ocr`. It speaks the
# OpenAI-compatible API, so no API key is required.
OCR = {
    "base_url": os.getenv("OCR_BASE_URL", "http://localhost:11434/v1"),
    "model": os.getenv("OCR_MODEL", "glm-ocr"),
    "api_key": os.getenv("OCR_API_KEY", "ollama"),   # placeholder; Ollama ignores it
    "max_pages": int(os.getenv("OCR_MAX_PAGES", "12")),
    "dpi": int(os.getenv("OCR_DPI", "150")),
}


def step3_jsonl(backend: str) -> Path:
    """Per-backend Step-3 output file, e.g. data/step3_gpt-5.5.jsonl."""
    safe = backend.replace("/", "_")
    return DATA_DIR / f"step3_{safe}.jsonl"
