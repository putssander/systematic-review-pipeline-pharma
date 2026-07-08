# Paper-screening pipeline (DDI / ADE / ADR / MRH systematic review)

Reproducibly fills the `include` / `reason` columns of **`25 papers SP.xlsx`** and
**cross-checks two models** (GPT-5.5 vs a local qwen3-8b) so the remaining mistakes
surface as disagreements instead of hiding in the sheet.

The notebook (`screening_notebook.ipynb`) only *orchestrates and verifies*. All the
work is in small, independently runnable scripts under `pipeline/`.

```
step0_load      xlsx            -> data/step0_papers.jsonl
step1_resolve   DOI             -> canonical URL + title/year checks   (catches wrong links)
step2_fetch     URL             -> raw text (HTML; PDF text-layer; local glm-ocr only if textless)
step3_screen    text+metadata   -> include / reason              (run once per backend)
merge_results   all backends    -> 25 papers SP_screened.xlsx    (with disagreement flags)
```

## Quick start (local)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pipeline.step0_load
python -m pipeline.step1_resolve
python -m pipeline.step2_fetch_text --sleep 1
python -m pipeline.step3_screen --backend gpt-5.5
python -m pipeline.merge_results --backends gpt-5.5
open "25 papers SP_screened.xlsx"
```

Every step reads/writes plain JSONL in `data/`, so runs are resumable
(`--only-missing`) and each stage is inspectable before the next.

## Comparing GPT-5.5 with a local qwen3-8b

Both backends screen against the exact same prompt + JSON schema, so their outputs
are directly comparable. Pick how qwen3-8b is served:

**A. HuggingFace transformers (default; best on a Colab GPU)**
```bash
pip install transformers torch accelerate     # + bitsandbytes for 4-bit on a T4
python -m pipeline.step3_screen --backend qwen3-8b
```

**B. Ollama / LM Studio (OpenAI-compatible local server)**
```bash
ollama pull qwen3:8b
QWEN_BACKEND=openai QWEN_BASE_URL=http://localhost:11434/v1 QWEN_MODEL=qwen3:8b \
  python -m pipeline.step3_screen --backend qwen3-8b
```

Then merge both and read the top of the sheet:
```bash
python -m pipeline.merge_results --backends gpt-5.5 qwen3-8b
```

`review_priority` sorts the mistakes to the top:

| priority | meaning |
|---|---|
| `DISAGREE-MODELS` | the two models disagree on include/exclude |
| `DISAGREE-HUMAN`  | models agree but differ from the current sheet |
| `FLAGGED-STEP1`   | wrong/duplicate link or DOI-title mismatch |
| `NEEDS-REVIEW`    | a model requested human review |

## Colab + VS Code

1. New Colab notebook, **Runtime -> GPU** (T4 fine in 4-bit; A100 for fp16).
2. Put this folder on Google Drive; open `screening_notebook.ipynb` (or attach VS Code
   to the Colab runtime).
3. Run the first cell - it mounts Drive, `cd`s in, and `pip install`s. qwen3-8b then
   runs locally on the GPU via transformers, no server needed.

## Configuration

All knobs live in `pipeline/config.py`, overridable via environment variables / `.env`:

| var | default | purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required for GPT) | OpenAI key |
| `GPT_MODEL` | `gpt-5.5` | OpenAI model id |
| `QWEN_BACKEND` | `hf` | `hf` (transformers) or `openai` (local server) |
| `QWEN_MODEL` | `Qwen/Qwen3-8B` | HF id, or Ollama tag when `openai` |
| `QWEN_BASE_URL` | `http://localhost:11434/v1` | local server URL |
| `QWEN_4BIT` | `0` | `1` to load qwen in 4-bit (small GPUs) |
| `OCR_MODEL` | `glm-ocr` | local Ollama OCR model (free, keyless) |
| `OCR_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `CROSSREF_MAILTO` | your email | Crossref "polite pool" contact |

### Scanned-PDF OCR (free, keyless)

OCR runs **only as a last resort** — a PDF is OCR'd only when it has essentially no
extractable text layer (a genuine scan). It uses local
[`glm-ocr` on Ollama](https://ollama.com/library/glm-ocr), so no API key is needed:

```bash
ollama pull glm-ocr        # one time
ollama serve               # usually already running
```

If Ollama isn't running, Step 2 just logs `OCR skipped` and keeps the abstract as the
source — the pipeline never stalls on it. Disable OCR entirely with
`step2_fetch_text --no-ocr`.

## ⚠️ Security note

`.env` currently contains a live-looking `OPENAI_API_KEY`. It is now covered by
`.gitignore`, but if this key was ever shared or pushed anywhere, **rotate it** at
platform.openai.com. Never commit `.env`.
