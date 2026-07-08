"""One screening interface, three transports.

    screen(backend, system, prompt) -> parsed dict (validated JSON)

- kind="openai", api_style="responses": OpenAI Responses API + strict json_schema
  (the GPT-5.5 path).
- kind="openai", api_style="chat": chat.completions with json_schema, degrading to
  json_object then plain-text extraction (Ollama / LM Studio / vLLM).
- kind="hf": HuggingFace transformers, loaded lazily and cached (qwen3-8b on Colab).

The heavy imports (openai, torch/transformers) are done lazily so the local Mac
environment can run Steps 0-2 without them installed.
"""
from __future__ import annotations

import os

from . import config
from .schema import SCREENING_SCHEMA
from .util import extract_json

_HF_CACHE: dict[str, tuple] = {}
_OPENAI_CACHE: dict[str, object] = {}


def get_backend(name: str) -> dict:
    if name not in config.BACKENDS:
        raise KeyError(f"Unknown backend '{name}'. Known: {list(config.BACKENDS)}")
    return config.BACKENDS[name]


# --------------------------------------------------------------------------- #
# OpenAI-compatible transport
# --------------------------------------------------------------------------- #
def _openai_client(cfg: dict):
    key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
    cache_key = f"{cfg.get('base_url')}|{key_env}"
    if cache_key in _OPENAI_CACHE:
        return _OPENAI_CACHE[cache_key]
    from openai import OpenAI  # lazy

    api_key = os.getenv(key_env) or "not-needed"  # local servers ignore the key
    kwargs = {"api_key": api_key}
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    client = OpenAI(**kwargs)
    _OPENAI_CACHE[cache_key] = client
    return client


def _screen_openai(cfg: dict, system: str, prompt: str) -> dict:
    client = _openai_client(cfg)
    model = cfg["model"]

    # Qwen3 (via Ollama/LM Studio) otherwise emits a long <think> block per paper,
    # which is slow and looks like a hang. The '/no_think' soft switch turns it off.
    if cfg.get("no_think"):
        system = f"{system}\n/no_think"

    if cfg.get("api_style") == "responses":
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            text={"format": {
                "type": "json_schema",
                "name": "paper_screening_result",
                "strict": True,
                "schema": SCREENING_SCHEMA,
            }},
            max_output_tokens=2000,
        )
        if getattr(resp, "status", None) == "incomplete":
            raise RuntimeError(f"Incomplete response: {resp.incomplete_details}")
        return extract_json(resp.output_text)

    # chat.completions path (OpenAI-compatible local servers)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    for response_format in (
        {"type": "json_schema", "json_schema": {
            "name": "paper_screening_result", "strict": True, "schema": SCREENING_SCHEMA}},
        {"type": "json_object"},
        None,
    ):
        try:
            kwargs = {"model": model, "messages": messages, "temperature": 0}
            if response_format is not None:
                kwargs["response_format"] = response_format
            resp = client.chat.completions.create(**kwargs)
            return extract_json(resp.choices[0].message.content or "")
        except Exception as e:  # server rejected this response_format; try the next
            last = e
    raise RuntimeError(f"chat.completions screening failed: {last}")


# --------------------------------------------------------------------------- #
# HuggingFace transformers transport (Colab GPU friendly)
# --------------------------------------------------------------------------- #
def _load_hf(cfg: dict):
    model_id = cfg["model"]
    if model_id in _HF_CACHE:
        return _HF_CACHE[model_id]
    import torch  # lazy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    load_kwargs = {"device_map": "auto"}
    dtype = cfg.get("hf_dtype", "auto")
    load_kwargs["torch_dtype"] = "auto" if dtype == "auto" else getattr(torch, dtype)
    if cfg.get("hf_load_in_4bit"):
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    _HF_CACHE[model_id] = (tok, model)
    return tok, model


def _screen_hf(cfg: dict, system: str, prompt: str) -> dict:
    tok, model = _load_hf(cfg)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    # Qwen3 supports a thinking mode; disable it for clean, deterministic JSON.
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tok(text, return_tensors="pt").to(model.device)
    gen = model.generate(
        **inputs,
        max_new_tokens=cfg.get("hf_max_new_tokens", 2048),
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    out = tok.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return extract_json(out)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def screen(backend: str, system: str, prompt: str) -> dict:
    cfg = get_backend(backend)
    if cfg["kind"] == "openai":
        return _screen_openai(cfg, system, prompt)
    if cfg["kind"] == "hf":
        return _screen_hf(cfg, system, prompt)
    raise ValueError(f"Unknown backend kind: {cfg['kind']}")
