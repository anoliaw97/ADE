from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import numpy as np

load_dotenv('.env')

# ---------------------------------------------------------------------------
# Ollama local backend configuration
# Ollama exposes an OpenAI-compatible REST API at /v1, so we reuse the
# openai Python client – no paid API keys required.
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434/v1")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",     "llama3.2")
OLLAMA_MODEL_ALT = os.getenv("OLLAMA_MODEL_ALT", "mistral")   # secondary / "llama" alias

# Single client pointed at the local Ollama server
client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama",  # required by the openai client but not used by Ollama
)

# ---------------------------------------------------------------------------
# Local embeddings via SentenceTransformers (replaces openai text-embedding-ada-002)
# ---------------------------------------------------------------------------

_embedding_model = None

def _get_embedding_model():
    """Lazily load the SentenceTransformer model (cached after first call)."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
    return _embedding_model


def local_embedding(text: str):
    """Return a float-list embedding for *text* using a local SentenceTransformer."""
    try:
        model = _get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


# Backward-compatible alias – any module that previously called openai_embedding
# will now use the local SentenceTransformer instead.
openai_embedding = local_embedding


# ---------------------------------------------------------------------------
# Core LLM call (Ollama via OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

def get_completion_ollama(
    messages: list[dict[str, str]],
    response_format=None,
    model: str = None,
    max_tokens: int = 16000,
    temperature: float = 0.1,
    stop=None,
    seed: int = 123,
    tools=None,
    logprobs=None,
    top_logprobs=None,
):
    """Call the local Ollama LLM through its OpenAI-compatible endpoint."""
    if model is None:
        model = OLLAMA_MODEL

    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
    }

    if response_format is not None:
        params["response_format"] = response_format
    if tools is not None:
        params["tools"] = tools
    if logprobs is not None:
        params["logprobs"] = logprobs
    if top_logprobs is not None:
        params["top_logprobs"] = top_logprobs

    return client.chat.completions.create(**params)


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# Previously "gpt" called GPT-4o-mini, "llama" called Together-AI Llama.
# Both now route to the local Ollama server (different default models).
# ---------------------------------------------------------------------------

def get_completion_gpt4(
    messages: list[dict[str, str]],
    response_format=None,
    model: str = None,
    max_tokens: int = 16000,
    temperature: float = 0.1,
    stop=None,
    seed: int = 123,
    tools=None,
    logprobs=None,
    top_logprobs=None,
):
    """Previously used GPT-4o-mini; now routes to local Ollama (OLLAMA_MODEL)."""
    return get_completion_ollama(
        messages=messages,
        response_format=response_format,
        model=model or OLLAMA_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        seed=seed,
        tools=tools,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
    )


def get_completion_llama(
    messages: list[dict[str, str]],
    response_format=None,
    temperature: float = 0.1,
    logprobs=None,
    model: str = None,
):
    """Previously used Together-AI Llama; now routes to local Ollama (OLLAMA_MODEL_ALT)."""
    return get_completion_ollama(
        messages=messages,
        response_format=response_format,
        model=model or OLLAMA_MODEL_ALT,
        temperature=temperature,
        logprobs=logprobs,
    )


# ---------------------------------------------------------------------------
# Unified dispatcher (used throughout the codebase)
# ---------------------------------------------------------------------------

def get_llm_completion(
    messages: list[dict[str, str]],
    llm_choice: str = "ollama",   # "ollama" | "gpt" | "llama" (all → local Ollama)
    response_format=None,
    temperature: float = 0.1,
    model: str = None,
    max_tokens: int = 16000,
    stop=None,
    seed: int = 123,
    tools=None,
    logprobs=None,
    top_logprobs=None,
) -> str:
    """
    Route to the local Ollama LLM regardless of *llm_choice*.

    *llm_choice* is kept for backward compatibility:
      - "gpt"   → OLLAMA_MODEL  (was GPT-4o-mini)
      - "llama" → OLLAMA_MODEL_ALT (was Together-AI Llama)
      - "ollama"→ OLLAMA_MODEL  (new default)
    """
    if llm_choice.lower() in ("llama",):
        selected_model = model or OLLAMA_MODEL_ALT
    else:
        # "gpt", "ollama", or any future value
        selected_model = model or OLLAMA_MODEL

    return get_completion_ollama(
        messages=messages,
        response_format=response_format,
        model=selected_model,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        seed=seed,
        tools=tools,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
    )
