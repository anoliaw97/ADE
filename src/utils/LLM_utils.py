from openai import OpenAI
import os
import json
from dotenv import load_dotenv
import numpy as np

load_dotenv('.env')

# ---------------------------------------------------------------------------
# OpenAI client  (GPT-4o-mini + text-embedding-ada-002)
# ---------------------------------------------------------------------------

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_completion_gpt4(
    messages: list[dict[str, str]],
    response_format=None,
    model: str = "gpt-4o-mini",
    max_tokens=16000,
    temperature=0.1,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,
    top_logprobs=None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion


def openai_embedding(text):
    """Get OpenAI embedding for text using text-embedding-ada-002."""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


# ---------------------------------------------------------------------------
# Groq client  (Llama via Groq's OpenAI-compatible API — free tier)
# ---------------------------------------------------------------------------
# Get a free API key at https://console.groq.com
# Groq supports the same OpenAI Python SDK; only base_url differs.

_GROQ_BASE_URL  = "https://api.groq.com/openai/v1"
_GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
_DEFAULT_LLAMA  = "llama-3.3-70b-versatile"   # free on Groq

client2 = OpenAI(
    api_key=_GROQ_API_KEY or "no-key",
    base_url=_GROQ_BASE_URL,
)


def get_completion_llama(
    messages: list[dict[str, str]],
    response_format=None,
    temperature=0.1,
    logprobs=None,
    top_logprobs=None,
    model: str = _DEFAULT_LLAMA,
) -> str:
    """
    Call Llama (or any other model) via Groq's OpenAI-compatible endpoint.
    Groq uses the standard OpenAI logprobs format:
        response.choices[0].logprobs.content[i].logprob
    """
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "logprobs": logprobs,
    }
    if top_logprobs is not None:
        params["top_logprobs"] = top_logprobs
    if response_format is not None:
        params["response_format"] = response_format

    completion = client2.chat.completions.create(**params)
    return completion


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def get_llm_completion(
    messages: list[dict[str, str]],
    llm_choice: str = "gpt",
    response_format=None,
    temperature=0.1,
    model: str = None,
    max_tokens=16000,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,
    top_logprobs=None,
) -> str:
    """
    Route to GPT (OpenAI) or Llama (Groq) based on llm_choice.

    Args:
        llm_choice: "gpt"   → OpenAI gpt-4o-mini
                    "llama" → Groq llama-3.3-70b-versatile (free tier)
    """
    if llm_choice.lower() == "gpt":
        return get_completion_gpt4(
            messages=messages,
            response_format=response_format,
            model=model or "gpt-4o-mini",
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            seed=seed,
            tools=tools,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
    elif llm_choice.lower() == "llama":
        return get_completion_llama(
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            model=model or _DEFAULT_LLAMA,
        )
    else:
        raise ValueError(f"Invalid llm_choice '{llm_choice}'. Use 'gpt' or 'llama'.")
