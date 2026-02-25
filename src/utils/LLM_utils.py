"""
LLM utilities — local inference via Qwen2.5-3B-Instruct.
No API keys required.  All calls route through LocalChatModel.complete().
"""

import os
import json
import numpy as np

from src.models.local_llm import _get_llm, MockCompletionResponse


# ---------------------------------------------------------------------------
# Public completion helpers  (drop-in replacements for the OpenAI / Together
# versions that the rest of the codebase calls)
# ---------------------------------------------------------------------------

def get_completion_gpt4(
    messages,
    response_format=None,
    model: str = None,          # ignored — local model used
    max_tokens: int = 16000,    # ignored — controlled by LocalChatModel
    temperature: float = 0.1,
    stop=None,
    seed=None,
    tools=None,
    logprobs=None,
    top_logprobs=None,
) -> MockCompletionResponse:
    """Drop-in replacement for the original OpenAI get_completion_gpt4."""
    return _get_llm().complete(
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
    )


def get_completion_llama(
    messages,
    response_format=None,
    temperature: float = 0.1,
    logprobs=None,
    model: str = None,          # ignored
) -> MockCompletionResponse:
    """Drop-in replacement for the original Together/Groq get_completion_llama."""
    return _get_llm().complete(
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        logprobs=logprobs,
    )


def get_llm_completion(
    messages,
    llm_choice: str = "local",  # "gpt", "llama", or "local" all route to local model
    response_format=None,
    temperature: float = 0.1,
    model: str = None,
    max_tokens: int = 16000,
    stop=None,
    seed=None,
    tools=None,
    logprobs=None,
    top_logprobs=None,
) -> MockCompletionResponse:
    """
    Unified completion entry-point.
    All llm_choice values route to the local Qwen model.
    """
    return _get_llm().complete(
        messages=messages,
        response_format=response_format,
        temperature=temperature,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
    )


def openai_embedding(text: str):
    """
    Returns a zero vector placeholder.
    The RL chain uses embeddings for contextual bandits;
    with a local model we use a simple fixed-size zero vector.
    """
    return [0.0] * 768
