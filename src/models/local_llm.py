"""
Local LLM wrapper using Qwen/Qwen2.5-3B-Instruct.

Provides:
  - LocalChatModel: LangChain-compatible chat model (for GymnasiumAgent)
  - MockCompletionResponse: OpenAI-style response object (for LLM_utils)

No API keys required — all inference runs locally via HuggingFace transformers.
"""

import os
import torch
from typing import List, Optional, Any

from langchain.schema import BaseMessage, AIMessage, SystemMessage, HumanMessage


# ---------------------------------------------------------------------------
# Mock response objects that mimic the OpenAI client response structure
# ---------------------------------------------------------------------------

class MockTopLogprob:
    def __init__(self, token: str = "", logprob: float = -0.1):
        self.token = token
        self.logprob = logprob


class MockLogprobItem:
    def __init__(self, logprob: float = -0.1):
        self.logprob = logprob
        self.top_logprobs = [MockTopLogprob(logprob=logprob)]


class MockLogprobs:
    def __init__(self):
        self.content = [MockLogprobItem()]
        self.token_logprobs = [-0.1]
        self.tokens = [""]


class MockMessage:
    def __init__(self, content: str):
        self.content = content


class MockChoice:
    def __init__(self, content: str):
        self.message = MockMessage(content)
        self.logprobs = MockLogprobs()


class MockCompletionResponse:
    """
    Mimics openai.ChatCompletion response so existing code that calls
    response.choices[0].message.content works without changes.
    """
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


# ---------------------------------------------------------------------------
# Module-level singleton to avoid reloading the model for every call
# ---------------------------------------------------------------------------

_llm_instance = None


def _get_llm():
    """Lazy-load the LocalChatModel singleton."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalChatModel()
    return _llm_instance


# ---------------------------------------------------------------------------
# LangChain-compatible chat model wrapping Qwen2.5-3B-Instruct
# ---------------------------------------------------------------------------

class LocalChatModel:
    """
    Wraps Qwen/Qwen2.5-3B-Instruct so it can be used anywhere a
    LangChain ChatOpenAI is expected.

    The GymnasiumAgent calls:
        response = self.model([SystemMessage(...), HumanMessage(...)])
        action = int(parser.parse(response.content)['action'])

    This class satisfies that interface via __call__ / invoke.
    """

    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(
        self,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"[LocalChatModel] Loading {self.MODEL_ID} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        print("[LocalChatModel] Model loaded.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_chat_format(self, messages: List[BaseMessage]) -> List[dict]:
        """Convert LangChain messages to Qwen chat-template format."""
        chat = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                chat.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                chat.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                chat.append({"role": "assistant", "content": msg.content})
            else:
                chat.append({"role": "user", "content": str(msg.content)})
        return chat

    def _generate(self, messages: List[BaseMessage]) -> str:
        """Run tokenization + generation, return decoded string."""
        chat = self._to_chat_format(messages)
        text = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # LangChain-compatible interface
    # ------------------------------------------------------------------

    def __call__(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """Called by GymnasiumAgent as self.model([...messages...])."""
        return self.invoke(messages, **kwargs)

    def invoke(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """Primary entry-point; returns AIMessage with .content."""
        content = self._generate(messages)
        return AIMessage(content=content)

    # ------------------------------------------------------------------
    # OpenAI-style interface (for LLM_utils.py compatibility)
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: List[dict],
        response_format=None,
        temperature: float = 0.1,
        logprobs: bool = None,
        top_logprobs: int = None,
        **kwargs,
    ) -> MockCompletionResponse:
        """
        Accepts messages in OpenAI dict format
        ({"role": "user", "content": "..."}) and returns a
        MockCompletionResponse that mirrors the OpenAI SDK response shape.
        """
        lc_messages: List[BaseMessage] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        content = self._generate(lc_messages)
        return MockCompletionResponse(content)
