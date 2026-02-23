import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.prompt_template import PromptTemplate
from src.utils.LLM_utils import get_llm_completion
import numpy as np


def _compute_perplexity(response) -> float:
    """
    Derive perplexity from the LLM response logprobs.

    Tries three formats:
      1. OpenAI / Ollama OpenAI-compatible  -> choices[0].logprobs.content
      2. Together-AI legacy format           -> choices[0].logprobs.token_logprobs
      3. Fallback                            -> 1.0  (neutral perplexity)
    """
    try:
        lp_list = [t.logprob for t in response.choices[0].logprobs.content]
        if lp_list:
            return float(np.exp(-np.mean(lp_list)))
    except (AttributeError, IndexError, TypeError):
        pass

    try:
        lp_list = response.choices[0].logprobs.token_logprobs
        if lp_list:
            return float(np.exp(-np.mean(lp_list)))
    except (AttributeError, IndexError, TypeError):
        pass

    return 1.0   # neutral fallback


def schema_building_with_llm(baseprompt: str, document_text: str, llm_choice: str = "ollama"):
    """
    Build a JSON schema for the given document using the LLM.

    Args:
        baseprompt:     Schema-builder system prompt.
        document_text:  Text content of the document (typically the first page).
        llm_choice:     "ollama" | "gpt" | "llama" (all route to local Ollama).

    Returns:
        Tuple (full_prompt: str, schema_json: str, perplexity: float)
    """
    schema_builder_prompt = PromptTemplate(
        template="""
{{baseprompt}}

{{document_text}}

    """,
        input_variables=["baseprompt", "document_text"],
    )

    prompt = schema_builder_prompt.format(
        baseprompt=baseprompt,
        document_text=document_text,
    )

    try:
        response = get_llm_completion(
            messages=[{"role": "user", "content": prompt}],
            llm_choice=llm_choice,
            response_format={"type": "json_object"},
            logprobs=True,
            temperature=0.1,
        )

        response_text    = response.choices[0].message.content
        perplexity_score = _compute_perplexity(response)

    except Exception as e:
        print(f"Error during Schema Building: {e}")
        return "Error", None, None

    return prompt, response_text, perplexity_score
