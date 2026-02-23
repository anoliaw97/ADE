import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.prompt_template import PromptTemplate
from src.utils.LLM_utils import get_llm_completion
import numpy as np


CLASSES = [
    "Invoice",
    "Purchase Order",
    "Utility Bill",
    "Receipt",
    "Financial Document",
    "Salary Slip",
    "Laboratory Report",
]


def _parse_confidence(response) -> float:
    """
    Extract a linear-probability confidence score from the LLM response.

    Tries three formats in order:
      1. OpenAI / Ollama OpenAI-compatible  → choices[0].logprobs.content[0].top_logprobs
      2. Together-AI legacy format           → choices[0].logprobs.token_logprobs[0]
      3. Fallback                            → 90.0 %
    """
    try:
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        if top_logprobs:
            return float(np.round(np.exp(top_logprobs[0].logprob) * 100, 2))
    except (AttributeError, IndexError, TypeError):
        pass

    try:
        token_logprob = response.choices[0].logprobs.token_logprobs[0]
        return float(np.round(np.exp(token_logprob) * 100, 2))
    except (AttributeError, IndexError, TypeError):
        pass

    return 90.0


def classify_document_with_llm(document_text: str, llm_choice: str = "ollama"):
    """
    Classify a document using the local Ollama LLM.

    Args:
        document_text: Text content of the document to classify.
        llm_choice:    LLM backend selector ("ollama" | "gpt" | "llama").
                       All values now route to the local Ollama server.

    Returns:
        Tuple (document_type: str, confidence: float)
    """
    CLASSIFICATION_PROMPT = PromptTemplate(
        template="""
I have a document, and I want you to classify it into one of the following categories:
- Invoice
- Purchase Order
- Utility Bill
- Receipt
- Financial Document
- Salary Slip
- Laboratory Report

A "Laboratory Report" contains experimental data, test results, measurement tables,
analytical figures, graphs, or scientific observations from a lab environment.

Here is the content of the document:

{{text}}

Based on the content, which category does this document belong to? Please reply with only the category name.
    """,
        input_variables=["text"],
    )

    prompt = CLASSIFICATION_PROMPT.format(text=document_text)

    try:
        response = get_llm_completion(
            messages=[{"role": "user", "content": prompt}],
            llm_choice=llm_choice,
            logprobs=True,
            top_logprobs=1,
            temperature=0.1,
        )

        classification = response.choices[0].message.content.strip()
        confidence = _parse_confidence(response)

        print(f"Classification: {classification}  |  Confidence: {confidence}%\n")

        if classification in CLASSES:
            return classification, confidence
        else:
            return "Unknown", 0.0

    except Exception as e:
        print(f"Error during LLM classification: {e}")
        return "Error", 0.0
