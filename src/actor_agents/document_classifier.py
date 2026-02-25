import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.prompt_template import PromptTemplate
from src.utils.LLM_utils import get_llm_completion
import numpy as np


# ---------------------------------------------------------------------------
# Document classes — original + scientific lab report subcategories
# ---------------------------------------------------------------------------

_BASE_CLASSES = [
    "Invoice", "Purchase Order", "Bill", "Receipt",
    "Financial Document", "Salary Slip",
]

LAB_REPORT_CLASSES = [
    "Chemical Analysis Report",
    "Environmental Analysis Report",
    "Microbiology Report",
    "Material Testing Report",
    "Clinical Laboratory Report",
    "Geotechnical Report",
    "Food Analysis Report",
    "General Laboratory Report",
]

CLASSES = [*_BASE_CLASSES, *LAB_REPORT_CLASSES]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify_document_with_llm(document_text: str, llm_choice: str = "local"):
    """
    Classify a document into one of the known CLASSES using the local LLM.

    Args:
        document_text: Text content of the first page.
        llm_choice:    Ignored (local model always used); kept for API compatibility.

    Returns:
        Tuple[str, float]: (document_type, confidence_percent)
    """
    classes_list = "\n".join(f"- {c}" for c in CLASSES)

    CLASSIFICATION_PROMPT = PromptTemplate(
        template="""
I have a document and I want you to classify it into ONE of the following categories:
{classes_list}

Here is the content of the document:

{{text}}

Based on the content, which category does this document belong to?
Reply with ONLY the exact category name from the list above — nothing else.
""",
        input_variables=["text"]
    )

    prompt = CLASSIFICATION_PROMPT.format(text=document_text).replace(
        "{classes_list}", classes_list
    )

    try:
        response = get_llm_completion(
            messages=[{"role": "user", "content": prompt}],
            llm_choice=llm_choice,
            temperature=0.1,
        )

        classification = response.choices[0].message.content.strip()

        # Use mock logprob (local model doesn't provide real logprobs)
        try:
            lp_val = response.choices[0].logprobs.content[0].logprob
            linear_probability = float(np.round(np.exp(lp_val) * 100, 2))
        except Exception:
            linear_probability = 95.0   # confident default for local model

        # Exact match first
        if classification in CLASSES:
            return classification, linear_probability

        # Case-insensitive fallback
        lower_map = {c.lower(): c for c in CLASSES}
        matched = lower_map.get(classification.lower())
        if matched:
            return matched, linear_probability

        return "Unknown", 0.0

    except Exception as e:
        print(f"Error during LLM classification: {e}")
        return "Error", 0.0
