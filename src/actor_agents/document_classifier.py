import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.prompt_template import PromptTemplate
from src.utils.LLM_utils import get_llm_completion
import numpy as np

# ---------------------------------------------------------------------------
# Supported document classes
# ---------------------------------------------------------------------------

# Scientific laboratory report subcategories
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

# All supported classes
CLASSES = [
    "Invoice",
    "Purchase Order",
    "Utility Bill",
    "Receipt",
    "Financial Document",
    "Salary Slip",
    # Lab report subcategories
    *LAB_REPORT_CLASSES,
]

# Convenience alias — any lab subcategory maps to the same extraction pipeline
LAB_REPORT_TYPES = set(LAB_REPORT_CLASSES)


def is_lab_report(doc_type: str) -> bool:
    """Return True if *doc_type* is any scientific laboratory report variant."""
    return doc_type in LAB_REPORT_TYPES


# ---------------------------------------------------------------------------
# Confidence extraction
# ---------------------------------------------------------------------------

def _parse_confidence(response) -> float:
    """
    Extract a linear-probability confidence score from the LLM response.
    Tries three formats:
      1. OpenAI / Ollama  → choices[0].logprobs.content[0].top_logprobs
      2. Together-AI      → choices[0].logprobs.token_logprobs[0]
      3. Fallback         → 90.0 %
    """
    try:
        top_lp = response.choices[0].logprobs.content[0].top_logprobs
        if top_lp:
            return float(np.round(np.exp(top_lp[0].logprob) * 100, 2))
    except (AttributeError, IndexError, TypeError):
        pass

    try:
        token_lp = response.choices[0].logprobs.token_logprobs[0]
        return float(np.round(np.exp(token_lp) * 100, 2))
    except (AttributeError, IndexError, TypeError):
        pass

    return 90.0


# ---------------------------------------------------------------------------
# Main classification function
# ---------------------------------------------------------------------------

def classify_document_with_llm(document_text: str, llm_choice: str = "ollama"):
    """
    Classify a document using the local Ollama LLM.

    Returns:
        Tuple (document_type: str, confidence: float)
    """
    CLASSIFICATION_PROMPT = PromptTemplate(
        template="""
I have a document and I want you to classify it into EXACTLY ONE of the following categories.
Reply with ONLY the category name — no explanation, no punctuation.

NON-LABORATORY DOCUMENTS:
- Invoice
- Purchase Order
- Utility Bill
- Receipt
- Financial Document
- Salary Slip

SCIENTIFIC LABORATORY REPORTS (documents containing experimental data, measurement
tables, test results, analytical figures, or scientific observations):
- Chemical Analysis Report   (chromatography, spectroscopy, titrations, elemental analysis)
- Environmental Analysis Report  (water quality, soil, air, effluent, wastewater)
- Microbiology Report        (plate counts, MPN, culture results, PCR, sensitivity tests)
- Material Testing Report    (tensile, hardness, compression, fatigue, corrosion, metallurgy)
- Clinical Laboratory Report (blood tests, urinalysis, pathology, medical diagnostics)
- Geotechnical Report        (soil classification, compaction, permeability, shear strength)
- Food Analysis Report       (nutritional analysis, contaminants, pesticide residues, allergens)
- General Laboratory Report  (any other scientific lab report not fitting the above)

Document content:
{{text}}

Category:""",
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
        confidence     = _parse_confidence(response)

        print(f"Classification: {classification}  |  Confidence: {confidence}%\n")

        if classification in CLASSES:
            return classification, confidence
        else:
            return "Unknown", 0.0

    except Exception as e:
        print(f"Error during LLM classification: {e}")
        return "Error", 0.0
