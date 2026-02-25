import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.prompt_template import PromptTemplate
from src.utils.LLM_utils import get_llm_completion
import numpy as np
from IPython.display import display, HTML


# Original document categories
_BASE_CLASSES = ["Invoice", "Purchase Order", "Bill", "Receipt", "Financial Document", "Salary Slip"]

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

CLASSES = [*_BASE_CLASSES, *LAB_REPORT_CLASSES]

# Set of all lab subcategory names for fast membership check
LAB_REPORT_TYPES = set(LAB_REPORT_CLASSES)


def is_lab_report(doc_type: str) -> bool:
    """Return True if *doc_type* is any recognised laboratory report subcategory."""
    return doc_type in LAB_REPORT_TYPES


def classify_document_with_llm(document_text, llm_choice):
    """
    Classify document using either GPT or Llama based on user choice
    Args:
        document_text: Text content of the document to classify
        llm_choice: Either "gpt" or "llama" (default: "gpt")
    """

    CLASSIFICATION_PROMPT = PromptTemplate(
        template="""
I have a document, and I want you to classify it into one of the following categories:

General documents:
- Invoice
- Purchase Order
- Utility Bill
- Receipt
- Financial Document
- Salary Slip

Scientific laboratory reports:
- Chemical Analysis Report  (concentrations, calibration, LOD/LOQ, titration)
- Environmental Analysis Report  (water/air/soil sampling, regulatory limits)
- Microbiology Report  (colony counts, MPN, zone-of-inhibition, sterility)
- Material Testing Report  (tensile/compression strength, hardness, fatigue)
- Clinical Laboratory Report  (patient test results, reference ranges, flags)
- Geotechnical Report  (soil classification, bearing capacity, SPT/CPT data)
- Food Analysis Report  (nutritional composition, contaminants, shelf-life)
- General Laboratory Report  (any other scientific or technical lab document)

Here is the content of the document:

{{text}}

Based on the content, which category does this document belong to?
Reply with ONLY the exact category name from the list above — nothing else.
    """,
    input_variables=["text"]
    )
    prompt = CLASSIFICATION_PROMPT.format(
        text = document_text)

    # Call the API to classify the document
    try:
        response = get_llm_completion(
            messages=[{"role": "user", "content": prompt}],
            llm_choice=llm_choice,
            logprobs=True,
            top_logprobs=1,  # supported by both OpenAI (GPT) and Groq (Llama)
            temperature=0.1,
        )

        classification = response.choices[0].message.content.strip()

        # Extract confidence (linear probability) from logprobs.
        # GPT returns top_logprobs; Groq/Llama returns logprobs.content[i].logprob
        # (same OpenAI format).  Together AI used token_logprobs — kept as fallback.
        linear_probability = 90.0  # safe default
        try:
            if llm_choice == "gpt":
                top_lp = response.choices[0].logprobs.content[0].top_logprobs
                html_content = ""
                for i, logprob in enumerate(top_lp, start=1):
                    html_content += (
                        f"<span style='color: cyan'>Output token {i}:</span> {logprob.token}, "
                        f"<span style='color: darkorange'>logprobs:</span> {logprob.logprob}, "
                        f"<span style='color: magenta'>linear probability:</span> "
                        f"{np.round(np.exp(logprob.logprob)*100, 2)}%<br>"
                    )
                display(HTML(html_content))
                linear_probability = np.round(np.exp(top_lp[0].logprob) * 100, 2)
            else:
                # Groq / OpenAI-compatible format: logprobs.content[i].logprob
                lp_content = response.choices[0].logprobs.content
                if lp_content:
                    lp_val = lp_content[0].logprob
                    linear_probability = np.round(np.exp(lp_val) * 100, 2)
                    print(f"Token: {lp_content[0].token}")
                    print(f"Logprob: {lp_val}")
                    print(f"Linear probability: {linear_probability}%")
                else:
                    # Fallback: Together AI legacy format
                    token_logprob = response.choices[0].logprobs.token_logprobs[0]
                    linear_probability = np.round(np.exp(token_logprob) * 100, 2)
                    print(f"Linear probability (legacy): {linear_probability}%")
        except Exception as lp_err:
            print(f"Could not extract logprob (using default 90%): {lp_err}")

        print("\n")

        if classification in CLASSES:
            return classification, linear_probability
        else:
            return "Unknown", 0.0

    except Exception as e:
        print(f"Error during LLM classification: {e}")
        return "Error", 0.0



