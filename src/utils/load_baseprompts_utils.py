import os

# Resolve project root: this file lives at <project_root>/src/utils/
_current_dir = os.path.dirname(os.path.abspath(__file__))          # .../src/utils
_project_root = os.path.dirname(os.path.dirname(_current_dir))     # .../ADE

_PROMPT_DIR = os.path.join(_project_root, "src", "actor_agents", "Prompts")

# Mapping from document type to extraction prompt filename
_PROMPT_MAPPING = {
    # --- original document types ---
    "Invoice":             "Invoice_prompt.txt",
    "Purchase Order":      "purchaseorders_prompt.txt",
    "Utility Bill":        "utilitybill_prompt.txt",
    "Financial Document":  "financial_prompt.txt",
    "Receipt":             "Invoice_prompt.txt",
    "Salary Slip":         "salaryslip_prompt.txt",
    "Unknown":             "financial_prompt.txt",
    # --- scientific laboratory report types ---
    "Chemical Analysis Report":     "labreport_prompt.txt",
    "Environmental Analysis Report":"labreport_prompt.txt",
    "Microbiology Report":          "labreport_prompt.txt",
    "Material Testing Report":      "labreport_prompt.txt",
    "Clinical Laboratory Report":   "labreport_prompt.txt",
    "Geotechnical Report":          "labreport_prompt.txt",
    "Food Analysis Report":         "labreport_prompt.txt",
    "General Laboratory Report":    "labreport_prompt.txt",
}

# Mapping from document type to schema-builder prompt filename
_SCHEMA_PROMPT_MAPPING = {
    "Chemical Analysis Report":     "labreport_schema_prompt.txt",
    "Environmental Analysis Report":"labreport_schema_prompt.txt",
    "Microbiology Report":          "labreport_schema_prompt.txt",
    "Material Testing Report":      "labreport_schema_prompt.txt",
    "Clinical Laboratory Report":   "labreport_schema_prompt.txt",
    "Geotechnical Report":          "labreport_schema_prompt.txt",
    "Food Analysis Report":         "labreport_schema_prompt.txt",
    "General Laboratory Report":    "labreport_schema_prompt.txt",
}


def load_prompt_from_file(filename: str = None, document_type: str = None) -> str:
    """
    Load a prompt template from the Prompts directory.

    Args:
        filename:      Specific file to load (e.g. "schema_builder_prompt.txt").
        document_type: Document type to look up in _PROMPT_MAPPING.

    Returns:
        The prompt text as a string.
    """
    if filename:
        prompt_file = filename
    elif document_type:
        prompt_file = _PROMPT_MAPPING.get(document_type, "financial_prompt.txt")
    else:
        raise ValueError("Either filename or document_type must be provided")

    prompt_path = os.path.join(_PROMPT_DIR, prompt_file)
    try:
        with open(prompt_path, "r", encoding="utf-8") as fh:
            print(f"Loading prompt from: {prompt_path}")
            return fh.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


def load_schema_prompt_for_type(document_type: str) -> str | None:
    """
    Return the schema-builder prompt for the given document type,
    or None if no specific schema prompt exists for that type.
    """
    prompt_file = _SCHEMA_PROMPT_MAPPING.get(document_type)
    if not prompt_file:
        return None
    prompt_path = os.path.join(_PROMPT_DIR, prompt_file)
    try:
        with open(prompt_path, "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return None
