import os

def load_prompt_from_file(filename: str = None, document_type: str = None) -> str:
    """
    Load prompt template from a text file based on document type or filename
    
    Args:
        filename: Optional specific prompt file to load
        document_type: Type of document to load prompt for
        
    Returns:
        str: Prompt template text
    """
    # Get the absolute path to the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))        # …/src/utils
    project_root = os.path.dirname(os.path.dirname(current_dir))    # …/ADE  (2 levels up)

    # Prompt filename mapping — covers original doc types and all lab subcategories
    _PROMPT_MAPPING = {
        # Original document types
        "Invoice":            "invoice_prompt.txt",
        "Purchase Order":     "purchaseorders_prompt.txt",
        "Utility Bill":       "utilitybill_prompt.txt",
        "Financial Document": "financial_prompt.txt",
        "Receipt":            "invoice_prompt.txt",
        "Salary Slip":        "salaryslip_prompt.txt",
        # Scientific laboratory report subcategories (all share the same extraction prompt)
        "Chemical Analysis Report":      "labreport_prompt.txt",
        "Environmental Analysis Report": "labreport_prompt.txt",
        "Microbiology Report":           "labreport_prompt.txt",
        "Material Testing Report":       "labreport_prompt.txt",
        "Clinical Laboratory Report":    "labreport_prompt.txt",
        "Geotechnical Report":           "labreport_prompt.txt",
        "Food Analysis Report":          "labreport_prompt.txt",
        "General Laboratory Report":     "labreport_prompt.txt",
        # Fallback
        "Unknown":            "financial_prompt.txt",
    }

    # If specific filename is provided, use it
    if filename:
        prompt_file = filename
    # Otherwise select prompt based on document type
    elif document_type:
        prompt_file = _PROMPT_MAPPING.get(document_type, "financial_prompt.txt")
    else:
        raise ValueError("Either filename or document_type must be provided")

    # Try paths relative to the resolved project root
    possible_paths = [
        os.path.join(project_root, 'src', 'actor_agents', 'Prompts', prompt_file),
        # Legacy path kept for backward compatibility
        os.path.join(os.path.dirname(project_root), 'Unstructured-Data-Extraction',
                     'src', 'actor_agents', 'Prompts', prompt_file),
    ]
    
    for prompt_path in possible_paths:
        try:
            with open(prompt_path, 'r', encoding='utf-8') as file:
                print(f"Loading prompt from: {prompt_path}")
                return file.read()
        except FileNotFoundError:
            continue

    raise FileNotFoundError(f"Prompt file not found. Tried paths: {possible_paths}")


def load_schema_prompt_for_type(document_type: str) -> str | None:
    """
    Return a document-type-specific schema-builder prompt, or None to fall back
    to the default schema_builder_prompt.txt.

    Currently only laboratory report subcategories have a dedicated schema prompt.
    """
    _SCHEMA_MAPPING = {
        "Chemical Analysis Report":      "labreport_schema_prompt.txt",
        "Environmental Analysis Report": "labreport_schema_prompt.txt",
        "Microbiology Report":           "labreport_schema_prompt.txt",
        "Material Testing Report":       "labreport_schema_prompt.txt",
        "Clinical Laboratory Report":    "labreport_schema_prompt.txt",
        "Geotechnical Report":           "labreport_schema_prompt.txt",
        "Food Analysis Report":          "labreport_schema_prompt.txt",
        "General Laboratory Report":     "labreport_schema_prompt.txt",
    }
    schema_file = _SCHEMA_MAPPING.get(document_type)
    if schema_file is None:
        return None
    try:
        return load_prompt_from_file(filename=schema_file)
    except FileNotFoundError:
        return None