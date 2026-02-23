import os


# ---------------------------------------------------------------------------
# Prompt file mapping: document type → prompt filename
# ---------------------------------------------------------------------------

PROMPT_MAPPING = {
    "Invoice":            "invoice_prompt.txt",
    "Purchase Order":     "purchaseorders_prompt.txt",
    "Utility Bill":       "utilitybill_prompt.txt",
    "Financial Document": "financial_prompt.txt",
    "Receipt":            "invoice_prompt.txt",        # receipts share invoice structure
    "Salary Slip":        "salaryslip_prompt.txt",
    "Laboratory Report":  "labreport_prompt.txt",      # table/graph-focused extraction
    "Unknown":            "financial_prompt.txt",      # safe fallback
}

SCHEMA_PROMPT_MAPPING = {
    "Laboratory Report":  "labreport_schema_prompt.txt",
}

# Directories to search for prompt files (in priority order)
def _prompt_dirs() -> list[str]:
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    return [
        os.path.join(project_root, "src", "actor_agents", "Prompts"),
        os.path.join(project_root, "Unstructured-Data-Extraction",
                     "src", "actor_agents", "Prompts"),
        os.path.join(os.path.dirname(os.path.dirname(current_dir)),
                     "actor_agents", "Prompts"),
    ]


def _load_file(filename: str) -> str:
    """Search all known prompt directories for *filename* and return its text."""
    tried = []
    for d in _prompt_dirs():
        path = os.path.join(d, filename)
        tried.append(path)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                print(f"Loading prompt from: {path}")
                return fh.read()
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"Prompt file '{filename}' not found. Tried:\n" + "\n".join(tried)
    )


def load_prompt_from_file(filename: str = None, document_type: str = None) -> str:
    """
    Load a prompt template from the Prompts directory.

    Args:
        filename:      Specific prompt filename (overrides document_type lookup).
        document_type: Document category; used to look up the default prompt file.

    Returns:
        Prompt text as a string.
    """
    if filename:
        return _load_file(filename)

    if document_type:
        prompt_file = PROMPT_MAPPING.get(document_type, "financial_prompt.txt")
        return _load_file(prompt_file)

    raise ValueError("Either 'filename' or 'document_type' must be provided.")


def load_schema_prompt_for_type(document_type: str) -> str | None:
    """
    Return a document-type-specific schema builder prompt if one exists,
    otherwise return None (caller should fall back to the default schema prompt).
    """
    schema_file = SCHEMA_PROMPT_MAPPING.get(document_type)
    if schema_file is None:
        return None
    try:
        return _load_file(schema_file)
    except FileNotFoundError:
        return None
