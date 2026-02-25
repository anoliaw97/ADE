from typing import Dict, List, Any, Tuple

from src.models.local_llm import _get_llm
from src.environments.data_extraction_env import DataExtractionEnvIterative
from src.rl_agents.gymnasium_extraction_agent import GymnasiumAgent as ExtractionAgent


def process_single_page(args: Tuple[str, str, str, dict, dict, int, int, int, str]) -> Dict[str, Any]:
    """
    Process a single page for data extraction.

    Args:
        args: Tuple of:
            page_text, doc_type, extraction_prompt, schema, groundtruth,
            page_num, total_pages, max_steps, llm_choice
    Returns:
        Dict with keys: page_num, results, steps
    """
    page_text, doc_type, extraction_prompt, schema, groundtruth, page_num, total_pages, max_steps, llm_choice = args

    print(f"\nProcessing page {page_num + 1}/{total_pages}")

    # Use local Qwen model (singleton — loaded once across all pages)
    chat_model = _get_llm()

    extraction_env = DataExtractionEnvIterative(
        baseprompt=extraction_prompt,
        document_type=doc_type,
        document=page_text,
        schema=schema,
        groundtruth=groundtruth,
        llm_choice=llm_choice,
        max_steps=max_steps,
    )

    extraction_agent = ExtractionAgent(chat_model, extraction_env)
    extraction_agent.interact()

    page_results = extraction_env.get_best_results()

    return {
        "page_num": page_num,
        "results": page_results,
        "steps": extraction_env.current_step,
    }
