from typing import Dict, List, Any, Tuple
import os
from langchain_openai import ChatOpenAI
from src.environments.data_extraction_env import DataExtractionEnvIterative
from src.rl_agents.gymnasium_extraction_agent import GymnasiumAgent as ExtractionAgent

# Ollama configuration (read at import time; overridable via environment)
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
_OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2")


def process_single_page(args: Tuple[str, str, str, dict, dict, int, int, int, str]) -> Dict[str, Any]:
    """
    Process a single page for data extraction in parallel.

    Args:
        args: Tuple containing:
            - page_text (str):        Text content of the page
            - doc_type (str):         Type of document
            - extraction_prompt (str): Prompt template for extraction
            - schema (dict):          Schema to use for extraction
            - groundtruth (dict):     Optional groundtruth for validation
            - page_num (int):         Current page number (0-indexed)
            - total_pages (int):      Total number of pages
            - max_steps (int):        Maximum steps for the extraction RL loop
            - llm_choice (str):       "ollama" | "gpt" | "llama"

    Returns:
        Dict with keys: page_num, results, steps
    """
    page_text, doc_type, extraction_prompt, schema, groundtruth, page_num, total_pages, max_steps, llm_choice = args

    print(f"\nProcessing page {page_num + 1}/{total_pages}")

    # Build a fresh LangChain model per worker process (avoids pickling issues)
    chat_model = ChatOpenAI(
        model=_OLLAMA_MODEL,
        openai_api_key="ollama",
        openai_api_base=_OLLAMA_BASE_URL,
        temperature=0.6,
    )
    
    # Create extraction environment
    extraction_env = DataExtractionEnvIterative(
        baseprompt=extraction_prompt,
        document_type=doc_type,
        document=page_text,
        schema=schema,
        groundtruth=groundtruth,
        llm_choice=llm_choice,  
        max_steps=max_steps
    )
    
    # Run extraction
    extraction_agent = ExtractionAgent(chat_model, extraction_env)
    extraction_agent.interact()
    
    # Get results
    page_results = extraction_env.get_best_results()
    
    return {
        'page_num': page_num,
        'results': page_results,
        'steps': extraction_env.current_step
    }