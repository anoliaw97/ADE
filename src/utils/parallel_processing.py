from typing import Dict, List, Any, Tuple
import os
from langchain_openai import ChatOpenAI
from src.environments.data_extraction_env import DataExtractionEnvIterative
from src.rl_agents.gymnasium_extraction_agent import GymnasiumAgent as ExtractionAgent


def _make_chat_model(llm_choice: str) -> ChatOpenAI:
    """Return a ChatOpenAI-compatible model for the given llm_choice.

    'gpt'   → OpenAI gpt-4o-mini  (requires OPENAI_API_KEY)
    'llama' → Groq llama-3.3-70b  (requires GROQ_API_KEY, free tier)
    """
    if llm_choice.lower() == "llama":
        return ChatOpenAI(
            model="llama-3.3-70b-versatile",
            openai_api_key=os.getenv("GROQ_API_KEY", "no-key"),
            openai_api_base="https://api.groq.com/openai/v1",
            temperature=0.6,
        )
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.6)

def process_single_page(args: Tuple[str, str, str, dict, dict, int, int, int, str]) -> Dict[str, Any]:
    """
    Process a single page for data extraction in parallel.
    
    Args:
        args: Tuple containing:
            - page_text (str): The text content of the page
            - doc_type (str): Type of document
            - extraction_prompt (str): The prompt template for extraction
            - schema (dict): The schema to use for extraction
            - groundtruth (dict): Optional groundtruth for validation
            - page_num (int): Current page number
            - total_pages (int): Total number of pages
            - max_steps (int): Maximum number of steps for extraction
            - llm_choice (str): llama or gpt
    
    Returns:
        Dict containing:
            - page_num: The processed page number
            - results: Extraction results for the page
            - steps: Number of steps taken
    """
    page_text, doc_type, extraction_prompt, schema, groundtruth, page_num, total_pages, max_steps, llm_choice = args
    
    print(f"\nProcessing page {page_num + 1}/{total_pages}")
    
    # Initialize chat model — routes to Groq (free) when llm_choice='llama'
    chat_model = _make_chat_model(llm_choice)
    
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