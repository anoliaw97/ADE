"""
Learned Prompt Optimizer using VowpalWabbit contextual bandits.

Previously depended on OpenAI's paid API (embeddings + GPT-4o-mini).
Now uses:
  - Local Ollama LLM  (via langchain_openai.ChatOpenAI with custom base_url)
  - SentenceTransformers for embeddings  (free, local)
"""

import sys
import os
from typing import List, Dict, Union, Any

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from langchain_experimental import rl_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from src.action_space.meta_prompting_agent import (
    clarity_strategy,
    best_practice_strategy,
    fewshot_strategy,
    no_change_strategy,
    LLm_feedback_strategy,
)
from dotenv import load_dotenv

load_dotenv(".env")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2")


def _build_ollama_llm() -> ChatOpenAI:
    """Return a LangChain ChatOpenAI client pointed at the local Ollama server."""
    return ChatOpenAI(
        model=OLLAMA_MODEL,
        openai_api_key="ollama",          # dummy key – not used by Ollama
        openai_api_base=OLLAMA_BASE_URL,
        temperature=0.1,
    )


class LocalEmbedder:
    """
    SentenceTransformer-backed embedder compatible with the rl_chain
    PickBestFeatureEmbedder interface.

    Replaces the previous OpenAIEmbedder that called text-embedding-ada-002.
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, convert_to_numpy=True).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LearnedPromptOptimizer:
    """
    Optimises prompts by learning from different meta-prompting strategies using RL.
    Uses VowpalWabbit contextual bandits to select the most effective strategy.
    """

    def __init__(self, llm=None, model_save_dir: str = None):
        self.llm = llm if llm is not None else _build_ollama_llm()
        self.model_save_dir = model_save_dir

        # Local embedder (no paid API needed)
        self.embedder = LocalEmbedder()

        # Prompt template for the RL chain
        self.prompt_template = PromptTemplate(
            template=(
                "Given the document type: {doc_type}\n"
                "And the current extraction results: {current_output}\n"
                "With groundtruth: {groundtruth}\n\n"
                "Here is a proposed prompt strategy:\n{prompt_strategy}\n\n"
                "Evaluate this prompt strategy and generate an optimised prompt that "
                "maximises extraction accuracy and maintains consistent formatting."
            ),
            input_variables=["doc_type", "current_output", "groundtruth", "prompt_strategy"],
        )

        # Scoring criteria for the automatic selection scorer
        scoring_criteria_template = (
            "Given the document type {doc_type} and groundtruth {groundtruth}, "
            "rank how effective this prompt strategy would be: {prompt_strategy}"
        )

        # VowpalWabbit configuration
        vw_cmd = [
            "--cb_explore_adf",
            "--interactions", "all",
            "--learning_rate", "0.1",
            "--cb_type", "mtr",
            "--epsilon", "0.2",
            "--quiet",
        ]

        self.chain = rl_chain.PickBest.from_llm(
            llm=self.llm,
            prompt=self.prompt_template,
            selection_scorer=rl_chain.AutoSelectionScorer(
                llm=self.llm,
                scoring_criteria_template_str=scoring_criteria_template,
            ),
            auto_embed=True,
            feature_embedder=rl_chain.PickBestFeatureEmbedder(
                auto_embed=True,
                model=self.embedder,
            ),
            model_save_dir=model_save_dir,
            vw_cmd=vw_cmd,
        )

    # ------------------------------------------------------------------
    def generate_strategy_variations(
        self,
        base_prompt: str,
        doc_type: str,
        current_output: str,
        groundtruth: str,
    ) -> List[str]:
        """Generate prompt variations using each meta-prompting strategy."""
        return [
            clarity_strategy(base_prompt),
            best_practice_strategy(base_prompt, doc_type),
            fewshot_strategy(base_prompt, doc_type),
            no_change_strategy(base_prompt),
            LLm_feedback_strategy(base_prompt, current_output, groundtruth),
        ]

    # ------------------------------------------------------------------
    def optimize_prompt(
        self,
        base_prompt: str,
        doc_type: str,
        current_output: str = None,
        groundtruth: str = None,
    ) -> Dict:
        """
        Select and combine the best meta-prompting strategy via RL.

        Returns a dict with keys:
          - optimized_prompt  (str)
          - base_variations   (list[str])
          - selection_metadata (dict)
        """
        variations = self.generate_strategy_variations(
            base_prompt, doc_type, current_output or "", groundtruth or ""
        )
        if not variations:
            variations = [base_prompt]

        try:
            chain_inputs = {
                "doc_type":       doc_type,
                "current_output": current_output or "",
                "groundtruth":    groundtruth or "",
                "prompt_strategy": variations,
            }
            chain_response = self.chain.invoke(chain_inputs)

            if isinstance(chain_response, dict) and "text" in chain_response:
                selected_prompt = chain_response["text"]
            else:
                selected_prompt = variations[0]
                print(f"Warning: Unexpected chain response type: {type(chain_response)}")

            selected_index = (
                variations.index(selected_prompt)
                if selected_prompt in variations
                else 0
            )

            selection_metadata = {
                "context": {
                    "doc_type":       doc_type,
                    "current_output": current_output or "",
                    "groundtruth":    groundtruth or "",
                },
                "to_select_from": {"prompt_strategy": variations},
                "selected": {
                    "index": selected_index,
                    "value": selected_prompt,
                    "score": None,
                },
            }
            response = {"response": selected_prompt, "selection_metadata": selection_metadata}

        except Exception as e:
            print(f"Warning: RL chain failed ({e}); falling back to first variation.")
            response = {
                "response": variations[0],
                "selection_metadata": {
                    "context": {"doc_type": doc_type, "current_output": current_output or "",
                                "groundtruth": groundtruth or ""},
                    "to_select_from": {"prompt_strategy": variations},
                    "selected": {"index": 0, "value": variations[0], "score": None},
                },
            }

        if self.model_save_dir:
            try:
                self.chain.save_progress()
            except Exception as e:
                print(f"Warning: Failed to save RL progress: {e}")

        return {
            "optimized_prompt":   response["response"],
            "base_variations":    variations,
            "selection_metadata": response["selection_metadata"],
        }

    # ------------------------------------------------------------------
    def update_with_results(
        self,
        optimization_response: Dict,
        extraction_success: float,
    ) -> None:
        """Update the VowpalWabbit policy with delayed reward feedback."""
        selection_metadata = optimization_response.get("selection_metadata")
        if selection_metadata is None:
            print("Warning: No selection metadata – skipping policy update.")
            return

        try:
            score = max(0.0, min(1.0, float(extraction_success)))
            if "selected" in selection_metadata:
                selection_metadata["selected"]["score"] = score

            chain_response = {
                "selection_metadata": selection_metadata,
                "text": optimization_response.get(
                    "optimized_prompt", selection_metadata["selected"]["value"]
                ),
            }

            try:
                self.chain.update_with_delayed_score(
                    score=score,
                    chain_response=chain_response,
                    force_score=True,
                )
            except AttributeError as e:
                print(f"Warning: update_with_delayed_score not available: {e}")
                if hasattr(self.chain, "update_policy"):
                    self.chain.update_policy(chain_response, score)
            except Exception as e:
                print(f"Warning: Failed to update policy: {e}")

            if self.model_save_dir:
                self.chain.save_progress()

        except Exception as e:
            print(f"Warning: Failed to process policy update: {e}")
