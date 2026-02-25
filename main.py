import sys
import os
import json
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import time
import logging
from multiprocessing import Pool, cpu_count

from src.models.local_llm import _get_llm
from src.utils.read_data_utils import DocumentReader
from src.utils.LLM_utils import get_completion_gpt4
from src.utils.load_baseprompts_utils import load_prompt_from_file
from src.utils.jsonparser_utils import clean_llm_output, json_to_dataframe
from src.actor_agents.document_classifier import classify_document_with_llm
from src.environments.schema_builder_env import SchemaBuilderEnv
from src.rl_agents.gymnasium_schemabuilder_agent import GymnasiumAgent as SchemaAgent
from src.utils.parallel_processing import process_single_page
from src.utils.logging_utils import setup_logging
from src.utils.cache_utils import cache_results


def update_metrics_excel(metrics_dict: dict, excel_path: str = "output/metrics/extraction_metrics.xlsx"):
    """Update or create Excel file with document processing metrics."""
    try:
        df_existing = pd.read_excel(excel_path)
        df_new = pd.DataFrame([metrics_dict])
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_updated = pd.DataFrame([metrics_dict])

    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    df_updated.to_excel(excel_path, index=False)
    logging.info(f"Metrics updated in: {excel_path}")


@cache_results
def process_document(
    file_path: str,
    extraction_groundtruth: dict,
    output_dir: str = None,
    schema_groundtruth: dict = None,
    max_workers: int = None,
    max_steps: int = 5,
    llm_choice: str = "local",
    force: bool = False,
    use_vlm: bool = False,
) -> dict:
    """
    Process a document through the complete ADE pipeline.

    Args:
        file_path:              Path to the input document.
        extraction_groundtruth: Groundtruth dict for data extraction.
        output_dir:             Directory to save results.
        schema_groundtruth:     Optional groundtruth for schema building.
        max_workers:            Max parallel workers (default: cpu count).
        max_steps:              Max RL steps per phase.
        llm_choice:             Ignored; local Qwen model always used.
        force:                  Force reprocessing even if cached.
        use_vlm:                Use olmOCR VLM for text extraction (PDF only).
    """
    if output_dir:
        os.makedirs(os.path.join(output_dir, "extracted_data"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "unknown_docs"), exist_ok=True)

    start_time = time.time()

    metrics = {
        "File_Path": file_path,
        "File_Name": os.path.basename(file_path),
        "Processing_Start_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Number_of_Pages": 0,
        "Document_Type": None,
        "Classification_Confidence": 0,
        "Best_Perplexity_Score": None,
        "Best_Complexity_Score": None,
        "Schema_Steps": 0,
        "Best_Exact_Match": None,
        "Best_Semantic_Match": None,
        "Best_Similarity": None,
        "Extraction_Steps": 0,
        "Schema_Groundtruth_Used": bool(schema_groundtruth),
        "Extraction_Groundtruth_Used": bool(extraction_groundtruth),
        "Total_Processing_Time": None,
        "Reading_Time": None,
        "Classification_Time": None,
        "Schema_Building_Time": None,
        "Extraction_Time": None,
        "llm_choice": "local",
        "vlm_used": use_vlm,
    }

    reader = DocumentReader()
    chat_model = _get_llm()   # local Qwen model

    logging.info(f"Processing document: {file_path}")

    # ------------------------------------------------------------------
    # 1. Read document
    # ------------------------------------------------------------------
    reading_start = time.time()
    try:
        file_extension = os.path.splitext(file_path)[1].lower()

        if use_vlm and file_extension == ".pdf":
            # VLM pathway: PDF → images → olmOCR → text
            from src.models.vlm_extractor import VLMExtractor, pdf_to_images
            images = pdf_to_images(file_path)
            vlm = VLMExtractor()
            pages_text = vlm.extract_from_pages(images)
            document_text = "\n\n".join(pages_text)
            metrics["Number_of_Pages"] = len(pages_text)
            logging.info(f"VLM extracted {len(pages_text)} pages")

        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                document_text = f.read()
            pages_text = [document_text]
            metrics["Number_of_Pages"] = 1

        elif reader.is_image_file(file_path):
            result = reader.read_image(file_path)
            document_text = result["text"]
            pages_text = [document_text]
            metrics["Number_of_Pages"] = 1
            logging.info(
                f"Image processed with confidence: "
                f"{sum(result['confidence_scores'])/len(result['confidence_scores']):.2f}"
            )
        else:
            result = reader.read_document(file_path)
            document_text = result["text"]
            pages_text = result["pages"]
            metrics["Number_of_Pages"] = result["num_pages"]

        metrics["Reading_Time"] = round(time.time() - reading_start, 2)
        logging.info(f"Successfully read document with {metrics['Number_of_Pages']} pages")

    except Exception as e:
        metrics["Reading_Time"] = round(time.time() - reading_start, 2)
        logging.error(f"Error reading document: {str(e)}")
        raise

    # ------------------------------------------------------------------
    # 2. Classify document
    # ------------------------------------------------------------------
    classification_start = time.time()
    try:
        doc_type, confidence = classify_document_with_llm(pages_text[0], llm_choice)
        metrics["Document_Type"] = doc_type
        metrics["Classification_Confidence"] = confidence
        logging.info(f"Document classified as: {doc_type} (confidence: {confidence}%)")

        if doc_type == "Unknown":
            unknown_dir = os.path.join(output_dir or "output", "unknown_docs")
            os.makedirs(unknown_dir, exist_ok=True)
            shutil.copy2(file_path, os.path.join(unknown_dir, os.path.basename(file_path)))
            logging.warning(f"Unrecognized document type. File copied to: {unknown_dir}")
            update_metrics_excel(metrics)
            return {
                "document_type": "Unknown",
                "confidence": confidence,
                "schema": None,
                "extracted_data": None,
                "num_pages": metrics["Number_of_Pages"],
            }

        metrics["Classification_Time"] = round(time.time() - classification_start, 2)

    except Exception as e:
        metrics["Classification_Time"] = round(time.time() - classification_start, 2)
        logging.error(f"Error classifying document: {str(e)}")
        raise

    # ------------------------------------------------------------------
    # 3. Build and optimise schema
    # ------------------------------------------------------------------
    schema_start = time.time()
    try:
        schema_prompt = load_prompt_from_file(filename="schema_builder_prompt.txt")
        schema_env = SchemaBuilderEnv(
            baseprompt=schema_prompt,
            document_text=pages_text[0],
            groundtruth=schema_groundtruth,
            llm_choice=llm_choice,
            max_steps=max_steps,
        )
        schema_agent = SchemaAgent(chat_model, schema_env)
        schema_agent.interact()

        schema_results = schema_env.get_best_results()
        best_schema = schema_results["best_schema"]
        metrics["Best_Perplexity_Score"] = schema_results["best_perplexity"]
        metrics["Best_Complexity_Score"] = schema_results["best_complexity"]
        metrics["Schema_Steps"] = schema_env.current_step

        if schema_groundtruth:
            metrics["Schema_Match_Score"] = schema_results.get("groundtruth_match_score")

        metrics["Schema_Building_Time"] = round(time.time() - schema_start, 2)
        logging.info("Schema created and optimised")

    except Exception as e:
        metrics["Schema_Building_Time"] = round(time.time() - schema_start, 2)
        logging.error(f"Error building schema: {str(e)}")
        raise

    # ------------------------------------------------------------------
    # 4. Parallel data extraction
    # ------------------------------------------------------------------
    extraction_start = time.time()
    try:
        extraction_prompt = load_prompt_from_file(document_type=doc_type)

        process_args = [
            (page_text, doc_type, extraction_prompt, best_schema,
             extraction_groundtruth, idx, len(pages_text), max_steps, llm_choice)
            for idx, page_text in enumerate(pages_text)
        ]

        try:
            available_cores = cpu_count()
        except Exception:
            available_cores = 2

        n_workers = min(max_workers or available_cores, len(pages_text))
        logging.info(f"\nStarting parallel processing with {n_workers} workers")

        with Pool(processes=n_workers) as pool:
            page_results = pool.map(process_single_page, process_args)

        combined_results = []
        max_extraction_steps = 0
        best_exact_match = best_semantic_match = best_similarity = 0

        for result in sorted(page_results, key=lambda x: x["page_num"]):
            page_data = result["results"]
            best_exact_match = max(best_exact_match, page_data["best_exact_match"])
            best_semantic_match = max(best_semantic_match, page_data["best_semantic_match"])
            best_similarity = max(best_similarity, page_data["best_similarity"])
            max_extraction_steps = max(max_extraction_steps, result["steps"])

            if isinstance(page_data["best_output"], str):
                try:
                    clean_llm_output(page_data["best_output"])
                    parsed_data = json.loads(page_data["best_output"].strip())
                    combined_results.append(parsed_data)
                except json.JSONDecodeError:
                    logging.warning(f"Could not parse JSON from page {result['page_num'] + 1}")
                    continue
            else:
                combined_results.append(page_data["best_output"])

        metrics["Best_Exact_Match"] = best_exact_match
        metrics["Best_Semantic_Match"] = best_semantic_match
        metrics["Best_Similarity"] = best_similarity
        metrics["Extraction_Steps"] = max_extraction_steps

        # Merge multi-page results
        final_results: dict = {}
        for page_result in combined_results:
            if not isinstance(page_result, dict):
                continue
            for key, value in page_result.items():
                if key not in final_results:
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        final_results[key] = []
                    else:
                        final_results[key] = None

                if isinstance(value, list) and value and isinstance(value[0], dict):
                    final_results[key].extend(value)
                else:
                    if isinstance(value, list):
                        value = value[0] if value else None
                    if final_results[key] is None:
                        final_results[key] = value

        logging.info("\nData extraction completed")

        if output_dir:
            extracted_data_dir = os.path.join(output_dir, "extracted_data")
            os.makedirs(extracted_data_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(file_path))[0]
            output_file = os.path.join(extracted_data_dir, f"{base}_local_extracted.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_results, f, indent=2)
            logging.info(f"Results saved to: {output_file}")

        metrics["Extraction_Time"] = round(time.time() - extraction_start, 2)
        metrics["Total_Processing_Time"] = round(time.time() - start_time, 2)
        metrics["Processing_End_Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if metrics["Number_of_Pages"] > 0:
            metrics["Average_Time_Per_Page"] = round(
                metrics["Extraction_Time"] / metrics["Number_of_Pages"], 2
            )

        update_metrics_excel(metrics)

        return {
            "document_type": doc_type,
            "confidence": confidence,
            "schema": best_schema,
            "extracted_data": final_results,
            "num_pages": metrics["Number_of_Pages"],
            "processing_times": {
                "total_time": metrics["Total_Processing_Time"],
                "reading_time": metrics["Reading_Time"],
                "classification_time": metrics["Classification_Time"],
                "schema_time": metrics["Schema_Building_Time"],
                "extraction_time": metrics["Extraction_Time"],
                "avg_time_per_page": metrics.get("Average_Time_Per_Page"),
            },
        }

    except Exception as e:
        metrics["Extraction_Time"] = round(time.time() - extraction_start, 2)
        metrics["Total_Processing_Time"] = round(time.time() - start_time, 2)
        metrics["Processing_End_Time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_metrics_excel(metrics)
        logging.error(f"Error in data extraction: {str(e)}")
        raise


def get_matching_groundtruth(input_file: str, groundtruth_dir: str) -> dict:
    """Find and load matching groundtruth file for a given input file."""
    if not groundtruth_dir or not isinstance(groundtruth_dir, str) or not os.path.isdir(groundtruth_dir):
        return None

    input_basename = os.path.splitext(os.path.basename(input_file))[0]

    for ext in [".json", ".txt"]:
        groundtruth_path = os.path.join(groundtruth_dir, input_basename + ext)
        if os.path.exists(groundtruth_path):
            try:
                with open(groundtruth_path, "r") as f:
                    if ext == ".json":
                        return json.load(f)
                    content = f.read().strip()
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return content
            except Exception as e:
                logging.warning(f"Error loading groundtruth {groundtruth_path}: {e}")
                return None

    logging.warning(f"No matching groundtruth file found for {input_basename}")
    return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ADE — Agentic Document Processing Pipeline (local VLM+LLM)"
    )
    parser.add_argument("input_path", help="Path to input document or directory")
    parser.add_argument("--output-dir", default="output", help="Directory to save results")
    parser.add_argument("--schema-groundtruth", help="Path to schema groundtruth JSON file")
    parser.add_argument("--extraction-groundtruth", help="Path to extraction groundtruth JSON file")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel workers (default: CPU count)")
    parser.add_argument("--max-steps", type=int, default=3,
                        help="Max RL steps before terminating (default: 3)")
    parser.add_argument("--use-vlm", action="store_true",
                        help="Use olmOCR VLM for PDF text extraction")
    parser.add_argument("--force", action="store_true",
                        help="Force reprocessing even if cache exists")
    args = parser.parse_args()

    try:
        log_file = setup_logging(args.output_dir, args.input_path)
        logging.info("Starting ADE pipeline (local VLM+LLM)")
        logging.info(f"Input: {args.input_path}")
        logging.info(f"Output: {args.output_dir}")

        if os.path.isfile(args.input_path):
            schema_gt = get_matching_groundtruth(args.input_path, args.schema_groundtruth) if args.schema_groundtruth else None
            extraction_gt = get_matching_groundtruth(args.input_path, args.extraction_groundtruth) if args.extraction_groundtruth else None

            try:
                result = process_document(
                    args.input_path,
                    extraction_gt,
                    args.output_dir,
                    schema_gt,
                    args.max_workers,
                    args.max_steps,
                    "local",
                    args.force,
                    args.use_vlm,
                )
                logging.info("\nProcessing complete!")
                logging.info(f"Document Type: {result['document_type']}")
                logging.info(f"Number of Pages: {result['num_pages']}")
                print("Extracted Data:", json.dumps(result["extracted_data"], indent=2))
            finally:
                for handler in logging.root.handlers[:]:
                    handler.close()
                    logging.root.removeHandler(handler)

        elif os.path.isdir(args.input_path):
            logging.info(f"Processing directory: {args.input_path}")
            for handler in logging.root.handlers[:]:
                handler.close()
                logging.root.removeHandler(handler)

            for file in os.listdir(args.input_path):
                file_path = os.path.join(args.input_path, file)
                if not os.path.isfile(file_path):
                    continue
                try:
                    log_file = setup_logging(args.output_dir, file_path)
                    logging.info(f"Processing: {file}")

                    s_gt = get_matching_groundtruth(file_path, args.schema_groundtruth) if args.schema_groundtruth else None
                    e_gt = get_matching_groundtruth(file_path, args.extraction_groundtruth) if args.extraction_groundtruth else None

                    result = process_document(
                        file_path, e_gt, args.output_dir, s_gt,
                        args.max_workers, args.max_steps, "local", args.force, args.use_vlm,
                    )
                    logging.info(f"Successfully processed {file}")
                except Exception as e:
                    logging.error(f"Error processing {file}: {str(e)}")
                finally:
                    for handler in logging.root.handlers[:]:
                        handler.close()
                        logging.root.removeHandler(handler)
        else:
            logging.error("Invalid input path")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise
