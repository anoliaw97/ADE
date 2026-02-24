"""
ADE – Agentic Document Extraction  |  Gradio GUI
=================================================
Launches a web-based interface for the full ADE pipeline.

Usage:
    python gui.py
    python gui.py --port 7860 --share          # public tunnel via Gradio share
    python gui.py --host 0.0.0.0 --port 8080   # custom host/port
    python gui.py --lab                        # open GUI in lab-report mode
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

import gradio as gr

# Make sure project root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Lazy pipeline import (lets the GUI start even if heavy deps are missing)
# ---------------------------------------------------------------------------

def _import_pipeline():
    from main import process_document, get_matching_groundtruth
    return process_document, get_matching_groundtruth


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class _QueueHandler(logging.Handler):
    """Forwards log records to a thread-safe queue for the GUI to poll."""

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                            datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord):
        try:
            self.log_queue.put_nowait(self.format(record))
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Ollama connectivity check
# ---------------------------------------------------------------------------

def check_ollama(base_url: str, model: str) -> str:
    """Return a human-readable status string for the Ollama connection."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key="ollama")
        # Minimal call: list models
        models = client.models.list()
        names = [m.id for m in models.data]
        available = ", ".join(names) if names else "(none)"
        if model in names:
            return f"Connected. Model **{model}** is available.  All models: {available}"
        else:
            return (f"Connected, but model **{model}** was NOT found.\n"
                    f"Available: {available}\n\n"
                    f"Run:  `ollama pull {model}`")
    except Exception as exc:
        return (f"Cannot reach Ollama at `{base_url}`.\n\n"
                f"Error: {exc}\n\n"
                "Make sure Ollama is running:  `ollama serve`")


# ---------------------------------------------------------------------------
# Core pipeline wrapper
# ---------------------------------------------------------------------------

def load_default_prompt(doc_type: str) -> str:
    """Return the default extraction prompt text for the chosen document type."""
    try:
        from src.utils.load_baseprompts_utils import load_prompt_from_file
        if not doc_type or doc_type == "— auto-detect —":
            return ""
        return load_prompt_from_file(document_type=doc_type)
    except Exception as exc:
        return f"# Could not load prompt: {exc}"


def run_pipeline(
    uploaded_files,          # list[tempfile paths] from gr.Files
    model_name: str,
    max_steps: int,
    max_workers_val: int,
    llm_choice: str,
    schema_gt_dir: str,
    extraction_gt_dir: str,
    force_reprocess: bool,
    output_dir: str,
    doc_type_override: str,
    custom_prompt: str,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Execute the ADE pipeline on *uploaded_files* and stream results back.

    Yields tuples of (log_text, results_json, metrics_text, download_path).
    """
    if not uploaded_files:
        yield "No files uploaded.", "{}", "—", None
        return

    # Apply model override via environment variable
    os.environ["OLLAMA_MODEL"] = model_name.strip() or "llama3.2"

    # Prepare output directory
    output_dir = output_dir.strip() or "output"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "extracted_data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    # Set up a log queue so we can surface pipeline messages in the UI
    log_queue: queue.Queue = queue.Queue(maxsize=500)
    queue_handler = _QueueHandler(log_queue)
    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO)

    log_lines: list[str] = []

    def drain_log() -> str:
        """Pull all pending log messages into log_lines and return joined text."""
        while True:
            try:
                msg = log_queue.get_nowait()
                log_lines.append(msg)
            except queue.Empty:
                break
        return "\n".join(log_lines[-200:])   # keep last 200 lines visible

    try:
        process_document, get_matching_groundtruth = _import_pipeline()
    except ImportError as exc:
        yield f"Import error: {exc}", "{}", "—", None
        root_logger.removeHandler(queue_handler)
        return

    all_results   = []
    metrics_rows  = []
    n_files       = len(uploaded_files)

    for idx, file_obj in enumerate(uploaded_files):
        # Gradio passes either a string path or an object with a .name attribute
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name
        fname     = os.path.basename(file_path)

        progress((idx) / n_files, desc=f"Processing {fname} …")
        log_lines.append(f"\n{'='*60}")
        log_lines.append(f"[{idx+1}/{n_files}] Processing: {fname}")
        log_lines.append(f"{'='*60}")
        yield drain_log(), "{}", "Processing …", None

        schema_gt     = None
        extraction_gt = None
        try:
            if schema_gt_dir.strip():
                schema_gt = get_matching_groundtruth(file_path, schema_gt_dir.strip())
            if extraction_gt_dir.strip():
                extraction_gt = get_matching_groundtruth(file_path, extraction_gt_dir.strip())
        except Exception:
            pass

        # Resolve custom prompt / doc-type override
        _custom_prompt   = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else None
        _doc_type_override = (doc_type_override if doc_type_override
                              and doc_type_override != "— auto-detect —" else None)

        try:
            result = process_document(
                file_path=file_path,
                extraction_groundtruth=extraction_gt,
                output_dir=output_dir,
                schema_groundtruth=schema_gt,
                max_workers=max_workers_val or None,
                max_steps=max_steps,
                llm_choice=llm_choice,
                force=force_reprocess,
                custom_extraction_prompt=_custom_prompt,
                doc_type_override=_doc_type_override,
            )

            all_results.append({"file": fname, "status": "success", "result": result})

            metrics_rows.append({
                "File":            fname,
                "Doc Type":        result.get("document_type", "—"),
                "Confidence":      f"{result.get('confidence', 0):.1f}%",
                "Pages":           result.get("num_pages", "—"),
                "Total Time (s)":  result.get("processing_times", {}).get("total_time", "—"),
                "Exact Match":     result.get("processing_times", {}) and "—",   # placeholder
            })

            log_lines.append(f"  ✓ Done: {fname}  [{result.get('document_type')}]")

        except Exception as exc:
            all_results.append({"file": fname, "status": "error", "error": str(exc)})
            log_lines.append(f"  ✗ Error: {fname}: {exc}")

        yield drain_log(), json.dumps(all_results, indent=2, default=str), _fmt_metrics(metrics_rows), None

    progress(1.0, desc="Done")

    # Write combined results JSON to a temp file for download
    download_path = None
    if all_results:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".json",
            prefix="ade_results_", dir=tempfile.gettempdir()
        )
        tmp.write(json.dumps(all_results, indent=2, default=str).encode())
        tmp.close()
        download_path = tmp.name

    root_logger.removeHandler(queue_handler)

    yield drain_log(), json.dumps(all_results, indent=2, default=str), _fmt_metrics(metrics_rows), download_path


# ---------------------------------------------------------------------------
# Lab report table / figure renderer
# ---------------------------------------------------------------------------

LAB_SUBCATEGORIES = [
    "Chemical Analysis Report",
    "Environmental Analysis Report",
    "Microbiology Report",
    "Material Testing Report",
    "Clinical Laboratory Report",
    "Geotechnical Report",
    "Food Analysis Report",
    "General Laboratory Report",
    "Laboratory Report",   # generic alias kept for backward compat
]

ALL_DOC_TYPES = [
    "— auto-detect —",
    # Non-lab
    "Invoice", "Purchase Order", "Utility Bill",
    "Receipt", "Financial Document", "Salary Slip",
    # Lab subcategories
    *LAB_SUBCATEGORIES,
]


def render_lab_tables(json_text: str):
    """
    Parse extracted JSON and return:
    - A list of (table_name, pd.DataFrame) for any embedded tables, OR
    - A plain text summary when no tables are found.

    Returns (status_md, list_of_gr_Dataframe_data)
    """
    import pandas as pd

    if not json_text or not json_text.strip():
        return "No JSON provided.", []

    try:
        data = json.loads(json_text) if isinstance(json_text, str) else json_text
    except json.JSONDecodeError as exc:
        return f"JSON parse error: {exc}", []

    if isinstance(data, list) and data and isinstance(data[0], dict) and "result" in data[0]:
        # gui result wrapper: [{file, status, result}, ...]
        data = data[0].get("result", data)

    tables_found = []

    def _try_table(obj, name=""):
        """Recursively hunt for list-of-dicts (tabular data)."""
        if isinstance(obj, list) and obj and all(isinstance(r, dict) for r in obj):
            try:
                df = pd.DataFrame(obj)
                tables_found.append((name, df))
                return
            except Exception:
                pass
        if isinstance(obj, dict):
            for key, val in obj.items():
                _try_table(val, name=key if not name else f"{name} › {key}")

    _try_table(data)

    if not tables_found:
        # no_table flag or genuinely empty
        if isinstance(data, dict) and data.get("no_table"):
            return "Document contains **no table** (the model returned `{\"no_table\": true}`).", []
        return "No tabular data found in the extracted JSON.", []

    status = f"Found **{len(tables_found)}** table(s) in the extracted data."
    # Return as list of (label, list-of-lists) for gr.Dataframe
    frames = [(lbl, df.values.tolist(), list(df.columns)) for lbl, df in tables_found]
    return status, frames


def _fmt_metrics(rows: list[dict]) -> str:
    """Format metrics rows as a simple text table."""
    if not rows:
        return "No results yet."
    header = ["File", "Doc Type", "Confidence", "Pages", "Total Time (s)"]
    lines  = ["  |  ".join(header), "-" * 80]
    for row in rows:
        lines.append("  |  ".join(str(row.get(h, "—")) for h in header))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick single-document preview (no RL, direct extraction)
# ---------------------------------------------------------------------------

def quick_extract(file_obj, model_name: str, progress=gr.Progress()):
    """
    Fast single-pass extraction without the RL optimisation loop.
    Useful for rapid testing / preview.
    """
    if file_obj is None:
        return "No file uploaded.", "{}"

    file_path = file_obj if isinstance(file_obj, str) else file_obj.name
    os.environ["OLLAMA_MODEL"] = model_name.strip() or "llama3.2"

    progress(0.1, desc="Reading document …")
    try:
        from src.utils.read_data_utils import DocumentReader
        from src.actor_agents.document_classifier import classify_document_with_llm
        from src.utils.load_baseprompts_utils import load_prompt_from_file
        from src.actor_agents.document_extractor import document_extractor_agent
        from src.utils.LLM_utils import get_llm_completion
        from src.utils.jsonparser_utils import clean_llm_output

        reader = DocumentReader()
        ext    = os.path.splitext(file_path)[1].lower()

        if ext == ".txt":
            with open(file_path, encoding="utf-8") as fh:
                text = fh.read()
        elif reader.is_image_file(file_path):
            text = reader.read_image(file_path)["text"]
        else:
            text = reader.read_document(file_path)["text"]

        progress(0.3, desc="Classifying document …")
        doc_type, confidence = classify_document_with_llm(text, "ollama")

        progress(0.5, desc="Building schema …")
        from src.actor_agents.schema_builder import schema_building_with_llm
        schema_prompt = load_prompt_from_file(filename="schema_builder_prompt.txt")
        _, schema, _ = schema_building_with_llm(schema_prompt, text[:4000], "ollama")

        progress(0.7, desc="Extracting data …")
        extraction_prompt = load_prompt_from_file(document_type=doc_type)
        full_prompt = document_extractor_agent(extraction_prompt, doc_type, text, schema)
        raw = get_llm_completion(
            [{"role": "user", "content": full_prompt}],
            llm_choice="ollama",
            response_format={"type": "json_object"},
        ).choices[0].message.content
        extracted = clean_llm_output(raw)

        progress(1.0, desc="Done")
        status = f"Document Type: **{doc_type}**  |  Confidence: {confidence:.1f}%"
        return status, json.dumps(json.loads(extracted) if isinstance(extracted, str) else extracted,
                                  indent=2, default=str)

    except Exception as exc:
        return f"Error: {exc}", "{}"


# ---------------------------------------------------------------------------
# Results browser
# ---------------------------------------------------------------------------

def browse_results(output_dir: str):
    """Return a formatted list of previously extracted JSON files."""
    out_path = Path(output_dir.strip() or "output") / "extracted_data"
    if not out_path.exists():
        return "No results directory found.", "{}"

    files = sorted(out_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return "No extracted results found.", "{}"

    file_list = "\n".join(f"  • {f.name}  ({f.stat().st_size} bytes)" for f in files[:50])
    return file_list, ""


def load_result_file(output_dir: str, filename: str):
    """Load and display a specific result file."""
    path = Path(output_dir.strip() or "output") / "extracted_data" / filename.strip()
    if not path.exists():
        return f"File not found: {path}"
    try:
        return json.dumps(json.loads(path.read_text()), indent=2)
    except Exception as exc:
        return f"Error loading file: {exc}"


# ---------------------------------------------------------------------------
# Gradio UI definition
# ---------------------------------------------------------------------------

DESCRIPTION = """
# ADE — Agentic Document Extraction

**Free, local AI pipeline** for extracting structured data from documents.

| Component | Technology |
|-----------|-----------|
| LLM | [Ollama](https://ollama.com) (local, free) |
| Embeddings | [SentenceTransformers](https://sbert.net) (local, free) |
| RL Optimization | [VowpalWabbit](https://vowpalwabbit.org) contextual bandits |
| OCR | PaddleOCR |

> Make sure Ollama is running (`ollama serve`) and a model is pulled (`ollama pull llama3.2`) before processing.
"""

LAB_DESCRIPTION = """
# ADE — Scientific Laboratory Report Extraction

**Local AI pipeline** specialised for extracting structured data from scientific lab reports.

Supported subcategories:
`Chemical Analysis` · `Environmental` · `Microbiology` · `Material Testing` ·
`Clinical` · `Geotechnical` · `Food Analysis` · `General Lab Report`

Extracts: **tables** (with detection limits, uncertainty, QC), **figures** (with equations / R²),
and **narrative conclusions** into structured JSON.

> Make sure Ollama is running (`ollama serve`) before processing.
"""

def build_ui(lab_mode: bool = False) -> gr.Blocks:
    with gr.Blocks(
        title="ADE – Agentic Document Extraction",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="""
        .result-box { font-family: monospace; font-size: 0.85em; }
        .log-box    { font-family: monospace; font-size: 0.8em; background: #1e1e2e; color: #cdd6f4; }
        """,
    ) as demo:

        gr.Markdown(LAB_DESCRIPTION if lab_mode else DESCRIPTION)

        # ── TABS ─────────────────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: Full Pipeline ─────────────────────────────────────
            with gr.TabItem("Process Documents"):
                with gr.Row():
                    # Left column – inputs
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Documents")
                        files_input = gr.Files(
                            label="Upload Documents",
                            file_types=[".pdf", ".docx", ".doc", ".png",
                                        ".jpg", ".jpeg", ".tiff", ".bmp", ".txt"],
                            file_count="multiple",
                        )

                        gr.Markdown("### LLM Settings")
                        with gr.Row():
                            model_input = gr.Textbox(
                                label="Ollama Model",
                                value=os.getenv("OLLAMA_MODEL", "llama3.2"),
                                placeholder="e.g. llama3.2, mistral, phi3",
                            )
                            llm_choice_dd = gr.Dropdown(
                                label="LLM Choice",
                                choices=["ollama", "gpt", "llama"],
                                value="ollama",
                                info="'gpt'/'llama' are legacy aliases, both route to Ollama.",
                            )

                        gr.Markdown("### Processing Options")
                        with gr.Row():
                            max_steps_sl = gr.Slider(
                                label="Max RL Steps", minimum=1, maximum=10,
                                value=3, step=1,
                            )
                            max_workers_sl = gr.Slider(
                                label="Max Workers", minimum=1, maximum=8,
                                value=2, step=1,
                            )
                        force_cb = gr.Checkbox(label="Force reprocess (ignore cache)", value=False)

                        # ── Custom Prompt / Doc-Type Override ────────────
                        _default_doctype = (
                            "General Laboratory Report" if lab_mode else "— auto-detect —"
                        )
                        with gr.Accordion(
                            "Custom Prompt & Document Type Override",
                            open=lab_mode,   # auto-open in lab mode
                        ):
                            gr.Markdown(
                                "Override the automatic classification and/or the default "
                                "extraction prompt.  \n"
                                "**Tip for Lab Reports:** select a lab subcategory below and "
                                "click *Load Default Prompt* to pre-fill the table/graph-focused prompt, "
                                "then edit it as needed."
                            )
                            with gr.Row():
                                doc_type_dd = gr.Dropdown(
                                    label="Override Document Type",
                                    choices=ALL_DOC_TYPES,
                                    value=_default_doctype,
                                    info="Leave on auto-detect unless you want to force a type.",
                                )
                                load_prompt_btn = gr.Button("Load Default Prompt", scale=1)

                            custom_prompt_tb = gr.Textbox(
                                label="Custom Extraction Prompt",
                                placeholder=(
                                    "Leave empty to use the built-in prompt for the selected "
                                    "document type.\n\n"
                                    "Paste or type your custom prompt here to override it.\n\n"
                                    "Example rules for lab reports:\n"
                                    "1. Extract the complete table including all headers, rows, columns.\n"
                                    "2. If table present, extract ALL rows in exact order as JSON array.\n"
                                    "3. Extract data ONLY if table present. If no table: {\"no_table\": true}"
                                ),
                                lines=14,
                                max_lines=30,
                            )

                            # Wire the "Load Default Prompt" button
                            load_prompt_btn.click(
                                fn=load_default_prompt,
                                inputs=[doc_type_dd],
                                outputs=[custom_prompt_tb],
                            )

                        gr.Markdown("### Optional Groundtruth (for evaluation)")
                        schema_gt_tb = gr.Textbox(
                            label="Schema Groundtruth Folder",
                            placeholder="/path/to/schema_gt/",
                        )
                        extraction_gt_tb = gr.Textbox(
                            label="Extraction Groundtruth Folder",
                            placeholder="/path/to/extraction_gt/",
                        )
                        output_dir_tb = gr.Textbox(
                            label="Output Directory",
                            value="output",
                        )

                        with gr.Row():
                            process_btn = gr.Button("Process Documents", variant="primary",
                                                    scale=2)
                            stop_btn    = gr.Button("Stop", variant="stop", scale=1)

                    # Right column – outputs
                    with gr.Column(scale=1):
                        gr.Markdown("### Processing Log")
                        log_out = gr.Textbox(
                            label="Live Log",
                            lines=18,
                            max_lines=18,
                            interactive=False,
                            elem_classes=["log-box"],
                        )

                        gr.Markdown("### Metrics")
                        metrics_out = gr.Textbox(
                            label="",
                            lines=5,
                            interactive=False,
                            elem_classes=["result-box"],
                        )

                        gr.Markdown("### Extracted Results (JSON)")
                        results_out = gr.Code(
                            label="",
                            language="json",
                            lines=14,
                            interactive=False,
                        )

                        download_btn = gr.File(label="Download Results JSON", visible=True)

                # Wire up the process button
                process_event = process_btn.click(
                    fn=run_pipeline,
                    inputs=[
                        files_input, model_input, max_steps_sl, max_workers_sl,
                        llm_choice_dd, schema_gt_tb, extraction_gt_tb,
                        force_cb, output_dir_tb,
                        doc_type_dd, custom_prompt_tb,     # custom prompt / override
                    ],
                    outputs=[log_out, results_out, metrics_out, download_btn],
                )
                stop_btn.click(fn=None, cancels=[process_event])

            # ── Tab 2: Quick Extract (no RL) ─────────────────────────────
            with gr.TabItem("Quick Extract"):
                gr.Markdown(
                    "Single-pass extraction **without** the RL optimisation loop.  "
                    "Faster, but results may be less refined."
                )
                with gr.Row():
                    with gr.Column():
                        qe_file    = gr.File(label="Upload Document", file_count="single")
                        qe_model   = gr.Textbox(
                            label="Ollama Model",
                            value=os.getenv("OLLAMA_MODEL", "llama3.2"),
                        )
                        qe_btn     = gr.Button("Extract", variant="primary")
                    with gr.Column():
                        qe_status  = gr.Markdown("—")
                        qe_results = gr.Code(label="Extracted JSON", language="json", lines=20)

                qe_btn.click(
                    fn=quick_extract,
                    inputs=[qe_file, qe_model],
                    outputs=[qe_status, qe_results],
                )

            # ── Tab 3: Results Browser ───────────────────────────────────
            with gr.TabItem("Results Browser"):
                gr.Markdown("Browse previously extracted results stored in the output directory.")
                with gr.Row():
                    rb_outdir = gr.Textbox(label="Output Directory", value="output")
                    rb_browse = gr.Button("Refresh List", variant="secondary")

                with gr.Row():
                    with gr.Column(scale=1):
                        rb_filelist = gr.Textbox(
                            label="Available Result Files",
                            lines=15,
                            interactive=False,
                            elem_classes=["result-box"],
                        )
                    with gr.Column(scale=1):
                        rb_filename = gr.Textbox(
                            label="File name to load",
                            placeholder="invoice_ollama_extracted.json",
                        )
                        rb_load     = gr.Button("Load File", variant="secondary")
                        rb_content  = gr.Code(
                            label="File Contents",
                            language="json",
                            lines=15,
                            interactive=False,
                        )

                rb_browse.click(
                    fn=browse_results,
                    inputs=[rb_outdir],
                    outputs=[rb_filelist, rb_content],
                )
                rb_load.click(
                    fn=load_result_file,
                    inputs=[rb_outdir, rb_filename],
                    outputs=[rb_content],
                )

            # ── Tab 4: Lab Table Viewer ──────────────────────────────────
            with gr.TabItem("Lab Table Viewer"):
                gr.Markdown(
                    "Paste extracted JSON (from the **Process Documents** tab or a saved file) "
                    "to visualise any embedded data tables."
                )
                with gr.Row():
                    ltv_json_in = gr.Textbox(
                        label="Extracted JSON",
                        lines=12,
                        placeholder='{"results_table": [{"Sample": "A1", "pH": 7.4}, ...]}'
                    )
                    ltv_btn = gr.Button("Render Tables", variant="primary", scale=0)

                ltv_status = gr.Markdown("—")

                # We render up to 5 tables dynamically
                _MAX_TABLES = 5
                ltv_frames = []
                for _i in range(_MAX_TABLES):
                    with gr.Group(visible=False) as _grp:
                        _lbl  = gr.Markdown(f"**Table {_i + 1}**")
                        _df   = gr.Dataframe(interactive=False, wrap=True)
                        ltv_frames.append((_grp, _lbl, _df))

                def _render_tables(json_text):
                    status_md, tables = render_lab_tables(json_text)
                    updates = [gr.update(value=status_md)]  # ltv_status
                    for i in range(_MAX_TABLES):
                        if i < len(tables):
                            lbl_txt, rows, cols = tables[i]
                            updates += [
                                gr.update(visible=True),           # group
                                gr.update(value=f"**{lbl_txt}**"), # label
                                gr.update(value=rows, headers=cols, visible=True),  # dataframe
                            ]
                        else:
                            updates += [
                                gr.update(visible=False),
                                gr.update(value=""),
                                gr.update(visible=False),
                            ]
                    return updates

                _render_outputs = [ltv_status]
                for grp, lbl, df in ltv_frames:
                    _render_outputs += [grp, lbl, df]

                ltv_btn.click(fn=_render_tables, inputs=[ltv_json_in], outputs=_render_outputs)

            # ── Tab 5: Ollama Health Check ───────────────────────────────
            with gr.TabItem("Ollama Status"):
                gr.Markdown("Check that the local Ollama server is reachable and the chosen model is available.")
                with gr.Row():
                    hc_url   = gr.Textbox(
                        label="Ollama Base URL",
                        value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                    )
                    hc_model = gr.Textbox(
                        label="Model Name",
                        value=os.getenv("OLLAMA_MODEL", "llama3.2"),
                    )
                hc_btn    = gr.Button("Test Connection", variant="primary")
                hc_status = gr.Markdown("—")

                hc_btn.click(
                    fn=check_ollama,
                    inputs=[hc_url, hc_model],
                    outputs=[hc_status],
                )

            # ── Tab 6: Help ──────────────────────────────────────────────
            with gr.TabItem("Help"):
                gr.Markdown("""
## Quick Start

### 1. Install Ollama
```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Then pull a model (choose one):
ollama pull llama3.2       # recommended – fast, good quality
ollama pull mistral        # alternative
ollama pull phi3           # lightweight
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the GUI
```bash
# General mode
python gui.py

# Scientific lab report mode (pre-sets lab defaults)
python gui.py --lab
```

### 4. CLI usage (without GUI)
```bash
# Single file (auto-detect type)
python main.py /path/to/document.pdf --output-dir output --max-steps 3

# Directory of files
python main.py /path/to/docs/ --output-dir output --llm-choice ollama

# Lab report mode — forces lab classification
python main.py /path/to/report.pdf --mode lab

# Lab baseline mode — fast single-pass extraction (no RL optimisation)
python main.py /path/to/report.pdf --mode lab-baseline

# Force a specific lab subcategory
python main.py /path/to/report.pdf --mode lab --lab-type "Chemical Analysis Report"

# With groundtruth evaluation
python main.py /path/to/docs/ \\
    --schema-groundtruth /path/to/schema_gt/ \\
    --extraction-groundtruth /path/to/extraction_gt/
```

### Document Types Supported
| File Format | Extensions |
|------------|-----------|
| PDF | `.pdf` (searchable & scanned via OCR) |
| Word | `.docx`, `.doc` |
| Images | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp` |
| Text | `.txt` |

### Extraction Categories

**General documents**
- Invoice · Purchase Order · Utility Bill · Receipt · Financial Document · Salary Slip

**Scientific laboratory reports**
| Subcategory | Typical contents |
|-------------|-----------------|
| Chemical Analysis Report | Concentration tables, calibration curves, LOD/LOQ |
| Environmental Analysis Report | Water/air/soil results, regulatory limits |
| Microbiology Report | Colony counts, MPN, zone-of-inhibition |
| Material Testing Report | Tensile/compression strength, hardness, fatigue |
| Clinical Laboratory Report | Patient results, reference ranges, flags |
| Geotechnical Report | Soil classification, bearing capacity, SPT data |
| Food Analysis Report | Nutritional content, contaminant limits |
| General Laboratory Report | Any other lab document |

> All lab subcategories share the same RL-optimised extraction pipeline with table and figure support.

### Architecture
```
Document → OCR/Reader → Classifier → Schema Builder (RL) → Data Extractor (RL) → JSON
                                          ↑                        ↑
                                   VowpalWabbit             VowpalWabbit
                                  contextual bandits       contextual bandits
```

### Environment Variables (`.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Primary LLM model |
| `OLLAMA_MODEL_ALT` | `mistral` | Secondary model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
""")

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ADE Gradio GUI")
    parser.add_argument("--host",  default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port",  type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    parser.add_argument("--lab",   action="store_true",
                        help="Open GUI in lab-report mode (pre-selects lab type, opens prompt editor)")
    args = parser.parse_args()

    demo = build_ui(lab_mode=args.lab)
    demo.queue(max_size=4)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
