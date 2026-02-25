"""
ADE – Agentic Document Extraction  |  Gradio GUI
=================================================
Launches a web-based interface for the full ADE pipeline.

Tech stack: OpenAI (gpt-4o-mini) · Groq/Llama (free) · LangChain · Gymnasium · VowpalWabbit · PaddleOCR

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
import sys
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr

# Make sure project root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Lab subcategory constants
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
]

ALL_DOC_TYPES = [
    "— auto-detect —",
    "Invoice", "Purchase Order", "Bill", "Receipt", "Financial Document", "Salary Slip",
    *LAB_SUBCATEGORIES,
]


# ---------------------------------------------------------------------------
# Lazy pipeline import
# ---------------------------------------------------------------------------

def _import_pipeline():
    from main import process_document, get_matching_groundtruth
    return process_document, get_matching_groundtruth


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                            datefmt="%H:%M:%S"))

    def emit(self, record):
        try:
            self.q.put_nowait(self.format(record))
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# OpenAI / Groq connectivity check
# ---------------------------------------------------------------------------

def check_api_status(openai_key: str, groq_key: str) -> str:
    lines = []

    # Check OpenAI
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key.strip() or os.getenv("OPENAI_API_KEY", ""))
        models = client.models.list()
        names = [m.id for m in models.data if "gpt" in m.id][:5]
        lines.append(f"**OpenAI** ✓ — available GPT models: {', '.join(names)}")
    except Exception as exc:
        lines.append(f"**OpenAI** ✗ — {exc}")

    # Check Groq
    try:
        from openai import OpenAI as _OAI
        groq = _OAI(
            api_key=groq_key.strip() or os.getenv("GROQ_API_KEY", "no-key"),
            base_url="https://api.groq.com/openai/v1",
        )
        models = groq.models.list()
        names = [m.id for m in models.data][:5]
        lines.append(f"**Groq (Llama)** ✓ — available models: {', '.join(names)}")
    except Exception as exc:
        lines.append(f"**Groq (Llama)** ✗ — {exc}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Default prompt loader
# ---------------------------------------------------------------------------

def load_default_prompt(doc_type: str) -> str:
    try:
        from src.utils.load_baseprompts_utils import load_prompt_from_file
        if not doc_type or doc_type == "— auto-detect —":
            return ""
        return load_prompt_from_file(document_type=doc_type)
    except Exception as exc:
        return f"# Could not load prompt: {exc}"


# ---------------------------------------------------------------------------
# Lab table renderer
# ---------------------------------------------------------------------------

def render_lab_tables(json_text: str):
    import pandas as pd

    if not json_text or not json_text.strip():
        return "No JSON provided.", []

    try:
        data = json.loads(json_text) if isinstance(json_text, str) else json_text
    except json.JSONDecodeError as exc:
        return f"JSON parse error: {exc}", []

    if isinstance(data, list) and data and isinstance(data[0], dict) and "result" in data[0]:
        data = data[0].get("result", data)

    tables_found = []

    def _try_table(obj, name=""):
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
        if isinstance(data, dict) and data.get("no_table"):
            return "Document contains **no table** (`{\"no_table\": true}` returned).", []
        return "No tabular data found in the extracted JSON.", []

    status = f"Found **{len(tables_found)}** table(s)."
    frames = [(lbl, df.values.tolist(), list(df.columns)) for lbl, df in tables_found]
    return status, frames


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------

def run_pipeline(
    uploaded_files,
    openai_key: str,
    groq_key: str,
    llm_choice: str,
    max_steps: int,
    max_workers_val: int,
    schema_gt_dir: str,
    extraction_gt_dir: str,
    force_reprocess: bool,
    output_dir: str,
    doc_type_override: str,
    custom_prompt: str,
    progress=gr.Progress(track_tqdm=True),
):
    if not uploaded_files:
        yield "No files uploaded.", "{}", "—", None
        return

    # Inject API keys into environment if provided
    if openai_key.strip():
        os.environ["OPENAI_API_KEY"] = openai_key.strip()
    if groq_key.strip():
        os.environ["GROQ_API_KEY"] = groq_key.strip()

    output_dir = output_dir.strip() or "output"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "extracted_data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)

    log_queue: queue.Queue = queue.Queue(maxsize=500)
    queue_handler = _QueueHandler(log_queue)
    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)
    root_logger.setLevel(logging.INFO)

    log_lines: list[str] = []

    def drain_log() -> str:
        while True:
            try:
                log_lines.append(log_queue.get_nowait())
            except queue.Empty:
                break
        return "\n".join(log_lines[-200:])

    try:
        process_document, get_matching_groundtruth = _import_pipeline()
    except ImportError as exc:
        yield f"Import error: {exc}", "{}", "—", None
        root_logger.removeHandler(queue_handler)
        return

    all_results = []
    metrics_rows = []
    n_files = len(uploaded_files)

    for idx, file_obj in enumerate(uploaded_files):
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name
        fname = os.path.basename(file_path)

        progress(idx / n_files, desc=f"Processing {fname} …")
        log_lines += [f"\n{'='*60}", f"[{idx+1}/{n_files}] {fname}", f"{'='*60}"]
        yield drain_log(), "{}", "Processing …", None

        schema_gt = extraction_gt = None
        try:
            if schema_gt_dir.strip():
                schema_gt = get_matching_groundtruth(file_path, schema_gt_dir.strip())
            if extraction_gt_dir.strip():
                extraction_gt = get_matching_groundtruth(file_path, extraction_gt_dir.strip())
        except Exception:
            pass

        _custom = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else None
        _override = (doc_type_override
                     if doc_type_override and doc_type_override != "— auto-detect —" else None)

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
                custom_extraction_prompt=_custom,
                doc_type_override=_override,
            )
            all_results.append({"file": fname, "status": "success", "result": result})
            metrics_rows.append({
                "File":           fname,
                "Doc Type":       result.get("document_type", "—"),
                "Confidence":     f"{result.get('confidence', 0):.1f}%",
                "Pages":          result.get("num_pages", "—"),
                "Total Time (s)": result.get("processing_times", {}).get("total_time", "—"),
            })
            log_lines.append(f"  ✓ Done: {fname}  [{result.get('document_type')}]")
        except Exception as exc:
            all_results.append({"file": fname, "status": "error", "error": str(exc)})
            log_lines.append(f"  ✗ Error: {fname}: {exc}")

        yield drain_log(), json.dumps(all_results, indent=2, default=str), _fmt_metrics(metrics_rows), None

    progress(1.0, desc="Done")
    download_path = None
    if all_results:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json",
                                          prefix="ade_results_", dir=tempfile.gettempdir())
        tmp.write(json.dumps(all_results, indent=2, default=str).encode())
        tmp.close()
        download_path = tmp.name

    root_logger.removeHandler(queue_handler)
    yield drain_log(), json.dumps(all_results, indent=2, default=str), _fmt_metrics(metrics_rows), download_path


def _fmt_metrics(rows: list[dict]) -> str:
    if not rows:
        return "No results yet."
    header = ["File", "Doc Type", "Confidence", "Pages", "Total Time (s)"]
    lines = ["  |  ".join(header), "-" * 80]
    for row in rows:
        lines.append("  |  ".join(str(row.get(h, "—")) for h in header))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Results browser
# ---------------------------------------------------------------------------

def browse_results(output_dir: str):
    out_path = Path(output_dir.strip() or "output") / "extracted_data"
    if not out_path.exists():
        return "No results directory found.", "{}"
    files = sorted(out_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return "No extracted results found.", "{}"
    return "\n".join(f"  • {f.name}  ({f.stat().st_size} bytes)" for f in files[:50]), ""


def load_result_file(output_dir: str, filename: str):
    path = Path(output_dir.strip() or "output") / "extracted_data" / filename.strip()
    if not path.exists():
        return f"File not found: {path}"
    try:
        return json.dumps(json.loads(path.read_text()), indent=2)
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# ADE — Agentic Document Extraction

**Agentic pipeline** for extracting structured data from documents using Reinforcement Learning.

| Gymnasium | LangChain | OpenAI | Groq (Llama) | PaddleOCR | OpenCV | Jinja2 |
|:---------:|:---------:|:------:|:------------:|:---------:|:------:|:------:|
| RL Environments | Prompt Optimization | GPT-4o-mini | Llama-3.3-70b (free) | OCR | Image Reading | Prompt Templates |

> Add your API keys in the **API Keys** tab before processing.
"""

LAB_DESCRIPTION = """
# ADE — Scientific Laboratory Report Extraction

**Agentic RL pipeline** specialised for extracting structured data from scientific lab reports.

Supported subcategories:
`Chemical Analysis` · `Environmental` · `Microbiology` · `Material Testing` ·
`Clinical` · `Geotechnical` · `Food Analysis` · `General Lab`

Extracts: **tables** (with detection limits, uncertainty, QC), **figures** (equations / R²), and **narrative conclusions**.

> Add your API keys in the **API Keys** tab before processing.
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

        with gr.Tabs():

            # ── Tab 1: Process Documents ─────────────────────────────────
            with gr.TabItem("Process Documents"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Input Documents")
                        files_input = gr.Files(
                            label="Upload Documents",
                            file_types=[".pdf", ".docx", ".doc", ".png",
                                        ".jpg", ".jpeg", ".tiff", ".bmp", ".txt"],
                            file_count="multiple",
                        )

                        gr.Markdown("### LLM Settings")
                        llm_choice_dd = gr.Dropdown(
                            label="LLM Backend",
                            choices=["gpt", "llama"],
                            value="gpt",
                            info='"gpt" uses OpenAI gpt-4o-mini. "llama" uses Groq Llama-3.3-70b (free).',
                        )

                        gr.Markdown("### Processing Options")
                        with gr.Row():
                            max_steps_sl = gr.Slider(
                                label="Max RL Steps", minimum=1, maximum=10, value=3, step=1)
                            max_workers_sl = gr.Slider(
                                label="Max Workers", minimum=1, maximum=8, value=2, step=1)
                        force_cb = gr.Checkbox(label="Force reprocess (ignore cache)", value=False)

                        # ── Custom Prompt / Doc-Type Override ────────────
                        _default_doctype = "General Laboratory Report" if lab_mode else "— auto-detect —"
                        with gr.Accordion("Custom Prompt & Document Type Override", open=lab_mode):
                            gr.Markdown(
                                "Override the automatic classification and/or the default "
                                "extraction prompt.\n\n"
                                "**Tip for Lab Reports:** select a subcategory and click "
                                "*Load Default Prompt* to pre-fill the scientific extraction prompt."
                            )
                            with gr.Row():
                                doc_type_dd = gr.Dropdown(
                                    label="Override Document Type",
                                    choices=ALL_DOC_TYPES,
                                    value=_default_doctype,
                                )
                                load_prompt_btn = gr.Button("Load Default Prompt", scale=1)
                            custom_prompt_tb = gr.Textbox(
                                label="Custom Extraction Prompt",
                                placeholder=(
                                    "Leave empty to use the built-in prompt.\n\n"
                                    "Paste your custom prompt here to override it.\n\n"
                                    "Lab report rules example:\n"
                                    "1. Extract the complete table with all headers, rows, columns.\n"
                                    "2. If table present, extract ALL rows in exact order as JSON array.\n"
                                    "3. If no table: {\"no_table\": true}"
                                ),
                                lines=14, max_lines=30,
                            )
                            load_prompt_btn.click(
                                fn=load_default_prompt,
                                inputs=[doc_type_dd],
                                outputs=[custom_prompt_tb],
                            )

                        gr.Markdown("### Optional Groundtruth (for evaluation)")
                        schema_gt_tb = gr.Textbox(label="Schema Groundtruth Folder",
                                                   placeholder="/path/to/schema_gt/")
                        extraction_gt_tb = gr.Textbox(label="Extraction Groundtruth Folder",
                                                       placeholder="/path/to/extraction_gt/")
                        output_dir_tb = gr.Textbox(label="Output Directory", value="output")

                        with gr.Row():
                            process_btn = gr.Button("Process Documents", variant="primary", scale=2)
                            stop_btn    = gr.Button("Stop", variant="stop", scale=1)

                    with gr.Column(scale=1):
                        gr.Markdown("### Processing Log")
                        log_out = gr.Textbox(label="Live Log", lines=18, max_lines=18,
                                              interactive=False, elem_classes=["log-box"])
                        gr.Markdown("### Metrics")
                        metrics_out = gr.Textbox(label="", lines=5, interactive=False,
                                                  elem_classes=["result-box"])
                        gr.Markdown("### Extracted Results (JSON)")
                        results_out = gr.Code(label="", language="json", lines=14, interactive=False)
                        download_btn = gr.File(label="Download Results JSON")

                # API keys are read from the API Keys tab
                _api_openai = gr.State(value="")
                _api_groq   = gr.State(value="")

                proc_evt = process_btn.click(
                    fn=run_pipeline,
                    inputs=[
                        files_input, _api_openai, _api_groq,
                        llm_choice_dd, max_steps_sl, max_workers_sl,
                        schema_gt_tb, extraction_gt_tb,
                        force_cb, output_dir_tb,
                        doc_type_dd, custom_prompt_tb,
                    ],
                    outputs=[log_out, results_out, metrics_out, download_btn],
                )
                stop_btn.click(fn=None, cancels=[proc_evt])

            # ── Tab 2: Lab Table Viewer ──────────────────────────────────
            with gr.TabItem("Lab Table Viewer"):
                gr.Markdown(
                    "Paste extracted JSON to visualise embedded data tables as interactive grids."
                )
                with gr.Row():
                    ltv_json_in = gr.Textbox(
                        label="Extracted JSON", lines=12,
                        placeholder='{"results_table": [{"Sample": "A1", "pH": 7.4}, ...]}')
                    ltv_btn = gr.Button("Render Tables", variant="primary", scale=0)
                ltv_status = gr.Markdown("—")

                _MAX_TABLES = 5
                ltv_frames = []
                for _i in range(_MAX_TABLES):
                    with gr.Group(visible=False) as _grp:
                        _lbl = gr.Markdown(f"**Table {_i + 1}**")
                        _df  = gr.Dataframe(interactive=False, wrap=True)
                        ltv_frames.append((_grp, _lbl, _df))

                def _render_tables(json_text):
                    status_md, tables = render_lab_tables(json_text)
                    updates = [gr.update(value=status_md)]
                    for i in range(_MAX_TABLES):
                        if i < len(tables):
                            lbl_txt, rows, cols = tables[i]
                            updates += [
                                gr.update(visible=True),
                                gr.update(value=f"**{lbl_txt}**"),
                                gr.update(value=rows, headers=cols, visible=True),
                            ]
                        else:
                            updates += [gr.update(visible=False), gr.update(value=""),
                                        gr.update(visible=False)]
                    return updates

                _render_outs = [ltv_status] + [c for grp, lbl, df in ltv_frames for c in (grp, lbl, df)]
                ltv_btn.click(fn=_render_tables, inputs=[ltv_json_in], outputs=_render_outs)

            # ── Tab 3: Results Browser ───────────────────────────────────
            with gr.TabItem("Results Browser"):
                gr.Markdown("Browse previously extracted results from the output directory.")
                with gr.Row():
                    rb_outdir = gr.Textbox(label="Output Directory", value="output")
                    rb_browse = gr.Button("Refresh List", variant="secondary")
                with gr.Row():
                    with gr.Column(scale=1):
                        rb_filelist = gr.Textbox(label="Available Result Files", lines=15,
                                                  interactive=False, elem_classes=["result-box"])
                    with gr.Column(scale=1):
                        rb_filename = gr.Textbox(label="File name to load",
                                                  placeholder="report_gpt_extracted.json")
                        rb_load    = gr.Button("Load File", variant="secondary")
                        rb_content = gr.Code(label="File Contents", language="json",
                                             lines=15, interactive=False)
                rb_browse.click(fn=browse_results, inputs=[rb_outdir],
                                outputs=[rb_filelist, rb_content])
                rb_load.click(fn=load_result_file, inputs=[rb_outdir, rb_filename],
                              outputs=[rb_content])

            # ── Tab 4: API Keys ──────────────────────────────────────────
            with gr.TabItem("API Keys"):
                gr.Markdown(
                    "Enter your API keys here. They are stored only in memory for this session "
                    "and are never saved to disk.\n\n"
                    "Alternatively, set them in the `.env` file before starting the GUI."
                )
                with gr.Row():
                    ak_openai   = gr.Textbox(label="OpenAI API Key",
                                              value=os.getenv("OPENAI_API_KEY", ""),
                                              type="password", placeholder="sk-...")
                    ak_groq = gr.Textbox(label="Groq API Key (free — llama choice)",
                                          value=os.getenv("GROQ_API_KEY", ""),
                                          type="password",
                                          placeholder="gsk_... — get free at console.groq.com")
                ak_check = gr.Button("Test Connections", variant="primary")
                ak_status = gr.Markdown("—")

                # Wire keys into the pipeline State components
                ak_openai.change(fn=lambda k: k, inputs=[ak_openai], outputs=[_api_openai])
                ak_groq.change(fn=lambda k: k, inputs=[ak_groq], outputs=[_api_groq])
                ak_check.click(fn=check_api_status, inputs=[ak_openai, ak_groq],
                               outputs=[ak_status])

            # ── Tab 5: Help ──────────────────────────────────────────────
            with gr.TabItem("Help"):
                gr.Markdown("""
## Quick Start

### 1. Get API Keys
- **OpenAI** (required for `--llm-choice gpt`): https://platform.openai.com/api-keys
- **Groq** (free, required for `--llm-choice llama`): https://console.groq.com
- **LangChain/LangSmith** (optional — only for tracing): https://smith.langchain.com

Add them to `.env`:
```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
LANGCHAIN_API_KEY=          # optional, leave blank to disable tracing
```

### 2. Install dependencies
```bash
conda create -n ade python=3.10 -y
conda activate ade
pip install -r requirements.txt
```

### 3. Launch the GUI
```bash
python gui.py          # general mode
python gui.py --lab    # pre-selects lab report type
```

### 4. CLI usage
```bash
# GPT extraction (default)
python main.py report.pdf --output-dir output

# Llama extraction via Groq (free)
python main.py report.pdf --llm-choice llama --output-dir output

# Force lab report mode
python main.py report.pdf --mode lab --output-dir output

# Force specific lab subcategory
python main.py report.pdf --mode lab --lab-type "Chemical Analysis Report"

# Fast baseline (no RL optimisation)
python main.py report.pdf --mode lab-baseline --output-dir output

# Learned prompt optimisation pipeline
python main_lr_op.py report.pdf --output-dir output --llm-choice gpt
```

### Document Formats
`.pdf` · `.docx` · `.doc` · `.png` · `.jpg` · `.jpeg` · `.tiff` · `.bmp` · `.txt`

### Extraction Categories

**General documents**
Invoice · Purchase Order · Bill · Receipt · Financial Document · Salary Slip

**Scientific laboratory reports**
| Subcategory | Typical contents |
|-------------|-----------------|
| Chemical Analysis Report | Concentrations, calibration curves, LOD/LOQ |
| Environmental Analysis Report | Water/air/soil results, regulatory limits |
| Microbiology Report | Colony counts, MPN, zone-of-inhibition |
| Material Testing Report | Tensile/compression strength, hardness |
| Clinical Laboratory Report | Patient results, reference ranges, flags |
| Geotechnical Report | Soil classification, bearing capacity, SPT/CPT |
| Food Analysis Report | Nutritional content, contaminants |
| General Laboratory Report | Any other scientific lab document |

### Architecture
```
Document → OCR/Reader → Classifier (GPT/Llama) → Schema Builder (RL: Gymnasium + VowpalWabbit)
                                                        ↓
                                              Data Extractor (RL: Gymnasium + VowpalWabbit)
                                                        ↓
                                           Learned Prompt Optimizer (LangChain + VowpalWabbit)
                                                        ↓
                                                  Structured JSON
```
""")

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ADE Gradio GUI")
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    parser.add_argument("--lab",   action="store_true",
                        help="Open in lab-report mode (pre-selects lab type, opens prompt editor)")
    args = parser.parse_args()

    demo = build_ui(lab_mode=args.lab)
    demo.queue(max_size=4)
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share, show_error=True)


if __name__ == "__main__":
    main()
