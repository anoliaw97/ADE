"""
ADE — Agentic Document Processing Pipeline
Tkinter GUI (local VLM + LLM, no API keys required)

Layout (SCAL-inspired):
  ┌──────────────────────┬────────────────────────────┐
  │  LEFT PANEL          │  RIGHT PANEL               │
  │  Controls + Chat     │  Document Preview          │
  │  (results log)       │  + Extracted Data Table    │
  └──────────────────────┴────────────────────────────┘

Run:
    python gui.py
"""

import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Optional

# Make sure project root is on PYTHONPATH
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ---------------------------------------------------------------------------
# Colour palette (dark theme, similar to SCAL Intelligent Assistant)
# ---------------------------------------------------------------------------
BG_DARK   = "#1e1e2e"
BG_MID    = "#2a2a3e"
BG_PANEL  = "#252535"
ACCENT    = "#7c6af7"
ACCENT2   = "#5af78e"
TEXT_FG   = "#cdd6f4"
TEXT_DIM  = "#6c7086"
TEXT_WARN = "#f38ba8"
BORDER    = "#45475a"
ENTRY_BG  = "#313244"


# ---------------------------------------------------------------------------
# Helper: run a callable in a background thread, post result to main thread
# ---------------------------------------------------------------------------

def _run_in_thread(fn, callback, *args, **kwargs):
    def _worker():
        try:
            result = fn(*args, **kwargs)
            callback(result, None)
        except Exception as exc:
            callback(None, exc)
    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------

class ADEApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("ADE — Agentic Document Extractor  (Local VLM + LLM)")
        self.geometry("1280x780")
        self.minsize(900, 600)
        self.configure(bg=BG_DARK)

        self._file_path: Optional[str] = None
        self._result: Optional[dict] = None
        self._processing = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Top toolbar
        self._build_toolbar()

        # Main split: left (controls + log) | right (preview + table)
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        left_frame = tk.Frame(paned, bg=BG_PANEL, bd=0)
        right_frame = tk.Frame(paned, bg=BG_PANEL, bd=0)
        paned.add(left_frame,  weight=2)
        paned.add(right_frame, weight=3)

        self._build_left(left_frame)
        self._build_right(right_frame)

        # Status bar
        self._status_var = tk.StringVar(value="Ready — select a document to begin.")
        status_bar = tk.Label(
            self, textvariable=self._status_var,
            bg=BG_MID, fg=TEXT_DIM, anchor="w", padx=10,
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # -- Toolbar ----------------------------------------------------------

    def _build_toolbar(self):
        bar = tk.Frame(self, bg=BG_MID, height=46)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        title_lbl = tk.Label(
            bar,
            text="⚙  ADE  |  Agentic Document Extractor",
            bg=BG_MID, fg=ACCENT,
            font=("Segoe UI", 13, "bold"),
        )
        title_lbl.pack(side=tk.LEFT, padx=14)

        model_lbl = tk.Label(
            bar,
            text="VLM: allenai/olmOCR-2-7B  |  LLM: Qwen2.5-3B-Instruct  |  No API key needed",
            bg=BG_MID, fg=TEXT_DIM,
            font=("Segoe UI", 9),
        )
        model_lbl.pack(side=tk.RIGHT, padx=14)

    # -- Left panel -------------------------------------------------------

    def _build_left(self, parent):
        # Section: File selection
        sec1 = self._section(parent, "Document")
        self._file_label = tk.Label(
            sec1, text="No file selected",
            bg=BG_PANEL, fg=TEXT_DIM,
            font=("Segoe UI", 9), wraplength=260, justify="left",
        )
        self._file_label.pack(anchor="w", padx=10, pady=(2, 4))

        btn_row = tk.Frame(sec1, bg=BG_PANEL)
        btn_row.pack(fill=tk.X, padx=10, pady=4)
        self._mk_btn(btn_row, "📂  Open File", self._on_open_file).pack(side=tk.LEFT)
        self._mk_btn(btn_row, "📁  Open Folder", self._on_open_folder).pack(side=tk.LEFT, padx=(6, 0))

        # Section: Options
        sec2 = self._section(parent, "Options")

        tk.Label(sec2, text="Document type override (optional):",
                 bg=BG_PANEL, fg=TEXT_DIM, font=("Segoe UI", 9)).pack(anchor="w", padx=10)
        self._doc_type_var = tk.StringVar(value="(auto-detect)")
        doc_types = [
            "(auto-detect)",
            "Invoice", "Purchase Order", "Utility Bill",
            "Financial Document", "Receipt", "Salary Slip",
            "Chemical Analysis Report", "Environmental Analysis Report",
            "Microbiology Report", "Material Testing Report",
            "Clinical Laboratory Report", "Geotechnical Report",
            "Food Analysis Report", "General Laboratory Report",
        ]
        doc_combo = ttk.Combobox(
            sec2, textvariable=self._doc_type_var,
            values=doc_types, state="readonly", width=32,
        )
        doc_combo.pack(anchor="w", padx=10, pady=(2, 6))

        opt_row = tk.Frame(sec2, bg=BG_PANEL)
        opt_row.pack(fill=tk.X, padx=10, pady=4)

        self._vlm_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            opt_row, text="Use VLM for PDF OCR (olmOCR)",
            variable=self._vlm_var,
            bg=BG_PANEL, fg=TEXT_FG,
            selectcolor=BG_MID, activebackground=BG_PANEL,
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT)

        steps_row = tk.Frame(sec2, bg=BG_PANEL)
        steps_row.pack(fill=tk.X, padx=10, pady=4)
        tk.Label(steps_row, text="Max RL steps:", bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self._steps_var = tk.IntVar(value=3)
        tk.Spinbox(
            steps_row, from_=1, to=10, textvariable=self._steps_var,
            width=4, bg=ENTRY_BG, fg=TEXT_FG, insertbackground=TEXT_FG,
            buttonbackground=BG_MID,
        ).pack(side=tk.LEFT, padx=(6, 0))

        # Run button
        self._run_btn = self._mk_btn(parent, "▶  Run Extraction", self._on_run, accent=True)
        self._run_btn.pack(fill=tk.X, padx=10, pady=8)

        # Section: Log / Chat
        sec3 = self._section(parent, "Processing Log")
        self._log_box = scrolledtext.ScrolledText(
            sec3,
            bg=BG_DARK, fg=TEXT_FG,
            font=("Consolas", 9),
            insertbackground=TEXT_FG,
            relief=tk.FLAT, bd=0,
            height=18,
        )
        self._log_box.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self._log_box.config(state=tk.DISABLED)

        clear_btn = self._mk_btn(sec3, "🗑  Clear log", self._clear_log)
        clear_btn.pack(anchor="e", padx=6, pady=4)

    # -- Right panel ------------------------------------------------------

    def _build_right(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Tab 1: Extracted JSON
        json_frame = tk.Frame(nb, bg=BG_DARK)
        nb.add(json_frame, text="  Extracted Data (JSON)  ")

        self._json_box = scrolledtext.ScrolledText(
            json_frame,
            bg=BG_DARK, fg=ACCENT2,
            font=("Consolas", 10),
            insertbackground=ACCENT2,
            relief=tk.FLAT, bd=0,
        )
        self._json_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._json_box.config(state=tk.DISABLED)

        # Tab 2: Results table (key–value)
        table_frame = tk.Frame(nb, bg=BG_DARK)
        nb.add(table_frame, text="  Results Table  ")

        cols = ("Field", "Value")
        self._tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=20)
        self._tree.heading("Field", text="Field")
        self._tree.heading("Value", text="Value")
        self._tree.column("Field", width=220, minwidth=120)
        self._tree.column("Value", width=460, minwidth=200)

        v_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self._tree.yview)
        h_scroll = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=self._tree.xview)
        self._tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # Tab 3: Metadata
        meta_frame = tk.Frame(nb, bg=BG_DARK)
        nb.add(meta_frame, text="  Metadata  ")

        self._meta_box = scrolledtext.ScrolledText(
            meta_frame,
            bg=BG_DARK, fg=TEXT_FG,
            font=("Consolas", 10),
            insertbackground=TEXT_FG,
            relief=tk.FLAT, bd=0,
        )
        self._meta_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._meta_box.config(state=tk.DISABLED)

        # Tab 4: Document text (raw OCR output)
        text_frame = tk.Frame(nb, bg=BG_DARK)
        nb.add(text_frame, text="  Raw Text  ")

        self._raw_text_box = scrolledtext.ScrolledText(
            text_frame,
            bg=BG_DARK, fg=TEXT_DIM,
            font=("Consolas", 9),
            insertbackground=TEXT_FG,
            relief=tk.FLAT, bd=0,
            wrap=tk.WORD,
        )
        self._raw_text_box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._raw_text_box.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _section(self, parent: tk.Widget, title: str) -> tk.LabelFrame:
        frame = tk.LabelFrame(
            parent, text=f"  {title}  ",
            bg=BG_PANEL, fg=ACCENT,
            font=("Segoe UI", 9, "bold"),
            bd=1, relief=tk.GROOVE,
        )
        frame.pack(fill=tk.X, padx=6, pady=(8, 2))
        return frame

    def _mk_btn(self, parent: tk.Widget, text: str, command, accent: bool = False) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=ACCENT if accent else BG_MID,
            fg="#ffffff",
            activebackground=ACCENT2 if accent else ACCENT,
            activeforeground="#000000" if accent else "#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            font=("Segoe UI", 9, "bold" if accent else "normal"),
            padx=10, pady=5,
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str, tag: str = "info"):
        """Append a message to the log box (thread-safe)."""
        def _append():
            self._log_box.config(state=tk.NORMAL)
            colour_map = {
                "info":  TEXT_FG,
                "ok":    ACCENT2,
                "warn":  "#f9e2af",
                "error": TEXT_WARN,
            }
            self._log_box.tag_config(tag, foreground=colour_map.get(tag, TEXT_FG))
            self._log_box.insert(tk.END, msg + "\n", tag)
            self._log_box.see(tk.END)
            self._log_box.config(state=tk.DISABLED)
        self.after(0, _append)

    def _set_status(self, msg: str):
        self.after(0, lambda: self._status_var.set(msg))

    def _clear_log(self):
        self._log_box.config(state=tk.NORMAL)
        self._log_box.delete("1.0", tk.END)
        self._log_box.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _show_json(self, data: dict):
        def _update():
            self._json_box.config(state=tk.NORMAL)
            self._json_box.delete("1.0", tk.END)
            self._json_box.insert(tk.END, json.dumps(data, indent=2, ensure_ascii=False))
            self._json_box.config(state=tk.DISABLED)
        self.after(0, _update)

    def _show_table(self, data: dict):
        def _update():
            for row in self._tree.get_children():
                self._tree.delete(row)
            self._populate_tree("", data)
        self.after(0, _update)

    def _populate_tree(self, prefix: str, data, parent_id: str = ""):
        if isinstance(data, dict):
            for key, value in data.items():
                label = f"{prefix}{key}" if not prefix else f"  {key}"
                if isinstance(value, (dict, list)):
                    node = self._tree.insert(parent_id, tk.END, values=(label, "▶ expand"))
                    self._populate_tree("", value, node)
                else:
                    self._tree.insert(parent_id, tk.END, values=(label, str(value)))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                label = f"[{i}]"
                if isinstance(item, (dict, list)):
                    node = self._tree.insert(parent_id, tk.END, values=(label, "▶ expand"))
                    self._populate_tree("", item, node)
                else:
                    self._tree.insert(parent_id, tk.END, values=(label, str(item)))
        else:
            self._tree.insert(parent_id, tk.END, values=(prefix or "value", str(data)))

    def _show_meta(self, result: dict):
        lines = [
            f"Document Type  : {result.get('document_type', 'N/A')}",
            f"Confidence     : {result.get('confidence', 'N/A')}%",
            f"Pages          : {result.get('num_pages', 'N/A')}",
            f"Schema         : {json.dumps(result.get('schema', {}), indent=2)}",
        ]
        times = result.get("processing_times", {})
        if times:
            lines += [
                "",
                "── Timing ──",
                f"Total          : {times.get('total_time', 'N/A')} s",
                f"Reading        : {times.get('reading_time', 'N/A')} s",
                f"Classification : {times.get('classification_time', 'N/A')} s",
                f"Schema build   : {times.get('schema_time', 'N/A')} s",
                f"Extraction     : {times.get('extraction_time', 'N/A')} s",
                f"Avg / page     : {times.get('avg_time_per_page', 'N/A')} s",
            ]
        def _update():
            self._meta_box.config(state=tk.NORMAL)
            self._meta_box.delete("1.0", tk.END)
            self._meta_box.insert(tk.END, "\n".join(lines))
            self._meta_box.config(state=tk.DISABLED)
        self.after(0, _update)

    def _show_raw(self, text: str):
        def _update():
            self._raw_text_box.config(state=tk.NORMAL)
            self._raw_text_box.delete("1.0", tk.END)
            self._raw_text_box.insert(tk.END, text or "(no text extracted)")
            self._raw_text_box.config(state=tk.DISABLED)
        self.after(0, _update)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_open_file(self):
        path = filedialog.askopenfilename(
            title="Select document",
            filetypes=[
                ("All supported", "*.pdf *.png *.jpg *.jpeg *.tiff *.bmp *.txt *.docx"),
                ("PDF files", "*.pdf"),
                ("Images", "*.png *.jpg *.jpeg *.tiff *.bmp"),
                ("Text files", "*.txt"),
                ("Word documents", "*.docx"),
            ]
        )
        if path:
            self._file_path = path
            self._file_label.config(text=os.path.basename(path), fg=TEXT_FG)
            self._set_status(f"Loaded: {path}")
            self._log(f"📄 File selected: {path}", "info")

    def _on_open_folder(self):
        folder = filedialog.askdirectory(title="Select folder of documents")
        if folder:
            self._file_path = folder
            self._file_label.config(text=f"[folder] {os.path.basename(folder)}", fg=ACCENT2)
            self._set_status(f"Folder: {folder}")
            self._log(f"📁 Folder selected: {folder}", "info")

    def _on_run(self):
        if self._processing:
            self._log("⚠ Already processing — please wait.", "warn")
            return
        if not self._file_path:
            messagebox.showwarning("No file", "Please select a document or folder first.")
            return

        self._processing = True
        self._run_btn.config(state=tk.DISABLED, text="⏳  Processing…")
        self._set_status("Processing document…")
        self._log("─" * 50, "info")
        self._log("▶ Starting extraction pipeline …", "info")

        # Collect options
        use_vlm   = self._vlm_var.get()
        max_steps = self._steps_var.get()
        doc_override = self._doc_type_var.get()
        if doc_override == "(auto-detect)":
            doc_override = None

        def _run():
            from main import process_document
            import logging

            logging.basicConfig(level=logging.INFO)

            result = process_document(
                self._file_path,
                extraction_groundtruth=None,
                output_dir="output",
                schema_groundtruth=None,
                max_workers=None,
                max_steps=max_steps,
                llm_choice="local",
                force=False,
                use_vlm=use_vlm,
            )
            return result

        def _done(result, err):
            self._processing = False
            self.after(0, lambda: self._run_btn.config(state=tk.NORMAL, text="▶  Run Extraction"))
            if err:
                self._log(f"✖ Error: {err}", "error")
                self._set_status(f"Error: {err}")
                messagebox.showerror("Error", str(err))
            else:
                self._result = result
                doc_type = result.get("document_type", "Unknown")
                n_pages  = result.get("num_pages", "?")
                self._log(f"✔ Done!  Type={doc_type}  Pages={n_pages}", "ok")
                self._set_status(f"Complete — {doc_type}  ({n_pages} pages)")

                extracted = result.get("extracted_data") or {}
                self._show_json(extracted)
                self._show_table(extracted)
                self._show_meta(result)

        _run_in_thread(_run, _done)


# ---------------------------------------------------------------------------
# Styling (ttk theme)
# ---------------------------------------------------------------------------

def _apply_theme(root: tk.Tk):
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", background=BG_PANEL, foreground=TEXT_FG, fieldbackground=ENTRY_BG)
    style.configure("TFrame", background=BG_PANEL)
    style.configure("TLabel", background=BG_PANEL, foreground=TEXT_FG)
    style.configure("TNotebook", background=BG_MID, tabmargins=[2, 4, 2, 0])
    style.configure("TNotebook.Tab",
                    background=BG_MID, foreground=TEXT_DIM,
                    padding=[10, 4], font=("Segoe UI", 9))
    style.map("TNotebook.Tab",
              background=[("selected", BG_PANEL)],
              foreground=[("selected", ACCENT)])
    style.configure("Treeview",
                    background=BG_DARK, foreground=TEXT_FG,
                    fieldbackground=BG_DARK, rowheight=22)
    style.configure("Treeview.Heading",
                    background=BG_MID, foreground=ACCENT,
                    font=("Segoe UI", 9, "bold"))
    style.map("Treeview", background=[("selected", ACCENT)])
    style.configure("TScrollbar", background=BG_MID, troughcolor=BG_DARK)
    style.configure("TCombobox",
                    fieldbackground=ENTRY_BG, background=BG_MID,
                    foreground=TEXT_FG, selectbackground=ACCENT)
    style.configure("TPanedwindow", background=BORDER)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = ADEApp()
    _apply_theme(app)
    app.mainloop()
