from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
import io
import base64
import json
import time
import pandas as pd
from datetime import datetime
from pdf2image import convert_from_path
from PIL import Image

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['PDF_CACHE'] = os.path.join(os.path.dirname(__file__), 'pdf_cache')
app.config['OUTPUT_DIR'] = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PDF_CACHE'], exist_ok=True)
os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)

MODEL_ID = "allenai/olmOCR-2-7B-1025-FP8"
LLM_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

DEFAULT_OLMOCR_PROMPT = """Attached is one page of a document that you must process. Just return the plain text representation of this document as if you were reading it naturally. Convert equations to LateX and tables to HTML.
If there are any figures or charts, label them with the following markdown syntax ![Alt text describing the contents of the figure](page_startx_starty_width_height.png)
Return your output as markdown, with a front matter section on top specifying values for the primary_language, is_rotation_valid, rotation_correction, is_table, and is_diagram parameters."""

TABLE_PROMPT = """Extract data from the document image. If table present, extract ALL rows in exact order with exact column names as JSON array. If no table found, return: {"no_table": true}. Provide clean JSON output."""

model_status = {'vlm_loaded': False, 'llm_loaded': False}
current_pdf = {'name': None, 'pages': [], 'page_count': 0, 'selected_pages': []}
extracted_data = []
selected_files = []
output_dir = None
stop_flag = False
chat_messages = []
error_messages = []

@app.route('/')
def index():
    return render_template('index.html', 
                           model_status=model_status,
                           current_pdf=current_pdf,
                           extracted_data=extracted_data,
                           DEFAULT_OLMOCR_PROMPT=DEFAULT_OLMOCR_PROMPT,
                           TABLE_PROMPT=TABLE_PROMPT)

@app.route('/status')
def status():
    return jsonify(model_status)

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global current_pdf
    f = request.files.get('pdf')
    if not f:
        return jsonify({'error': 'no file'}), 400
    
    path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(path)
    
    cache_folder = os.path.join(app.config['PDF_CACHE'], f.filename.replace('.pdf', ''))
    os.makedirs(cache_folder, exist_ok=True)
    
    try:
        images = convert_from_path(path, dpi=150)
        for i, img in enumerate(images):
            img.save(os.path.join(cache_folder, f'page_{i+1}.png'), 'PNG')
        current_pdf = {
            'name': f.filename,
            'path': path,
            'pages': images,
            'page_count': len(images),
            'selected_pages': list(range(len(images)))
        }
        return jsonify({'name': f.filename, 'pages': len(images), 'path': path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/files', methods=['POST'])
def select_files():
    global selected_files
    data = request.get_json()
    mode = data.get('mode', 'single')
    
    if mode == 'single':
        if current_pdf.get('name'):
            selected_files = [current_pdf['path']]
            return jsonify({'files': selected_files, 'count': 1})
    return jsonify({'files': selected_files, 'count': len(selected_files)})

@app.route('/output', methods=['POST'])
def select_output():
    global output_dir
    data = request.get_json()
    output_dir = data.get('path') or app.config['OUTPUT_DIR']
    os.makedirs(output_dir, exist_ok=True)
    return jsonify({'output_dir': output_dir})

@app.route('/pdf/page/<name>/<int:page_num>')
def pdf_page(name, page_num):
    cache_folder = os.path.join(app.config['PDF_CACHE'], name.replace('.pdf', ''))
    page_path = os.path.join(cache_folder, f'page_{page_num}.png')
    if os.path.exists(page_path):
        with open(page_path, 'rb') as f:
            img_data = base64.b64encode(f.read()).decode()
        return jsonify({'image': img_data, 'page': page_num})
    return jsonify({'error': 'page not found'}), 404

@app.route('/pdf/thumb/<name>/<int:page_num>')
def pdf_thumb(name, page_num):
    cache_folder = os.path.join(app.config['PDF_CACHE'], name.replace('.pdf', ''))
    page_path = os.path.join(cache_folder, f'page_{page_num}.png')
    if os.path.exists(page_path):
        img = Image.open(page_path)
        img.thumbnail((80, 100))
        buf = io.BytesIO()
        img.save(buf, 'PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    return 'not found', 404

@app.route('/load_vlm', methods=['POST'])
def load_vlm():
    global model_status
    model_status['vlm_loaded'] = True
    return jsonify({'status': 'ok', 'message': f'loaded {MODEL_ID}'})

@app.route('/load_llm', methods=['POST'])
def load_llm():
    global model_status
    model_status['llm_loaded'] = True
    return jsonify({'status': 'ok', 'message': f'loaded {LLM_MODEL_ID}'})

@app.route('/extract', methods=['POST'])
def extract():
    global extracted_data, stop_flag
    data = request.get_json()
    prompt = data.get('prompt', DEFAULT_OLMOCR_PROMPT)
    pages = data.get('pages', [])
    
    if not current_pdf.get('pages'):
        return jsonify({'error': 'No PDF loaded'}), 400
    
    extracted_data = []
    pages_to_process = pages if pages else list(range(current_pdf['page_count']))
    
    for i, page_idx in enumerate(pages_to_process):
        if stop_flag:
            break
        
        result = {
            'raw_response': f'Simulated extraction for page {page_idx + 1}\n\nPrimary Language: English\nIs Rotation Valid: true\nRotation Correction: 0\nIs Table: false\nIs Diagram: false\n\n[Content extracted from document...]',
            'prompt_used': prompt,
            'input_tokens': 1000 + i * 50,
            'output_tokens': 500 + i * 20,
            'total_tokens': 1500 + i * 70,
            'duration': 12.5 + i * 0.5
        }
        extracted_data.append(result)
    
    stop_flag = False
    return jsonify({'status': 'ok', 'pages': len(extracted_data), 'total_tokens': sum(r['total_tokens'] for r in extracted_data)})

@app.route('/stop', methods=['POST'])
def stop_extraction():
    global stop_flag
    stop_flag = True
    return jsonify({'status': 'ok', 'message': 'Stopping...'})

@app.route('/results')
def get_results():
    return jsonify(extracted_data)

@app.route('/result/<int:page_idx>')
def get_result_page(page_idx):
    if page_idx < len(extracted_data):
        return jsonify(extracted_data[page_idx])
    return jsonify({'error': 'No result for page'}), 404

@app.route('/optimize_prompt', methods=['POST'])
def optimize_prompt():
    data = request.get_json()
    doc_type = data.get('doc_type', '')
    goal = data.get('goal', '')
    optimized = f"""Extract {doc_type} data from the document. Focus on: {goal}
Return as structured JSON with all relevant fields."""
    return jsonify({'prompt': optimized})

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global chat_messages
    if request.method == 'POST':
        data = request.get_json() or {}
        message = data.get('message', '')
        
        response = None
        msg_lower = message.lower()
        
        if 'load vlm' in msg_lower:
            load_vlm()
            response = "VLM loading initiated..."
        elif 'load llm' in msg_lower:
            load_llm()
            response = "LLM loading initiated..."
        elif msg_lower in ['extract', 'start', 'run']:
            response = "Use START EXTRACTION button to begin"
        elif 'stop' in msg_lower:
            stop_extraction()
            response = "Stopping..."
        elif 'help' in msg_lower:
            response = "Commands: 'load vlm', 'load llm', 'extract', 'stop', 'optimize'"
        elif not model_status.get('llm_loaded'):
            response = "Load LLM first: 'load llm'"
        
        if message:
            chat_messages.append({'role': 'user', 'content': message, 'timestamp': datetime.now().strftime('%H:%M:%S')})
        if response:
            chat_messages.append({'role': 'assistant', 'content': response, 'timestamp': datetime.now().strftime('%H:%M:%S')})
        
        return jsonify({'status': 'ok', 'messages': chat_messages})
    
    return jsonify(chat_messages)

@app.route('/chat/clear', methods=['POST'])
def clear_chat():
    global chat_messages
    chat_messages = []
    return jsonify({'status': 'ok'})

@app.route('/error_log', methods=['GET', 'POST'])
def error_log():
    global error_messages
    if request.method == 'POST':
        data = request.get_json() or {}
        error = data.get('error', '')
        if error:
            error_messages.append({'error': error, 'timestamp': datetime.now().strftime('%H:%M:%S')})
        return jsonify({'status': 'ok', 'errors': error_messages})
    
    return jsonify(error_messages)

@app.route('/error_log/clear', methods=['POST'])
def clear_error_log():
    global error_messages
    error_messages = []
    return jsonify({'status': 'ok'})

@app.route('/health')
def health():
    return jsonify({'status': 'up', 'models': model_status})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
