"""
chat_flask.py - Flask backend (REST API) + Gradio user interface.

    pip install flask gradio requests ollama numpy pypdf pdfplumber
    (optional, better ranking: pip install sentence-transformers)
    python chat_flask.py

Then open http://127.0.0.1:7860

Architecture:
    Gradio UI (port 7860)  --HTTP-->  Flask API (port 5000)  -->  rag_vw_pep_flask.RagEngine -->  Ollama
Flask endpoints:
    GET  /health
    GET  /docs                      list indexed documents
    POST /upload   (multipart file) index a PDF/TXT
    POST /ask      {doc_id, question} -> streamed NDJSON ({"token":..} ... {"sources":[..]})
"""

import json
import threading
import time
from pathlib import Path

import gradio as gr
import requests
from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.utils import secure_filename

import rag_vw_pep_flask as rag
from rag_vw_pep_flask import RagEngine


def _llm_label() -> str:
    """Name of the model for the page title. Works with the newer engine (local Ollama, LLM_MODEL)
    and with the older one (model chosen in llm_client.py)."""
    if hasattr(rag, "LLM_MODEL"):
        return f"local Ollama: {rag.LLM_MODEL}"
    try:
        from llm_client import model_label
        return model_label()
    except Exception:  # noqa: BLE001
        return "local LLM"


FLASK_HOST, FLASK_PORT = "127.0.0.1", 5000
API = f"http://{FLASK_HOST}:{FLASK_PORT}"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ============================ Flask backend ==================================
app = Flask(__name__)
engine = RagEngine()


@app.get("/health")
def health():
    return jsonify(status="ok", documents=len(engine.docs))


@app.get("/docs")
def docs():
    return jsonify(engine.list_documents())


@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="No file uploaded"), 400
    if Path(f.filename).suffix.lower() not in {".pdf", ".txt", ".md"}:
        return jsonify(error="Only .pdf, .txt, .md supported"), 400
    path = UPLOAD_DIR / secure_filename(f.filename)
    f.save(path)
    try:
        return jsonify(engine.add_document(path, name=f.filename))
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e)), 500


@app.post("/ask")
def ask():
    data = request.get_json(force=True)
    doc_id, question = data.get("doc_id"), (data.get("question") or "").strip()
    history = data.get("history") or []
    if doc_id not in engine.docs:
        return jsonify(error="Unknown document"), 404
    if not question:
        return jsonify(error="Empty question"), 400

    @stream_with_context
    def generate():
        try:
            for event in engine.ask_stream(doc_id, question, history):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:  # noqa: BLE001
            yield json.dumps({"error": str(e)}) + "\n"

    return Response(generate(), mimetype="application/x-ndjson")


def run_flask():
    app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True, use_reloader=False)


# ============================ Gradio frontend ================================
def fetch_docs():
    """Return (dropdown choices as (label, doc_id))."""
    try:
        r = requests.get(f"{API}/docs", timeout=10)
        return [(f"{d['name']} ({d['chunks']} chunks)", d["doc_id"]) for d in r.json()]
    except requests.RequestException:
        return []


def on_upload(file_path):
    choices = fetch_docs()
    if not file_path:
        return gr.update(choices=choices), "Please choose a file first."
    try:
        with open(file_path, "rb") as fh:
            r = requests.post(
                f"{API}/upload",
                files={"file": (Path(file_path).name, fh)},
                timeout=3600,  # embedding a big document can take a while
            )
        data = r.json()
    except requests.RequestException as e:
        return gr.update(choices=choices), f"Backend error: {e}"
    if r.status_code != 200:
        return gr.update(choices=choices), f"Error: {data.get('error')}"
    choices = fetch_docs()
    msg = f"Indexed **{data['name']}** – {data['chunks']} chunks" + (" (from cache)" if data["cached"] else "")
    return gr.update(choices=choices, value=data["doc_id"]), msg


def _text(content) -> str:
    """Chatbot message content can be a str or a list of blocks, depending on the Gradio version."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_text(c) for c in content)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return ""


def on_ask(doc_id, message, history):
    """Outputs: chatbot, sources, question textbox. The chat history lives in the
    browser session (gr.Chatbot) and is sent to the backend with every question."""
    history = list(history or [])
    message = (message or "").strip()
    if not message:
        yield history, "", ""
        return
    if not doc_id:
        history += [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Upload and select a document first."},
        ]
        yield history, "", ""
        return

    # previous turns (plain text only) -> sent to the backend as memory
    prior = [
        {"role": m["role"], "content": _text(m["content"])}
        for m in history
        if m.get("role") in ("user", "assistant") and _text(m.get("content"))
    ]
    history += [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "*Thinking …*"},
    ]
    yield history, "", ""  # shows the question immediately and clears the textbox

    answer, sources_md, head = "", "", ""
    try:
        with requests.post(
            f"{API}/ask",
            json={"doc_id": doc_id, "question": message, "history": prior},
            stream=True,
            timeout=600,
        ) as r:
            if r.status_code != 200:
                history[-1]["content"] = f"Error: {r.json().get('error')}"
                yield history, "", ""
                return
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                ev = json.loads(line)
                if "token" in ev:
                    answer += ev["token"]
                    history[-1]["content"] = answer
                    yield history, sources_md, ""
                elif "search_query" in ev:
                    if ev["search_query"].strip() != message:  # only shown for follow-ups that were rewritten
                        head = f"**Search query used:** {ev['search_query']}\n\n"
                        sources_md = head
                elif "sources" in ev:
                    sources_md = head + "\n\n".join(
                        f"**[{s['n']}] pages {s['pages']}**" + (f" – {s['heading']}" if s.get("heading") else "")
                        + f" (score {s['score']})\n\n> "
                        + s["text"][:500].replace("\n", " ")
                        + (" …" if len(s["text"]) > 500 else "")
                        for s in ev["sources"]
                    )
                    yield history, sources_md, ""
                elif "error" in ev:
                    history[-1]["content"] = f"{answer}\n\nError: {ev['error']}"
                    yield history, sources_md, ""
    except requests.RequestException as e:
        history[-1]["content"] = f"Backend error: {e}"
        yield history, "", ""


def build_ui():
    with gr.Blocks(title="VW PEP – Document Chat") as demo:
        gr.Markdown("# VW PEP – Document Chat\nUpload a PDF, pick it, and ask questions (" + _llm_label() + ").")
        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(label="PDF / TXT", file_types=[".pdf", ".txt", ".md"], type="filepath")
                upload_btn = gr.Button("Upload & index", variant="primary")
                status = gr.Markdown()
                doc_dd = gr.Dropdown(choices=fetch_docs(), label="Document", interactive=True)
            with gr.Column(scale=2):
                try:  # Gradio 5 needs type="messages"; newer versions removed the argument
                    chatbot = gr.Chatbot(label="Conversation", height=480, type="messages")
                except TypeError:
                    chatbot = gr.Chatbot(label="Conversation", height=480)
                question = gr.Textbox(
                    label="Your message", lines=2,
                    placeholder="z. B. Welche Meilensteine gibt es im PEP?  (Enter = send)",
                )
                with gr.Row():
                    ask_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear conversation")
                with gr.Accordion("Sources of the last answer", open=False):
                    sources = gr.Markdown()

        upload_btn.click(on_upload, inputs=file, outputs=[doc_dd, status])
        ask_btn.click(on_ask, inputs=[doc_dd, question, chatbot], outputs=[chatbot, sources, question])
        question.submit(on_ask, inputs=[doc_dd, question, chatbot], outputs=[chatbot, sources, question])
        clear_btn.click(lambda: ([], "", ""), outputs=[chatbot, sources, question])
        # a new document means a new conversation
        doc_dd.change(lambda: ([], ""), outputs=[chatbot, sources])
        demo.load(lambda: gr.update(choices=fetch_docs()), outputs=doc_dd)
    return demo


if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    for _ in range(50):  # wait until Flask is up
        try:
            requests.get(f"{API}/health", timeout=1)
            break
        except requests.RequestException:
            time.sleep(0.2)
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860)