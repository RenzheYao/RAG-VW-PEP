#ollama pull qwen2.5vl:7b
#pip install ollama pillow
#python parse_diagram.py v_model.png --refine --index

"""
parse_diagram.py - read a diagram (e.g. the ASPICE V-model) with a local vision LLM and turn it into
text that your RAG can search: every box (with its text) and every arrow/connection (with its meaning).

    ollama pull qwen2.5vl:7b
    pip install ollama pillow
    python parse_diagram.py v_model.png                  # -> v_model.md + v_model.json next to the image
    python parse_diagram.py v_model.png --refine         # second pass: the model checks its result against the image
    python parse_diagram.py *.png --out diagrams/        # several images, write into a folder
    python parse_diagram.py v_model.png --index          # also add the .md to your data pool (rag_vw_pep_flask.py)

What comes out
  <name>.json  nodes (id, label, description, group) and edges (from, to, directed, kind, label)
  <name>.md    the same as readable text: a summary, then one section per box that lists what goes into it
               and out of it. Upload this .md in chat_flask.py like any other document.

Why not plain OCR: OCR gives you the words but not which box an arrow starts from, where it ends, or
what the arrow means. Here the model is asked for that structure explicitly (as JSON, enforced by Ollama).

Knobs (environment variables): VL_MODEL, VL_NUM_CTX, IMG_MAX_SIDE, KEEP_ALIVE.
"""

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path

import ollama

VL_MODEL = os.getenv("VL_MODEL", "qwen2.5vl:7b")
NUM_CTX = int(os.getenv("VL_NUM_CTX", "8192"))          # image tokens + prompt + JSON answer must fit
IMG_MAX_SIDE = int(os.getenv("IMG_MAX_SIDE", "1568"))   # larger = reads small text better, but is slower
KEEP_ALIVE = os.getenv("KEEP_ALIVE", "30m")
KINDS = ["flow", "verification", "traceability", "hierarchy", "other"]

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "group": {"type": "string"},
                },
                "required": ["id", "label"],
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "directed": {"type": "boolean"},
                    "kind": {"type": "string", "enum": KINDS},
                    "label": {"type": "string"},
                },
                "required": ["from", "to"],
            },
        },
    },
    "required": ["nodes", "edges"],
}

PROMPT = """You are reading a diagram (it may be an ASPICE / automotive V-model, a process or architecture diagram).
Describe ONLY what is drawn. Do not add boxes, arrows or text from your own knowledge of ASPICE or V-models.

Return JSON with:
- "title": the title of the diagram (empty string if none)
- "summary": 2-4 sentences: what the diagram shows and how it is organised (e.g. left branch = design / right branch = verification)
- "nodes": EVERY box, shape, phase or label that carries text. For each:
    "id": short unique id (n1, n2, ...)
    "label": the text of the box exactly as written (keep codes such as SYS.2, SWE.4 and the original language)
    "description": any further text inside or directly attached to the box, else ""
    "group": where it sits or which group it belongs to, in a few words (e.g. "left branch, system level", "right branch, software level", "lane: Supplier"), else ""
- "edges": EVERY arrow, line or connection between two nodes. For each:
    "from", "to": node ids. For an arrow, "from" is the tail and "to" is the arrowhead.
    "directed": true if the line has an arrowhead, false if it is a plain line or has arrowheads on both ends
    "kind": "flow" (sequence / hand-over of work from one step to the next), "verification" (a design/requirements step and the test step that checks it, often a horizontal link), "traceability" (bidirectional traceability), "hierarchy" (one box contains or is split into others), or "other"
    "label": the text written on or next to the arrow, else ""

Check before answering: every box with text is in "nodes", every arrow is in "edges", every "from"/"to" is an existing node id."""

REFINE_PROMPT = """Here is a JSON description of the diagram in the image:

{current}

Compare it with the image and return the corrected COMPLETE JSON in the same format:
add boxes and arrows that are missing, remove ones that are not in the image, fix wrong arrow directions, labels and kinds.
Describe only what is drawn; do not add anything from your own knowledge."""

# sentence templates for the .md (so every relation is a self-contained sentence a RAG chunk can match)
PHRASES = {
    "en": {
        "flow": "{a} leads to {b}.",
        "verification": "{a} and {b} form a verification pair (one defines, the other tests it).",
        "traceability": "{a} is traceable to {b}.",
        "hierarchy": "{a} contains {b}.",
        "other": "{a} is related to {b}.",
        "undirected": "{a} and {b} are connected (no direction drawn).",
        "label": ' Label on the connection: "{x}".',
        "elements": "Elements", "in": "Incoming", "out": "Outgoing", "none": "none", "both": "Connected", "group": "Group",
        "source": "Source image", "diagram": "Diagram", "links": "Connections",
    },
    "de": {
        "flow": "{a} führt zu {b}.",
        "verification": "{a} und {b} bilden ein Verifikationspaar (eines definiert, das andere prüft es).",
        "traceability": "{a} ist rückverfolgbar zu {b}.",
        "hierarchy": "{a} enthält {b}.",
        "other": "{a} steht in Beziehung zu {b}.",
        "undirected": "{a} und {b} sind verbunden (ohne Richtung).",
        "label": ' Beschriftung der Verbindung: "{x}".',
        "elements": "Elemente", "in": "Eingehend", "out": "Ausgehend", "none": "keine", "both": "Verbunden", "group": "Gruppe",
        "source": "Quellbild", "diagram": "Diagramm", "links": "Verbindungen",
    },
}


# ----------------------------- image + model --------------------------------
def load_image_bytes(path: Path) -> bytes:
    """PNG bytes, shrunk so the longest side is at most IMG_MAX_SIDE (keeps the number of image tokens in check)."""
    try:
        from PIL import Image
    except ImportError:
        print("(Pillow not installed - sending the image unchanged. pip install pillow)", file=sys.stderr)
        return path.read_bytes()
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = IMG_MAX_SIDE / max(im.size)
        if scale < 1:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)  # model wrapped the JSON in prose / code fences
        if not m:
            raise ValueError("The model did not return JSON:\n" + text[:500])
        return json.loads(m.group(0))


def ask_model(image: bytes, prompt: str) -> dict:
    resp = ollama.chat(
        model=VL_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [image]}],
        format=SCHEMA,  # Ollama forces the answer into this structure
        options={"temperature": 0, "num_ctx": NUM_CTX},
        keep_alive=KEEP_ALIVE,
    )
    return extract_json(resp["message"]["content"])


# ----------------------------- cleaning -------------------------------------
def normalize(data: dict) -> tuple[dict, list[str]]:
    """Unique node ids, edges that point to existing nodes only, no duplicates. Returns (clean data, warnings)."""
    warnings: list[str] = []
    nodes, by_id, by_label = [], {}, {}
    for i, n in enumerate(data.get("nodes") or [], 1):
        if not isinstance(n, dict) or not str(n.get("label", "")).strip():
            continue
        label = re.sub(r"\s+", " ", str(n["label"])).strip()
        if label.lower() in by_label:  # same box listed twice
            continue
        nid = str(n.get("id") or f"n{i}").strip()
        if nid in by_id:
            nid = f"n{i}_{len(nodes)}"
        node = {"id": nid, "label": label, "description": str(n.get("description") or "").strip(),
                "group": str(n.get("group") or "").strip()}
        nodes.append(node)
        by_id[nid] = node
        by_label[label.lower()] = node

    def resolve(ref) -> dict | None:
        ref = str(ref).strip()
        return by_id.get(ref) or by_label.get(ref.lower())

    edges, seen = [], set()
    for e in data.get("edges") or []:
        if not isinstance(e, dict):
            continue
        a, b = resolve(e.get("from", "")), resolve(e.get("to", ""))
        if not a or not b:
            warnings.append(f"dropped an edge with an unknown end: {e.get('from')!r} -> {e.get('to')!r}")
            continue
        if a is b:
            continue
        directed = bool(e.get("directed", True))
        kind = e.get("kind") if e.get("kind") in KINDS else "other"
        label = str(e.get("label") or "").strip()
        key = (a["id"], b["id"], label.lower()) if directed else (*sorted([a["id"], b["id"]]), label.lower())
        if key in seen:
            continue
        seen.add(key)
        edges.append({"from": a["id"], "to": b["id"], "directed": directed, "kind": kind, "label": label})
    linked = {e["from"] for e in edges} | {e["to"] for e in edges}
    for n in nodes:
        if n["id"] not in linked:
            warnings.append(f"box without any connection: {n['label']!r} (check the image: missed arrow?)")
    return {"title": str(data.get("title") or "").strip(), "summary": str(data.get("summary") or "").strip(),
            "nodes": nodes, "edges": edges}, warnings


# ----------------------------- markdown for the RAG -------------------------
def to_markdown(data: dict, source: str, lang: str = "en") -> str:
    P = PHRASES[lang]
    names = {n["id"]: n["label"] for n in data["nodes"]}

    def q(nid):  # box name inside a sentence
        return f"\"{names[nid]}\""

    def sentence(e) -> str:
        t = P[e["kind"]] if e["directed"] else P["undirected"]
        s = t.format(a=q(e["from"]), b=q(e["to"]))
        return s + (P["label"].format(x=e["label"]) if e["label"] else "")

    title = data["title"] or Path(source).stem
    out = [f"# {P['diagram']}: {title}", f"{P['source']}: {Path(source).name}", ""]
    if data["summary"]:
        out += [data["summary"], ""]
    out += [f"## {P['elements']}", ""]
    for n in data["nodes"]:
        # every section repeats the diagram title, so a retrieved chunk is understandable on its own
        out.append(f"### {n['label']}  ({P['diagram']}: {title})")
        if n["group"]:
            out.append(f"{P['group']}: {n['group']}")
        if n["description"]:
            out.append(n["description"])
        ins = [e for e in data["edges"] if e["directed"] and e["to"] == n["id"]]
        outs = [e for e in data["edges"] if e["directed"] and e["from"] == n["id"]]
        both = [e for e in data["edges"] if not e["directed"] and n["id"] in (e["from"], e["to"])]
        out.append(f"{P['in']}: " + (" ".join(sentence(e) for e in ins) if ins else P["none"]))
        out.append(f"{P['out']}: " + (" ".join(sentence(e) for e in outs) if outs else P["none"]))
        if both:
            out.append(f"{P['both']}: " + " ".join(sentence(e) for e in both))
        out.append("")
    out += [f"## {P['links']}", ""] + [f"- {sentence(e)}" for e in data["edges"]] + [""]
    return "\n".join(out)


# ----------------------------- main -----------------------------------------
def parse_image(path: Path, refine: bool = False) -> tuple[dict, list[str]]:
    image = load_image_bytes(path)
    data = ask_model(image, PROMPT)
    data, warnings = normalize(data)
    if refine:
        data2 = ask_model(image, REFINE_PROMPT.format(current=json.dumps(data, ensure_ascii=False, indent=1)))
        data2, warnings2 = normalize(data2)
        if data2["nodes"]:  # keep the first result if the refinement came back empty
            data, warnings = data2, warnings2
    return data, warnings


def main():
    ap = argparse.ArgumentParser(description="Diagram image -> nodes, relations, RAG-ready markdown")
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, help="output folder (default: next to each image)")
    ap.add_argument("--lang", choices=sorted(PHRASES), default="en", help="language of the generated sentences")
    ap.add_argument("--refine", action="store_true", help="second pass: model checks its JSON against the image")
    ap.add_argument("--index", action="store_true", help="add the .md to your data pool (rag_vw_pep_flask.py)")
    args = ap.parse_args()

    engine = None
    if args.index:
        from rag_vw_pep_flask import RagEngine
        engine = RagEngine()

    for img in args.images:
        if not img.exists():
            print(f"{img}: file not found", file=sys.stderr)
            continue
        print(f"Reading {img.name} with {VL_MODEL} ...", flush=True)
        try:
            data, warnings = parse_image(img, args.refine)
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}", file=sys.stderr)
            continue
        out_dir = args.out or img.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{img.stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path = out_dir / f"{img.stem}.md"
        md_path.write_text(to_markdown(data, img.name, args.lang), encoding="utf-8")
        print(f"  {len(data['nodes'])} boxes, {len(data['edges'])} connections -> {md_path}")
        for w in warnings:
            print(f"  ! {w}")
        if engine:
            r = engine.add_document(md_path, name=md_path.name)
            print(f"  added to the pool: {r['name']} ({r['chunks']} chunks)")


if __name__ == "__main__":
    main()