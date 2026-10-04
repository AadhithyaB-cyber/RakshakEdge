"""
RakshakEdge — On-device scam & phishing detection.

Loads a DistilBERT-based scam/phishing classifier (fine-tuned checkpoint,
architecture-compatible with Qualcomm AI Hub's distil_bert_base_uncased_hf
export pipeline for Snapdragon NPU deployment) and classifies input text
as SCAM or SAFE, entirely on-device — no network calls at inference time.

Usage:
    python app.py                # interactive CLI mode
    python app.py --web          # simple local Flask web UI
    python app.py --text "..."   # classify a single string and exit
"""

import argparse
import sys
import time
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def load_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.eval()
    return tokenizer, model


def classify(text: str, tokenizer, model):
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed_ms = (time.perf_counter() - start) * 1000

    probs = torch.softmax(outputs.logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    label_map = getattr(model.config, "id2label", {0: "SAFE", 1: "SCAM"})
    label = label_map.get(pred_id, str(pred_id))
    confidence = float(probs[pred_id].item())

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "latency_ms": round(elapsed_ms, 2),
    }


def run_cli(tokenizer, model):
    print("RakshakEdge — on-device scam/phishing detector")
    print("Type a message or URL to check (or 'quit' to exit)\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break
        result = classify(text, tokenizer, model)
        verdict = "⚠️  SCAM" if result["label"].upper() == "SCAM" else "✅ SAFE"
        print(
            f"  {verdict}  (confidence: {result['confidence']}%, "
            f"inference: {result['latency_ms']} ms)\n"
        )


def run_web(tokenizer, model):
    from flask import Flask, request, jsonify, render_template_string

    app = Flask(__name__)

    PAGE = """
    <!doctype html>
    <html>
    <head>
      <title>RakshakEdge</title>
      <style>
        body { font-family: system-ui, sans-serif; max-width: 560px; margin: 60px auto; }
        textarea { width: 100%; height: 100px; font-size: 15px; padding: 10px; }
        button { padding: 10px 20px; font-size: 15px; margin-top: 10px; cursor: pointer; }
        #result { margin-top: 20px; padding: 14px; border-radius: 8px; font-weight: 600; }
        .scam { background: #fde2e2; color: #a11; }
        .safe { background: #e2f7e2; color: #17701a; }
      </style>
    </head>
    <body>
      <h2>RakshakEdge</h2>
      <p>Paste a message, SMS, or URL below to check it for scam/phishing content — runs entirely on-device.</p>
      <textarea id="text" placeholder="Paste message or URL here..."></textarea><br>
      <button onclick="check()">Check</button>
      <div id="result"></div>
      <script>
        async function check() {
          const text = document.getElementById('text').value;
          const res = await fetch('/check', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text})
          });
          const data = await res.json();
          const el = document.getElementById('result');
          el.className = data.label.toUpperCase() === 'SCAM' ? 'scam' : 'safe';
          el.innerText = (data.label.toUpperCase() === 'SCAM' ? '⚠️ SCAM' : '✅ SAFE')
            + ' — confidence ' + data.confidence + '%, inference ' + data.latency_ms + ' ms';
        }
      </script>
    </body>
    </html>
    """

    @app.route("/")
    def index():
        return render_template_string(PAGE)

    @app.route("/check", methods=["POST"])
    def check():
        text = request.json.get("text", "")
        return jsonify(classify(text, tokenizer, model))

    print("Serving RakshakEdge at http://127.0.0.1:5000")
    app.run(port=5000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--web", action="store_true", help="run local web UI")
    parser.add_argument("--text", type=str, help="classify a single string and exit")
    args = parser.parse_args()

    tokenizer, model = load_model()

    if args.text:
        result = classify(args.text, tokenizer, model)
        print(result)
        sys.exit(0)
    elif args.web:
        run_web(tokenizer, model)
    else:
        run_cli(tokenizer, model)
