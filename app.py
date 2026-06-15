"""
Flask API server — bridges the HTML frontend with classifier.py
Run: python app.py
Visit: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys

# Ensure classifier module is importable
sys.path.insert(0, os.path.dirname(__file__))
from classifier import ensemble_classify, classify_batch

app = Flask(__name__, static_folder=".")
CORS(app)  # allow browser fetch from same origin or localhost


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json(force=True)
    comment = data.get("comment", "")
    if not comment:
        return jsonify({"error": "No comment provided"}), 400
    result = ensemble_classify(comment)
    return jsonify(result)


@app.route("/classify-batch", methods=["POST"])
def classify_batch_route():
    data = request.get_json(force=True)
    comments = data.get("comments", [])
    if not isinstance(comments, list) or not comments:
        return jsonify({"error": "Provide a non-empty list under 'comments'"}), 400
    results = classify_batch(comments)
    return jsonify(results)


@app.route("/health")
def health():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader_ok = True
    except ImportError:
        vader_ok = False
    try:
        from textblob import TextBlob
        textblob_ok = True
    except ImportError:
        textblob_ok = False
    return jsonify({
        "status": "ok",
        "vader": vader_ok,
        "textblob": textblob_ok,
        "rule_based": True
    })


if __name__ == "__main__":
    print("=" * 50)
    print("  Sentiment Classifier API")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)
