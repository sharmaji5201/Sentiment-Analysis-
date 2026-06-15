# SentimentLens — Social Media Sentiment Classifier

A sentiment classification system for social media comments.  
Labels: **+1 (Positive)**, **0 (Neutral)**, **-1 (Negative)**

---

## 📁 Project Structure

```
sentiment_classifier/
├── classifier.py       ← Core Python classifier (VADER + TextBlob + rule-based)
├── app.py              ← Flask API server
├── index.html          ← Standalone HTML/CSS/JS frontend
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Option A — Open HTML directly (no Python needed)
Just open `index.html` in your browser.  
It uses a built-in JavaScript rule-based classifier so **no installation required**.

---

### Option B — Full Python ensemble (recommended)

**Step 1: Install dependencies**
```bash
pip install -r requirements.txt
python -m textblob.download_corpora   # one-time download
```

**Step 2: Start the Flask server**
```bash
python app.py
```

**Step 3: Open in browser**
```
http://localhost:5000
```
The HTML UI auto-detects the running API and switches from JS mode → full Python ensemble (VADER + TextBlob + rule-based).

---

## 🧠 How the Classifier Works

| Method      | Description                                      |
|-------------|--------------------------------------------------|
| **VADER**   | Optimized for social media, handles emojis/slang |
| **TextBlob**| Great for grammatically correct text             |
| **Rule-based** | Dictionary lookup with negation handling      |
| **Ensemble**| Majority vote across all available methods       |

### Label System

| Label | Sentiment | When                              |
|-------|-----------|-----------------------------------|
| `+1`  | Positive  | Compound score > 0.05             |
|  `0`  | Neutral   | Compound score between -0.05–0.05 |
| `-1`  | Negative  | Compound score < -0.05            |

---

## 🐍 Use as Python Library

```python
from classifier import ensemble_classify, classify_batch

# Single comment
result = ensemble_classify("This is absolutely amazing! 😍")
print(result)
# {
#   "label": 1,
#   "sentiment": "Positive",
#   "score": 0.7865,
#   "method": "ensemble",
#   "votes": {"positive": 3, "neutral": 0, "negative": 0},
#   "details": { "vader": {...}, "textblob": {...}, "rule_based": {...} }
# }

# Batch classify
results = classify_batch([
    "Love this product!",
    "Worst experience ever.",
    "It was okay I guess."
])
for r in results:
    print(f"{r['label']:+d}  {r['sentiment']:10s}  {r['comment']}")
```

---

## 🌐 REST API Endpoints

| Endpoint           | Method | Body                               | Response          |
|--------------------|--------|------------------------------------|-------------------|
| `/classify`        | POST   | `{"comment": "text here"}`         | Single result     |
| `/classify-batch`  | POST   | `{"comments": ["c1", "c2", ...]}`  | Array of results  |
| `/health`          | GET    | —                                  | Library status    |

### Example API call
```bash
curl -X POST http://localhost:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"comment": "This is terrible, I hate it!"}'
```

---

## 💡 Features

- ✅ Ensemble classifier (VADER + TextBlob + rule-based)
- ✅ Negation handling ("not good" → Negative)
- ✅ Emoji support 😊😡
- ✅ Batch processing
- ✅ REST API via Flask
- ✅ Standalone HTML UI (works without Python)
- ✅ Session history with statistics
- ✅ Dark-mode polished interface

---

## 📦 Dependencies

- `vaderSentiment` — Valence Aware Dictionary and sEntiment Reasoner
- `textblob` — Simple NLP library
- `flask` — Lightweight web framework
- `flask-cors` — Cross-origin support
