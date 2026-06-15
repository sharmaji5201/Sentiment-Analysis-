"""
Sentiment Classification System for Social Media Comments
Labels: 1 (Positive), 0 (Neutral), -1 (Negative)
Uses VADER (optimized for social media) + TextBlob as ensemble
"""

import json
import sys
import re

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


# ─── Simple fallback classifier (no external deps) ───────────────────────────
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "awesome", "love", "loved",
    "wonderful", "fantastic", "happy", "best", "perfect", "beautiful",
    "enjoy", "enjoyed", "brilliant", "outstanding", "superb", "nice",
    "helpful", "positive", "like", "liked", "cool", "incredible", "wow",
    "splendid", "terrific", "delightful", "pleasant", "glad", "thankful",
    "grateful", "excited", "thrilled", "impressive", "recommend", "top",
    "winning", "win", "success", "successful", "yay", "yes", "✅", "👍",
    "😊", "😍", "🎉", "❤️", "🔥", "💯", "👏"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "hate", "hated", "worst",
    "ugly", "poor", "useless", "garbage", "trash", "disappointing",
    "disappointed", "failure", "fail", "failed", "annoying", "annoyed",
    "frustrating", "frustrated", "angry", "mad", "sad", "unhappy",
    "disgusting", "disgusted", "pathetic", "stupid", "idiotic", "wrong",
    "error", "broken", "scam", "fraud", "lies", "lie", "lied", "fake",
    "never", "waste", "wasted", "boring", "bored", "dislike", "no",
    "nope", "dreadful", "nightmare", "terrible", "toxic", "ridiculous",
    "shame", "shameful", "😡", "😢", "👎", "🤮", "💔", "😤"
}

NEGATION_WORDS = {"not", "no", "never", "dont", "don't", "isn't", "isnt",
                  "wasn't", "wasn't", "couldn't", "couldn't", "won't",
                  "wont", "neither", "nobody", "nothing", "nowhere"}


def simple_classify(text: str) -> dict:
    """Rule-based fallback classifier."""
    words = re.findall(r"[\w']+|[^\w\s]", text.lower())
    pos_count = 0
    neg_count = 0
    i = 0
    while i < len(words):
        word = words[i]
        negated = (i > 0 and words[i - 1] in NEGATION_WORDS)
        if word in POSITIVE_WORDS:
            if negated:
                neg_count += 1
            else:
                pos_count += 1
        elif word in NEGATIVE_WORDS:
            if negated:
                pos_count += 1
            else:
                neg_count += 1
        i += 1

    total = pos_count + neg_count
    if total == 0:
        score = 0.0
    else:
        score = (pos_count - neg_count) / total

    if score > 0.1:
        label = 1
        sentiment = "Positive"
    elif score < -0.1:
        label = -1
        sentiment = "Negative"
    else:
        label = 0
        sentiment = "Neutral"

    return {
        "label": label,
        "sentiment": sentiment,
        "score": round(score, 4),
        "method": "rule-based"
    }


def vader_classify(text: str) -> dict:
    """VADER classifier — great for social media slang/emoji."""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = 1
        sentiment = "Positive"
    elif compound <= -0.05:
        label = -1
        sentiment = "Negative"
    else:
        label = 0
        sentiment = "Neutral"

    return {
        "label": label,
        "sentiment": sentiment,
        "score": round(compound, 4),
        "pos": round(scores["pos"], 4),
        "neu": round(scores["neu"], 4),
        "neg": round(scores["neg"], 4),
        "method": "VADER"
    }


def textblob_classify(text: str) -> dict:
    """TextBlob classifier — good for grammatically correct text."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity        # -1.0 to 1.0
    subjectivity = blob.sentiment.subjectivity  # 0 to 1.0

    if polarity > 0.05:
        label = 1
        sentiment = "Positive"
    elif polarity < -0.05:
        label = -1
        sentiment = "Negative"
    else:
        label = 0
        sentiment = "Neutral"

    return {
        "label": label,
        "sentiment": sentiment,
        "score": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "method": "TextBlob"
    }


def ensemble_classify(text: str) -> dict:
    """
    Combine all available classifiers via majority vote.
    Falls back gracefully if libraries aren't installed.
    """
    if not text or not text.strip():
        return {
            "label": 0,
            "sentiment": "Neutral",
            "score": 0.0,
            "method": "ensemble",
            "details": {},
            "error": "Empty input"
        }

    results = {}
    labels = []

    # VADER (preferred for social media)
    if VADER_AVAILABLE:
        v = vader_classify(text)
        results["vader"] = v
        labels.append(v["label"])

    # TextBlob
    if TEXTBLOB_AVAILABLE:
        t = textblob_classify(text)
        results["textblob"] = t
        labels.append(t["label"])

    # Rule-based (always available)
    r = simple_classify(text)
    results["rule_based"] = r
    labels.append(r["label"])

    # Majority vote
    pos_votes = labels.count(1)
    neg_votes = labels.count(-1)
    neu_votes = labels.count(0)

    if pos_votes > neg_votes and pos_votes > neu_votes:
        final_label = 1
        final_sentiment = "Positive"
    elif neg_votes > pos_votes and neg_votes > neu_votes:
        final_label = -1
        final_sentiment = "Negative"
    else:
        final_label = 0
        final_sentiment = "Neutral"

    # Average compound score
    scores = [results[k]["score"] for k in results]
    avg_score = round(sum(scores) / len(scores), 4)

    return {
        "label": final_label,
        "sentiment": final_sentiment,
        "score": avg_score,
        "method": "ensemble",
        "votes": {"positive": pos_votes, "neutral": neu_votes, "negative": neg_votes},
        "details": results
    }


def classify_batch(comments: list) -> list:
    """Classify a list of comment strings."""
    return [{"comment": c, **ensemble_classify(c)} for c in comments]


# ─── CLI entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classifier.py 'Your comment here'")
        print("       python classifier.py --batch '[\"comment1\", \"comment2\"]'")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        comments = json.loads(sys.argv[2])
        results = classify_batch(comments)
        print(json.dumps(results, indent=2))
    else:
        text = " ".join(sys.argv[1:])
        result = ensemble_classify(text)
        print(json.dumps(result, indent=2))
