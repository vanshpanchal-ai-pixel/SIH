"""
Category detection: keyword-match first (fast, no training), DistilBERT
fallback for ambiguous cases (wired in later once fine-tuned).
"""

_KEYWORDS = {
    "Food": ["fssai", "net qty", "mfg date", "ingredients", "nutrition", "veg", "non-veg"],
    "Cosmetics": ["cosmetic", "sunscreen", "spf", "shampoo", "lotion", "cream", "skin"],
}


def keyword_categorize(text: str):
    text_lower = text.lower()
    scores = {cat: sum(1 for kw in kws if kw in text_lower) for cat, kws in _KEYWORDS.items()}
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else None


# Placeholder for DistilBERT fine-tuned model — wire in once trained.
_bert_model = None


def bert_categorize(text: str):
    global _bert_model
    if _bert_model is None:
        return None  # not trained/loaded yet
    # TODO: tokenize + predict once model is fine-tuned
    return None


def detect_category(text: str, default: str = "Food") -> str:
    cat = keyword_categorize(text)
    if cat:
        return cat
    cat = bert_categorize(text)
    return cat or default