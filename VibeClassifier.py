from transformers import pipeline

# Pre-load the zero-shot classifier (optional fallback)
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# List of predefined vibes
VIBE_LABELS = [
    "Coquette",
    "Clean Girl",
    "Cottagecore",
    "Streetcore",
    "Y2K",
    "Boho",
    "Party Glam"
]

# Rule-based keywords
VIBE_KEYWORDS = {
    "Coquette": ["lace", "pink", "ribbon", "feminine", "doll", "soft", "blush", "corset", "frilly"],
    "Clean Girl": ["minimal", "slick bun", "gold hoop", "glow", "fresh", "clean aesthetic", "neutrals"],
    "Cottagecore": ["vintage", "pastoral", "cottage", "farm", "mushroom", "floral", "lantern", "meadow"],
    "Streetcore": ["urban", "baggy", "streetwear", "sneakers", "techwear", "grunge", "cargo pants"],
    "Y2K": ["retro", "chrome", "low rise", "bling", "juicy couture", "butterfly clip", "2000s"],
    "Boho": ["bohemian", "flowy", "earthy", "fringe", "turquoise", "hippie", "tribal", "ethnic"],
    "Party Glam": ["sparkle", "sequins", "night out", "heels", "bold makeup", "glam", "shimmer"]
}

def classify_vibes_rule_based(text, max_vibes=3):
    text_lower = text.lower()
    scores = {vibe: 0 for vibe in VIBE_KEYWORDS}

    for vibe, keywords in VIBE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[vibe] += 1

    sorted_vibes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_vibes = [v for v, score in sorted_vibes if score > 0][:max_vibes]
    return top_vibes

def classify_vibes_zeroshot(text, threshold=0.5):
    result = zero_shot_classifier(text, VIBE_LABELS, multi_label=True)
    return [label for label, score in zip(result['labels'], result['scores']) if score > threshold]

def classify_vibes(text, use_transformer=False):
    rule_based_vibes = classify_vibes_rule_based(text)
    zero_shot_vibes = classify_vibes_zeroshot(text) if use_transformer else []
    combined = list(dict.fromkeys(rule_based_vibes + zero_shot_vibes))[:3]
    print(combined)
    return combined

