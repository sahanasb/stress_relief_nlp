from transformers import pipeline


topic_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)


TOPIC_LABELS = [
    "academic",
    "work or career",
    "relationships",
    "family",
    "financial",
    "health or wellness",
    "general"
]


def detect_topic(text):
    result = topic_classifier(
        text,
        candidate_labels=TOPIC_LABELS
    )

    return result