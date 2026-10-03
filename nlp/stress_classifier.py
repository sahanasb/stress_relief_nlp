from transformers import pipeline


stress_classifier = pipeline(
    "text-classification",
    model="abdulmuksith/textthreat-distilbert-dreaddit",
    top_k=None
)


def detect_stress(text):
    results = stress_classifier(text)[0]
    return results