from transformers import pipeline


emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)


def detect_emotion(text):
    results = emotion_classifier(text)[0]

    return results