import joblib

from utils import preprocess_text

# ----------------------------------
# Load Model Files
# ----------------------------------

model = joblib.load(
    "models/emotion_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

label_names = joblib.load(
    "models/label_names.pkl"
)

# ----------------------------------
# Test Samples
# ----------------------------------

samples = [

    "I am extremely happy today.",

    "I feel lonely and hopeless.",

    "I am terrified of tomorrow.",

    "I cannot believe this happened!",

    "I love spending time with my family.",

    "This makes me furious."

]

# ----------------------------------
# Predictions
# ----------------------------------

for text in samples:

    cleaned = preprocess_text(text)

    vector = vectorizer.transform(
        [cleaned]
    )

    prediction = model.predict(
        vector
    )[0]

    probabilities = model.predict_proba(
        vector
    )[0]

    emotion = label_names[
        prediction
    ]

    confidence = probabilities.max()

    print("\n" + "=" * 60)

    print(f"\nText:\n{text}")

    print(
        f"\nPredicted Emotion: {emotion.upper()}"
    )

    print(
        f"Confidence: {confidence:.2%}"
    )

    print("\nAll Emotion Scores:")

    scores = list(
        zip(
            label_names,
            probabilities
        )
    )

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    for label, score in scores:

        print(
            f"{label:<10} : {score:.2%}"
        )