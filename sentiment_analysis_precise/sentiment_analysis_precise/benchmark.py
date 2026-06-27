import joblib

from utils import preprocess_text

tests = {

    "I am extremely happy today": "joy",

    "I feel lonely and hopeless": "sadness",

    "I am terrified of tomorrow": "fear",

    "This makes me furious": "anger",

    "I love spending time with my family": "love",

    "I was completely shocked when they threw me a surprise party": "surprise",

    "I am nervous about tomorrow's presentation": "fear",

    "I miss my best friend every day": "sadness",

    "I adore spending time with you": "love",

    "I am laughing so hard at this joke": "joy"
}

model = joblib.load(
    "models/emotion_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

label_names = joblib.load(
    "models/label_names.pkl"
)

correct = 0

print()

for text, expected in tests.items():

    cleaned = preprocess_text(text)

    vector = vectorizer.transform(
        [cleaned]
    )

    pred = model.predict(
        vector
    )[0]

    predicted = label_names[pred]

    if predicted == expected:

        correct += 1

    print(f"Input: {text}")

    print(f"Expected: {expected}")

    print(f"Predicted: {predicted}")

    print()

print(

    f"Benchmark Accuracy: "

    f"{correct}/{len(tests)}"

)