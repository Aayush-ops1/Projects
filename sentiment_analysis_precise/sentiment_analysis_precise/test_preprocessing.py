from utils import preprocess_text

examples = [
    "I am VERY happy today!!!",
    "I am not happy.",
    "Visit https://google.com now!",
    "I am feeling terrified and anxious.",
    "This is AMAZING!!!!",
]

for sentence in examples:

    cleaned = preprocess_text(sentence)

    print("\nOriginal:")
    print(sentence)

    print("Processed:")
    print(cleaned)