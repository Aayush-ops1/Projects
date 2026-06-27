import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

# Load stopwords
stop_words = set(stopwords.words("english"))

# Keep important negation words
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("never")


def preprocess_text(text):
    """
    Clean and preprocess text.

    Steps:
    1. Lowercase
    2. Remove URLs
    3. Remove special characters
    4. Tokenize
    5. Remove stopwords
    6. Lemmatize
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Keep letters and spaces only
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


def input_validator(text):

    text = text.strip()

    # Empty input

    if not text:

        return False, "Please enter some text."

    # Emoji/symbol only

    if not re.search(r'[A-Za-z0-9\u0980-\u09FF]', text):

        return False, (
            "Emoji-only or symbol-only inputs are not supported."
        )

    # Non-English text

    if re.search(r'[\u0980-\u09FF]', text):

        return False, (
            "Only English text is currently supported."
        )

    # Very short input

    if len(text.split()) < 3:

        return False, (
            "Input is too short. Please provide a complete sentence for reliable emotion detection."
        )

    # Very long input

    if len(text.split()) > 150:

        return False, (
            "Very long paragraphs may contain multiple emotions and reduce prediction reliability."
        )

    return True, None