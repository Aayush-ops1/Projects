import streamlit as st
import joblib

from utils import preprocess_text, input_validator

# ----------------------------------
# Page Config
# ----------------------------------

st.set_page_config(
    page_title="Emotion Detection",
    page_icon="😊",
    layout="centered"
)

# ----------------------------------
# Load Artifacts
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
# Emojis
# ----------------------------------

emotion_emoji = {

    "sadness": "😢",

    "joy": "😄",

    "love": "❤️",

    "anger": "😡",

    "fear": "😨",

    "surprise": "😲"

}

# ----------------------------------
# Title
# ----------------------------------

st.title(
    "AI Emotion Detection from Text"
)

st.write(
    "Enter any sentence and the model will predict the emotion."
)

# ----------------------------------
# Input
# ----------------------------------

user_text = st.text_area(
    "Enter Text",
    height=150
)

# ----------------------------------
# Prediction
# ----------------------------------

if st.button(
    "Predict Emotion"
):
    is_valid, message = input_validator(user_text)

    if not is_valid:
        st.warning(message)
    else:
        cleaned = preprocess_text(user_text)
        vector = vectorizer.transform([cleaned])

    if user_text.strip():

        cleaned = preprocess_text(
            user_text
        )

        st.subheader("Processed Text")

        st.code(cleaned)

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

        st.success(
            f"{emotion.upper()} "
            f"{emotion_emoji.get(emotion, '')}"
        )

        st.write(
            f"Confidence: {confidence:.2%}"
        )

        st.subheader(
            "Emotion Probabilities"
        )

        for label, score in sorted(

            zip(
                label_names,
                probabilities
            ),

            key=lambda x: x[1],
            reverse=True

        ):

            st.write(
                f"{label}: {score:.2%}"
            )

            st.progress(
                float(score)
            )

    else:

        st.warning(
            "Please enter some text."
        )