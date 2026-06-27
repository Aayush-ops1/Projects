from datasets import load_dataset
import pandas as pd
import joblib
import os

from utils import preprocess_text

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Create folders
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("Loading dataset...")

dataset = load_dataset("dair-ai/emotion")

train_df = dataset["train"].to_pandas()
valid_df = dataset["validation"].to_pandas()
test_df = dataset["test"].to_pandas()

label_names = dataset["train"].features["label"].names

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

print("Preprocessing text...")

train_df["clean_text"] = train_df["text"].apply(preprocess_text)

valid_df["clean_text"] = valid_df["text"].apply(preprocess_text)

test_df["clean_text"] = test_df["text"].apply(preprocess_text)

# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

print("Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(
    train_df["clean_text"]
)

X_valid = vectorizer.transform(
    valid_df["clean_text"]
)

X_test = vectorizer.transform(
    test_df["clean_text"]
)

y_train = train_df["label"]

y_valid = valid_df["label"]

y_test = test_df["label"]

# --------------------------------------------------
# Model 1 : Naive Bayes
# --------------------------------------------------

print("\nTraining Naive Bayes...")

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_predictions = nb_model.predict(X_valid)

nb_accuracy = accuracy_score(
    y_valid,
    nb_predictions
)

nb_f1 = f1_score(
    y_valid,
    nb_predictions,
    average="weighted"
)

print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")
print(f"Naive Bayes F1 Score: {nb_f1:.4f}")

# --------------------------------------------------
# Model 2 : Logistic Regression
# --------------------------------------------------

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight="balanced"
)

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_valid)

lr_accuracy = accuracy_score(
    y_valid,
    lr_predictions
)

lr_f1 = f1_score(
    y_valid,
    lr_predictions,
    average="weighted"
)

print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print(f"Logistic Regression F1 Score: {lr_f1:.4f}")

# --------------------------------------------------
# Select Best Model
# --------------------------------------------------

if lr_f1 >= nb_f1:

    best_model = lr_model
    model_name = "Logistic Regression"

else:

    best_model = nb_model
    model_name = "Naive Bayes"

print(f"\nBest Model: {model_name}")

# --------------------------------------------------
# Final Evaluation on Test Set
# --------------------------------------------------

print("\nEvaluating on Test Set...")

test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted"
)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        test_predictions,
        target_names=label_names
    )
)

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    test_predictions
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.show()

# --------------------------------------------------
# Save Artifacts
# --------------------------------------------------

joblib.dump(
    best_model,
    "models/emotion_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

joblib.dump(
    label_names,
    "models/label_names.pkl"
)

print("\nSaved Model Files:")
print("models/emotion_model.pkl")
print("models/tfidf_vectorizer.pkl")
print("models/label_names.pkl")