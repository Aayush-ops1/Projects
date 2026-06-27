from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load dataset
dataset = load_dataset("dair-ai/emotion")

train_df = dataset["train"].to_pandas()

# -------------------------------
# Basic Dataset Information
# -------------------------------

print("\nDataset Shape:")
print(train_df.shape)

print("\nFirst 5 Rows:")
print(train_df.head())

print("\nMissing Values:")
print(train_df.isnull().sum())

# -------------------------------
# Emotion Distribution
# -------------------------------

label_names = dataset["train"].features["label"].names

emotion_counts = train_df["label"].value_counts().sort_index()

print("\nEmotion Distribution:\n")

for label, count in emotion_counts.items():
    print(f"{label_names[label]} : {count}")

# Plot Emotion Distribution

plt.figure(figsize=(8, 5))

sns.barplot(
    x=label_names,
    y=emotion_counts.values
)

plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "outputs/emotion_distribution.png",
    dpi=300
)

plt.show()

# -------------------------------
# Sentence Length Analysis
# -------------------------------

train_df["text_length"] = train_df["text"].apply(
    lambda x: len(x.split())
)

print("\nSentence Length Statistics:\n")

print(train_df["text_length"].describe())

print(
    f"\nAverage sentence length: "
    f"{train_df['text_length'].mean():.2f} words"
)

print(
    f"Shortest sentence: "
    f"{train_df['text_length'].min()} words"
)

print(
    f"Longest sentence: "
    f"{train_df['text_length'].max()} words"
)

# Plot Sentence Length Distribution

plt.figure(figsize=(8, 5))

sns.histplot(
    train_df["text_length"],
    bins=30
)

plt.title("Sentence Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "outputs/sentence_length_distribution.png",
    dpi=300
)

plt.show()