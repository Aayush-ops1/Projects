# AI-Powered Emotion Detection from Text

## Overview

This project is an AI-based Emotion Detection System that identifies the emotion expressed in a piece of text using Natural Language Processing (NLP) and Machine Learning techniques.

The system analyzes textual input and classifies it into one of the following emotions:

* Sadness
* Joy
* Love
* Anger
* Fear
* Surprise

The project was developed as part of an Artificial Intelligence Internship to demonstrate the complete NLP workflow, including data preprocessing, feature extraction, model training, evaluation, and prediction.

---

# Problem Statement

Build an AI model that can read a sentence or paragraph and predict the underlying emotion expressed by the user.

Example:

Input:

```text
I am feeling very happy today.
```

Output:

```text
Emotion: Joy
```

---

# Objectives

* Understand NLP preprocessing techniques.
* Convert textual data into machine-readable features.
* Train and compare machine learning models.
* Evaluate model performance using standard metrics.
* Predict emotions from unseen text.

---

# Dataset

Dataset Source:

```text
dair-ai/emotion
```

Loaded using Hugging Face Datasets.

Dataset Statistics:

| Split      | Samples |
| ---------- | ------- |
| Train      | 16,000  |
| Validation | 2,000   |
| Test       | 2,000   |

Emotion Classes:

| Label | Emotion  |
| ----- | -------- |
| 0     | Sadness  |
| 1     | Joy      |
| 2     | Love     |
| 3     | Anger    |
| 4     | Fear     |
| 5     | Surprise |

Example Record:

```python
{
    "text": "i didnt feel humiliated",
    "label": 0
}
```

---

# Project Workflow

```text
Dataset
   ↓
Data Analysis
   ↓
Text Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Emotion Prediction
```

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* NLTK
* Hugging Face Datasets
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

---

# Project Structure

```text
EmotionDetection/
│
├── Emotion_Detection.ipynb
│
├── data_analysis.py
├── train.py
├── test.py
├── utils.py
│
├── models/
│   ├── emotion_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_names.pkl
│
├── outputs/
│   ├── emotion_distribution.png
│   ├── sentence_length_distribution.png
│   └── confusion_matrix.png
│
├── requirements.txt
│
└── README.md
```

---

# Data Preprocessing

The following preprocessing techniques were applied:

### 1. Lowercasing

Example:

```text
HAPPY → happy
```

### 2. URL Removal

Example:

```text
Visit https://google.com
→ Visit
```

### 3. Special Character Removal

Example:

```text
happy!!!
→ happy
```

### 4. Tokenization

Sentence is split into individual words.

### 5. Stopword Removal

Common words such as:

```text
the, is, are, a, an
```

are removed.

Important emotion-related words:

```text
not
no
never
```

are preserved.

### 6. Lemmatization

Example:

```text
running → run
cars → car
```

---

# Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert text into numerical vectors.

Configuration:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2)
)
```

Benefits:

* Captures important words.
* Reduces influence of common words.
* Efficient for text classification tasks.

---

# Machine Learning Models

Two models were trained and compared.

## 1. Multinomial Naive Bayes

Advantages:

* Fast training
* Lightweight
* Good baseline performance

## 2. Logistic Regression

Advantages:

* Strong classification performance
* Effective with TF-IDF features
* Better generalization

---

# Model Performance

## Validation Results

| Model               | Accuracy | F1 Score |
| ------------------- | -------- | -------- |
| Naive Bayes         | 75.15%   | 71.16%   |
| Logistic Regression | 88.85%   | 88.50%   |

Logistic Regression achieved the best results and was selected as the final model.

---

# Final Test Results

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 88.05% |
| F1 Score | 87.58% |

Classification Performance:

| Emotion  | Precision | Recall | F1 Score |
| -------- | --------- | ------ | -------- |
| Sadness  | 0.90      | 0.95   | 0.93     |
| Joy      | 0.85      | 0.96   | 0.90     |
| Love     | 0.85      | 0.62   | 0.72     |
| Anger    | 0.91      | 0.84   | 0.87     |
| Fear     | 0.90      | 0.82   | 0.86     |
| Surprise | 0.89      | 0.48   | 0.63     |

---

# Visualizations

The project generates the following visual outputs:

### Emotion Distribution

Shows class distribution across all emotions.

File:

```text
outputs/emotion_distribution.png
```

### Sentence Length Distribution

Shows text length distribution.

File:

```text
outputs/sentence_length_distribution.png
```

### Confusion Matrix

Visualizes model prediction performance.

File:

```text
outputs/confusion_matrix.png
```

---

# Sample Predictions

### Example 1

Input:

```text
I am extremely happy today.
```

Prediction:

```text
Joy
```

---

### Example 2

Input:

```text
I feel lonely and hopeless.
```

Prediction:

```text
Sadness
```

---

### Example 3

Input:

```text
I am terrified of tomorrow.
```

Prediction:

```text
Fear
```

---

### Example 4

Input:

```text
This makes me furious.
```

Prediction:

```text
Anger
```

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd EmotionDetection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# How to Run

## Step 1: Data Analysis

```bash
python data_analysis.py
```

Generates:

* Emotion Distribution Graph
* Sentence Length Distribution Graph

---

## Step 2: Train Model

```bash
python train.py
```

Generates:

* Trained Model
* TF-IDF Vectorizer
* Label Mapping
* Confusion Matrix

---

## Step 3: Test Model

```bash
python test.py
```

Runs emotion predictions on sample inputs.

---

# Real-World Applications

* Customer Support Analytics
* Social Media Monitoring
* Mental Health Support Systems
* Conversational AI
* Chatbots
* Feedback Analysis
* User Sentiment Tracking

---

# Future Improvements

Possible enhancements include:

* Deep Learning (LSTM)
* Transformer Models (BERT, RoBERTa)
* Real-Time Emotion Detection
* Streamlit Web Application
* Multilingual Emotion Detection
* Emotion Intensity Prediction

---

# Conclusion

This project successfully demonstrates the complete NLP pipeline for emotion classification. Using TF-IDF feature extraction and Logistic Regression, the model achieved an accuracy of 88.05% and an F1 score of 87.58% on unseen test data.

The system is capable of accurately identifying emotions from text and can serve as a foundation for more advanced emotion-aware AI applications.
