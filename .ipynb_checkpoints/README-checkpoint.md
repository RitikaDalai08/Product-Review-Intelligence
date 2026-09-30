# AI-Powered Product Review Intelligence System

An end-to-end Machine Learning NLP application that analyzes product reviews and classifies customer sentiment into **Positive**, **Negative**, or **Neutral**.

## 📌 Features
* **Dataset Size:** 154,475 processed reviews[cite: 26]
* **Best Model:** Logistic Regression (~90.27% accuracy)[cite: 26]
* **Interactive UI:** Real-time single-review predictor and batch CSV analysis[cite: 26]

## 📊 Model Performance

| Model | Accuracy |
| :--- | :--- |
| **Logistic Regression** | **90.27%** |
| Multinomial Naive Bayes | 89.78% |

## 📁 Directory Structure
```text
Product-Review-Intelligence/
├── data/
│   ├── Dataset-SA.csv
│   └── cleaned_reviews.csv
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── sentiment_analysis.ipynb
├── app.py
├── requirements.txt
└── README.md