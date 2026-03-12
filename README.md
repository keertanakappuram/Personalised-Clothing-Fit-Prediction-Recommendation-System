# 👗 Personalized Clothing Size & Fit Recommendation System

> Predicting whether clothing items run **Small, Fit, or Large** to reduce e-commerce return rates — trained on 192,000+ real rental transactions.

---

## 🎯 Problem

Clothing fit is one of the top reasons for e-commerce returns. Customers can't try items before buying, and size labels vary wildly across brands and categories. This project builds a personalized fit prediction system that combines customer body profiles, product attributes, and review text to recommend the right size — before the customer buys.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| ROC-AUC | **0.81** |
| Dataset Size | **192,000+ transactions** |
| Target Classes | Small / Fit / Large |
| Top Ranking Metric | Recall@K, NDCG |

---

## 🔍 Approach

### 1. Exploratory Data Analysis
- Identified size inconsistency patterns across brands, categories, and user body profiles
- Analyzed distribution of fit outcomes by item type and customer demographics

### 2. Feature Engineering
- **Customer signals**: height, weight, body type, historical fit preferences
- **Product attributes**: category, brand, fabric type, size label
- **Review text**: TF-IDF features extracted from customer reviews to capture qualitative fit signals

### 3. Models Trained
- Logistic Regression (baseline)
- Gradient Boosted Trees (XGBoost)
- Neural Network (PyTorch)

### 4. Evaluation
- Classification performance measured via ROC-AUC
- Recommendation quality measured via **Recall@K** and **NDCG** to evaluate usefulness of top-K size suggestions

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data Processing | Python, Pandas, SQL |
| Distributed Computing | PySpark, Databricks |
| ML Models | Scikit-learn, XGBoost, PyTorch |
| NLP | TF-IDF (Scikit-learn) |
| Evaluation | Ranking metrics (Recall@K, NDCG) |

---

## 📁 Repository Structure

```
├── Personalised_Clothing_And_Fit_Recommendation_System.ipynb  # Full pipeline notebook
└── README.md
```

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/keertanakappuram/Personalised-Clothing-And-Fit-Recommendation-System.git
cd Personalised-Clothing-And-Fit-Recommendation-System
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Open the notebook
```bash
jupyter notebook Personalised_Clothing_And_Fit_Recommendation_System.ipynb
```

---

## 📦 Dataset

This project uses the [Rent the Runway dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit) — a publicly available dataset of clothing rental transactions with customer fit feedback.
