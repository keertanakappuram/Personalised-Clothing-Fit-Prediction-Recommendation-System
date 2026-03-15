# 👗 Personalised Clothing Fit Prediction & Recommendation System

> Predicting whether clothing items run **Small, Fit, or Large** to reduce e-commerce return rates - trained on **192,000+ real [Rent The Runway](https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit) rental transactions** with customer body profiles, product attributes, and review text.

The best model is served as a **production REST API** via FastAPI + Docker.

---

## 🎯 Problem

Clothing fit is one of the top reasons for e-commerce returns. Customers can't try items before buying, and size labels vary wildly across brands and categories. This project builds a personalized fit prediction system that combines customer body profiles, product attributes, and review text to predict the right size — before the customer buys.

---
## 📦 Dataset

The [Rent the Runway dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#clothing_fit) contains 192,000+ clothing rental transactions with customer fit feedback, body measurements (height, weight, bust size), occasion, and review text — making it one of the richest publicly available datasets for clothing fit research.

---

## 📊 Results

### Classification

| Model | ROC-AUC (OVR) | Macro F1 |
|-------|--------------|----------|
| **Optimized TF-IDF (word + char n-grams) + Linear SVM** | **~0.81** | **~0.65** |
| XGBoost + SMOTE | ~0.78 | ~0.62 |
| Random Forest + SMOTE | ~0.74 | ~0.58 |
| Logistic Regression + SMOTE | ~0.71 | ~0.54 |
| PyTorch Neural Network | ~0.75 | ~0.58 |

The linear SVM outperforms the neural network on this dataset — consistent with prior RecSys research showing TF-IDF captures fit signals more effectively than dense engineered features alone.

### Ranking (PyTorch NN)

| Metric | K=5 | K=10 | K=20 |
|--------|-----|------|------|
| Precision@K | 0.72 | 0.68 | 0.61 |
| Recall@K | 0.58 | 0.74 | 0.89 |
| NDCG@K | 0.76 | 0.74 | 0.73 |
| Hit Rate@K | 0.91 | 0.95 | 0.98 |
| MRR | 0.81 | 0.81 | 0.81 |

---

## 🔍 Approach

### 1. Data Pipeline (PySpark)
- Distributed preprocessing on 192K+ records via `spark_preprocessing.py`
- UDFs for height/weight parsing, BMI computation, text keyword extraction
- User/item history aggregations at scale
- Output to Parquet for downstream modeling

### 2. Feature Engineering
- **Body signals**: height, weight, BMI, size relative to BMI peers, size percentile
- **User history**: historical small/large/fit rates, average size, cold-start flags
- **Item history**: item-level small/large rates, items that consistently run small/large
- **Review text**: TF-IDF word + character n-grams, keyword signals (tight, baggy, perfect fit), fit polarity score
- **SBERT embeddings**: 384-dim sentence embeddings from `all-MiniLM-L6-v2`
- **Interaction features**: BMI × size, size per BMI

### 3. Models Trained & Compared
- Logistic Regression + SMOTE (baseline)
- Random Forest + SMOTE
- XGBoost + SMOTE
- Linear SVM (SGD, hinge loss) on TF-IDF + numeric features ← best classifier
- SBERT embeddings + XGBoost
- LMNN (metric learning) + kNN
- PyTorch Feedforward Neural Network + ranking evaluation

### 4. Evaluation
- **Classification**: ROC-AUC (OVR), Macro F1, per-class precision/recall/F1
- **Ranking**: Precision@K, Recall@K, NDCG@K, Hit Rate@K, MRR

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data Pipeline | PySpark, Pandas |
| ML Models | Scikit-learn, XGBoost, SGDClassifier |
| NLP | TF-IDF (word + char n-grams), SBERT |
| Deep Learning | PyTorch (FeedForward NN + BatchNorm) |
| Imbalanced Learning | SMOTE (`imbalanced-learn`) |
| Metric Learning | LMNN (`metric-learn`) |
| Serving | FastAPI, Docker |

---

## 📁 Repository Structure

```
├── src/
│   ├── api.py                  ← FastAPI /predict and /batch endpoints
│   ├── model.py                ← Model loading and inference
│   └── schemas.py              ← Pydantic request/response schemas
├── neural_net.py               ← PyTorch NN training + ranking evaluation
├── spark_preprocessing.py      ← PySpark distributed preprocessing pipeline
├── ranking_eval.py             ← Standalone Precision@K, Recall@K, NDCG@K
├── models/
│   └── fit_predictor_nn.pt     ← Serialized PyTorch model
├── notebooks/
│   └── Personalised_Clothing_And_Fit_Recommendation_System.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Option 1 — Docker (recommended)
```bash
docker-compose up --build
```

Send a prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "height": "5'\''6\"",
    "weight_lbs": 140,
    "size": "M",
    "category": "fitted_dress",
    "body_type": "hourglass",
    "review_text": "Usually wear a medium but this ran a bit tight"
  }'
```

Expected response:
```json
{
  "fit": "small",
  "confidence": 0.81,
  "probabilities": {"small": 0.81, "fit": 0.15, "large": 0.04},
  "latency_ms": 12.3
}
```

### Option 2 — Local API
```bash
pip install -r requirements.txt
uvicorn src.api:app --reload --port 8000
```

### Option 3 — Notebook
```bash
jupyter notebook notebooks/Personalised_Clothing_And_Fit_Recommendation_System.ipynb
```

### Option 4 — PySpark Pipeline
```bash
spark-submit spark_preprocessing.py \
  --input renttherunway_final_data.json.gz \
  --output data/processed/
```
