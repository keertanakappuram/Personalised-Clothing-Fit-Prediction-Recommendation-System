"""
model.py — Model Loading and Inference
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import Optional

LABEL_NAMES = ["small", "fit", "large"]

class FitPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], n_classes=3, dropout=0.3):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FitModel:
    def __init__(self):
        self.model       = None
        self.scaler      = None
        self.feature_cols = []
        self.is_loaded   = False
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self, path: str):
        checkpoint       = torch.load(path, map_location=self.device)
        self.scaler      = checkpoint["scaler"]
        self.feature_cols = checkpoint["feature_cols"]
        hidden_dims      = checkpoint.get("hidden_dims", [256, 128, 64])
        dropout          = checkpoint.get("dropout", 0.3)
        self.model       = FitPredictor(
            input_dim=len(self.feature_cols),
            hidden_dims=hidden_dims,
            dropout=dropout
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.is_loaded = True

    def _parse_height(self, h):
        try:
            if not h: return 65.0
            h = str(h).strip().replace('"', '')
            if "'" in h:
                p = h.split("'")
                return float(int(p[0]) * 12 + (int(p[1].strip()) if p[1].strip() else 0))
        except: return 65.0

    def _parse_bust(self, bust):
        try:
            if not bust: return 36.0, 3.0
            import re
            band = float(re.search(r'(\d+)', str(bust)).group(1))
            cup_map = {'A':1,'B':2,'C':3,'D':4,'DD':5,'DDD':6,'E':5,'F':6}
            cup_match = re.search(r'([A-Z]+)', str(bust))
            cup = float(cup_map.get(cup_match.group(1), 3)) if cup_match else 3.0
            return band, cup
        except: return 36.0, 3.0

    def _size_to_numeric(self, size):
        size_map = {'xxs':0,'xs':1,'s':2,'sm':2,'m':3,'md':3,
                    'l':4,'lg':4,'xl':5,'xxl':6,'1x':5,'2x':6,'3x':7}
        return float(size_map.get(str(size).lower().strip(), 3))

    def predict(self, request):
        from src.schemas import PredictResponse

        height_inches = self._parse_height(request.height)
        weight_lbs    = request.weight_lbs or 140.0
        age           = request.age or 35.0
        bmi           = 703 * weight_lbs / (height_inches ** 2)
        size_numeric  = self._size_to_numeric(request.size or "M")
        bust_band, bust_cup = self._parse_bust(request.bust_size)

        text = str(request.review_text or "").lower()
        text_small   = int(any(w in text for w in ['too small','tight','snug','ran small','size up']))
        text_large   = int(any(w in text for w in ['too big','large','loose','baggy','ran large']))
        text_perfect = int(any(w in text for w in ['perfect','true to size','fit perfectly','just right']))

        rating         = request.rating or 5.0
        is_high_rating = int(rating >= 9)

        feature_dict = {
            'age': age, 'bmi': bmi, 'height_inches': height_inches,
            'weight_lbs': weight_lbs, 'size_numeric': size_numeric,
            'size_relative_to_peers': 0.0, 'size_percentile': 0.5,
            'bmi_size_interaction': bmi * size_numeric,
            'size_per_bmi': size_numeric / (bmi + 1e-9),
            'user_small_rate': 0.0, 'user_large_rate': 0.0, 'user_fit_rate': 0.75,
            'user_avg_size': size_numeric, 'user_total_rentals': 1,
            'item_small_rate': 0.0, 'item_large_rate': 0.0, 'item_total_rentals': 1,
            'item_runs_small': 0, 'item_runs_large': 0,
            'rating_numeric': rating, 'bust_band_size': bust_band,
            'bust_cup_numeric': bust_cup, 'is_high_rating': is_high_rating,
            'is_cold_user': 1, 'is_cold_item': 1,
            'text_small': text_small, 'text_large': text_large, 'text_perfect': text_perfect
        }

        x = np.array([[feature_dict.get(f, 0.0) for f in self.feature_cols]], dtype=np.float32)
        x = self.scaler.transform(x)
        x_t = torch.tensor(x, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x_t)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        return PredictResponse(
            fit=LABEL_NAMES[pred_idx],
            confidence=round(float(probs[pred_idx]), 4),
            probabilities={
                LABEL_NAMES[i]: round(float(probs[i]), 4)
                for i in range(len(LABEL_NAMES))
            }
        )
