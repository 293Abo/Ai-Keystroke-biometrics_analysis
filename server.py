import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

app = FastAPI(title="Apple Kinetic Biometrics Gateway")

TARGET_PASSPHRASE = "Welcome Guest"
MODEL_PATH = "biometric_model.pkl"

class SystemState:
    def __init__(self):
        self.owner_name = "Verified Owner"
        self.model = None
        self.features = ['dwell_ratio', 'avg_hold_ratio', 'std_hold_ratio', 'avg_flight_ratio', 'std_flight_ratio'] + [f'rel_digraph_{i}' for i in range(1, 12)]
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                artifact = joblib.load(MODEL_PATH)
                if isinstance(artifact, dict) and 'model' in artifact:
                    self.model = artifact['model']
                    self.features = artifact.get('features', self.features)
                elif hasattr(artifact, 'predict'):
                    self.model = artifact
                else:
                    df_train = pd.DataFrame(artifact).fillna(0)
                    pipe = Pipeline([
                        ('scaler', RobustScaler()),
                        ('svm', OneClassSVM(kernel='rbf', gamma=0.01, nu=0.05))
                    ])
                    pipe.fit(df_train)
                    self.model = pipe
                    self.features = list(df_train.columns)
                print("Colab Biometric Model loaded successfully!")
            except Exception as e:
                print(f"Error loading Colab model: {e}")
                self.init_fallback()
        else:
            self.init_fallback()

    def init_fallback(self):
        pipe = Pipeline([
            ('scaler', RobustScaler()),
            ('svm', OneClassSVM(kernel='rbf', gamma=0.01, nu=0.05))
        ])
        dummy_data = np.random.normal(0.25, 0.04, (20, len(self.features)))
        pipe.fit(pd.DataFrame(dummy_data, columns=self.features))
        self.model = pipe

state = SystemState()

class VerifyPayload(BaseModel):
    keystrokes: List[Dict[str, Any]]

class EnrollPayload(BaseModel):
    username: str
    attempts: List[List[Dict[str, Any]]]

def extract_features(keystrokes: List[Dict[str, Any]]) -> Dict[str, float]:
    holds = [float(k.get('hold', 0.1)) for k in keystrokes]
    flights = [float(k.get('flight', 0.1)) for k in keystrokes]

    total_hold = float(np.sum(holds)) if holds else 1.0
    total_flight = float(np.sum(flights)) if flights else 1.0

    dwell_ratio = total_hold / max(0.0001, total_flight)
    relative_flights = [f / max(0.001, total_hold + total_flight) for f in flights] if flights else [0.1]
    relative_holds = [h / max(0.001, total_hold) for h in holds] if holds else [0.1]

    f_dict = {
        'dwell_ratio': dwell_ratio,
        'avg_hold_ratio': float(np.mean(relative_holds)),
        'std_hold_ratio': float(np.std(relative_holds)) if len(relative_holds) > 1 else 0.0,
        'avg_flight_ratio': float(np.mean(relative_flights)),
        'std_flight_ratio': float(np.std(relative_flights)) if len(relative_flights) > 1 else 0.0
    }

    for i in range(1, 12):
        val = relative_flights[i % len(relative_flights)] if relative_flights else 0.1
        f_dict[f'rel_digraph_{i}'] = float(val)

    return f_dict

@app.get("/", response_class=HTMLResponse)
def serve_portal():
    if os.path.exists("portal.html"):
        with open("portal.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h1 style='color:white;text-align:center;'>portal.html is missing.</h1>"

@app.post("/api/verify")
def verify_attempt(payload: VerifyPayload):
    feat_dict = extract_features(payload.keystrokes)
    df_eval = pd.DataFrame([feat_dict]).fillna(0)

    for col in state.features:
        if col not in df_eval.columns:
            df_eval[col] = 0.0
    df_eval = df_eval[state.features]

    pred = int(state.model.predict(df_eval)[0])
    score = float(state.model.decision_function(df_eval)[0])
    
    # الاعتماد الحقيقي على تنبؤ نموذج الـ SVM المستخرج من كولاب
    is_auth = (pred == 1)

    return {
        "authorized": is_auth,
        "score": round(score, 4),
        "dwell_ratio": round(feat_dict['dwell_ratio'], 3),
        "owner": state.owner_name
    }

@app.post("/api/enroll")
def enroll_user(payload: EnrollPayload):
    if not payload.attempts:
        return {"success": False, "message": "No attempts"}

    training_rows = [extract_features(a) for a in payload.attempts]
    df_train = pd.DataFrame(training_rows).fillna(0)

    new_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('svm', OneClassSVM(kernel='rbf', gamma=0.01, nu=0.05))
    ])
    new_pipeline.fit(df_train)

    state.model = new_pipeline
    state.features = list(df_train.columns)
    state.owner_name = payload.username.strip() if payload.username.strip() else "Verified Owner"

    joblib.dump({'model': state.model, 'features': state.features}, MODEL_PATH)

    return {"success": True, "owner": state.owner_name}
