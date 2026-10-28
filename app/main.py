from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Titanic Survival API")


class Passenger(BaseModel):
    Pclass: int
    Sex: Literal["male", "female"]
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: Literal["S", "C", "Q"]


@app.get("/")
def home():
    return {"message": "Titanic API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: Passenger):
    df = pd.DataFrame([p.model_dump()])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {"survived": pred, "probability": round(prob, 4)}