from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

MODEL_PATH = "model.pkl"

app = FastAPI()

class TrainRow(BaseModel):
    # историческая запись (можно грузить массивом)
    base_daily_price_usd: float
    demand_index: float       # 0..1
    location_tier: int        # 1..3
    events_multiplier: float  # >0
    sol_usd: float            # котировка (Pyth)
    season_low_mid_high: int  # 0/1/2
    y_price_usd: float        # фактическая цена (таргет)

class TrainBatch(BaseModel):
    rows: list[TrainRow]

class PriceInput(BaseModel):
    base_daily_price_usd: float
    demand_index: float       # 0..1
    location_tier: int        # 1..3
    events_multiplier: float  # >0
    sol_usd: float            # Pyth котировка (или другой рынок)
    season_low_mid_high: int  # 0 low, 1 mid, 2 high

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def save_model(model):
    joblib.dump(model, MODEL_PATH)

@app.post("/train")
def train(batch: TrainBatch):
    # формируем матрицу
    X, y = [], []
    for r in batch.rows:
        X.append([
            r.base_daily_price_usd,
            r.demand_index,
            r.location_tier,
            r.events_multiplier,
            r.sol_usd,
            r.season_low_mid_high
        ])
        y.append(r.y_price_usd)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(Xtr, ytr)
    r2 = model.score(Xte, yte)

    save_model(model)
    return {"status": "ok", "r2": r2}

@app.post("/price")
def price(inp: PriceInput):
    model = load_model()
    if model is None:
        # fallback — если не обучались: эвристика (как раньше, но +sol_usd)
        p = inp.base_daily_price_usd
        # мягкая корректировка спросом/сезоном/событиями/локацией
        p *= (0.9 + 0.4*max(0, min(1, inp.demand_index)))  # 0.9..1.3
        p *= (0.95 + 0.05*(inp.location_tier-2))           # 0.95..1.05
        p *= max(0.7, min(1.5, inp.events_multiplier))
        # якорим на рыночный контекст через sol_usd
        # например, повышаем/понижаем на 10% при сильном движении SOL
        anchor_sol = 100.0  # опорное значение
        p *= (0.9 + 0.2 * (inp.sol_usd / anchor_sol))
        return {"daily_price_usd": round(float(np.clip(p, 300, 100000)), 2), "mode": "heuristic"}

    X = np.array([[
        inp.base_daily_price_usd,
        inp.demand_index,
        inp.location_tier,
        inp.events_multiplier,
        inp.sol_usd,
        inp.season_low_mid_high
    ]], dtype=float)

    pred = float(model.predict(X)[0])
    pred = float(np.clip(pred, 300.0, 100000.0))
    pred = 5 * round(pred / 5.0)
    return {"daily_price_usd": pred, "mode": "ml"}
