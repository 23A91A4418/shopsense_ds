import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import mlflow
import mlflow.sklearn
import os

app = FastAPI(title="ShopSense Analytics Churn Serving API")

# Global model and metadata variables
model = None
model_name = "shopsense_churn_model"
model_version = "fallback"

class DummyModel:
    """
    Fallback model to ensure API health checks work even before initial training is executed.
    """
    def predict_proba(self, X):
        n_samples = len(X)
        # Returns high probability of class 0 (non-churn) and low of class 1 (churn)
        return np.column_stack([np.ones(n_samples) * 0.85, np.ones(n_samples) * 0.15])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

@app.on_event("startup")
def load_model_on_startup():
    global model, model_version
    
    # Try different tracking URIs, starting with environment variable if present
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    uris = []
    if env_uri:
        uris.append(env_uri)
    uris.extend([None, "sqlite:///mlflow.db", "mlruns"])
    
    for tracking_uri in uris:
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Revert to default/initial active URI if no env uri
            if not env_uri:
                mlflow.set_tracking_uri("sqlite:///mlflow.db")
            else:
                continue
            
        model_uri_staging = f"models:/{model_name}/Staging"
        model_uri_alias = f"models:/{model_name}@staging"

        # Try loading native sklearn model first to support predict_proba
        for uri in [model_uri_staging, model_uri_alias]:
            try:
                model = mlflow.sklearn.load_model(uri)
                model_version = "registered"
                print(f"Model successfully loaded from registry using URI: {uri}")
                return
            except Exception:
                continue

        # Fallback to pyfunc model if native sklearn fails
        for uri in [model_uri_staging, model_uri_alias]:
            try:
                model = mlflow.pyfunc.load_model(uri)
                model_version = "pyfunc_registered"
                print(f"PyFunc model successfully loaded from registry using URI: {uri}")
                return
            except Exception:
                continue

        # Try loading the best run model directly if not registered
        try:
            from shopsense.evaluation import get_best_mlflow_run
            best_run = get_best_mlflow_run()
            if best_run and "run_id" in best_run:
                run_id = best_run["run_id"]
                model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                model_version = f"run_{run_id[:8]}"
                print(f"Model successfully loaded from run ID: {run_id}")
                return
        except Exception:
            pass

    # Use DummyModel as absolute fallback
    model = DummyModel()
    model_version = "dummy_fallback"
    print("Warning: Loaded dummy fallback model.")

class SingleCustomerRequest(BaseModel):
    customer_features: Dict[str, Any]

class BatchCustomerItem(BaseModel):
    customer_id: str
    features: Dict[str, Any]

class BatchCustomerRequest(BaseModel):
    customers: List[BatchCustomerItem]

@app.get("/health")
def health_check():
    """
    API Health check endpoint returning status and model version.
    """
    return {
        "status": "ok",
        "model_version": model_version,
        "model_name": model_name
    }

@app.post("/predict/churn")
def predict_churn(request: SingleCustomerRequest):
    """
    Predict churn for a single customer.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    features = request.customer_features
    customer_id = features.get("customer_id", "unknown")

    # Create DataFrame
    df = pd.DataFrame([features])
    if "customer_id" in df.columns:
        df = df.set_index("customer_id")
    if "churn_label" in df.columns:
        df = df.drop(columns=["churn_label"])

    try:
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(df)[0, 1])
        else:
            # Fallback if pyfunc has no predict_proba
            pred = model.predict(df)
            prob = float(pred[0]) if isinstance(pred, np.ndarray) else float(pred)

        # Churn prediction at 0.5 threshold
        prediction = 1 if prob >= 0.5 else 0

        # Risk levels
        if prob < 0.3:
            risk = "Low"
        elif prob <= 0.6:
            risk = "Medium"
        else:
            risk = "High"

        return {
            "customer_id": customer_id,
            "churn_probability": prob,
            "churn_prediction": prediction,
            "risk_level": risk
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/churn/batch")
def predict_churn_batch(request: BatchCustomerRequest):
    """
    Predict churn for a batch of customers.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    customers = request.customers
    if not customers:
        return {
            "predictions": [],
            "summary": {"total": 0, "high_risk": 0, "medium_risk": 0, "low_risk": 0}
        }

    rows = []
    cids = []
    for cust in customers:
        feat = cust.features.copy()
        cid = cust.customer_id
        feat["customer_id"] = cid
        rows.append(feat)
        cids.append(cid)

    df = pd.DataFrame(rows).set_index("customer_id")
    if "churn_label" in df.columns:
        df = df.drop(columns=["churn_label"])

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)[:, 1]
        else:
            preds = model.predict(df)
            probs = preds.values if hasattr(preds, "values") else preds

        predictions = []
        high = 0
        med = 0
        low = 0

        for i, cid in enumerate(cids):
            prob = float(probs[i])
            prediction = 1 if prob >= 0.5 else 0

            if prob < 0.3:
                risk = "Low"
                low += 1
            elif prob <= 0.6:
                risk = "Medium"
                med += 1
            else:
                risk = "High"
                high += 1

            predictions.append({
                "customer_id": cid,
                "churn_probability": prob,
                "churn_prediction": prediction,
                "risk_level": risk
            })

        return {
            "predictions": predictions,
            "summary": {
                "total": len(cids),
                "high_risk": high,
                "medium_risk": med,
                "low_risk": low
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
