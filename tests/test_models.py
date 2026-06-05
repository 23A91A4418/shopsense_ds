import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from shopsense.data_generator import generate_customers, generate_transactions, generate_events
from shopsense.features import (
    compute_rfm_features, compute_behavioral_features,
    compute_transaction_features, build_master_feature_table
)
from shopsense.models.churn_model import (
    build_churn_preprocessing_pipeline, train_churn_model,
    find_optimal_threshold, evaluate_churn_model
)
from shopsense.models.segmentation_model import (
    prepare_clustering_features, find_optimal_k,
    train_segmentation_model, profile_clusters
)
from shopsense.models.revenue_model import (
    prepare_revenue_timeseries, test_stationarity as check_stationarity,
    train_sarima_model, forecast_sarima, evaluate_forecast
)

@pytest.fixture(scope="module")
def processed_data():
    cust = generate_customers(n_customers=50, random_state=42)
    tx = generate_transactions(cust, random_state=42)
    evt = generate_events(cust, random_state=42)
    
    snapshot_date = "2023-12-31"
    rfm = compute_rfm_features(tx, snapshot_date)
    behavioral = compute_behavioral_features(evt, tx, snapshot_date)
    transaction = compute_transaction_features(tx, snapshot_date)
    
    master = build_master_feature_table(cust, rfm, behavioral, transaction)
    return master, tx

def test_churn_modeling(processed_data):
    master, _ = processed_data
    
    # Define numeric and categorical features
    numeric_features = ["age", "recency_days", "frequency", "monetary_total", "avg_session_duration"]
    categorical_features = ["gender", "city", "acquisition_channel"]
    
    # Preprocessor
    preprocessor = build_churn_preprocessing_pipeline(numeric_features, categorical_features)
    assert isinstance(preprocessor, Pipeline)
    
    # Train
    X = master[numeric_features + categorical_features]
    y = master["churn_label"]
    
    model = train_churn_model(X, y, preprocessor, model_type="logistic_regression", random_state=42)
    assert isinstance(model, Pipeline)
    
    # Evaluate
    eval_dict = evaluate_churn_model(model, X, y, threshold=0.5)
    assert "roc_auc" in eval_dict
    assert "confusion_matrix" in eval_dict
    
    # Threshold curve
    curve_dict = find_optimal_threshold(model, X, y, metric="f1")
    assert "optimal_threshold" in curve_dict
    assert "threshold_curve" in curve_dict

def test_segmentation_modeling(processed_data):
    master, _ = processed_data
    
    X_scaled, feature_names, scaler = prepare_clustering_features(master)
    assert isinstance(X_scaled, np.ndarray)
    
    # Optimal K
    k_dict = find_optimal_k(X_scaled, k_range=range(2, 4))
    assert "optimal_k_silhouette" in k_dict
    
    # Train
    kmeans_model, labels = train_segmentation_model(X_scaled, n_clusters=2, random_state=42)
    assert len(labels) == len(X_scaled)
    
    # Profile
    profile_df = profile_clusters(master, labels, feature_names)
    assert isinstance(profile_df, pd.DataFrame)
    assert "cluster_size" in profile_df.columns

def test_revenue_modeling(processed_data):
    _, tx = processed_data
    
    series = prepare_revenue_timeseries(tx, freq="M")
    assert isinstance(series, pd.Series)
    
    # ADF test
    adf_dict = check_stationarity(series)
    assert "is_stationary" in adf_dict
    
    # Train SARIMA
    # Use very small order for fast tests
    res, fitted = train_sarima_model(series, order=(1,0,0), seasonal_order=(0,0,0,0))
    assert fitted is not None
    
    # Forecast
    fc = forecast_sarima(res, steps=2)
    assert "predicted_revenue" in fc.columns
    
    # Eval
    eval_res = evaluate_forecast(series, fitted)
    assert "mape" in eval_res
