import pytest
import numpy as np
import pandas as pd
from shopsense.data_generator import generate_customers, generate_transactions, generate_events
from shopsense.features import (
    compute_rfm_features, compute_behavioral_features,
    compute_transaction_features, build_master_feature_table
)
from shopsense.models.churn_model import (
    build_churn_preprocessing_pipeline, train_churn_model
)
from shopsense.evaluation import (
    compute_shap_values, explain_single_prediction,
    compute_shap_dependence, compare_models,
    estimate_churn_business_impact, detect_feature_drift
)

@pytest.fixture(scope="module")
def full_setup():
    cust = generate_customers(n_customers=50, random_state=42)
    tx = generate_transactions(cust, random_state=42)
    evt = generate_events(cust, random_state=42)
    
    snapshot_date = "2023-12-31"
    rfm = compute_rfm_features(tx, snapshot_date)
    behavioral = compute_behavioral_features(evt, tx, snapshot_date)
    transaction = compute_transaction_features(tx, snapshot_date)
    
    master = build_master_feature_table(cust, rfm, behavioral, transaction)
    
    numeric_features = ["age", "recency_days", "frequency", "monetary_total"]
    categorical_features = []
    
    preprocessor = build_churn_preprocessing_pipeline(numeric_features, categorical_features)
    X = master[numeric_features]
    y = master["churn_label"]
    
    # Train a fast LogisticRegression model
    model = train_churn_model(X, y, preprocessor, model_type="logistic_regression", random_state=42)
    return cust, tx, model, X, y

def test_shap_explainability(full_setup):
    _, _, model, X, _ = full_setup
    
    # Transform data
    preprocessor = model.steps[0][1]
    X_prep = pd.DataFrame(preprocessor.transform(X), columns=preprocessor.get_feature_names_out())
    
    shap_results = compute_shap_values(model, X_prep, model_type="logistic_regression")
    assert "shap_values" in shap_results
    assert "mean_abs_shap" in shap_results
    
    # explain single
    single = explain_single_prediction(
        shap_results["shap_values"],
        shap_results["feature_names"],
        shap_results["expected_value"],
        instance_index=0,
        top_n=2,
        X_sample=X_prep
    )
    assert len(single) == 2
    assert "feature" in single.columns
    
    # dependence
    dep = compute_shap_dependence(
        shap_results["shap_values"],
        shap_results["feature_names"],
        X_prep,
        feature_name=shap_results["feature_names"][0]
    )
    assert len(dep) == len(X_prep)

def test_compare_models(full_setup):
    _, _, model, X, y = full_setup
    models_dict = {"logistic_regression": model}
    df_comp = compare_models(models_dict, X, y)
    assert isinstance(df_comp, pd.DataFrame)
    assert "roc_auc" in df_comp.columns

def test_estimate_churn_business_impact(full_setup):
    cust, tx, model, X, _ = full_setup
    impact = estimate_churn_business_impact(cust, tx, model, X, threshold=0.5)
    assert "predicted_churners_count" in impact
    assert "total_revenue_at_risk" in impact

def test_detect_feature_drift(full_setup):
    _, _, _, X, _ = full_setup
    ref_df = X.iloc[:25]
    cur_df = X.iloc[25:]
    
    df_drift = detect_feature_drift(ref_df, cur_df, numeric_features=X.columns.tolist(), categorical_features=[])
    assert isinstance(df_drift, pd.DataFrame)
    assert "drift_detected" in df_drift.columns
