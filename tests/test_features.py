import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from shopsense.data_generator import generate_customers, generate_transactions, generate_events
from shopsense.features import (
    compute_rfm_features,
    compute_behavioral_features,
    compute_transaction_features,
    build_master_feature_table
)

@pytest.fixture(scope="module")
def sample_data():
    # Generate a small sample dataset for testing
    cust_df = generate_customers(n_customers=50, random_state=42)
    tx_df = generate_transactions(cust_df, random_state=42)
    evt_df = generate_events(cust_df, random_state=42)
    return cust_df, tx_df, evt_df

def test_compute_rfm_features(sample_data):
    _, tx_df, _ = sample_data
    snapshot_date = "2023-12-31"
    rfm_df = compute_rfm_features(tx_df, snapshot_date)
    
    # Check index and expected columns
    assert isinstance(rfm_df, pd.DataFrame)
    assert rfm_df.index.name == "customer_id"
    expected_cols = [
        "recency_days", "frequency", "monetary_total", "monetary_avg",
        "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score",
        "rfm_total_score", "rfm_segment"
    ]
    for col in expected_cols:
        assert col in rfm_df.columns

def test_compute_behavioral_features(sample_data):
    _, tx_df, evt_df = sample_data
    snapshot_date = "2023-12-31"
    behavioral_df = compute_behavioral_features(evt_df, tx_df, snapshot_date)
    
    assert isinstance(behavioral_df, pd.DataFrame)
    assert behavioral_df.index.name == "customer_id"
    expected_cols = [
        "total_sessions", "avg_session_duration", "total_page_views",
        "cart_add_count", "cart_to_purchase_ratio", "wishlist_count",
        "preferred_device", "preferred_category", "days_since_last_event",
        "event_recency_trend"
    ]
    for col in expected_cols:
        assert col in behavioral_df.columns

def test_compute_transaction_features(sample_data):
    _, tx_df, _ = sample_data
    snapshot_date = "2023-12-31"
    tx_features = compute_transaction_features(tx_df, snapshot_date)
    
    assert isinstance(tx_features, pd.DataFrame)
    assert tx_features.index.name == "customer_id"
    expected_cols = [
        "category_diversity", "preferred_payment", "avg_discount_received",
        "return_rate", "revenue_last_30d", "revenue_last_90d",
        "revenue_last_180d", "purchase_gap_mean", "purchase_gap_std",
        "peak_season_purchase_ratio"
    ]
    for col in expected_cols:
        assert col in tx_features.columns

def test_build_master_feature_table(sample_data):
    cust_df, tx_df, evt_df = sample_data
    snapshot_date = "2023-12-31"
    
    rfm = compute_rfm_features(tx_df, snapshot_date)
    behavioral = compute_behavioral_features(evt_df, tx_df, snapshot_date)
    transaction = compute_transaction_features(tx_df, snapshot_date)
    
    master = build_master_feature_table(cust_df, rfm, behavioral, transaction)
    
    assert isinstance(master, pd.DataFrame)
    assert master.index.name == "customer_id"
    assert "churn_label" in master.columns
    # churn_label must be the last column
    assert master.columns[-1] == "churn_label"
    assert not master.isnull().any().any()
