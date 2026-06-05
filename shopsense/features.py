import pandas as pd
import numpy as np

def compute_rfm_features(transactions_df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """
    Compute Recency, Frequency, and Monetary value features.
    """
    df = transactions_df.copy()
    snapshot_dt = pd.to_datetime(snapshot_date)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # Compute net revenue per transaction: quantity * unit_price * (1 - discount_pct)
    # If returned (return_flag == 1), net spend is 0
    df["net_revenue"] = df["quantity"] * df["unit_price"] * (1.0 - df["discount_pct"])
    df.loc[df["return_flag"] == 1, "net_revenue"] = 0.0

    # Aggregate by customer
    agg = df.groupby("customer_id").agg(
        last_purchase=("transaction_date", "max"),
        frequency=("transaction_id", "count"),
        monetary_total=("net_revenue", "sum")
    )

    agg["recency_days"] = (snapshot_dt - agg["last_purchase"]).dt.days
    agg["monetary_avg"] = agg["monetary_total"] / agg["frequency"]
    agg["monetary_avg"] = agg["monetary_avg"].fillna(0.0)

    # Quintile scores (1 to 5) using rank to handle duplicates gracefully
    agg["rfm_recency_score"] = pd.qcut(agg["recency_days"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    agg["rfm_frequency_score"] = pd.qcut(agg["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    agg["rfm_monetary_score"] = pd.qcut(agg["monetary_total"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    agg["rfm_total_score"] = agg["rfm_recency_score"] + agg["rfm_frequency_score"] + agg["rfm_monetary_score"]

    # RFM Segment mapping
    def get_segment(score):
        if score >= 13:
            return "Champions"
        elif score >= 10:
            return "Loyal"
        elif score >= 7:
            return "At Risk"
        elif score >= 4:
            return "Hibernating"
        else:
            return "Lost"

    agg["rfm_segment"] = agg["rfm_total_score"].apply(get_segment)

    return agg[[
        "recency_days", "frequency", "monetary_total", "monetary_avg",
        "rfm_recency_score", "rfm_frequency_score", "rfm_monetary_score",
        "rfm_total_score", "rfm_segment"
    ]]

def compute_behavioral_features(events_df: pd.DataFrame, transactions_df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """
    Compute customer behavioral features from clickstream events.
    """
    df_evt = events_df.copy()
    snapshot_dt = pd.to_datetime(snapshot_date)
    df_evt["event_date"] = pd.to_datetime(df_evt["event_date"])

    # Aggregate basic features
    grouped = df_evt.groupby("customer_id")
    total_sessions = grouped["event_date"].nunique()
    avg_session_duration = grouped["session_duration_sec"].mean()

    # Pivot event types counts
    event_type_counts = df_evt.groupby(["customer_id", "event_type"]).size().unstack(fill_value=0)
    
    total_page_views = event_type_counts.get("page_view", pd.Series(0, index=event_type_counts.index))
    cart_add_count = event_type_counts.get("add_to_cart", pd.Series(0, index=event_type_counts.index))
    purchase_count = event_type_counts.get("purchase", pd.Series(0, index=event_type_counts.index))
    wishlist_count = event_type_counts.get("wishlist_add", pd.Series(0, index=event_type_counts.index))

    cart_to_purchase_ratio = purchase_count / cart_add_count
    cart_to_purchase_ratio = cart_to_purchase_ratio.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    # Preferred Device and Category
    def get_mode(series):
        mode_res = series.mode()
        return mode_res.iloc[0] if not mode_res.empty else "unknown"

    preferred_device = grouped["device_type"].agg(get_mode)
    preferred_category = grouped["page_category"].agg(get_mode)

    # Event recency
    last_event_date = grouped["event_date"].max()
    days_since_last_event = (snapshot_dt - last_event_date).dt.days

    # Event recency trend: linear regression slope over the last 12 weeks
    start_dt = snapshot_dt - pd.Timedelta(weeks=12)
    df_recent = df_evt[(df_evt["event_date"] > start_dt) & (df_evt["event_date"] <= snapshot_dt)].copy()
    
    # Assign week index from 0 to 11
    df_recent["week_idx"] = ((df_recent["event_date"] - start_dt).dt.days // 7).clip(0, 11)

    recent_counts = df_recent.groupby(["customer_id", "week_idx"]).size().unstack(fill_value=0)
    recent_counts = recent_counts.reindex(columns=range(12), fill_value=0)

    # Compute trend slope using precalculated linear regression slope formula
    slopes = pd.Series(0.0, index=df_evt["customer_id"].unique())
    for i in range(12):
        slopes += (i - 5.5) * recent_counts[i]
    slopes = slopes / 143.0

    behavioral_df = pd.DataFrame({
        "total_sessions": total_sessions,
        "avg_session_duration": avg_session_duration,
        "total_page_views": total_page_views,
        "cart_add_count": cart_add_count,
        "cart_to_purchase_ratio": cart_to_purchase_ratio,
        "wishlist_count": wishlist_count,
        "preferred_device": preferred_device,
        "preferred_category": preferred_category,
        "days_since_last_event": days_since_last_event,
        "event_recency_trend": slopes
    })

    behavioral_df = behavioral_df.reindex(df_evt["customer_id"].unique())
    behavioral_df.index.name = "customer_id"
    return behavioral_df

def compute_transaction_features(transactions_df: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """
    Compute transaction-level features.
    """
    df = transactions_df.copy()
    snapshot_dt = pd.to_datetime(snapshot_date)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # Compute net revenue per transaction
    df["net_revenue"] = df["quantity"] * df["unit_price"] * (1.0 - df["discount_pct"])
    df.loc[df["return_flag"] == 1, "net_revenue"] = 0.0

    grouped = df.groupby("customer_id")

    category_diversity = grouped["category"].nunique()
    
    def get_mode(series):
        mode_res = series.mode()
        return mode_res.iloc[0] if not mode_res.empty else "unknown"
        
    preferred_payment = grouped["payment_method"].agg(get_mode)
    avg_discount_received = grouped["discount_pct"].mean()
    return_rate = grouped["return_flag"].mean()

    # rolling net revenue
    rev_30 = df[df["transaction_date"] >= (snapshot_dt - pd.Timedelta(days=30))].groupby("customer_id")["net_revenue"].sum()
    rev_90 = df[df["transaction_date"] >= (snapshot_dt - pd.Timedelta(days=90))].groupby("customer_id")["net_revenue"].sum()
    rev_180 = df[df["transaction_date"] >= (snapshot_dt - pd.Timedelta(days=180))].groupby("customer_id")["net_revenue"].sum()

    # Gap between consecutive purchases
    def calc_gap_stats(dates):
        if len(dates) < 2:
            return pd.Series([np.nan, np.nan], index=["mean", "std"])
        sorted_dates = sorted(dates)
        gaps = [(sorted_dates[i] - sorted_dates[i-1]).days for i in range(1, len(sorted_dates))]
        return pd.Series([float(np.mean(gaps)), float(np.std(gaps))], index=["mean", "std"])

    gap_stats = grouped["transaction_date"].apply(calc_gap_stats).unstack()

    purchase_gap_mean = gap_stats["mean"]
    purchase_gap_std = gap_stats["std"]

    # Peak season (Oct-Dec) purchase ratio
    df["is_peak"] = df["transaction_date"].dt.month.isin([10, 11, 12])
    peak_season_purchase_ratio = df.groupby("customer_id")["is_peak"].mean()

    tx_features = pd.DataFrame({
        "category_diversity": category_diversity,
        "preferred_payment": preferred_payment,
        "avg_discount_received": avg_discount_received,
        "return_rate": return_rate,
        "revenue_last_30d": rev_30,
        "revenue_last_90d": rev_90,
        "revenue_last_180d": rev_180,
        "purchase_gap_mean": purchase_gap_mean,
        "purchase_gap_std": purchase_gap_std,
        "peak_season_purchase_ratio": peak_season_purchase_ratio
    })

    tx_features["revenue_last_30d"] = tx_features["revenue_last_30d"].fillna(0.0)
    tx_features["revenue_last_90d"] = tx_features["revenue_last_90d"].fillna(0.0)
    tx_features["revenue_last_180d"] = tx_features["revenue_last_180d"].fillna(0.0)

    tx_features.index.name = "customer_id"
    return tx_features

def build_master_feature_table(customers_df: pd.DataFrame, rfm_df: pd.DataFrame, behavioral_df: pd.DataFrame, transaction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge RFM, behavioral, and transaction features with customer metadata into a single master table.
    """
    master = customers_df.copy()
    if "customer_id" in master.columns:
        master = master.set_index("customer_id")

    master = master.join(rfm_df, how="left")
    master = master.join(behavioral_df, how="left")
    master = master.join(transaction_df, how="left", lsuffix="_beh", rsuffix="_txn")

    master = master.loc[:, ~master.columns.duplicated()]

    # Fill numeric columns with 0.0
    num_cols = master.select_dtypes(include=[np.number]).columns.tolist()
    if "churn_label" in num_cols:
        num_cols.remove("churn_label")
    master[num_cols] = master[num_cols].fillna(0.0)

    # Fill categorical/string columns
    cat_cols = master.select_dtypes(exclude=[np.number]).columns.tolist()
    if "is_premium" in cat_cols:
        master["is_premium"] = master["is_premium"].fillna(False)
        cat_cols.remove("is_premium")
    master[cat_cols] = master[cat_cols].fillna("unknown")

    # Move churn_label to the end
    if "churn_label" in master.columns:
        cols = [c for c in master.columns if c != "churn_label"] + ["churn_label"]
        master = master[cols]

    return master

# =====================================================================
# Legacy Helper Functions for Compatibility with test_features.py
# =====================================================================

def create_rfm_features(transactions_df):
    snapshot_date = transactions_df["transaction_date"].max().strftime("%Y-%m-%d")
    rfm = compute_rfm_features(transactions_df, snapshot_date).reset_index()
    rfm = rfm.rename(columns={"monetary_total": "monetary"})
    return rfm[["customer_id", "recency_days", "frequency", "monetary"]]

def create_transaction_features(transactions_df):
    df = transactions_df.copy()
    if "revenue" not in df.columns:
        df["revenue"] = df["quantity"] * df["unit_price"] * (1.0 - df["discount_pct"])
        df.loc[df["return_flag"] == 1, "revenue"] = 0.0
    return df.groupby("customer_id").agg(
        avg_discount=("discount_pct", "mean"),
        return_rate=("return_flag", "mean"),
        avg_order_value=("revenue", "mean"),
        total_quantity=("quantity", "sum")
    ).reset_index()

def create_event_features(events_df):
    return events_df.groupby("customer_id").agg(
        total_events=("event_id", "count"),
        avg_session_duration=("session_duration_sec", "mean")
    ).reset_index()

def create_master_feature_table(customers_df, rfm_df, transaction_features_df, event_features_df):
    master_df = customers_df.merge(rfm_df, on="customer_id")
    master_df = master_df.merge(transaction_features_df, on="customer_id")
    master_df = master_df.merge(event_features_df, on="customer_id")
    return master_df