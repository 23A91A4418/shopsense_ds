import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def prepare_clustering_features(master_df: pd.DataFrame) -> tuple:
    """
    Prepare numeric features for clustering by scaling them.
    Excludes labels, customer_id, and string-typed columns.
    """
    df = master_df.copy()
    
    # Drop known non-feature columns
    columns_to_drop = ["churn_label", "customer_id", "customer_profile"]
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Select only numeric, non-boolean columns
    numeric_cols = []
    for col in df.columns:
        col_series = df[col]
        if pd.api.types.is_numeric_dtype(col_series) and not pd.api.types.is_bool_dtype(col_series):
            numeric_cols.append(col)

    numeric_df = df[numeric_cols]
    feature_names = list(numeric_df.columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    return X_scaled, feature_names, scaler

def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> dict:
    """
    Evaluate KMeans for different values of K and return scores.
    """
    inertia_values = []
    silhouette_scores = []
    calinski_harabasz_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertia_values.append(float(kmeans.inertia_))
        silhouette_scores.append(float(silhouette_score(X_scaled, labels)))
        calinski_harabasz_scores.append(float(calinski_harabasz_score(X_scaled, labels)))

    optimal_k_silhouette = int(k_range[np.argmax(silhouette_scores)])
    optimal_k_calinski = int(k_range[np.argmax(calinski_harabasz_scores)])

    return {
        "inertia_values": inertia_values,
        "silhouette_scores": silhouette_scores,
        "calinski_harabasz_scores": calinski_harabasz_scores,
        "optimal_k_silhouette": optimal_k_silhouette,
        "optimal_k_calinski": optimal_k_calinski
    }

def train_segmentation_model(X_scaled: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple:
    """
    Train a KMeans segmentation model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return kmeans, cluster_labels

def profile_clusters(master_df: pd.DataFrame, cluster_labels: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Profile each cluster by calculating mean of numeric features and mode of categorical features.
    """
    df = master_df.copy()
    df["cluster"] = cluster_labels

    profile_rows = []
    unique_clusters = sorted(np.unique(cluster_labels))
    all_cols = [c for c in df.columns if c != "cluster"]

    for cluster_id in unique_clusters:
        cluster_df = df[df["cluster"] == cluster_id]
        row_profile = {
            "cluster_size": int(len(cluster_df))
        }

        # Calculate churn rate
        if "churn_label" in df.columns:
            row_profile["churn_rate"] = float(cluster_df["churn_label"].mean())
        else:
            row_profile["churn_rate"] = 0.0

        for col in all_cols:
            if col == "churn_label":
                continue
            col_series = cluster_df[col]
            if pd.api.types.is_numeric_dtype(col_series) and not pd.api.types.is_bool_dtype(col_series):
                row_profile[col] = float(col_series.mean())
            else:
                mode_res = col_series.mode()
                row_profile[col] = mode_res.iloc[0] if not mode_res.empty else "unknown"

        profile_rows.append(row_profile)

    profile_df = pd.DataFrame(profile_rows, index=unique_clusters)
    profile_df.index.name = "cluster"
    return profile_df
