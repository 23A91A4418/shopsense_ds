import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import shap
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, precision_recall_curve, auc
)

def compute_shap_values(model, X_sample: pd.DataFrame, model_type: str = "xgboost") -> dict:
    """
    Compute SHAP values for a given model and preprocessed sample matrix.
    """
    # Extract the classifier from the pipeline if it is a pipeline
    if hasattr(model, "steps"):
        classifier = model.steps[-1][1]
    else:
        classifier = model

    # Check model type to select the right explainer
    if model_type == "logistic_regression":
        explainer = shap.LinearExplainer(classifier, X_sample)
        shap_vals = explainer.shap_values(X_sample)
    else:
        explainer = shap.TreeExplainer(classifier)
        shap_vals = explainer.shap_values(X_sample)

    # Handle multi-class output (list of arrays)
    if isinstance(shap_vals, list):
        if len(shap_vals) == 2:
            shap_vals = shap_vals[1]
        else:
            shap_vals = shap_vals[0]
            
    # Handle the new SHAP Explanation object
    if hasattr(shap_vals, "values"):
        shap_values_array = shap_vals.values
        expected_value = float(shap_vals.base_values[0]) if isinstance(shap_vals.base_values, (list, np.ndarray)) else float(shap_vals.base_values)
    else:
        shap_values_array = shap_vals
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = float(expected_value[1]) if len(expected_value) == 2 else float(expected_value[0])
        else:
            expected_value = float(expected_value)

    feature_names = X_sample.columns.tolist()
    mean_abs = np.abs(shap_values_array).mean(axis=0)
    mean_abs_shap = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)

    return {
        "shap_values": shap_values_array,
        "expected_value": expected_value,
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs_shap
    }

def explain_single_prediction(shap_values: np.ndarray, feature_names: list, expected_value: float, instance_index: int, top_n: int = 10, X_sample: pd.DataFrame = None) -> pd.DataFrame:
    """
    Explain an individual prediction by returning top contributing features.
    """
    instance_shap = shap_values[instance_index]
    records = []
    
    for i, val in enumerate(instance_shap):
        f_name = feature_names[i]
        f_val = X_sample.iloc[instance_index, i] if X_sample is not None else np.nan
        records.append({
            "feature": f_name,
            "shap_value": float(val),
            "feature_value": f_val
        })

    df = pd.DataFrame(records)
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).drop(columns=["abs_shap"])
    return df.head(top_n).reset_index(drop=True)

def compute_shap_dependence(shap_values: np.ndarray, feature_names: list, X_sample: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    Extract feature values and SHAP values for a feature to perform dependence analysis.
    """
    f_idx = feature_names.index(feature_name)
    feat_values = X_sample.iloc[:, f_idx].values if isinstance(X_sample, pd.DataFrame) else X_sample[:, f_idx]
    shap_vals = shap_values[:, f_idx]
    
    return pd.DataFrame({
        "feature_value": feat_values,
        "shap_value": shap_vals
    })

def compare_models(models_dict: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Compare multiple fitted classification pipelines on the test set.
    """
    rows = []
    for name, pipeline in models_dict.items():
        y_probs = pipeline.predict_proba(X_test)[:, 1]
        y_preds = (y_probs >= 0.5).astype(int)

        roc_auc = float(roc_auc_score(y_test, y_probs))
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = float(auc(recall_curve, precision_curve))
        acc = float(accuracy_score(y_test, y_preds))
        prec = float(precision_score(y_test, y_preds, zero_division=0))
        rec = float(recall_score(y_test, y_preds, zero_division=0))
        f1 = float(f1_score(y_test, y_preds, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_test, y_preds))

        rows.append({
            "model_name": name,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "balanced_accuracy": bal_acc,
            "fit_time_sec": 0.0
        })

    df = pd.DataFrame(rows).set_index("model_name")
    return df.sort_values("roc_auc", ascending=False)

def estimate_churn_business_impact(customers_df: pd.DataFrame, transactions_df: pd.DataFrame, model, X: pd.DataFrame, threshold: float = 0.5) -> dict:
    """
    Estimate the business and financial impact of predicted customer churn.
    """
    y_probs = model.predict_proba(X)[:, 1]
    predicted_churners_mask = y_probs >= threshold
    predicted_churner_ids = X.index[predicted_churners_mask].tolist()
    
    # 12-month net revenue calculation
    tx = transactions_df.copy()
    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
    max_date = tx["transaction_date"].max()
    cutoff_date = max_date - pd.Timedelta(days=365)

    tx_12m = tx[tx["transaction_date"] >= cutoff_date].copy()
    tx_12m["net_rev"] = tx_12m["quantity"] * tx_12m["unit_price"] * (1.0 - tx_12m["discount_pct"])
    tx_12m.loc[tx_12m["return_flag"] == 1, "net_rev"] = 0.0

    customer_rev_12m = tx_12m.groupby("customer_id")["net_rev"].sum().to_dict()

    churner_revs = [float(customer_rev_12m.get(cid, 0.0)) for cid in predicted_churner_ids]
    
    avg_rev = float(np.mean(churner_revs)) if len(churner_revs) > 0 else 0.0
    total_rev_at_risk = float(np.sum(churner_revs))
    pot_save_30 = total_rev_at_risk * 0.30
    pot_save_50 = total_rev_at_risk * 0.50

    # Sort predicted churners by revenue descending
    churner_df = pd.DataFrame({
        "customer_id": predicted_churner_ids,
        "revenue_12m": churner_revs
    })
    top_10 = churner_df.sort_values("revenue_12m", ascending=False).head(10)["customer_id"].tolist()

    return {
        "predicted_churners_count": int(len(predicted_churner_ids)),
        "predicted_churner_ids": predicted_churner_ids,
        "avg_revenue_per_churner_12m": avg_rev,
        "total_revenue_at_risk": total_rev_at_risk,
        "potential_save_revenue_30pct": pot_save_30,
        "potential_save_revenue_50pct": pot_save_50,
        "top_10_revenue_at_risk_customers": top_10
    }

def detect_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, numeric_features: list, categorical_features: list) -> pd.DataFrame:
    """
    Perform Kolmogorov-Smirnov test (numeric) and Chi-Square test (categorical) to detect feature drift.
    """
    rows = []

    # Kolmogorov-Smirnov test for numeric features
    for col in numeric_features:
        if col in reference_df.columns and col in current_df.columns:
            ref_vals = reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values
            if len(ref_vals) > 0 and len(cur_vals) > 0:
                stat, p_val = stats.ks_2samp(ref_vals, cur_vals)
                stat, p_val = float(stat), float(p_val)
            else:
                stat, p_val = 0.0, 1.0
            
            rows.append({
                "feature": col,
                "feature_type": "numeric",
                "test_name": "Kolmogorov-Smirnov",
                "statistic": stat,
                "p_value": p_val,
                "drift_detected": p_val < 0.05
            })

    # Chi-Square test for categorical features
    for col in categorical_features:
        if col in reference_df.columns and col in current_df.columns:
            ref_series = reference_df[col].dropna()
            cur_series = current_df[col].dropna()
            categories = list(set(ref_series.unique()) | set(cur_series.unique()))
            
            if len(categories) <= 1:
                rows.append({
                    "feature": col,
                    "feature_type": "categorical",
                    "test_name": "Chi-Square",
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "drift_detected": False
                })
                continue

            ref_counts = ref_series.value_counts().reindex(categories, fill_value=0).values
            cur_counts = cur_series.value_counts().reindex(categories, fill_value=0).values
            
            contingency = np.array([ref_counts, cur_counts])
            if contingency.sum() == 0:
                stat, p_val = 0.0, 1.0
            else:
                try:
                    stat, p_val, _, _ = stats.chi2_contingency(contingency)
                    stat, p_val = float(stat), float(p_val)
                except Exception:
                    stat, p_val = 0.0, 1.0

            rows.append({
                "feature": col,
                "feature_type": "categorical",
                "test_name": "Chi-Square",
                "statistic": stat,
                "p_value": p_val,
                "drift_detected": p_val < 0.05
            })

    df_drift = pd.DataFrame(rows)
    if not df_drift.empty:
        df_drift = df_drift.sort_values("p_value").reset_index(drop=True)
    return df_drift

def get_best_mlflow_run(experiment_name: str = "shopsense_churn_experiment", metric: str = "roc_auc") -> dict:
    """
    Search and retrieve the best MLflow run based on a given metric.
    """
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return {}

    # Map standard metric to the validation metric logged
    logged_metric = f"val_{metric}"
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{logged_metric} DESC"],
        max_results=1
    )

    if not runs:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=1
        )

    if not runs:
        return {}

    run = runs[0]
    val = run.data.metrics.get(logged_metric, run.data.metrics.get(metric, 0.0))

    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "best_metric_value": float(val),
        "params": run.data.params,
        "tags": run.data.tags
    }

def register_best_model(experiment_name: str = "shopsense_churn_experiment", model_name: str = "shopsense_churn_model") -> str:
    """
    Register the best model from MLflow run registry and transition it to Staging.
    """
    run_info = get_best_mlflow_run(experiment_name, "roc_auc")
    if not run_info:
        raise ValueError("No runs found in the experiment to register.")

    run_id = run_info["run_id"]
    model_uri = f"runs:/{run_id}/model"
    
    # Register model
    model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
    version = model_details.version

    # Transition model version to Staging stage
    client = mlflow.tracking.MlflowClient()
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )
    except Exception:
        # Fallback to aliases for MLflow 2.9+
        try:
            client.set_registered_model_alias(model_name, "staging", version)
        except Exception:
            pass

    return str(version)

def generate_model_report(churn_eval_dict: dict, forecast_eval_dict: dict, cluster_profile_df: pd.DataFrame, shap_results: dict, output_path: str = "reports/model_report.md") -> str:
    """
    Generate an automated model performance report in Markdown.
    """
    # Ensure directory exists
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # Build Executive Summary and Recommendations
    auc_val = churn_eval_dict.get("roc_auc", 0.0)
    mape_val = forecast_eval_dict.get("mape", 100.0)
    
    recs = []
    if auc_val < 0.75:
        status_auc = "WARNING"
        recs.append("- **Churn Model Action Needed**: Churn prediction AUC is below target threshold of 0.75. We recommend collecting more temporal event features or engineering custom seasonal transaction statistics.")
    else:
        status_auc = "PASS"
        recs.append("- **Churn Model Status**: Churn model meets production quality standards (ROC AUC >= 0.75). Proceed to serve predictions in production serving environment.")

    if mape_val > 25.0:
        status_mape = "WARNING"
        recs.append("- **Revenue Forecast Action Needed**: Monthly forecasting error (MAPE) exceeds 25%. We recommend adjusting the seasonal ARIMA parameters or including exogenous holiday marketing variables.")
    else:
        status_mape = "PASS"
        recs.append("- **Revenue Forecast Status**: Forecast accuracy is within safe operational margin (MAPE <= 25%). Can be safely used for financial and logistics planning.")

    # Format Churn table
    churn_table = f"""
| Metric | Value |
|---|---|
| ROC AUC | {churn_eval_dict.get('roc_auc', 0.0):.4f} |
| PR AUC | {churn_eval_dict.get('pr_auc', 0.0):.4f} |
| Accuracy | {churn_eval_dict.get('accuracy', 0.0):.4f} |
| Precision | {churn_eval_dict.get('precision', 0.0):.4f} |
| Recall | {churn_eval_dict.get('recall', 0.0):.4f} |
| F1 Score | {churn_eval_dict.get('f1', 0.0):.4f} |
| Balanced Accuracy | {churn_eval_dict.get('balanced_accuracy', 0.0):.4f} |
"""

    # Format Forecast table
    forecast_table = f"""
| Metric | Value |
|---|---|
| MAE | ₹{forecast_eval_dict.get('mae', 0.0):,.2f} |
| RMSE | ₹{forecast_eval_dict.get('rmse', 0.0):,.2f} |
| MAPE | {forecast_eval_dict.get('mape', 0.0):.2f}% |
| sMAPE | {forecast_eval_dict.get('smape', 0.0):.2f}% |
| R² Score | {forecast_eval_dict.get('r2', 0.0):.4f} |
"""

    # Format Cluster Summary
    cluster_table = "| Cluster ID | Size | Churn Rate | Key Features (Mode / Mean) |\n|---|---|---|---|\n"
    for cid, row in cluster_profile_df.iterrows():
        size = row.get("cluster_size", 0)
        crate = row.get("churn_rate", 0.0)
        # Gather top features
        details = []
        for col in cluster_profile_df.columns:
            if col in ["cluster_size", "churn_rate"]:
                continue
            val = row[col]
            if isinstance(val, float):
                details.append(f"{col}: {val:.2f}")
            else:
                details.append(f"{col}: {val}")
        details_str = ", ".join(details[:3]) + "..."
        cluster_table += f"| {cid} | {size} | {crate:.2%} | {details_str} |\n"

    # Format SHAP features
    shap_series = shap_results.get("mean_abs_shap", pd.Series())
    shap_list = ""
    for idx, (feat, val) in enumerate(shap_series.head(15).items()):
        shap_list += f"{idx+1}. **{feat}** (mean |SHAP| = {val:.4f})\n"

    recs_str = "\n".join(recs)

    markdown_content = f"""# ShopSense Analytics - Automated Model Performance Report

## Executive Summary
This report summarizes the operational accuracy and financial outcomes of the ShopSense machine learning models.

- **Churn Classifier Status**: {status_auc} (ROC AUC = {auc_val:.4f})
- **Revenue Forecast Status**: {status_mape} (MAPE = {mape_val:.2f}%)

---

## Churn Model Performance
The predictive model was evaluated on a held-out test set:

{churn_table}

### Confusion Matrix
```
{np.array(churn_eval_dict.get('confusion_matrix', [[0, 0], [0, 0]]))}
```

---

## Revenue Forecast Accuracy
SARIMA time-series model performance evaluated on a 3-month forecast:

{forecast_table}

---

## Customer Segments Summary
Clustering profile of the customer base:

{cluster_table}

---

## Top 15 Most Important Features
Global feature importance derived from SHAP values:

{shap_list}

---

## Recommendations
{recs_str}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    return output_path
