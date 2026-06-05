import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sqlalchemy

def create_and_execute_notebooks():
    os.makedirs("notebooks", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # -------------------------------------------------------------
    # Notebook 1: EDA
    # -------------------------------------------------------------
    nb1 = nbformat.v4.new_notebook()
    nb1.cells = [
        nbformat.v4.new_markdown_cell("# 01 - Exploratory Data Analysis"),
        nbformat.v4.new_code_cell("""
import sqlalchemy
import pandas as pd
from shopsense.data_generator import generate_products, generate_customers, generate_transactions, generate_events
from shopsense.database import create_schema_and_tables, load_dataframe_to_db, execute_query
from shopsense.eda import compute_univariate_stats, churn_distribution_summary, compute_monthly_revenue, compute_cohort_retention, detect_outliers_iqr

engine = sqlalchemy.create_engine('postgresql://postgres:admin123@localhost:5432/postgres')
"""),
        nbformat.v4.new_markdown_cell("## Generate Synthetic Data"),
        nbformat.v4.new_code_cell("""
products_df = generate_products(random_state=42)
customers_df = generate_customers(n_customers=1000, random_state=42)
transactions_df = generate_transactions(customers_df, products_df, random_state=42)
events_df = generate_events(customers_df, products_df, random_state=42)

print(f"Products: {products_df.shape}")
print(f"Customers: {customers_df.shape}")
print(f"Transactions: {transactions_df.shape}")
print(f"Events: {events_df.shape}")
"""),
        nbformat.v4.new_markdown_cell("## Load Data into PostgreSQL Database"),
        nbformat.v4.new_code_cell("""
create_schema_and_tables(engine)
p_rows = load_dataframe_to_db(products_df, "products", engine)
c_rows = load_dataframe_to_db(customers_df, "customers", engine)
t_rows = load_dataframe_to_db(transactions_df, "transactions", engine)
e_rows = load_dataframe_to_db(events_df, "events", engine)

print(f"Loaded {p_rows} products, {c_rows} customers, {t_rows} transactions, {e_rows} events.")
"""),
        nbformat.v4.new_markdown_cell("## Univariate Statistics"),
        nbformat.v4.new_code_cell("""
uni_stats = compute_univariate_stats(customers_df, ["age", "gender", "city", "acquisition_channel"])
uni_stats
"""),
        nbformat.v4.new_markdown_cell("## Churn Distribution Summary"),
        nbformat.v4.new_code_cell("""
churn_summary = churn_distribution_summary(customers_df)
for k, v in churn_summary.items():
    if not isinstance(v, dict):
        print(f"{k}: {v}")
    else:
        print(f"{k}:")
        for sub_k, sub_v in v.items():
            print(f"  {sub_k}: {sub_v}")
"""),
        nbformat.v4.new_markdown_cell("## Monthly Revenue Trend"),
        nbformat.v4.new_code_cell("""
monthly_rev = compute_monthly_revenue(transactions_df)
monthly_rev.head()
"""),
        nbformat.v4.new_markdown_cell("## Cohort Retention Matrix"),
        nbformat.v4.new_code_cell("""
cohort_matrix = compute_cohort_retention(customers_df, transactions_df)
cohort_matrix.head()
"""),
        nbformat.v4.new_markdown_cell("## Outlier Detection using IQR"),
        nbformat.v4.new_code_cell("""
price_outliers = detect_outliers_iqr(transactions_df, "unit_price")
price_outliers
"""),
        nbformat.v4.new_markdown_cell("## Data Visualization"),
        nbformat.v4.new_code_cell("""
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_rev, x="year_month", y="return_adjusted_revenue", marker="o")
plt.title("Return-Adjusted Monthly Revenue Trend")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("reports/revenue_trend.png")
plt.show()
""")
    ]

    # -------------------------------------------------------------
    # Notebook 2: Feature Engineering
    # -------------------------------------------------------------
    nb2 = nbformat.v4.new_notebook()
    nb2.cells = [
        nbformat.v4.new_markdown_cell("# 02 - Feature Engineering and Hypothesis Testing"),
        nbformat.v4.new_code_cell("""
import sqlalchemy
import pandas as pd
from shopsense.database import execute_query
from shopsense.features import compute_rfm_features, compute_behavioral_features, compute_transaction_features, build_master_feature_table
from shopsense.eda import test_premium_vs_nonpremium_aov, test_channel_churn_association, test_revenue_seasonality

engine = sqlalchemy.create_engine('postgresql://postgres:admin123@localhost:5432/postgres')
"""),
        nbformat.v4.new_markdown_cell("## Load Raw Tables from Database"),
        nbformat.v4.new_code_cell("""
customers_df = execute_query("SELECT * FROM shopsense.customers", engine)
transactions_df = execute_query("SELECT * FROM shopsense.transactions", engine)
events_df = execute_query("SELECT * FROM shopsense.events", engine)

print(f"Loaded {len(customers_df)} customers, {len(transactions_df)} transactions, {len(events_df)} events.")
"""),
        nbformat.v4.new_markdown_cell("## Compute Customer-Level Features"),
        nbformat.v4.new_code_cell("""
snapshot_date = "2023-12-31"

rfm_features = compute_rfm_features(transactions_df, snapshot_date)
behavioral_features = compute_behavioral_features(events_df, transactions_df, snapshot_date)
transaction_features = compute_transaction_features(transactions_df, snapshot_date)

print(f"RFM Features Shape: {rfm_features.shape}")
print(f"Behavioral Features Shape: {behavioral_features.shape}")
print(f"Transaction Features Shape: {transaction_features.shape}")
"""),
        nbformat.v4.new_markdown_cell("## Build Master Feature Table"),
        nbformat.v4.new_code_cell("""
master_df = build_master_feature_table(customers_df, rfm_features, behavioral_features, transaction_features)
print(f"Master Feature Table Shape: {master_df.shape}")
master_df.head()
"""),
        nbformat.v4.new_markdown_cell("## Statistical Hypothesis Testing"),
        nbformat.v4.new_code_cell("""
test1 = test_premium_vs_nonpremium_aov(transactions_df, customers_df)
print("Test 1 Result:")
for k, v in test1.items():
    print(f"  {k}: {v}")

test2 = test_channel_churn_association(customers_df)
print("\\nTest 2 Result:")
for k, v in test2.items():
    print(f"  {k}: {v}")

test3 = test_revenue_seasonality(transactions_df)
print("\\nTest 3 Result:")
for k, v in test3.items():
    print(f"  {k}: {v}")
""")
    ]

    # -------------------------------------------------------------
    # Notebook 3: Modeling
    # -------------------------------------------------------------
    nb3 = nbformat.v4.new_notebook()
    nb3.cells = [
        nbformat.v4.new_markdown_cell("# 03 - Machine Learning Model Development"),
        nbformat.v4.new_code_cell("""
import sqlalchemy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from shopsense.database import execute_query
from shopsense.models.churn_model import build_churn_preprocessing_pipeline, train_churn_model, find_optimal_threshold, evaluate_churn_model
from shopsense.models.segmentation_model import prepare_clustering_features, find_optimal_k, train_segmentation_model, profile_clusters
from shopsense.models.revenue_model import prepare_revenue_timeseries, test_stationarity, train_sarima_model, forecast_sarima, evaluate_forecast
from shopsense.evaluation import register_best_model

engine = sqlalchemy.create_engine('postgresql://postgres:admin123@localhost:5432/postgres')
"""),
        nbformat.v4.new_markdown_cell("## Load Master Features"),
        nbformat.v4.new_code_cell("""
# Re-compute to ensure we have the fresh table from Notebook 2
customers_df = execute_query("SELECT * FROM shopsense.customers", engine)
transactions_df = execute_query("SELECT * FROM shopsense.transactions", engine)
events_df = execute_query("SELECT * FROM shopsense.events", engine)

from shopsense.features import compute_rfm_features, compute_behavioral_features, compute_transaction_features, build_master_feature_table
snapshot_date = "2023-12-31"

rfm = compute_rfm_features(transactions_df, snapshot_date)
beh = compute_behavioral_features(events_df, transactions_df, snapshot_date)
txn = compute_transaction_features(transactions_df, snapshot_date)
master_df = build_master_feature_table(customers_df, rfm, beh, txn)
"""),
        nbformat.v4.new_markdown_cell("## Churn Prediction Model (XGBoost)"),
        nbformat.v4.new_code_cell("""
numeric_features = [
    "age", "recency_days", "frequency", "monetary_total", "monetary_avg",
    "total_sessions", "avg_session_duration", "total_page_views",
    "cart_add_count", "cart_to_purchase_ratio", "wishlist_count",
    "days_since_last_event", "event_recency_trend", "category_diversity",
    "avg_discount_received", "return_rate", "revenue_last_30d",
    "revenue_last_90d", "revenue_last_180d", "purchase_gap_mean",
    "purchase_gap_std", "peak_season_purchase_ratio"
]
categorical_features = ["gender", "city", "acquisition_channel", "preferred_device", "preferred_category", "preferred_payment", "rfm_segment"]

# Setup preprocessing
preprocessor = build_churn_preprocessing_pipeline(numeric_features, categorical_features)

X = master_df[numeric_features + categorical_features]
y = master_df["churn_label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit Churn Model (XGBoost)
churn_pipeline = train_churn_model(X_train, y_train, preprocessor, model_type="xgboost", random_state=42)
print("Model trained and logged to MLflow successfully.")
"""),
        nbformat.v4.new_markdown_cell("## Optimize Classification Threshold"),
        nbformat.v4.new_code_cell("""
threshold_results = find_optimal_threshold(churn_pipeline, X_test, y_test, metric="f1")
opt_thresh = threshold_results["optimal_threshold"]
print(f"Optimal Threshold (F1): {opt_thresh:.2f}")
print(f"Optimal F1 Value: {threshold_results['optimal_metric_value']:.4f}")
"""),
        nbformat.v4.new_markdown_cell("## Churn Model Evaluation"),
        nbformat.v4.new_code_cell("""
eval_metrics = evaluate_churn_model(churn_pipeline, X_test, y_test, threshold=opt_thresh)
for k, v in eval_metrics.items():
    if k != "classification_report":
        print(f"{k}: {v}")
print("\\nClassification Report:\\n", eval_metrics["classification_report"])
"""),
        nbformat.v4.new_markdown_cell("## Unsupervised Customer Segmentation (K-Means)"),
        nbformat.v4.new_code_cell("""
X_scaled, cluster_features, scaler = prepare_clustering_features(master_df)

# Optimal K
k_selection = find_optimal_k(X_scaled, range(2, 6))
print(f"Optimal K (Silhouette): {k_selection['optimal_k_silhouette']}")
print(f"Optimal K (Calinski-Harabasz): {k_selection['optimal_k_calinski']}")

# Fit K-Means
kmeans_model, cluster_labels = train_segmentation_model(X_scaled, n_clusters=4, random_state=42)

# Profile Clusters
profile_df = profile_clusters(master_df, cluster_labels, cluster_features)
profile_df
"""),
        nbformat.v4.new_markdown_cell("## Revenue Time Series Forecasting (SARIMA)"),
        nbformat.v4.new_code_cell("""
timeseries = prepare_revenue_timeseries(transactions_df, freq="M")

# Stationarity ADF Test
adf_res = test_stationarity(timeseries)
print("ADF Statistic:", adf_res["adf_statistic"])
print("p-value:", adf_res["p_value"])
print("Is Stationary:", adf_res["is_stationary"])

# Fit SARIMA Model (p=1, d=1, q=1, s=12)
sarima_res, fitted = train_sarima_model(timeseries, order=(1, adf_res["recommended_differencing"], 1), seasonal_order=(1, 0, 0, 12))

# Forecast next 3 months
forecast_df = forecast_sarima(sarima_res, steps=3)
print("\\n3-Month Forecast:")
print(forecast_df)

# Evaluate Forecast
forecast_metrics = evaluate_forecast(timeseries, fitted)
print("\\nForecast Training Metrics:")
for k, v in forecast_metrics.items():
    print(f"  {k}: {v}")
"""),
        nbformat.v4.new_markdown_cell("## Register Model to MLflow Model Registry"),
        nbformat.v4.new_code_cell("""
try:
    version = register_best_model(model_name="shopsense_churn_model")
    print(f"Successfully registered model to registry. Version: {version}")
except Exception as e:
    print(f"Could not register model: {e}")
""")
    ]

    # -------------------------------------------------------------
    # Notebook 4: Explainability
    # -------------------------------------------------------------
    nb4 = nbformat.v4.new_notebook()
    nb4.cells = [
        nbformat.v4.new_markdown_cell("# 04 - Model Explainability and Business Impact"),
        nbformat.v4.new_code_cell("""
import sqlalchemy
import pandas as pd
import numpy as np
import mlflow
from shopsense.database import execute_query
from shopsense.features import compute_rfm_features, compute_behavioral_features, compute_transaction_features, build_master_feature_table
from shopsense.models.churn_model import evaluate_churn_model
from shopsense.evaluation import compute_shap_values, explain_single_prediction, compute_shap_dependence, estimate_churn_business_impact, detect_feature_drift, generate_model_report

engine = sqlalchemy.create_engine('postgresql://postgres:admin123@localhost:5432/postgres')
"""),
        nbformat.v4.new_markdown_cell("## Rebuild Master Feature Table"),
        nbformat.v4.new_code_cell("""
customers_df = execute_query("SELECT * FROM shopsense.customers", engine)
transactions_df = execute_query("SELECT * FROM shopsense.transactions", engine)
events_df = execute_query("SELECT * FROM shopsense.events", engine)

snapshot_date = "2023-12-31"
rfm = compute_rfm_features(transactions_df, snapshot_date)
beh = compute_behavioral_features(events_df, transactions_df, snapshot_date)
txn = compute_transaction_features(transactions_df, snapshot_date)
master_df = build_master_feature_table(customers_df, rfm, beh, txn)
"""),
        nbformat.v4.new_markdown_cell("## Load Best Registered Model"),
        nbformat.v4.new_code_cell("""
# Load staging model
try:
    model = mlflow.sklearn.load_model("models:/shopsense_churn_model/Staging")
    print("Model loaded from MLflow Model Registry Staging stage.")
except Exception as e:
    # Try alias
    try:
        model = mlflow.sklearn.load_model("models:/shopsense_churn_model@staging")
        print("Model loaded from MLflow Model Registry using staging alias.")
    except Exception:
        # Fallback to local runs or direct training for explainability
        print("Staging model not found in registry. Training a local XGBoost pipeline for explainability.")
        from sklearn.pipeline import Pipeline
        from shopsense.models.churn_model import build_churn_preprocessing_pipeline, train_churn_model
        numeric_features = [
            "age", "recency_days", "frequency", "monetary_total", "monetary_avg",
            "total_sessions", "avg_session_duration", "total_page_views",
            "cart_add_count", "cart_to_purchase_ratio", "wishlist_count",
            "days_since_last_event", "event_recency_trend", "category_diversity",
            "avg_discount_received", "return_rate", "revenue_last_30d",
            "revenue_last_90d", "revenue_last_180d", "purchase_gap_mean",
            "purchase_gap_std", "peak_season_purchase_ratio"
        ]
        categorical_features = ["gender", "city", "acquisition_channel", "preferred_device", "preferred_category", "preferred_payment", "rfm_segment"]
        preprocessor = build_churn_preprocessing_pipeline(numeric_features, categorical_features)
        X = master_df[numeric_features + categorical_features]
        y = master_df["churn_label"]
        model = train_churn_model(X, y, preprocessor, model_type="xgboost", random_state=42)
"""),
        nbformat.v4.new_markdown_cell("## Compute SHAP Explanations"),
        nbformat.v4.new_code_cell("""
# Extract preprocessor and transform features
preprocessor = model.steps[0][1]
numeric_features = [
    "age", "recency_days", "frequency", "monetary_total", "monetary_avg",
    "total_sessions", "avg_session_duration", "total_page_views",
    "cart_add_count", "cart_to_purchase_ratio", "wishlist_count",
    "days_since_last_event", "event_recency_trend", "category_diversity",
    "avg_discount_received", "return_rate", "revenue_last_30d",
    "revenue_last_90d", "revenue_last_180d", "purchase_gap_mean",
    "purchase_gap_std", "peak_season_purchase_ratio"
]
categorical_features = ["gender", "city", "acquisition_channel", "preferred_device", "preferred_category", "preferred_payment", "rfm_segment"]
X = master_df[numeric_features + categorical_features]

X_prep = pd.DataFrame(preprocessor.transform(X), columns=preprocessor.get_feature_names_out())

shap_results = compute_shap_values(model, X_prep, model_type="xgboost")
print("SHAP expected value:", shap_results["expected_value"])
print("\\nTop 5 contributing features (mean absolute SHAP):")
print(shap_results["mean_abs_shap"].head())
"""),
        nbformat.v4.new_markdown_cell("## Single Prediction Local Explanation"),
        nbformat.v4.new_code_cell("""
local_exp = explain_single_prediction(
    shap_results["shap_values"],
    shap_results["feature_names"],
    shap_results["expected_value"],
    instance_index=0,
    top_n=5,
    X_sample=X_prep
)
print("Local explanation for customer at index 0:")
local_exp
"""),
        nbformat.v4.new_markdown_cell("## Business Impact Estimation"),
        nbformat.v4.new_code_cell("""
business_impact = estimate_churn_business_impact(customers_df, transactions_df, model, X, threshold=0.5)
print(f"Predicted Churners: {business_impact['predicted_churners_count']}")
print(f"Total Revenue At Risk: ₹{business_impact['total_revenue_at_risk']:,.2f}")
print(f"Potential Revenue Saved (30% retention): ₹{business_impact['potential_save_revenue_30pct']:,.2f}")
print(f"Potential Revenue Saved (50% retention): ₹{business_impact['potential_save_revenue_50pct']:,.2f}")
print("\\nTop 5 highest risk customers (by revenue):")
print(business_impact["top_10_revenue_at_risk_customers"][:5])
"""),
        nbformat.v4.new_markdown_cell("## Feature Drift Detection"),
        nbformat.v4.new_code_cell("""
# Split master table into reference (past) and current (simulating new data)
ref_df = master_df.iloc[:500]
cur_df = master_df.iloc[500:]

drift_report = detect_feature_drift(ref_df, cur_df, numeric_features, categorical_features)
print("Feature Drift Analysis (p-value < 0.05 indicates drift):")
drift_report
"""),
        nbformat.v4.new_markdown_cell("## Generate Automated Model Report"),
        nbformat.v4.new_code_cell("""
# Re-evaluate models to pass to report
from shopsense.models.churn_model import evaluate_churn_model
from shopsense.models.revenue_model import evaluate_forecast
from shopsense.models.segmentation_model import prepare_clustering_features, train_segmentation_model, profile_clusters

y_test = master_df["churn_label"]
churn_eval = evaluate_churn_model(model, X, y_test, threshold=0.5)

# Forecast metrics
tx = transactions_df.copy()
tx["transaction_date"] = pd.to_datetime(tx["transaction_date"])
tx["net_revenue"] = tx["quantity"] * tx["unit_price"] * (1.0 - tx["discount_pct"])
tx.loc[tx["return_flag"] == 1, "net_revenue"] = 0.0
ts = tx.set_index("transaction_date")["net_revenue"].resample("M").sum()
from shopsense.models.revenue_model import train_sarima_model
res_ts, fitted_ts = train_sarima_model(ts, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12))
forecast_eval = evaluate_forecast(ts, fitted_ts)

# Clustering profiling
X_cl_scaled, cl_features, _ = prepare_clustering_features(master_df)
kmeans, labels = train_segmentation_model(X_cl_scaled, n_clusters=3, random_state=42)
cluster_profile = profile_clusters(master_df, labels, cl_features)

report_path = generate_model_report(churn_eval, forecast_eval, cluster_profile, shap_results)
print(f"Generated automated report successfully at: {report_path}")
""")
    ]

    # Save all notebooks
    notebook_files = {
        "notebooks/01_eda.ipynb": nb1,
        "notebooks/02_feature_engineering.ipynb": nb2,
        "notebooks/03_modeling.ipynb": nb3,
        "notebooks/04_explainability.ipynb": nb4
    }

    for path, nb in notebook_files.items():
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Created notebook: {path}")

    # Execute all notebooks using ExecutePreprocessor
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    for path in notebook_files.keys():
        print(f"Executing notebook: {path}...")
        with open(path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        
        try:
            ep.preprocess(nb, {"metadata": {"path": "."}})
            with open(path, "w", encoding="utf-8") as f:
                nbformat.write(nb, f)
            print(f"Successfully executed and saved: {path}")
        except Exception as e:
            print(f"Error executing {path}: {e}")

if __name__ == "__main__":
    create_and_execute_notebooks()
