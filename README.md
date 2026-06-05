# ShopSense Analytics - Customer Intelligence Data Science Pipeline

An end-to-end, production-ready data science pipeline addressing key business challenges at ShopSense Analytics: customer churn prediction, customer segmentation, and monthly revenue forecasting.

---

## 1. Project Overview & Architecture

The pipeline consists of the following components:
1. **Synthetic Data Generator** (`shopsense/data_generator.py`): Vectorized generator producing realistic demographic data for 10,000 customers, 150,000+ transactions (with seasonal peaks in Oct-Dec), and clickstream events.
2. **Database Ingestion** (`shopsense/database.py`): Sets up a PostgreSQL schema (`shopsense`) and loads dataframes using SQLAlchemy 2.0.
3. **Exploratory Data Analysis (EDA)** (`shopsense/eda.py`): Performs demographic summaries, monthly revenue aggregation, cohort retention analysis, and runs statistical hypothesis tests (Mann-Whitney U, Chi-Square, Kruskal-Wallis).
4. **Feature Engineering** (`shopsense/features.py`): Computes customer-level RFM parameters, behavioral clickstream features, and rolling transaction aggregations into a unified master feature table.
5. **Machine Learning Models** (`shopsense/models/`):
   - Churn Classification (XGBoost with class balance weighting and threshold optimization).
   - Unsupervised Segmentation (K-Means clustering and profile aggregation).
   - Time Series Forecasting (Seasonal ARIMA for revenue forecasting).
6. **Model Evaluation & Registry** (`shopsense/evaluation.py`): Tracks experiments with MLflow, registers model versions to "Staging", computes SHAP feature importances, and compiles performance metrics.
7. **Serving API** (`shopsense/serving/api.py`): REST API built with FastAPI that loads the registered staging model from the MLflow SQLite backend to serve single and batch churn predictions.

---

## 2. Tech Stack

- **Core**: Python 3.10+, PostgreSQL, SQLAlchemy 2.0, psycopg2-binary
- **Analytics & ML**: pandas, numpy, scikit-learn, XGBoost, statsmodels, shap, scipy
- **Experiment Tracking**: MLflow
- **Model Serving**: FastAPI, uvicorn
- **Reports**: fpdf2 (PDF generation)
- **Testing**: pytest

---

## 3. Local Setup & Installation

### 1. Database Configuration
Ensure a local PostgreSQL instance is running on `localhost:5432` with username `postgres` and password `admin123`.

### 2. Package Installation
Clone the repository and install the package in editable mode:
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 4. How to Run the Pipeline Locally

### 1. Run All Notebooks (Data Generation, Ingestion, Modeling, & Explainability)
Run the automated notebook creator and runner script. This will generate, execute, and save all four notebooks under the `notebooks/` directory and log runs/models to the MLflow registry:
```bash
python generate_notebooks.py
```

This creates and executes:
- `notebooks/01_eda.ipynb`
- `notebooks/02_feature_engineering.ipynb`
- `notebooks/03_modeling.ipynb`
- `notebooks/04_explainability.ipynb`

### 2. Generate PDF Report
Run the PDF compiler to produce a print-ready exploratory analysis report under `reports/eda_report.pdf`:
```bash
python generate_pdf_report.py
```

### 3. Start the Churn Serving API
Run the FastAPI application:
```bash
uvicorn shopsense.serving.api:app --reload
```
The interactive Swagger API documentation will be available at `http://127.0.0.1:8000/docs`.

---

## 5. Running in Docker Containers

You can build and spin up the complete environment (PostgreSQL Database, MLflow Tracking Server, and FastAPI API) using Docker Compose:

1. **Spin up the services**:
   ```bash
   docker-compose up --build
   ```
2. **Access the services**:
   - **FastAPI API Service**: `http://localhost:8000` (interactive documentation at `http://localhost:8000/docs`)
   - **MLflow Tracking Server**: `http://localhost:5000`
   - **PostgreSQL Database**: `localhost:5432`

---

## 6. How to Query the Churn API

### 1. Health Check
```bash
curl -X GET http://localhost:8000/health
```

### 2. Single Customer Churn Prediction
```bash
curl -X POST http://localhost:8000/predict/churn \
     -H "Content-Type: application/json" \
     -d '{
       "customer_features": {
         "customer_id": "CUST_000001",
         "age": 35,
         "gender": "M",
         "city": "Mumbai",
         "acquisition_channel": "organic",
         "is_premium": true,
         "recency_days": 15,
         "frequency": 25,
         "monetary_total": 45000.0,
         "monetary_avg": 1800.0,
         "total_sessions": 120,
         "avg_session_duration": 340.5,
         "total_page_views": 450,
         "cart_add_count": 35,
         "cart_to_purchase_ratio": 0.714,
         "wishlist_count": 12,
         "days_since_last_event": 2,
         "event_recency_trend": 0.15,
         "category_diversity": 4,
         "preferred_device": "mobile",
         "preferred_category": "Electronics",
         "preferred_payment": "UPI",
         "avg_discount_received": 0.10,
         "return_rate": 0.04,
         "revenue_last_30d": 12000.0,
         "revenue_last_90d": 35000.0,
         "revenue_last_180d": 45000.0,
         "purchase_gap_mean": 12.5,
         "purchase_gap_std": 3.4,
         "peak_season_purchase_ratio": 0.28,
         "rfm_segment": "Champions"
       }
     }'
```

### 3. Batch Churn Prediction
```bash
curl -X POST http://localhost:8000/predict/churn/batch \
     -H "Content-Type: application/json" \
     -d '{
       "customers": [
         {
           "customer_id": "CUST_000001",
           "features": {
             "age": 35,
             "gender": "M",
             "city": "Mumbai",
             "acquisition_channel": "organic",
             "is_premium": true,
             "recency_days": 15,
             "frequency": 25,
             "monetary_total": 45000.0,
             "monetary_avg": 1800.0,
             "total_sessions": 120,
             "avg_session_duration": 340.5,
             "total_page_views": 450,
             "cart_add_count": 35,
             "cart_to_purchase_ratio": 0.714,
             "wishlist_count": 12,
             "days_since_last_event": 2,
             "event_recency_trend": 0.15,
             "category_diversity": 4,
             "preferred_device": "mobile",
             "preferred_category": "Electronics",
             "preferred_payment": "UPI",
             "avg_discount_received": 0.10,
             "return_rate": 0.04,
             "revenue_last_30d": 12000.0,
             "revenue_last_90d": 35000.0,
             "revenue_last_180d": 45000.0,
             "purchase_gap_mean": 12.5,
             "purchase_gap_std": 3.4,
             "peak_season_purchase_ratio": 0.28,
             "rfm_segment": "Champions"
           }
         }
       ]
     }'
```

---

## 7. Testing & Verification

Run the automated test suite using pytest:
```bash
pytest -v
```

The pytest runner is configured via `pytest.ini` to isolate tests under the `tests/` directory, verifying:
- RFM and behavioral feature compilation
- Churn model prep, training, and threshold curves
- Clustering profile generation
- Time series stationarity and forecasting
- SHAP explainability calculations and feature drift tests