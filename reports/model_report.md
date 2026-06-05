# ShopSense Analytics - Automated Model Performance Report

## Executive Summary
This report summarizes the operational accuracy and financial outcomes of the ShopSense machine learning models.

- **Churn Classifier Status**: PASS (ROC AUC = 1.0000)
- **Revenue Forecast Status**: WARNING (MAPE = 39.67%)

---

## Churn Model Performance
The predictive model was evaluated on a held-out test set:


| Metric | Value |
|---|---|
| ROC AUC | 1.0000 |
| PR AUC | 1.0000 |
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 Score | 1.0000 |
| Balanced Accuracy | 1.0000 |


### Confusion Matrix
```
[[771   0]
 [  0 229]]
```

---

## Revenue Forecast Accuracy
SARIMA time-series model performance evaluated on a 3-month forecast:


| Metric | Value |
|---|---|
| MAE | ₹709,727.93 |
| RMSE | ₹1,008,885.34 |
| MAPE | 39.67% |
| sMAPE | 33.73% |
| R² Score | 0.7281 |


---

## Customer Segments Summary
Clustering profile of the customer base:

| Cluster ID | Size | Churn Rate | Key Features (Mode / Mean) |
|---|---|---|---|
| 0 | 229 | 100.00% | signup_date: 2021-04-09 00:00:00, age: 35.25, gender: M... |
| 1 | 400 | 0.00% | signup_date: 2021-01-28 00:00:00, age: 34.87, gender: M... |
| 2 | 371 | 0.00% | signup_date: 2021-08-05 00:00:00, age: 34.99, gender: F... |


---

## Top 15 Most Important Features
Global feature importance derived from SHAP values:

1. **num__frequency** (mean |SHAP| = 6.4410)
2. **num__age** (mean |SHAP| = 0.0000)
3. **num__recency_days** (mean |SHAP| = 0.0000)
4. **num__monetary_total** (mean |SHAP| = 0.0000)
5. **num__monetary_avg** (mean |SHAP| = 0.0000)
6. **num__total_sessions** (mean |SHAP| = 0.0000)
7. **num__avg_session_duration** (mean |SHAP| = 0.0000)
8. **num__total_page_views** (mean |SHAP| = 0.0000)
9. **num__cart_add_count** (mean |SHAP| = 0.0000)
10. **num__cart_to_purchase_ratio** (mean |SHAP| = 0.0000)
11. **num__wishlist_count** (mean |SHAP| = 0.0000)
12. **num__days_since_last_event** (mean |SHAP| = 0.0000)
13. **num__event_recency_trend** (mean |SHAP| = 0.0000)
14. **num__category_diversity** (mean |SHAP| = 0.0000)
15. **num__avg_discount_received** (mean |SHAP| = 0.0000)


---

## Recommendations
- **Churn Model Status**: Churn model meets production quality standards (ROC AUC >= 0.75). Proceed to serve predictions in production serving environment.
- **Revenue Forecast Action Needed**: Monthly forecasting error (MAPE) exceeds 25%. We recommend adjusting the seasonal ARIMA parameters or including exogenous holiday marketing variables.
