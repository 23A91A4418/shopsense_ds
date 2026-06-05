import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    classification_report, precision_recall_curve, auc
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def build_churn_preprocessing_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """
    Build a scikit-learn ColumnTransformer for preprocessing features.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return Pipeline(steps=[("preprocessor", preprocessor)])

def train_churn_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessing_pipeline, model_type: str = "xgboost", random_state: int = 42) -> Pipeline:
    """
    Train a churn model and log metrics and parameters to MLflow.
    """
    # Split for validation metrics logging
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )

    # Compute class ratio for imbalanced data
    ratio = (len(y_tr) - sum(y_tr)) / sum(y_tr) if sum(y_tr) > 0 else 1.0

    # Instantiate classifier
    if model_type == "xgboost":
        classifier = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=ratio,
            random_state=random_state,
            eval_metric="logloss"
        )
    elif model_type == "lightgbm":
        classifier = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1
        )
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight="balanced",
            random_state=random_state
        )
    elif model_type == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Build validation pipeline
    val_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessing_pipeline.steps[0][1]),
        ("classifier", classifier)
    ])

    # Enable autolog
    mlflow.set_experiment("shopsense_churn_experiment")
    
    with mlflow.start_run() as run:
        # Fit on validation train
        val_pipeline.fit(X_tr, y_tr)

        # Predict on validation val
        y_probs = val_pipeline.predict_proba(X_val)[:, 1]
        y_preds = (y_probs >= 0.5).astype(int)

        # Compute validation metrics
        roc_auc = float(roc_auc_score(y_val, y_probs))
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_probs)
        pr_auc = float(auc(recall_curve, precision_curve))
        acc = float(accuracy_score(y_val, y_preds))
        prec = float(precision_score(y_val, y_preds, zero_division=0))
        rec = float(recall_score(y_val, y_preds, zero_division=0))
        f1 = float(f1_score(y_val, y_preds, zero_division=0))

        # Log params
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(classifier.get_params())
        
        # Log validation metrics
        mlflow.log_metrics({
            "val_roc_auc": roc_auc,
            "val_pr_auc": pr_auc,
            "val_accuracy": acc,
            "val_precision": prec,
            "val_recall": rec,
            "val_f1": f1
        })
        
        # Log tags
        mlflow.set_tags({"model_type": model_type, "phase": "3.1"})

        # Train on full X_train, y_train
        full_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessing_pipeline.steps[0][1]),
            ("classifier", classifier)
        ])
        full_pipeline.fit(X_train, y_train)

        # Log full model artifact
        mlflow.sklearn.log_model(full_pipeline, "model")

    return full_pipeline

def find_optimal_threshold(model, X_val: pd.DataFrame, y_val: pd.Series, metric: str = "f1") -> dict:
    """
    Search thresholds from 0.05 to 0.95 to find the one maximizing a given metric.
    """
    y_probs = model.predict_proba(X_val)[:, 1]
    
    thresholds = np.arange(0.05, 0.96, 0.01)
    records = []

    for t in thresholds:
        y_preds = (y_probs >= t).astype(int)
        prec = float(precision_score(y_val, y_preds, zero_division=0))
        rec = float(recall_score(y_val, y_preds, zero_division=0))
        f1 = float(f1_score(y_val, y_preds, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_val, y_preds))

        records.append({
            "threshold": float(t),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "balanced_accuracy": bal_acc
        })

    df_curve = pd.DataFrame(records)
    best_row = df_curve.loc[df_curve[metric].idxmax()]

    return {
        "optimal_threshold": float(best_row["threshold"]),
        "optimal_metric_value": float(best_row[metric]),
        "threshold_curve": df_curve
    }

def evaluate_churn_model(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict:
    """
    Evaluate a model on test set using a specified classification threshold.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_probs))
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = float(auc(recall_curve, precision_curve))
    acc = float(accuracy_score(y_test, y_preds))
    prec = float(precision_score(y_test, y_preds, zero_division=0))
    rec = float(recall_score(y_test, y_preds, zero_division=0))
    f1 = float(f1_score(y_test, y_preds, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_test, y_preds))
    conf_mat = confusion_matrix(y_test, y_preds).tolist()
    class_rep = classification_report(y_test, y_preds)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": conf_mat,
        "classification_report": class_rep
    }

def run_rfe_feature_selection(X_train: pd.DataFrame, y_train: pd.Series, preprocessing_pipeline, n_features_to_select: int = 20) -> dict:
    """
    Run RFECV on preprocessed training data.
    """
    preprocessor = preprocessing_pipeline.steps[0][1]
    X_train_prep = preprocessor.fit_transform(X_train, y_train)
    feature_names = list(preprocessor.get_feature_names_out())

    # Estimator for RFE (XGBoost)
    estimator = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, eval_metric="logloss")

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=5,
        scoring="roc_auc",
        min_features_to_select=n_features_to_select,
        n_jobs=-1
    )
    rfecv.fit(X_train_prep, y_train)

    selected_features = [name for name, support in zip(feature_names, rfecv.support_) if support]
    cv_scores_mean = list(rfecv.cv_results_["mean_test_score"])
    cv_scores_std = list(rfecv.cv_results_["std_test_score"])

    return {
        "selected_features": selected_features,
        "optimal_n_features": int(rfecv.n_features_),
        "cv_scores_mean": cv_scores_mean,
        "cv_scores_std": cv_scores_std
    }

def tune_churn_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessing_pipeline, model_type: str = "xgboost", n_trials: int = 30, random_state: int = 42) -> dict:
    """
    Perform hyperparameter tuning and log to MLflow.
    """
    # Build a unified pipeline
    preprocessor = preprocessing_pipeline.steps[0][1]
    
    if model_type == "xgboost":
        classifier = XGBClassifier(random_state=random_state, eval_metric="logloss")
        param_grid = {
            "classifier__n_estimators": [50, 100, 150, 200],
            "classifier__max_depth": [3, 5, 7, 9],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0]
        }
    elif model_type == "lightgbm":
        classifier = LGBMClassifier(random_state=random_state, verbosity=-1)
        param_grid = {
            "classifier__n_estimators": [50, 100, 150, 200],
            "classifier__max_depth": [3, 5, 7, 9],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.6, 0.8, 1.0]
        }
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(random_state=random_state)
        param_grid = {
            "classifier__n_estimators": [50, 100, 150, 200],
            "classifier__max_depth": [3, 5, 7, 10],
            "classifier__min_samples_split": [2, 5, 10]
        }
    elif model_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=1000, random_state=random_state)
        param_grid = {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l2"]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    tune_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    search = RandomizedSearchCV(
        estimator=tune_pipeline,
        param_distributions=param_grid,
        n_iter=n_trials,
        scoring="roc_auc",
        cv=5,
        random_state=random_state,
        n_jobs=-1
    )

    mlflow.set_experiment("shopsense_churn_experiment")
    
    with mlflow.start_run() as run:
        search.fit(X_train, y_train)
        
        # Log params
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))
        mlflow.set_tags({"model_type": model_type, "phase": "4.1"})
        
        # Log model artifact
        mlflow.sklearn.log_model(search.best_estimator_, "best_model")

    return {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "best_model": search.best_estimator_
    }
