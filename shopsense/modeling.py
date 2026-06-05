import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def prepare_model_data(
    master_df
):

    features = [
        "age",
        "is_premium",

        "recency_days",
        "frequency",
        "monetary",

        "avg_discount",
        "return_rate",
        "avg_order_value",
        "total_quantity",

        "total_events",
        "avg_session_duration"
    ]

    X = master_df[features]

    y = master_df["churn_label"]

    return X, y


def split_data(
    X,
    y,
    test_size=0.2,
    random_state=42
):

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def scale_features(
    X_train,
    X_test
):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )
def train_xgboost(
    X_train,
    y_train
):

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(
        X_train,
        y_train
    )

    return model
def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            predictions
        )
    )

    print(
        "Precision:",
        precision_score(
            y_test,
            predictions
        )
    )

    print(
        "Recall:",
        recall_score(
            y_test,
            predictions
        )
    )

    print(
        "F1 Score:",
        f1_score(
            y_test,
            predictions
        )
    )

    print(
        "ROC AUC:",
        roc_auc_score(
            y_test,
            probabilities
        )
    )

    print(
        "\nConfusion Matrix\n"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )