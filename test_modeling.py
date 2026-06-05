from shopsense.modeling import (
    prepare_model_data,
    split_data,
    scale_features
)
from shopsense.data_generator import (
    generate_products,
    generate_customers,
    generate_transactions,
    generate_events,
    generate_churn_labels
)

from shopsense.features import (
    create_rfm_features,
    create_transaction_features,
    create_event_features,
    create_master_feature_table
)
from shopsense.modeling import (
    prepare_model_data,
    split_data,
    scale_features,
    train_xgboost,
    evaluate_model
)
products_df = generate_products()

customers_df = generate_customers()

transactions_df = generate_transactions(
    customers_df,
    products_df
)

events_df = generate_events(
    customers_df,
    products_df
)

customers_df = generate_churn_labels(
    customers_df,
    transactions_df
)

rfm_df = create_rfm_features(
    transactions_df
)

transaction_features_df = (
    create_transaction_features(
        transactions_df
    )
)

event_features_df = (
    create_event_features(
        events_df
    )
)

master_df = create_master_feature_table(
    customers_df,
    rfm_df,
    transaction_features_df,
    event_features_df
)
X, y = prepare_model_data(
    master_df
)

print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = (
    split_data(
        X,
        y
    )
)

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)
X_train_scaled, X_test_scaled, scaler = (
    scale_features(
        X_train,
        X_test
    )
)

print(X_train_scaled.shape)
print(X_test_scaled.shape)
model = train_xgboost(
    X_train_scaled,
    y_train
)
evaluate_model(
    model,
    X_test_scaled,
    y_test
)