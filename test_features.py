from shopsense.data_generator import (
    generate_products,
    generate_customers,
    generate_transactions,
    generate_events
)
from shopsense.features import (
    create_rfm_features,
    create_transaction_features
)
from shopsense.features import (
    create_rfm_features
)
from shopsense.features import (
    create_event_features
)
from shopsense.features import create_master_feature_table

products_df = generate_products()

customers_df = generate_customers()

transactions_df = generate_transactions(
    customers_df,
    products_df
)

rfm_df = create_rfm_features(
    transactions_df
)

print(rfm_df.head())

print(rfm_df.shape)
transaction_features_df = (
    create_transaction_features(
        transactions_df
    )
)

print(
    transaction_features_df.head()
)

print(
    transaction_features_df.shape
)
events_df = generate_events(
    customers_df,
    products_df
)
event_features_df = (
    create_event_features(
        events_df
    )
)

print(
    event_features_df.head()
)

print(
    event_features_df.shape
)
master_df = create_master_feature_table(
    customers_df,
    rfm_df,
    transaction_features_df,
    event_features_df
)

print(master_df.head())

print(master_df.shape)

print(master_df.columns.tolist())