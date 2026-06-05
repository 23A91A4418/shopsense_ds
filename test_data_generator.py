import pandas as pd
from shopsense.data_generator import generate_products
from shopsense.data_generator import generate_customers
from shopsense.data_generator import assign_customer_profile
from shopsense.data_generator import get_transaction_count
from shopsense.data_generator import generate_payment_method
from shopsense.data_generator import generate_quantity
from shopsense.data_generator import generate_transaction_date
from shopsense.data_generator import generate_transaction_date
from shopsense.data_generator import get_event_count
from shopsense.data_generator import generate_device
from shopsense.data_generator import generate_event_type
from shopsense.data_generator import generate_session_duration
from shopsense.data_generator import generate_events
from shopsense.data_generator import generate_churn_labels

products_df = generate_products()

print(products_df.head())
print(products_df.shape)
print(products_df["category"].value_counts())
customers_df = generate_customers()

print(customers_df.head())

print(customers_df.shape)

print(customers_df["gender"].value_counts(normalize=True))

print(customers_df["is_premium"].mean())

print(customers_df["city"].value_counts().head())
profiles = [
    assign_customer_profile()
    for _ in range(10000)
]

print(pd.Series(profiles).value_counts(normalize=True))
for profile in ["high", "medium", "low"]:
    print(
        profile,
        get_transaction_count(profile)
    )


dates = [
    generate_transaction_date()
    for _ in range(10000)
]

dates_df = pd.DataFrame({
    "date": dates
})

print(
    dates_df["date"]
    .dt.month
    .value_counts()
    .sort_index()
)
from shopsense.data_generator import generate_quantity

quantities = [
    generate_quantity()
    for _ in range(10000)
]

print(
    pd.Series(quantities)
    .value_counts(normalize=True)
    .sort_index()
)
methods = [
    generate_payment_method()
    for _ in range(10000)
]

print(
    pd.Series(methods)
    .value_counts(normalize=True)
)
from shopsense.data_generator import (
    generate_transactions
)

transactions_df = generate_transactions(
    customers_df,
    products_df
)
print(transactions_df.columns.tolist())
print(transactions_df.shape)
print(
    transactions_df["category"]
    .value_counts()
)
print(
    transactions_df["payment_method"]
    .value_counts(normalize=True)
)
print(
    transactions_df["transaction_date"]
    .dt.month
    .value_counts()
    .sort_index()
)
print(transactions_df.head())

print(
    transactions_df["return_flag"].mean()
)

print(
    transactions_df["discount_pct"].describe()
)

print(
    transactions_df["revenue"].describe()
)
for p in ["high","medium","low"]:
    print(
        p,
        get_event_count(p)
    )
print(
    customers_df["customer_profile"]
    .value_counts(normalize=True)
)
for p in ["high", "medium", "low"]:
    print(
        p,
        get_event_count(p)
    )

print(generate_event_type())
print(generate_device())
print(generate_session_duration())

events_df = generate_events(
    customers_df,
    products_df
)

print(events_df.head())

print(events_df.shape)
print(transactions_df.groupby(
    "customer_id"
).size())
freq = transactions_df.groupby(
    "customer_id"
).size()

print(freq.describe())
latest_date = (
    transactions_df["transaction_date"]
    .max()
)
customers_df = generate_churn_labels(
    customers_df,
    transactions_df
)

print(
    customers_df["churn_label"]
    .value_counts()
)
print(customers_df.head())

print(customers_df.columns.tolist())

print(customers_df["customer_profile"].value_counts())

print(customers_df["churn_label"].value_counts())