from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_products(random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic products dataset.
    """
    np.random.seed(random_state)

    categories = [
        "Electronics",
        "Fashion",
        "Home",
        "Beauty",
        "Sports",
        "Books"
    ]

    product_names = {
        "Electronics": ["Smartphone", "Laptop", "Tablet", "Headphones", "Smartwatch"],
        "Fashion": ["T-Shirt", "Jeans", "Sneakers", "Jacket", "Dress"],
        "Home": ["Sofa", "Dining Table", "Chair", "Mattress", "Lamp"],
        "Beauty": ["Face Wash", "Perfume", "Lipstick", "Shampoo", "Moisturizer"],
        "Sports": ["Cricket Bat", "Football", "Yoga Mat", "Dumbbell", "Tennis Racket"],
        "Books": ["Python Guide", "Data Science Handbook", "AI Fundamentals", "Machine Learning Basics", "Deep Learning Guide"]
    }

    rows = []
    for i in range(1, 201):
        category = np.random.choice(categories)
        product_id = f"PROD_{i:03d}"
        product_name = np.random.choice(product_names[category])
        brand_tier = np.random.choice(
            ["budget", "mid", "premium"],
            p=[0.50, 0.35, 0.15]
        )

        # Category-wise pricing (log-normal)
        if category == "Electronics":
            base_price = round(np.random.lognormal(mean=9.5, sigma=0.5), 2)
        elif category == "Fashion":
            base_price = round(np.random.lognormal(mean=7.0, sigma=0.4), 2)
        elif category == "Books":
            base_price = round(np.random.lognormal(mean=5.5, sigma=0.3), 2)
        elif category == "Beauty":
            base_price = round(np.random.lognormal(mean=6.5, sigma=0.3), 2)
        elif category == "Sports":
            base_price = round(np.random.lognormal(mean=7.5, sigma=0.4), 2)
        else:  # Home
            base_price = round(np.random.lognormal(mean=8.0, sigma=0.4), 2)

        # avg_rating: beta distribution mean ~3.8, range [3.0, 5.0]
        avg_rating = round(3.0 + np.random.beta(5, 2) * 2.0, 1)

        rows.append({
            "product_id": product_id,
            "product_name": product_name,
            "category": category,
            "base_price": base_price,
            "brand_tier": brand_tier,
            "avg_rating": avg_rating
        })

    return pd.DataFrame(rows)

def assign_customer_profile():
    """
    Assign a hidden profile for testing purposes.
    """
    return np.random.choice(
        ["high", "medium", "low"],
        p=[0.20, 0.60, 0.20]
    )

def generate_customers(n_customers: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customers dataset.
    """
    np.random.seed(random_state)

    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]

    # Signup Dates (Jan 2021 - Dec 2022)
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2022, 12, 31)
    total_days = (end_date - start_date).days

    signup_dates = [
        start_date + timedelta(days=int(np.random.randint(0, total_days + 1)))
        for _ in range(n_customers)
    ]

    # Age (Normal Distribution, mean=35, std=10, clipped to [18, 70])
    ages = np.random.normal(loc=35, scale=10, size=n_customers)
    ages = np.clip(ages, 18, 70).astype(int)

    # Gender (approx 48/48/4 split)
    genders = np.random.choice(
        ["M", "F", "Other"],
        size=n_customers,
        p=[0.48, 0.48, 0.04]
    )

    # 20 Indian Cities (weighted by population)
    cities = [
        "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
        "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
        "Surat", "Kanpur", "Nagpur", "Indore", "Bhopal",
        "Patna", "Visakhapatnam", "Vijayawada", "Coimbatore", "Kochi"
    ]
    city_weights = np.array([
        10, 10, 8, 8, 6,
        6, 5, 5, 4, 4,
        4, 3, 3, 3, 3,
        3, 3, 2, 2, 1
    ])
    city_weights = city_weights / city_weights.sum()

    cities_selected = np.random.choice(
        cities,
        size=n_customers,
        p=city_weights
    )

    # Acquisition Channel
    channels = np.random.choice(
        ["organic", "paid_search", "social_media", "referral", "email"],
        size=n_customers,
        p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )

    # Premium Users (~20%)
    is_premium = np.random.choice(
        [True, False],
        size=n_customers,
        p=[0.20, 0.80]
    )

    # Churn Label (~25% churn rate)
    churn_labels = np.random.choice(
        [0, 1],
        size=n_customers,
        p=[0.75, 0.25]
    )

    # For compatibility with legacy test scripts, assign a customer_profile column
    customer_profiles = [assign_customer_profile() for _ in range(n_customers)]

    customers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": signup_dates,
        "age": ages,
        "gender": genders,
        "city": cities_selected,
        "acquisition_channel": channels,
        "is_premium": is_premium,
        "customer_profile": customer_profiles,
        "churn_label": churn_labels
    })

    # Return exactly the required columns
    return customers_df[["customer_id", "signup_date", "age", "gender", "city", "acquisition_channel", "is_premium", "customer_profile", "churn_label"]]

def get_transaction_count(profile):
    if profile == "high":
        return np.random.randint(30, 61)
    elif profile == "medium":
        return np.random.randint(10, 26)
    else:
        return np.random.randint(1, 9)

def generate_transaction_date(random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    months = np.arange(1, 13)
    month_weights = np.array([
        1.0, 0.9, 1.0, 1.0, 0.95, 0.9,
        0.95, 1.0, 1.1, 1.5, 1.8, 2.0
    ])
    month_weights = month_weights / month_weights.sum()

    year = np.random.choice([2021, 2022, 2023])
    month = np.random.choice(months, p=month_weights)
    day = np.random.randint(1, 29)
    return pd.Timestamp(year=year, month=month, day=day)

def generate_quantity():
    return int(np.random.choice([1, 2, 3, 4, 5], p=[0.50, 0.25, 0.15, 0.07, 0.03]))

def generate_payment_method():
    return np.random.choice(
        ["UPI", "credit_card", "debit_card", "COD", "wallet"],
        p=[0.45, 0.20, 0.15, 0.10, 0.10]
    )

def generate_discount(category):
    if category in ["Fashion", "Beauty"]:
        return float(np.random.choice([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
                                       p=[0.10, 0.10, 0.15, 0.25, 0.20, 0.15, 0.05]))
    elif category == "Electronics":
        return float(np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20],
                                       p=[0.35, 0.30, 0.20, 0.10, 0.05]))
    else:
        return float(np.random.choice([0.0, 0.05, 0.10, 0.20],
                                       p=[0.30, 0.30, 0.25, 0.15]))

def generate_return_flag(category):
    if category in ["Fashion", "Electronics"]:
        return int(np.random.choice([0, 1], p=[0.88, 0.12]))
    elif category == "Books":
        return int(np.random.choice([0, 1], p=[0.97, 0.03]))
    else:
        return int(np.random.choice([0, 1], p=[0.94, 0.06]))

def generate_transactions(customers_df: pd.DataFrame, products_df: pd.DataFrame = None, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic transactions dataset (Optimized).
    """
    np.random.seed(random_state)
    if products_df is None:
        products_df = generate_products(random_state)

    # Convert products to list of dicts for ultra-fast sampling
    products_list = products_df.to_dict('records')
    budget_products_df = products_df[products_df["brand_tier"] == "budget"]
    budget_products_list = budget_products_df.to_dict('records') if len(budget_products_df) > 0 else products_list

    rows = []
    txn_counter = 1
    end_date_limit = datetime(2023, 12, 31)

    month_weights = {
        1: 1.0, 2: 0.9, 3: 1.0, 4: 1.0, 5: 0.95, 6: 0.9,
        7: 0.95, 8: 1.0, 9: 1.1, 10: 1.8, 11: 1.8, 12: 2.0
    }
    max_month_weight = 2.0

    for idx, customer in customers_df.iterrows():
        customer_id = customer["customer_id"]
        signup_date = pd.to_datetime(customer["signup_date"])
        churn_label = customer["churn_label"]

        if churn_label == 1:
            n_txns = np.random.randint(1, 8)
            max_active_days = (end_date_limit - timedelta(days=181) - signup_date).days
            if max_active_days <= 1:
                last_active_date = signup_date + timedelta(days=1)
            else:
                active_days = np.random.randint(1, min(max_active_days, 365))
                last_active_date = signup_date + timedelta(days=active_days)
        else:
            n_txns = np.random.randint(8, 41)
            last_active_date = end_date_limit - timedelta(days=np.random.randint(0, 90))
            if last_active_date < signup_date:
                last_active_date = signup_date + timedelta(days=1)

        active_range_days = (last_active_date - signup_date).days

        for _ in range(n_txns):
            if churn_label == 1:
                product = budget_products_list[np.random.randint(0, len(budget_products_list))]
            else:
                product = products_list[np.random.randint(0, len(products_list))]

            if active_range_days <= 0:
                txn_date = signup_date
            else:
                while True:
                    candidate = signup_date + timedelta(days=int(np.random.randint(0, active_range_days + 1)))
                    m_weight = month_weights[candidate.month]
                    if np.random.uniform(0, max_month_weight) <= m_weight:
                        txn_date = candidate
                        break

            quantity = generate_quantity()
            payment_method = generate_payment_method()
            discount_pct = generate_discount(product["category"])
            return_flag = generate_return_flag(product["category"])

            transaction_id = f"TXN_{txn_counter:07d}"
            txn_counter += 1

            revenue = float(quantity * product["base_price"] * (1.0 - discount_pct))
            if return_flag == 1:
                revenue = 0.0

            rows.append({
                "transaction_id": transaction_id,
                "customer_id": customer_id,
                "transaction_date": txn_date,
                "product_id": product["product_id"],
                "category": product["category"],
                "quantity": quantity,
                "unit_price": product["base_price"],
                "discount_pct": discount_pct,
                "payment_method": payment_method,
                "return_flag": return_flag,
                "revenue": revenue
            })

    txns_df = pd.DataFrame(rows)
    if not txns_df.empty:
        txns_df = txns_df.sort_values("transaction_date").reset_index(drop=True)
    return txns_df

def get_event_count(profile):
    if profile == "high":
        return np.random.randint(500, 1201)
    elif profile == "medium":
        return np.random.randint(150, 501)
    else:
        return np.random.randint(20, 151)

def generate_session_duration():
    return int(np.clip(np.random.normal(loc=240, scale=120), 5, 1200))

def generate_event_type():
    return np.random.choice(
        ["page_view", "search", "wishlist_add", "add_to_cart", "checkout_start", "purchase"],
        p=[0.50, 0.20, 0.10, 0.12, 0.05, 0.03]
    )

def generate_device():
    return np.random.choice(
        ["mobile", "desktop", "tablet"],
        p=[0.65, 0.30, 0.05]
    )

def generate_events(customers_df: pd.DataFrame, products_df: pd.DataFrame = None, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic events dataset (Optimized Vectorized).
    """
    np.random.seed(random_state)
    if products_df is None:
        products_df = generate_products(random_state)

    categories = products_df["category"].unique().tolist()
    end_date_limit = datetime(2023, 12, 31)

    # First pass: pre-calculate number of events per customer
    n_events_list = []
    for _, customer in customers_df.iterrows():
        churn_label = customer["churn_label"]
        if churn_label == 1:
            n = np.random.randint(20, 101)
        else:
            n = np.random.randint(150, 801)
        n_events_list.append(n)

    total_events = sum(n_events_list)

    # Vectorized pre-generation of parameters
    event_types = np.random.choice(
        ["page_view", "search", "wishlist_add", "add_to_cart", "checkout_start", "purchase"],
        size=total_events,
        p=[0.50, 0.20, 0.10, 0.12, 0.05, 0.03]
    )
    device_types = np.random.choice(
        ["mobile", "desktop", "tablet"],
        size=total_events,
        p=[0.65, 0.30, 0.05]
    )
    session_durations = np.clip(np.random.normal(loc=240, scale=120, size=total_events), 5, 1200).astype(int)
    page_categories = np.random.choice(categories, size=total_events)

    rows = []
    evt_counter = 1
    event_idx = 0

    for idx, customer in customers_df.iterrows():
        customer_id = customer["customer_id"]
        signup_date = pd.to_datetime(customer["signup_date"])
        churn_label = customer["churn_label"]
        n_events = n_events_list[idx]

        if churn_label == 1:
            max_active_days = (end_date_limit - timedelta(days=181) - signup_date).days
            if max_active_days <= 1:
                last_active_date = signup_date + timedelta(days=1)
            else:
                active_days = np.random.randint(1, min(max_active_days, 365))
                last_active_date = signup_date + timedelta(days=active_days)
        else:
            last_active_date = end_date_limit - timedelta(days=np.random.randint(0, 90))
            if last_active_date < signup_date:
                last_active_date = signup_date + timedelta(days=1)

        active_range_days = (last_active_date - signup_date).days
        if active_range_days <= 0:
            active_range_days = 1

        # Vectorized event date calculation
        if churn_label == 1:
            t = np.random.beta(2, 5, size=n_events)
        else:
            t = np.random.uniform(0, 1, size=n_events)

        deltas = (t * active_range_days).astype(int)
        event_dates = signup_date + pd.to_timedelta(deltas, unit='D')

        cust_event_types = event_types[event_idx : event_idx + n_events]
        cust_device_types = device_types[event_idx : event_idx + n_events]
        cust_durations = session_durations[event_idx : event_idx + n_events]
        cust_page_categories = page_categories[event_idx : event_idx + n_events]

        for i in range(n_events):
            rows.append({
                "event_id": f"EVT_{evt_counter:07d}",
                "customer_id": customer_id,
                "event_date": event_dates[i],
                "event_type": cust_event_types[i],
                "session_duration_sec": int(cust_durations[i]),
                "device_type": cust_device_types[i],
                "page_category": cust_page_categories[i]
            })
            evt_counter += 1

        event_idx += n_events

    events_df = pd.DataFrame(rows)
    if not events_df.empty:
        events_df = events_df.sort_values("event_date").reset_index(drop=True)
    return events_df

def generate_churn_labels(customers_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to determine churn labels based on frequency and recency.
    For compatibility with test scripts.
    """
    latest_date = transactions_df["transaction_date"].max()
    agg_df = transactions_df.groupby("customer_id").agg(
        frequency=("transaction_id", "count"),
        last_purchase=("transaction_date", "max")
    ).reset_index()
    agg_df["recency_days"] = (latest_date - agg_df["last_purchase"]).dt.days

    churn_rules = (agg_df["frequency"] < 8) & (agg_df["recency_days"] > 180)
    agg_df["churn_label"] = churn_rules.astype(int)

    if "churn_label" in customers_df.columns:
        customers_df = customers_df.drop(columns=["churn_label"])

    customers_df = customers_df.merge(
        agg_df[["customer_id", "churn_label"]],
        on="customer_id",
        how="left"
    )
    customers_df["churn_label"] = customers_df["churn_label"].fillna(0).astype(int)
    return customers_df