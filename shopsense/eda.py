import pandas as pd
import numpy as np
import scipy.stats as stats

def compute_univariate_stats(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Compute univariate statistics for a list of columns in a DataFrame.
    """
    stats_list = []
    for col in columns:
        col_series = df[col]
        missing_pct = float(col_series.isnull().mean() * 100)
        unique_count = int(col_series.nunique())

        if pd.api.types.is_numeric_dtype(col_series) and not pd.api.types.is_bool_dtype(col_series):
            mean_val = float(col_series.mean())
            median_val = float(col_series.median())
            std_val = float(col_series.std())
            skew_val = float(col_series.skew())
            kurt_val = float(col_series.kurtosis())
        else:
            mean_val = np.nan
            median_val = np.nan
            std_val = np.nan
            skew_val = np.nan
            kurt_val = np.nan

        stats_list.append({
            "mean": mean_val,
            "median": median_val,
            "std": std_val,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "missing_pct": missing_pct,
            "unique_count": unique_count
        })

    return pd.DataFrame(stats_list, index=columns)

def churn_distribution_summary(customers_df: pd.DataFrame) -> dict:
    """
    Provide summary of churn distribution across various categories.
    """
    total_customers = int(len(customers_df))
    churned_count = int(customers_df["churn_label"].sum())
    churn_rate = round(float(churned_count / total_customers), 4)

    churn_by_channel = {
        str(k): round(float(v), 4)
        for k, v in customers_df.groupby("acquisition_channel")["churn_label"].mean().to_dict().items()
    }
    churn_by_gender = {
        str(k): round(float(v), 4)
        for k, v in customers_df.groupby("gender")["churn_label"].mean().to_dict().items()
    }
    churn_by_premium = {
        bool(k): round(float(v), 4)
        for k, v in customers_df.groupby("is_premium")["churn_label"].mean().to_dict().items()
    }

    mean_age_churned = float(customers_df[customers_df["churn_label"] == 1]["age"].mean())
    mean_age_retained = float(customers_df[customers_df["churn_label"] == 0]["age"].mean())

    return {
        "total_customers": total_customers,
        "churned_count": churned_count,
        "churn_rate": churn_rate,
        "churn_by_channel": churn_by_channel,
        "churn_by_gender": churn_by_gender,
        "churn_by_premium": churn_by_premium,
        "mean_age_churned": round(mean_age_churned, 4),
        "mean_age_retained": round(mean_age_retained, 4)
    }

def compute_monthly_revenue(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute aggregated monthly revenue metrics.
    """
    df = transactions_df.copy()
    # Net revenue is unit_price * quantity * (1 - discount_pct)
    df["revenue"] = df["quantity"] * df["unit_price"] * (1.0 - df["discount_pct"])
    # Adjusted revenue drops returned items (assumes full refund)
    df["return_adjusted_revenue"] = df["revenue"] * (1.0 - df["return_flag"])

    df["year_month"] = pd.to_datetime(df["transaction_date"]).dt.strftime("%Y-%m")

    grouped = df.groupby("year_month").agg(
        total_revenue=("revenue", "sum"),
        transaction_count=("transaction_id", "count"),
        unique_customers=("customer_id", "nunique"),
        return_adjusted_revenue=("return_adjusted_revenue", "sum")
    ).reset_index()

    grouped["avg_order_value"] = grouped["total_revenue"] / grouped["transaction_count"]
    grouped = grouped.sort_values("year_month").reset_index(drop=True)

    return grouped[[
        "year_month", "total_revenue", "transaction_count",
        "unique_customers", "avg_order_value", "return_adjusted_revenue"
    ]]

def compute_cohort_retention(customers_df: pd.DataFrame, transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cohort analysis to calculate monthly retention rates.
    """
    customer_cohorts = customers_df[["customer_id", "signup_date"]].copy()
    customer_cohorts["cohort_month"] = pd.to_datetime(customer_cohorts["signup_date"]).dt.strftime("%Y-%m")

    tx_cohort = transactions_df.merge(customer_cohorts, on="customer_id", how="inner")
    tx_cohort["tx_month"] = pd.to_datetime(tx_cohort["transaction_date"])
    tx_cohort["cohort_month_dt"] = pd.to_datetime(tx_cohort["cohort_month"], format="%Y-%m")

    # Months difference since signup cohort
    tx_cohort["period_idx"] = (tx_cohort["tx_month"].dt.year - tx_cohort["cohort_month_dt"].dt.year) * 12 + \
                              (tx_cohort["tx_month"].dt.month - tx_cohort["cohort_month_dt"].dt.month)

    cohort_group = tx_cohort.groupby(["cohort_month", "period_idx"])["customer_id"].nunique().reset_index()
    cohort_sizes = customer_cohorts.groupby("cohort_month")["customer_id"].nunique().to_dict()

    cohort_group["retention_rate"] = cohort_group.apply(
        lambda row: row["customer_id"] / cohort_sizes[row["cohort_month"]], axis=1
    )

    retention_matrix = cohort_group.pivot(index="cohort_month", columns="period_idx", values="retention_rate")

    # Enforce period 0 is always 1.0
    retention_matrix[0] = 1.0
    cols = sorted(retention_matrix.columns)
    retention_matrix = retention_matrix[cols]

    return retention_matrix

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> dict:
    """
    Detect outliers in a numeric column using IQR.
    """
    series = df[column].dropna()
    if len(series) == 0:
        return {
            "q1": 0.0, "q3": 0.0, "iqr": 0.0,
            "lower_bound": 0.0, "upper_bound": 0.0,
            "outlier_count": 0, "outlier_pct": 0.0
        }

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = series[(series < lower_bound) | (series > upper_bound)]
    outlier_count = int(len(outliers))
    outlier_pct = round(float(outlier_count / len(series)), 4)

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "outlier_pct": outlier_pct
    }

def test_premium_vs_nonpremium_aov(transactions_df: pd.DataFrame, customers_df: pd.DataFrame) -> dict:
    """
    Test if premium customers have significantly higher average order value using Mann-Whitney U.
    """
    df = transactions_df.copy()
    df["revenue"] = df["quantity"] * df["unit_price"] * (1.0 - df["discount_pct"])
    # Calculate AOV per customer
    customer_aov = df.groupby("customer_id")["revenue"].mean().reset_index(name="aov")

    merged = customer_aov.merge(customers_df[["customer_id", "is_premium"]], on="customer_id", how="inner")

    premium_aov = merged[merged["is_premium"] == True]["aov"]
    nonpremium_aov = merged[merged["is_premium"] == False]["aov"]

    stat, p_val = stats.mannwhitneyu(premium_aov, nonpremium_aov, alternative="two-sided")
    p_val = float(p_val)
    stat = float(stat)

    premium_mean = float(premium_aov.mean()) if len(premium_aov) > 0 else 0.0
    nonpremium_mean = float(nonpremium_aov.mean()) if len(nonpremium_aov) > 0 else 0.0

    reject_null = p_val < 0.05
    interpretation = (
        f"Premium customers have a significantly higher AOV (₹{premium_mean:.2f}) "
        f"compared to non-premium customers (₹{nonpremium_mean:.2f}) (p={p_val:.4e})."
        if reject_null and premium_mean > nonpremium_mean else
        f"There is no statistically significant difference in AOV between premium (₹{premium_mean:.2f}) "
        f"and non-premium (₹{nonpremium_mean:.2f}) customers (p={p_val:.4f})."
    )

    return {
        "test_name": "Mann-Whitney U",
        "statistic": stat,
        "p_value": p_val,
        "premium_mean_aov": premium_mean,
        "nonpremium_mean_aov": nonpremium_mean,
        "reject_null": reject_null,
        "interpretation": interpretation
    }

def test_channel_churn_association(customers_df: pd.DataFrame) -> dict:
    """
    Test if there is a significant association between acquisition channel and churn using Chi-Square.
    """
    contingency_table = pd.crosstab(customers_df["acquisition_channel"], customers_df["churn_label"])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    chi2 = float(chi2)
    p_val = float(p_val)
    dof = int(dof)

    n = len(customers_df)
    k = min(contingency_table.shape)
    cramers_v = float(np.sqrt(chi2 / (n * (k - 1)))) if n > 0 and k > 1 else 0.0

    reject_null = p_val < 0.05
    interpretation = (
        f"There is a statistically significant association between acquisition channel and churn "
        f"(p={p_val:.4e}, Cramér's V={cramers_v:.4f})."
        if reject_null else
        f"No statistically significant association was found between acquisition channel and churn "
        f"(p={p_val:.4f})."
    )

    return {
        "test_name": "Chi-Square",
        "statistic": chi2,
        "p_value": p_val,
        "degrees_of_freedom": dof,
        "cramers_v": cramers_v,
        "reject_null": reject_null,
        "interpretation": interpretation
    }

def test_revenue_seasonality(transactions_df: pd.DataFrame) -> dict:
    """
    Test if daily revenue shows significant seasonality differences across months using Kruskal-Wallis.
    """
    df = transactions_df.copy()
    df["revenue"] = df["quantity"] * df["unit_price"] * (1.0 - df["discount_pct"])
    df["date"] = pd.to_datetime(df["transaction_date"]).dt.date

    # Daily revenue
    daily_rev = df.groupby("date")["revenue"].sum().reset_index()
    daily_rev["month"] = pd.to_datetime(daily_rev["date"]).dt.month

    # Group daily revenues by month
    grouped_months = {m: group["revenue"].values for m, group in daily_rev.groupby("month")}

    # Get median revenue per month
    medians = daily_rev.groupby("month")["revenue"].median()
    peak_month = int(medians.idxmax())
    trough_month = int(medians.idxmin())

    # Kruskal-Wallis test across months
    groups = list(grouped_months.values())
    if len(groups) > 1:
        stat, p_val = stats.kruskal(*groups)
        stat = float(stat)
        p_val = float(p_val)
    else:
        stat, p_val = 0.0, 1.0

    reject_null = p_val < 0.05
    interpretation = (
        f"Daily revenue shows statistically significant differences across months "
        f"(p={p_val:.4e}), with month {peak_month} showing peak median revenue and month {trough_month} showing the trough."
        if reject_null else
        f"No statistically significant seasonal monthly differences in daily revenue were found (p={p_val:.4f})."
    )

    return {
        "test_name": "Kruskal-Wallis",
        "statistic": stat,
        "p_value": p_val,
        "reject_null": reject_null,
        "peak_month": peak_month,
        "trough_month": trough_month,
        "interpretation": interpretation
    }
