import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_revenue_timeseries(transactions_df: pd.DataFrame, freq: str = "M") -> pd.Series:
    """
    Aggregate transaction data into a net revenue time series.
    """
    df = transactions_df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    
    # Calculate net revenue (excluding returned items)
    df["net_revenue"] = df["quantity"] * df["unit_price"] * (1.0 - df["discount_pct"])
    df.loc[df["return_flag"] == 1, "net_revenue"] = 0.0

    series = df.set_index("transaction_date")["net_revenue"].resample(freq).sum()
    return series

def test_stationarity(series: pd.Series) -> dict:
    """
    Perform Augmented Dickey-Fuller test to check time-series stationarity.
    """
    # Handle short or empty series
    if len(series) < 5:
        return {
            "adf_statistic": 0.0,
            "p_value": 1.0,
            "is_stationary": False,
            "critical_values": {"1%": 0.0, "5%": 0.0, "10%": 0.0},
            "recommended_differencing": 1
        }
        
    result = adfuller(series)
    adf_statistic = float(result[0])
    p_value = float(result[1])
    critical_values = {k: float(v) for k, v in result[4].items()}
    is_stationary = p_value < 0.05
    recommended_differencing = 0 if is_stationary else 1

    return {
        "adf_statistic": adf_statistic,
        "p_value": p_value,
        "is_stationary": is_stationary,
        "critical_values": critical_values,
        "recommended_differencing": recommended_differencing
    }

def train_sarima_model(series: pd.Series, order: tuple, seasonal_order: tuple) -> tuple:
    """
    Fit a SARIMAX model to the time series.
    """
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    sarima_result = model.fit(disp=False)
    fitted_values = sarima_result.fittedvalues
    return sarima_result, fitted_values

def forecast_sarima(sarima_result, steps: int) -> pd.DataFrame:
    """
    Generate future forecasts along with 95% confidence intervals.
    """
    forecast_res = sarima_result.get_forecast(steps=steps)
    predicted_revenue = forecast_res.predicted_mean
    ci = forecast_res.conf_int(alpha=0.05)

    forecast_df = pd.DataFrame({
        "forecast_date": predicted_revenue.index,
        "predicted_revenue": predicted_revenue.values,
        "lower_ci_95": ci.iloc[:, 0].values,
        "upper_ci_95": ci.iloc[:, 1].values
    })
    return forecast_df

def evaluate_forecast(actual: pd.Series, predicted: pd.Series) -> dict:
    """
    Evaluate forecast accuracy using standard error metrics.
    """
    # Align indices to match matching dates
    combined = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    act = combined["actual"].values
    pred = combined["predicted"].values

    if len(act) == 0:
        return {
            "mae": 0.0, "rmse": 0.0, "mape": 0.0, "smape": 0.0, "r2": 0.0
        }

    mae = float(mean_absolute_error(act, pred))
    rmse = float(np.sqrt(mean_squared_error(act, pred)))
    
    # MAPE: exclude periods where actual = 0
    non_zero = act != 0
    if np.sum(non_zero) > 0:
        mape = float(np.mean(np.abs((act[non_zero] - pred[non_zero]) / act[non_zero])) * 100.0)
    else:
        mape = 0.0

    # sMAPE: handle zeros in denom gracefully
    denom = (np.abs(act) + np.abs(pred))
    valid = denom != 0
    if np.sum(valid) > 0:
        smape = float(np.mean(2.0 * np.abs(act[valid] - pred[valid]) / denom[valid]) * 100.0)
    else:
        smape = 0.0

    # r2 score (min 1 sample)
    if len(act) > 1:
        r2 = float(r2_score(act, pred))
    else:
        r2 = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "r2": r2
    }
