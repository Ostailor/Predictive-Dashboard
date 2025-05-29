import pandas as pd
import numpy as np
from app import db
from app.models.data_models import SalesData
from sqlalchemy import func
import pmdarima as pm
import warnings

print(f"DEBUG: pmdarima version being used: {pm.__version__}")
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.", category=FutureWarning)

def get_daily_sales_data():
    """
    Fetches sales data, aggregates it by day, and ensures a complete daily series.
    Returns a pandas DataFrame with a DatetimeIndex ('date') and 'total_sales'.
    """
    daily_sales_query = db.session.query(
        func.date(SalesData.timestamp).label('date'),
        func.sum(SalesData.amount).label('total_sales')
    ).group_by(func.date(SalesData.timestamp)).order_by(func.date(SalesData.timestamp))

    df = pd.read_sql_query(daily_sales_query.statement, db.engine)

    if df.empty:
        print("DEBUG: get_daily_sales_data is returning an EMPTY DataFrame.") # New debug
        empty_idx = pd.DatetimeIndex([], name='date')
        return pd.DataFrame({'total_sales': []}, index=empty_idx)

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    if not df.empty:
        if not isinstance(df.index, pd.DatetimeIndex):
             df.index = pd.to_datetime(df.index)
        idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
        df = df.reindex(idx, fill_value=0) 
        df.index.name = 'date'
    
    # print(f"DEBUG: get_daily_sales_data, df head:\n{df.head()}") # Optional: very verbose
    # print(f"DEBUG: get_daily_sales_data, df info:") # Optional: very verbose
    # df.info() # Optional: very verbose
    return df


def train_sales_forecasting_model(daily_sales_df):
    """
    Trains an ARIMA model using auto_arima to find the best parameters.
    Assumes daily_sales_df has a 'total_sales' column and a daily DatetimeIndex.
    """
    print(f"DEBUG: train_sales_forecasting_model called. daily_sales_df empty? {daily_sales_df.empty}") # New debug
    if not daily_sales_df.empty:
        print(f"DEBUG: daily_sales_df.shape: {daily_sales_df.shape}") # New debug
        print(f"DEBUG: daily_sales_df non-NA 'total_sales' count: {len(daily_sales_df['total_sales'].dropna())}") # New debug
        # print(f"DEBUG: daily_sales_df head:\n{daily_sales_df.head()}") # New debug - can be verbose

    if daily_sales_df.empty or len(daily_sales_df['total_sales'].dropna()) < 30:
        warnings.warn("Not enough data to train model. Need at least 30 non-NA observations.")
        return None
    
    if daily_sales_df.index.freq is None:
        warnings.warn("Dataframe index does not have frequency. Attempting to set to 'D'.")
        daily_sales_df = daily_sales_df.asfreq('D', fill_value=0)

    y_transformed = np.log1p(daily_sales_df['total_sales'])
    print(f"DEBUG: y_transformed head for auto_arima:\n{y_transformed.head()}") # See what data looks like
    print(f"DEBUG: y_transformed describe for auto_arima:\n{y_transformed.describe()}")


    QUICK_TEST_MODE = True 

    try:
        print("INFO: Attempting to fit simple AutoARIMA (no Fourier terms).")
        if QUICK_TEST_MODE:
            print("INFO: Running AutoARIMA in QUICK TEST MODE with reduced parameters.")
            model = pm.auto_arima(y_transformed, # Fit directly to y_transformed
                start_p=1, start_q=1,
                max_p=2, max_q=2, 
                m=7, seasonal=True, # Just weekly seasonality
                max_P=1, max_Q=1, 
                d=None, D=None, 
                test='adf', 
                information_criterion='aic', 
                trace=False, error_action='ignore', 
                suppress_warnings=True, stepwise=True
            )
        else:
            print("INFO: Running AutoARIMA with full search parameters (no Fourier).")
            model = pm.auto_arima(y_transformed,
                start_p=1, start_q=1,
                max_p=5, max_q=5,
                m=7, seasonal=True, 
                max_P=3, max_Q=3,
                d=None, D=None, 
                test='adf',
                information_criterion='aic',
                trace=True, error_action='ignore',
                suppress_warnings=True, stepwise=True
            )
        
        print(f"Simple AutoARIMA selected model on log-transformed data: {model.order} {model.seasonal_order if model.seasonal_order else ''}")
        return model # Return the simple model

    except Exception as e:
        warnings.warn(f"Error training simple AutoARIMA model: {e}") 
        return None

# Modify predict_future_sales to accept either a model or a pipeline
def predict_future_sales(fitted_model_or_pipeline, periods=30, alpha=0.05):
    """
    Predicts future sales and confidence intervals using the fitted pipeline.
    The pipeline should handle Fourier term generation for the forecast horizon.
    """
    if fitted_model_or_pipeline is None:
        return None, None, None

    try:
        # The predict method is the same for a fitted AutoARIMA model and a Pipeline
        forecast_values_transformed, conf_int_transformed = fitted_model_or_pipeline.predict(
            n_periods=periods,
            return_conf_int=True,
            alpha=alpha
        )
        
        forecast_values = np.expm1(forecast_values_transformed)
        forecast_values[forecast_values < 0] = 0 
        
        lower_bound = np.expm1(conf_int_transformed[:, 0])
        upper_bound = np.expm1(conf_int_transformed[:, 1])
        lower_bound[lower_bound < 0] = 0
        upper_bound[upper_bound < 0] = 0
            
        return forecast_values, lower_bound, upper_bound
    except Exception as e:
        warnings.warn(f"Error during sales prediction: {e}")
        return None, None, None