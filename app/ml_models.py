import pandas as pd
import numpy as np
from app import db
from app.models.data_models import SalesData
from sqlalchemy import func
import pmdarima as pm
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # <--- ADD THIS LINE
from sklearn.metrics import mean_squared_error # Already there, but good to check
import joblib
import pandas as pd
import json # For saving MSE

print(f"DEBUG: pmdarima version being used: {pm.__version__}")
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.", category=FutureWarning)

# --- CSV Loading and Importing Functions ---

def load_sales_csv(csv_path):
    """
    Load sales data from a CSV file.
    Assumes CSV has columns: 'date', 'store', 'item', 'sales'
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        print(f"Successfully loaded CSV from {csv_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return pd.DataFrame() # Return empty DataFrame on error
    except Exception as e:
        print(f"Error loading CSV from {csv_path}: {e}")
        return pd.DataFrame()

def import_sales_csv_to_db(csv_path):
    """
    Import sales data from a CSV file into the SalesData table in the database.
    Optimized for speed by reducing per-row DB queries.
    """
    df = load_sales_csv(csv_path)
    if df.empty:
        print("CSV data is empty. Aborting import to DB.")
        return

    try:
        # 1. Fetch existing unique keys (date, store, item) from the database into a set
        print("Fetching existing sales records identifiers from DB...")
        existing_records_query = db.session.query(SalesData.date, SalesData.store, SalesData.item).all()
        # Ensure dates from DB are datetime.date objects if they aren't already
        existing_records_set = set(
            (
                record.date if hasattr(record.date, 'year') else pd.to_datetime(record.date).date(), # Ensure it's a date object
                record.store,
                record.item
            ) for record in existing_records_query
        )
        print(f"Found {len(existing_records_set)} existing unique records in the database.")

        new_records_to_insert = []
        skipped_count = 0
        
        # Prepare data from DataFrame for comparison and insertion
        # Ensure 'date' column is datetime.date objects
        # Pandas read_csv with parse_dates creates Timestamps; .dt.date converts to datetime.date
        df['date_obj'] = df['date'].dt.date 

        print("Processing CSV data and identifying new records...")
        for row_dict in df.to_dict(orient='records'):
            # Create a tuple for the current record's unique key
            # row_dict['date'] is a pandas Timestamp, use 'date_obj' we created
            current_record_key = (
                row_dict['date_obj'], 
                row_dict['store'],
                row_dict['item']
            )

            if current_record_key not in existing_records_set:
                new_records_to_insert.append({
                    'date': row_dict['date_obj'], # Use the datetime.date object
                    'store': row_dict['store'],
                    'item': row_dict['item'],
                    'sales': row_dict['sales']
                })
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} records that already exist in the database.")

        # 3. Bulk insert new records if any
        if new_records_to_insert:
            print(f"Attempting to bulk insert {len(new_records_to_insert)} new records...")
            db.session.bulk_insert_mappings(SalesData, new_records_to_insert)
            db.session.commit()
            print(f"Successfully bulk inserted {len(new_records_to_insert)} new records into the database.")
        else:
            print("No new records to insert.")
        
        print(f"Data import process from {csv_path} completed.")

    except Exception as e:
        db.session.rollback()
        print(f"Error importing CSV to database: {e}")
        import traceback
        traceback.print_exc()

def get_daily_sales_data():
    """
    Fetches sales data, aggregates it by day (summing 'sales' column), 
    and ensures a complete daily series.
    Returns a pandas DataFrame with a DatetimeIndex ('date') and 'total_sales'.
    """
    # Updated to use SalesData.date and SalesData.sales
    daily_sales_query = db.session.query(
        func.date(SalesData.date).label('date'), # Changed from SalesData.timestamp
        func.sum(SalesData.sales).label('total_sales') # Changed from SalesData.amount
    ).group_by(func.date(SalesData.date)).order_by(func.date(SalesData.date))

    df = pd.read_sql_query(daily_sales_query.statement, db.engine)

    if df.empty:
        print("DEBUG: get_daily_sales_data is returning an EMPTY DataFrame.")
        empty_idx = pd.DatetimeIndex([], name='date')
        return pd.DataFrame({'total_sales': []}, index=empty_idx)

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    if not df.empty:
        if not isinstance(df.index, pd.DatetimeIndex):
             df.index = pd.to_datetime(df.index)
        # Ensure daily frequency by creating a full date range and reindexing
        idx = pd.date_range(df.index.min(), df.index.max(), freq='D') 
        df = df.reindex(idx, fill_value=0) 
        df.index.name = 'date' # Ensure index name is set
    
    return df

def generate_fourier_features(time_steps, period, k_harmonics, prefix=''):
    """
    Generates Fourier series terms.
    :param time_steps: A 1D numpy array representing the time steps.
    :param period: The period of the seasonality.
    :param k_harmonics: The number of Fourier pairs (harmonics) to generate.
    :param prefix: A string prefix for the feature names (e.g., 'monthly_', 'annual_').
    :return: A pandas DataFrame with 2*k_harmonics columns.
    """
    features = {}
    for i in range(1, k_harmonics + 1):
        features[f'{prefix}sin_{period}_{i}'] = np.sin(2 * np.pi * i * time_steps / period)
        features[f'{prefix}cos_{period}_{i}'] = np.cos(2 * np.pi * i * time_steps / period)
    return pd.DataFrame(features)

def generate_calendar_features(date_index):
    """
    Generates calendar-based features (day of week, simple holiday).
    Month dummies are removed to reduce redundancy with monthly Fourier terms.
    :param date_index: A pandas DatetimeIndex.
    :return: A pandas DataFrame with calendar features.
    """
    calendar_df = pd.DataFrame(index=date_index)
    # Day of week dummies (Monday=0, Sunday=6)
    calendar_df['dayofweek'] = date_index.dayofweek
    day_dummies = pd.get_dummies(calendar_df['dayofweek'], prefix='day', drop_first=True) # 6 features
    
    # Simple holiday flag example: Christmas
    calendar_df['is_christmas'] = ((date_index.month == 12) & (date_index.day == 25)).astype(int) # 1 feature
    
    # Combine remaining calendar features
    all_calendar_features = pd.concat([day_dummies, calendar_df[['is_christmas']]], axis=1)
    return all_calendar_features

# This is the pmdarima version of train_sales_forecasting_model
def train_sales_forecasting_model_pmdarima(daily_sales_df): # Renamed for clarity
    print(f"DEBUG: train_sales_forecasting_model_pmdarima called. daily_sales_df empty? {daily_sales_df.empty}")
    if not daily_sales_df.empty:
        print(f"DEBUG: daily_sales_df.shape: {daily_sales_df.shape}")
        print(f"DEBUG: daily_sales_df non-NA 'total_sales' count: {len(daily_sales_df['total_sales'].dropna())}")

    # Adjusted min observations based on features
    if daily_sales_df.empty or len(daily_sales_df['total_sales'].dropna()) < 70: 
        warnings.warn("Not enough data to train pmdarima model. Need at least 70 non-NA observations for calendar and Fourier features.")
        return None
    
    if daily_sales_df.index.freq is None: 
        warnings.warn("Dataframe index does not have frequency for pmdarima model. Attempting to set to 'D'.")
        # Attempt to infer frequency or set to daily. This is crucial for time series models.
        daily_sales_df = daily_sales_df.asfreq('D', fill_value=0) # Or another appropriate frequency

    y_transformed = np.log1p(daily_sales_df['total_sales'])
    
    QUICK_TEST_MODE = True # Set to False for more thorough search

    # --- Calendar Features (Day of week, Christmas) ---
    X_calendar_train = generate_calendar_features(daily_sales_df.index)
    
    # --- Monthly Fourier Terms ---
    M_MONTHLY = 30.4375 
    K_FOURIER_MONTHLY = 2  
    # --- Annual Fourier Terms ---
    M_ANNUAL = 365.25
    K_FOURIER_ANNUAL = 3 

    n_train = len(y_transformed)
    train_time_steps = np.arange(n_train)

    X_fourier_monthly_train = generate_fourier_features(train_time_steps, M_MONTHLY, K_FOURIER_MONTHLY, prefix='monthly_')
    X_fourier_annual_train = generate_fourier_features(train_time_steps, M_ANNUAL, K_FOURIER_ANNUAL, prefix='annual_')
    
    X_fourier_monthly_train.index = y_transformed.index
    X_fourier_annual_train.index = y_transformed.index

    # Combine all exogenous features
    X_train_combined = pd.concat([X_calendar_train, X_fourier_monthly_train, X_fourier_annual_train], axis=1)
    print(f"DEBUG: X_train_combined shape: {X_train_combined.shape}, Columns: {X_train_combined.columns.tolist()}")

    try:
        if QUICK_TEST_MODE:
            print(f"INFO: Running AutoARIMA in QUICK TEST MODE with {X_train_combined.shape[1]} combined exogenous features (relaxed parameters).")
            model = pm.auto_arima(y_transformed,
                X=X_train_combined, 
                start_p=1, start_q=1,
                max_p=4, max_q=4, 
                m=7, seasonal=True, 
                start_P=1, start_Q=1, 
                max_P=2, max_Q=2, 
                d=None, D=None,   
                test='adf',      
                information_criterion='aic',
                trace=False, error_action='ignore', 
                suppress_warnings=True, stepwise=True,
                with_intercept=True 
            )
        else:
            print(f"INFO: Running AutoARIMA with full search parameters and {X_train_combined.shape[1]} combined exogenous features.")
            model = pm.auto_arima(y_transformed,
                X=X_train_combined,
                start_p=1, start_q=1,
                max_p=5, max_q=5, 
                m=7, seasonal=True, 
                max_P=3, max_Q=3, 
                d=None, D=None,   
                test='adf',
                information_criterion='aic',
                trace=True, error_action='ignore', 
                suppress_warnings=True, stepwise=True,
                with_intercept=True
            )
        
        print(f"AutoARIMA (with calendar & Fourier) selected model: {model.order} {model.seasonal_order}")
        
        model.n_train_ = n_train 
        model.fourier_monthly_period_ = M_MONTHLY
        model.fourier_monthly_k_ = K_FOURIER_MONTHLY
        model.fourier_annual_period_ = M_ANNUAL
        model.fourier_annual_k_ = K_FOURIER_ANNUAL
        # Store original index reference if not already stored by pmdarima
        if not hasattr(model, '_index_reference') or model._index_reference is None:
             model._index_reference = daily_sales_df.index 
        return model

    except Exception as e:
        warnings.warn(f"Error training AutoARIMA model with calendar & Fourier terms: {e}") 
        import traceback 
        traceback.print_exc() 
        return None

# This is the pmdarima version of predict_future_sales
def predict_future_sales_pmdarima(fitted_model, periods=30, alpha=0.05): # Renamed for clarity
    if fitted_model is None:
        return None, None, None

    last_train_date = None
    if hasattr(fitted_model, '_index_reference') and fitted_model._index_reference is not None and not fitted_model._index_reference.empty:
        last_train_date = fitted_model._index_reference[-1]
    
    if last_train_date is None:
        warnings.warn("Could not get last_train_date from model._index_reference. Cannot generate future features.")
        return None, None, None

    future_date_index = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=periods, freq='D')

    X_calendar_future = generate_calendar_features(future_date_index)

    X_fourier_monthly_future = pd.DataFrame()
    X_fourier_annual_future = pd.DataFrame()
    
    if hasattr(fitted_model, 'n_train_') and \
       hasattr(fitted_model, 'fourier_monthly_period_') and hasattr(fitted_model, 'fourier_monthly_k_'):
        future_time_steps = np.arange(fitted_model.n_train_, fitted_model.n_train_ + periods)
        X_fourier_monthly_future = generate_fourier_features(
            future_time_steps, 
            fitted_model.fourier_monthly_period_, 
            fitted_model.fourier_monthly_k_,
            prefix='monthly_'
        )
        X_fourier_monthly_future.index = future_date_index

    if hasattr(fitted_model, 'n_train_') and \
       hasattr(fitted_model, 'fourier_annual_period_') and hasattr(fitted_model, 'fourier_annual_k_'):
        if 'future_time_steps' not in locals(): 
            future_time_steps = np.arange(fitted_model.n_train_, fitted_model.n_train_ + periods)
        X_fourier_annual_future = generate_fourier_features(
            future_time_steps,
            fitted_model.fourier_annual_period_,
            fitted_model.fourier_annual_k_,
            prefix='annual_'
        )
        X_fourier_annual_future.index = future_date_index
    
    all_future_X_parts = [X_calendar_future]
    if not X_fourier_monthly_future.empty:
        all_future_X_parts.append(X_fourier_monthly_future)
    if not X_fourier_annual_future.empty:
        all_future_X_parts.append(X_fourier_annual_future)
    
    X_future_combined = pd.concat(all_future_X_parts, axis=1)

    try:
        forecast_values_transformed, conf_int_transformed = fitted_model.predict(
            n_periods=periods,
            X=X_future_combined, 
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
        warnings.warn(f"Error during pmdarima sales prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# --- RandomForest Model Training and Prediction Functions ---

def get_sales_data_from_db_for_rf(): # Renamed for clarity if needed, or ensure SalesData model is consistent
    """
    Fetch sales data from the database (date, store, item, sales).
    """
    query = SalesData.query.all()
    # Ensure SalesData model has date, store, item, sales columns
    df = pd.DataFrame([(d.date, d.store, d.item, d.sales) for d in query],
                      columns=['date', 'store', 'item', 'sales'])
    df['date'] = pd.to_datetime(df['date'])
    return df

def preprocess_data_for_rf(df): # Renamed for clarity
    """
    Preprocess data for RandomForest model.
    """
    if df.empty or 'date' not in df.columns:
        return pd.DataFrame()
    
    df_processed = df.copy()
    df_processed['date'] = pd.to_datetime(df_processed['date'])
    df_processed['dayofweek'] = df_processed['date'].dt.dayofweek
    df_processed['month'] = df_processed['date'].dt.month
    df_processed['year'] = df_processed['date'].dt.year
    df_processed['dayofyear'] = df_processed['date'].dt.dayofyear
    # Add more features as needed, e.g., lag features, rolling means
    return df_processed

# This is the RandomForest version
def train_sales_forecasting_model_rf(data_source='db', csv_path=None):
    """
    Train a sales forecasting model using RandomForest.
    Saves the model, training plot data, test plot data, and test MSE.
    Returns the model, and DataFrames for plotting training actuals, 
    test actuals, and test predictions, plus the test MSE.
    """
    # Define paths for saving artifacts
    MODEL_PATH = 'sales_forecaster_rf_model.joblib'
    TRAIN_DF_PATH = 'train_plot_df.pkl'
    TEST_DF_PATH = 'test_plot_df.pkl'
    TEST_MSE_PATH = 'test_mse.json'

    if data_source == 'csv' and csv_path:
        df = load_sales_csv(csv_path)
    elif data_source == 'db':
        df = get_sales_data_from_db_for_rf()
    else:
        print("Invalid data source or missing CSV path for RF model.")
        return None, pd.DataFrame(), pd.DataFrame(), float('nan')

    if df.empty:
        print("No data to train the RF model.")
        return None, pd.DataFrame(), pd.DataFrame(), float('nan')

    df_processed = preprocess_data_for_rf(df.copy())
    if df_processed.empty or 'date' not in df_processed.columns:
        print("Data preprocessing failed or 'date' column is missing after preprocessing.")
        return None, pd.DataFrame(), pd.DataFrame(), float('nan')

    features = ['store', 'item', 'dayofweek', 'month', 'year', 'dayofyear'] 
    target = 'sales'

    if not all(feature in df_processed.columns for feature in features) or target not in df_processed.columns:
        print(f"Missing required columns for RF training. Need: {features + [target]}")
        return None, pd.DataFrame(), pd.DataFrame(), float('nan')
    
    X = df_processed[features]
    y = df_processed[target]
    dates_for_split = df_processed['date']

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates_for_split, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42) 
    model.fit(X_train, y_train)

    predictions_on_test = model.predict(X_test)
    test_mse = mean_squared_error(y_test, predictions_on_test)
    print(f"RF Model trained on training set. Test MSE: {test_mse}")

    # Save the model and plot artifacts
    try:
        joblib.dump(model, MODEL_PATH)
        print(f"RF Model saved to {MODEL_PATH}")
        
        train_plot_df = pd.DataFrame({'date': dates_train, 'actual_sales': y_train}).sort_values(by='date')
        test_plot_df = pd.DataFrame({
            'date': dates_test,
            'actual_sales': y_test,
            'predicted_sales': predictions_on_test
        }).sort_values(by='date')

        train_plot_df.to_pickle(TRAIN_DF_PATH)
        print(f"Training plot data saved to {TRAIN_DF_PATH}")
        test_plot_df.to_pickle(TEST_DF_PATH)
        print(f"Test plot data saved to {TEST_DF_PATH}")
        with open(TEST_MSE_PATH, 'w') as f:
            json.dump({'test_mse': test_mse}, f)
        print(f"Test MSE saved to {TEST_MSE_PATH}")

    except Exception as e:
        print(f"Error saving model or plot artifacts: {e}")
        # Decide if you want to return None or the data even if saving fails
        # For now, we'll still return the data for the current request.

    return model, train_plot_df, test_plot_df, test_mse

# This is the RandomForest version
def predict_future_sales_rf(future_data_df, model_path='sales_forecaster_rf_model.joblib'): # Renamed for clarity
    """
    Predict future sales using the trained RandomForest model.
    future_data_df should have 'date', 'store', 'item' for preprocessing.
    """
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: RF Model file {model_path} not found. Train the model first.")
        return None
    except Exception as e:
        print(f"Error loading RF model: {e}")
        return None

    if 'date' not in future_data_df.columns or 'store' not in future_data_df.columns or 'item' not in future_data_df.columns:
        print("Error: future_data_df for RF model must contain 'date', 'store', and 'item' columns.")
        return None
        
    df_processed = preprocess_data_for_rf(future_data_df.copy()) 

    expected_features = ['store', 'item', 'dayofweek', 'month', 'year', 'dayofyear'] 
    
    missing_features = [f for f in expected_features if f not in df_processed.columns]
    if missing_features:
        print(f"Error: Missing features in preprocessed data for RF prediction: {missing_features}")
        return None
        
    X_future = df_processed[expected_features]
    
    predictions = model.predict(X_future)
    return predictions

# Example of how you might call these:
if __name__ == '__main__':
    # This block runs if the script is executed directly.
    # Ensure Flask app context for db operations.
    # from app import create_app # Assuming you have a create_app function
    # app = create_app()
    # with app.app_context():
    #     # db.create_all() # Ensure tables are created if they don't exist
    
    #     # --- To import CSV to DB ---
    #     # import_sales_csv_to_db('data/your_sales_data.csv') # Replace with your CSV file path

    #     # --- To train pmdarima model from DB (after importing and aggregation) ---
    #     # daily_data_for_pmdarima = get_daily_sales_data()
    #     # if not daily_data_for_pmdarima.empty:
    #     #    pmdarima_model = train_sales_forecasting_model_pmdarima(daily_data_for_pmdarima)
    #     #    if pmdarima_model:
    #     #        joblib.dump(pmdarima_model, 'sales_forecaster_pmdarima_model.joblib')
    #     #        print("Pmdarima model saved as sales_forecaster_pmdarima_model.joblib")
    #     #        # preds, lower, upper = predict_future_sales_pmdarima(pmdarima_model, periods=30)
    #     #        # if preds is not None:
    #     #        #     print("Pmdarima Predictions:", preds)

    #     # --- To train RandomForest model from CSV ---
    #     # rf_model_csv = train_sales_forecasting_model_rf(data_source='csv', csv_path='data/your_sales_data.csv')
    #     # if rf_model_csv:
    #     #     # --- To make a prediction with RF model ---
    #     #     future_dates_rf = pd.to_datetime(['2025-07-01', '2025-07-02'])
    #     #     sample_future_data_rf = pd.DataFrame({
    #     #         'date': future_dates_rf,
    #     #         'store': [1, 1], 
    #     #         'item': [1, 2]   
    #     #     })
    #     #     rf_predictions = predict_future_sales_rf(sample_future_data_rf, model_path='sales_forecaster_rf_model.joblib')
    #     #     if rf_predictions is not None:
    #     #        print("RandomForest Future Sales Predictions (from CSV model):", rf_predictions)

    #     # --- To train RandomForest model from DB (after importing) ---
    #     # rf_model_db = train_sales_forecasting_model_rf(data_source='db')
    pass