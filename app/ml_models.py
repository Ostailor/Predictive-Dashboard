import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from flask import current_app, jsonify # Ensure jsonify is imported if you plan to use it directly here, though typically done in routes
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from .models.data_models import SalesData # Assuming this is how you import SalesData
from .extensions import db # Assuming this is how you import db
from sqlalchemy import distinct
import pmdarima as pm
import warnings

# print(f"DEBUG: pmdarima version being used: {pm.__version__}") # Removed
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.", category=FutureWarning)

MODEL_CACHE_DIR_NAME = 'rf_models_cache'

def _get_rf_cache_paths(store_filter, item_filter):
    """Generates cache file paths for a given store/item cohort."""
    instance_path = current_app.instance_path
    cache_base_dir = os.path.join(instance_path, MODEL_CACHE_DIR_NAME)
    os.makedirs(cache_base_dir, exist_ok=True)

    store_str = str(store_filter) if store_filter is not None else 'all'
    item_str = str(item_filter) if item_filter is not None else 'all'
    
    base_filename = f"rf_model_store_{store_str}_item_{item_str}"
    
    paths = {
        "model": os.path.join(cache_base_dir, f"{base_filename}.joblib"),
        "train_df": os.path.join(cache_base_dir, f"{base_filename}_train_df.pkl"),
        "test_df": os.path.join(cache_base_dir, f"{base_filename}_test_df.pkl"),
        "mse": os.path.join(cache_base_dir, f"{base_filename}_mse.json"),
    }
    return paths

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

def get_sales_data_from_db_for_rf(store_filter=None, item_filter=None):
    """
    Fetch sales data from the database (date, store, item, sales),
    optionally filtered by store and/or item.
    """
    query_obj = SalesData.query

    if store_filter and store_filter != 'all':
        query_obj = query_obj.filter(SalesData.store == store_filter)
    if item_filter and item_filter != 'all':
        query_obj = query_obj.filter(SalesData.item == item_filter)
    
    sales_records = query_obj.all()
    
    df = pd.DataFrame([(d.date, d.store, d.item, d.sales) for d in sales_records],
                      columns=['date', 'store', 'item', 'sales'])
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

def get_distinct_filter_options_from_db():
    """
    Gets distinct store and item values from the SalesData table.
    """
    stores = ['all'] + [s[0] for s in db.session.query(distinct(SalesData.store)).order_by(SalesData.store).all()]
    items = ['all'] + [i[0] for i in db.session.query(distinct(SalesData.item)).order_by(SalesData.item).all()]
    return {"stores": stores, "items": items}

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
def train_sales_forecasting_model_rf(data_source='db', csv_path=None, store_filter=None, item_filter=None):
    cache_paths = _get_rf_cache_paths(store_filter, item_filter)
    MODEL_PATH = cache_paths["model"]
    TRAIN_DF_PATH = cache_paths["train_df"]
    TEST_DF_PATH = cache_paths["test_df"]
    MSE_PATH = cache_paths["mse"]
    # Add a path for feature importances
    IMPORTANCES_PATH = os.path.join(os.path.dirname(MODEL_PATH), f"rf_model_store_{store_filter if store_filter is not None else 'all'}_item_{item_filter if item_filter is not None else 'all'}_importances.json")


    if os.path.exists(MODEL_PATH) and \
       os.path.exists(TRAIN_DF_PATH) and \
       os.path.exists(TEST_DF_PATH) and \
       os.path.exists(MSE_PATH) and \
       os.path.exists(IMPORTANCES_PATH): # Check for importances cache
        try:
            model = joblib.load(MODEL_PATH)
            train_plot_df = pd.read_pickle(TRAIN_DF_PATH)
            test_plot_df = pd.read_pickle(TEST_DF_PATH)
            with open(MSE_PATH, 'r') as f:
                mse_data = json.load(f)
            test_mse = mse_data.get('test_mse', float('nan'))
            with open(IMPORTANCES_PATH, 'r') as f: # Load cached importances
                feature_importances_data = json.load(f)
            
            return model, train_plot_df, test_plot_df, test_mse, MODEL_PATH, feature_importances_data
        except Exception as e:
            print(f"Error loading cached model or artifacts (including importances): {e}. Retraining.")

    if data_source == 'db':
        df = get_sales_data_from_db_for_rf(store_filter=store_filter, item_filter=item_filter)
        if df.empty:
            return None, pd.DataFrame(), pd.DataFrame(), float('nan'), None, None # Added None for importances
    elif data_source == 'csv' and csv_path:
        df = load_sales_csv(csv_path) 
        if df.empty:
            return None, pd.DataFrame(), pd.DataFrame(), float('nan'), None, None
    else:
        return None, pd.DataFrame(), pd.DataFrame(), float('nan'), None, None

    if df.shape[0] < 20: 
        return None, pd.DataFrame(), pd.DataFrame(), float('nan'), None, None
        
    df_processed = preprocess_data_for_rf(df.copy())
    if df_processed.empty or 'sales' not in df_processed.columns or df_processed.shape[0] < 2:
        return None, pd.DataFrame(), pd.DataFrame(), float('nan'), None, None

    X = df_processed.drop(['sales', 'date'], axis=1)
    y = df_processed['sales']
    dates_for_split = df_processed['date']
    
    feature_names = X.columns.tolist() # Get feature names

    if len(X) < 2 or len(y) < 2: 
        return None, pd.DataFrame(), pd.DataFrame(), float('nan'), None, None

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates_for_split, test_size=0.2, random_state=42, stratify=None
    )
    
    if X_train.empty or X_test.empty:
        return None, pd.DataFrame(), pd.DataFrame(), float('nan'), None, None

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) 
    model.fit(X_train, y_train)

    predictions_on_test = model.predict(X_test)
    test_mse = mean_squared_error(y_test, predictions_on_test)
    
    # Get feature importances
    importances = model.feature_importances_
    feature_importances_data = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)


    try:
        joblib.dump(model, MODEL_PATH)
        
        train_plot_df = pd.DataFrame({'date': dates_train, 'sales': y_train}).sort_values(by='date') # Changed 'actual_sales' to 'sales'
        test_plot_df = pd.DataFrame({
            'date': dates_test,
            'sales': y_test, # Changed 'actual_sales' to 'sales'
            'predictions': predictions_on_test # Changed 'predicted_sales' to 'predictions' for consistency with routes.py if needed, or keep as is if routes.py uses 'predictions' for this df
        }).sort_values(by='date')

        train_plot_df.to_pickle(TRAIN_DF_PATH)
        test_plot_df.to_pickle(TEST_DF_PATH)
        with open(MSE_PATH, 'w') as f:
            json.dump({'test_mse': test_mse}, f)
        with open(IMPORTANCES_PATH, 'w') as f: # Save feature importances
            json.dump(feature_importances_data, f)

    except Exception as e:
        print(f"Error saving model or plot artifacts: {e}")
        return model, train_plot_df, test_plot_df, test_mse, None, feature_importances_data # Return importances even if save fails

    return model, train_plot_df, test_plot_df, test_mse, MODEL_PATH, feature_importances_data

# This is the RandomForest version
def predict_future_sales_rf(model, train_df, n_periods, store_id, item_id):
    """
    Predict future sales using the trained RandomForest model.
    - model: The trained RandomForest model object.
    - train_df: DataFrame of the training data, used to get the last date.
                Must contain a 'date' column.
    - n_periods: Number of future periods (days) to predict.
    - store_id: The store for which to predict. Can be None if model is for 'all stores'.
    - item_id: The item for which to predict. Can be None if model is for 'all items'.
    """
    if model is None:
        print("Error: Model is None. Cannot make predictions.")
        return pd.DataFrame(), []

    if not hasattr(model, 'feature_names_in_'):
        print("Error: Model does not have 'feature_names_in_'. Cannot determine features for prediction.")
        return pd.DataFrame(), []
        
    model_expected_features = model.feature_names_in_

    if train_df.empty or 'date' not in train_df.columns:
        print("Error: Training data is empty or 'date' column missing. Cannot determine start for future dates.")
        return pd.DataFrame(), []

    last_date = pd.to_datetime(train_df['date']).max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_periods, freq='D')

    # Create a base DataFrame for future predictions
    future_df = pd.DataFrame({'date': future_dates})

    # Add store and item columns if they were features in the model
    if 'store' in model_expected_features:
        if store_id is None:
            # This model was trained with 'store' as a feature (e.g., for an "all stores" filter where store IDs varied).
            # Predicting a single series for "all stores" is problematic without a specific store value for the feature.
            print("Error: Model expects 'store' feature, but no specific store_id provided for prediction (e.g. 'all stores' selected). "
                  "This requires a strategy for setting the 'store' feature value for future dates.")
            return pd.DataFrame(), []
        future_df['store'] = store_id
    
    if 'item' in model_expected_features:
        if item_id is None:
            # Similar to 'store', if 'item' is a feature and item_id is None.
            print("Error: Model expects 'item' feature, but no specific item_id provided for prediction (e.g. 'all items' selected). "
                  "This requires a strategy for setting the 'item' feature value for future dates.")
            return pd.DataFrame(), []
        future_df['item'] = item_id
    
    # Preprocess the future_df to create date-based features
    # Uses the existing preprocess_data_for_rf from your file (lines 394-407)
    future_features_df = preprocess_data_for_rf(future_df.copy())
    
    if future_features_df.empty:
        print("Preprocessing future data resulted in an empty DataFrame.")
        return pd.DataFrame(), []

    # Ensure X_future has the same columns as the model was trained on, in the correct order.
    # The preprocess_data_for_rf keeps the original 'date' column, but it's not part of model_expected_features.
    # Selecting by model_expected_features handles this.
    try:
        X_future = future_features_df[model_expected_features]
    except KeyError as e:
        print(f"Error creating feature set for prediction. Missing columns: {e}. "
              f"Model expects: {model_expected_features}. Available: {future_features_df.columns.tolist()}")
        return pd.DataFrame(), []

    if X_future.empty:
        print("Future features DataFrame is empty after selection. Cannot make predictions.")
        return pd.DataFrame(), []

    try:
        predicted_sales = model.predict(X_future)
    except Exception as e:
        print(f"Error during model prediction: {e}")
        print(f"Columns in X_future submitted for prediction: {X_future.columns.tolist()}")
        return pd.DataFrame(), []

    results_df = pd.DataFrame({
        'date': future_dates,
        'predicted_sales': predicted_sales
    })
    
    return results_df, X_future.columns.tolist()

def perform_seasonal_decomposition(series, model='additive', period=365):
    """
    Performs seasonal decomposition on a time series.

    Args:
        series (pd.Series): Time series data, indexed by date.
        model (str): Type of decomposition ('additive' or 'multiplicative').
        period (int): The period of the seasonality. Default is 365 for daily data.
                      For weekly data with yearly seasonality, it might be 52.
                      Adjust based on your data's characteristics.

    Returns:
        tuple: (trend, seasonal, residual) pandas Series.
               Returns (None, None, None) if decomposition fails.
    """
    if series.empty or len(series) < 2 * period: # Ensure enough data for decomposition
        # print(f"DEBUG: Series too short for decomposition. Length: {len(series)}, Required: {2 * period}")
        return None, None, None
    try:
        decomposition = seasonal_decompose(series, model=model, period=period, extrapolate_trend='freq')
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        resid = decomposition.resid
        return trend, seasonal, resid
    except Exception as e:
        print(f"Error during seasonal decomposition: {e}")
        return None, None, None

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
