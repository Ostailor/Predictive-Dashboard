import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from flask import current_app, jsonify # Ensure jsonify is imported if you plan to use it directly here, though typically done in routes
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # Add MAE and R2Score
from .models.data_models import SalesData # Assuming this is how you import SalesData
from .extensions import db # Assuming this is how you import db
from sqlalchemy import distinct
import pmdarima as pm
import warnings
import re # Import the regular expression module
from datetime import datetime # To display timestamps

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
        "metrics": os.path.join(cache_base_dir, f"{base_filename}_metrics.json"), # Renamed from mse
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
    Returns a dictionary with import status:
    {
        'imported_count': int,
        'skipped_count': int,
        'total_processed': int,
        'error_message': str or None
    }
    """
    df = load_sales_csv(csv_path)
    if df.empty:
        current_app.logger.warning("CSV data is empty. Aborting import to DB.")
        return {
            'imported_count': 0,
            'skipped_count': 0,
            'total_processed': 0,
            'error_message': "CSV file was empty or could not be loaded."
        }

    total_processed_in_csv = len(df)
    imported_count = 0
    skipped_count = 0
    error_message = None

    try:
        # 1. Fetch existing unique keys (date, store, item) from the database into a set
        current_app.logger.info("Fetching existing sales records identifiers from DB...")
        existing_records_query = db.session.query(SalesData.date, SalesData.store, SalesData.item).all()
        existing_records_set = set(
            (
                record.date if hasattr(record.date, 'year') else pd.to_datetime(record.date).date(),
                record.store,
                record.item
            ) for record in existing_records_query
        )
        current_app.logger.info(f"Found {len(existing_records_set)} existing unique records in the database.")

        new_records_to_insert = []
        
        df['date_obj'] = df['date'].dt.date 

        current_app.logger.info("Processing CSV data and identifying new records...")
        for row_dict in df.to_dict(orient='records'):
            current_record_key = (
                row_dict['date_obj'], 
                row_dict['store'],
                row_dict['item']
            )

            if current_record_key not in existing_records_set:
                new_records_to_insert.append({
                    'date': row_dict['date_obj'],
                    'store': row_dict['store'],
                    'item': row_dict['item'],
                    'sales': row_dict['sales']
                })
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            current_app.logger.info(f"Skipped {skipped_count} records that already exist in the database.")

        if new_records_to_insert:
            current_app.logger.info(f"Attempting to bulk insert {len(new_records_to_insert)} new records...")
            db.session.bulk_insert_mappings(SalesData, new_records_to_insert)
            db.session.commit()
            imported_count = len(new_records_to_insert)
            current_app.logger.info(f"Successfully bulk inserted {imported_count} new records into the database.")
        else:
            current_app.logger.info("No new records to insert.")
        
        current_app.logger.info(f"Data import process from {csv_path} completed.")

    except Exception as e:
        db.session.rollback()
        error_message = f"Error importing CSV to database: {str(e)}"
        current_app.logger.error(error_message)
        import traceback
        traceback.print_exc()
    
    return {
        'imported_count': imported_count,
        'skipped_count': skipped_count,
        'total_processed': total_processed_in_csv,
        'error_message': error_message
    }

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
def train_sales_forecasting_model_rf(data_source='db', csv_path=None, store_filter=None, item_filter=None, force_retrain=False):
    cache_paths = _get_rf_cache_paths(store_filter, item_filter)
    MODEL_PATH = cache_paths["model"]
    TRAIN_DF_PATH = cache_paths["train_df"]
    TEST_DF_PATH = cache_paths["test_df"]
    METRICS_PATH = cache_paths["metrics"] # Use new path
    IMPORTANCES_PATH = os.path.join(os.path.dirname(MODEL_PATH), f"rf_model_store_{store_filter if store_filter is not None else 'all'}_item_{item_filter if item_filter is not None else 'all'}_importances.json")

    if not force_retrain:
        if os.path.exists(MODEL_PATH) and \
           os.path.exists(TRAIN_DF_PATH) and \
           os.path.exists(TEST_DF_PATH) and \
           os.path.exists(METRICS_PATH) and \
           os.path.exists(IMPORTANCES_PATH):
            try:
                current_app.logger.info(f"Attempting to load cached RF model and artifacts for store '{store_filter}', item '{item_filter}'.")
                model = joblib.load(MODEL_PATH)
                train_plot_df = pd.read_pickle(TRAIN_DF_PATH)
                test_plot_df = pd.read_pickle(TEST_DF_PATH)
                with open(METRICS_PATH, 'r') as f:
                    metrics_data = json.load(f)
                # test_mse = metrics_data.get('test_mse', float('nan')) # Old way
                with open(IMPORTANCES_PATH, 'r') as f:
                    feature_importances_data = json.load(f)
                current_app.logger.info(f"Successfully loaded cached RF model and artifacts.")
                return model, train_plot_df, test_plot_df, metrics_data, MODEL_PATH, feature_importances_data # Return metrics_data
            except Exception as e:
                current_app.logger.warning(f"Error loading cached RF model or artifacts: {e}. Proceeding to retrain.")
    else:
        current_app.logger.info(f"Force retrain is True for RF model (store: '{store_filter}', item: '{item_filter}'). Skipping cache load.")

    current_app.logger.info(f"Training new RF model for store '{store_filter}', item '{item_filter}'. Data source: {data_source}")
    if data_source == 'db':
        df = get_sales_data_from_db_for_rf(store_filter=store_filter, item_filter=item_filter)
        if df.empty:
            return None, pd.DataFrame(), pd.DataFrame(), {}, None, None # Return empty dict for metrics
    elif data_source == 'csv' and csv_path:
        df = load_sales_csv(csv_path) 
        if df.empty:
            return None, pd.DataFrame(), pd.DataFrame(), {}, None, None
    else:
        return None, pd.DataFrame(), pd.DataFrame(), {}, None, None

    if df.shape[0] < 20: 
        current_app.logger.warning(f"Not enough data points ({df.shape[0]}) for RF model training. Required >= 20.")
        return None, pd.DataFrame(), pd.DataFrame(), {}, None, None
        
    df_processed = preprocess_data_for_rf(df.copy())
    if df_processed.empty or 'sales' not in df_processed.columns or df_processed.shape[0] < 2:
        return None, pd.DataFrame(), pd.DataFrame(), {}, None, None

    X = df_processed.drop(['sales', 'date'], axis=1)
    y = df_processed['sales']
    dates_for_split = df_processed['date']
    
    feature_names = X.columns.tolist()

    if len(X) < 2 or len(y) < 2: 
        current_app.logger.warning("Not enough data after processing for train/test split in RF model.")
        return None, pd.DataFrame(), pd.DataFrame(), {}, None, None

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates_for_split, test_size=0.2, random_state=42, stratify=None
    )
    
    if X_train.empty or X_test.empty:
        current_app.logger.warning("Training or testing set is empty after split for RF model.")
        return None, pd.DataFrame(), pd.DataFrame(), {}, None, None

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) 
    model.fit(X_train, y_train)
    current_app.logger.info(f"RF Model fitting complete for store '{store_filter}', item: '{item_filter}'.")

    predictions_on_test = model.predict(X_test)
    
    # Calculate all metrics
    test_mse = mean_squared_error(y_test, predictions_on_test)
    test_mae = mean_absolute_error(y_test, predictions_on_test)
    test_r2 = r2_score(y_test, predictions_on_test)
    
    metrics_data = {
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }
    
    importances = model.feature_importances_
    feature_importances_data = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    try:
        joblib.dump(model, MODEL_PATH)
        
        train_plot_df = pd.DataFrame({'date': dates_train, 'sales': y_train}).sort_values(by='date')
        test_plot_df = pd.DataFrame({
            'date': dates_test,
            'sales': y_test, 
            'predictions': predictions_on_test 
        }).sort_values(by='date')

        train_plot_df.to_pickle(TRAIN_DF_PATH)
        test_plot_df.to_pickle(TEST_DF_PATH)
        with open(METRICS_PATH, 'w') as f: # Save to new metrics path
            json.dump(metrics_data, f)
        with open(IMPORTANCES_PATH, 'w') as f: 
            json.dump(feature_importances_data, f)
        current_app.logger.info(f"Successfully saved new RF model and artifacts to cache for store '{store_filter}', item '{item_filter}'.")
    except Exception as e:
        current_app.logger.error(f"Error saving RF model or plot artifacts to cache: {e}")
        return model, train_plot_df, test_plot_df, metrics_data, None, feature_importances_data

    return model, train_plot_df, test_plot_df, metrics_data, MODEL_PATH, feature_importances_data # Return metrics_data

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

def list_cached_rf_models():
    """
    Scans the RF model cache directory and lists cached models with their details.
    Returns a list of dictionaries, e.g.:
    [{'store': '1', 'item': '101', 'last_modified': 'YYYY-MM-DD HH:MM:SS', 'filename': '...'}]
    """
    instance_path = current_app.instance_path
    cache_base_dir = os.path.join(instance_path, MODEL_CACHE_DIR_NAME)
    
    cached_models = []
    if not os.path.exists(cache_base_dir):
        current_app.logger.info(f"Cache directory {cache_base_dir} does not exist. No models to list.")
        return cached_models

    # Regex to parse store and item from filenames like:
    # rf_model_store_1_item_101.joblib
    # rf_model_store_all_item_all.joblib
    # rf_model_store_1_item_all.joblib
    # rf_model_store_all_item_101.joblib
    filename_pattern = re.compile(r"rf_model_store_(?P<store>[^_]+)_item_(?P<item>[^_]+)\.joblib")

    for filename in os.listdir(cache_base_dir):
        match = filename_pattern.match(filename)
        if match:
            store = match.group('store')
            item = match.group('item')
            
            file_path = os.path.join(cache_base_dir, filename)
            try:
                timestamp = os.path.getmtime(file_path)
                last_modified = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                current_app.logger.error(f"Could not get timestamp for {file_path}: {e}")
                last_modified = "N/A"
                
            cached_models.append({
                'store': store,
                'item': item,
                'last_modified': last_modified,
                'filename': filename # Could be useful for a delete action later
            })
            
    # Sort by store, then item for consistent display
    cached_models.sort(key=lambda x: (x['store'], x['item']))
    current_app.logger.info(f"Found {len(cached_models)} cached RF model files.")
    return cached_models

def delete_cached_rf_model(store_filter, item_filter):
    """
    Deletes all cached files associated with a specific RF model cohort.
    Returns True if deletion was attempted (even if some files were already missing),
    False if a critical error occurred.
    """
    cache_paths = _get_rf_cache_paths(store_filter, item_filter)
    # Also need the importances path, which is not directly in _get_rf_cache_paths output
    store_str = str(store_filter) if store_filter is not None else 'all'
    item_str = str(item_filter) if item_filter is not None else 'all'
    instance_path = current_app.instance_path
    cache_base_dir = os.path.join(instance_path, MODEL_CACHE_DIR_NAME)
    importances_filename = f"rf_model_store_{store_str}_item_{item_str}_importances.json"
    IMPORTANCES_PATH = os.path.join(cache_base_dir, importances_filename)

    files_to_delete = [
        cache_paths["model"],
        cache_paths["train_df"],
        cache_paths["test_df"],
        cache_paths["metrics"], # Use new path name
        IMPORTANCES_PATH
    ]

    all_successful = True
    current_app.logger.info(f"Attempting to delete cached model for store '{store_str}', item '{item_str}'.")
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                current_app.logger.info(f"Successfully deleted cached file: {file_path}")
            else:
                current_app.logger.info(f"Cached file not found (already deleted or never existed): {file_path}")
        except Exception as e:
            current_app.logger.error(f"Error deleting cached file {file_path}: {e}")
            all_successful = False # Consider this a partial failure
    
    if all_successful:
        current_app.logger.info(f"Cache deletion process completed for store '{store_str}', item '{item_str}'.")
    else:
        current_app.logger.warning(f"Cache deletion process completed with some errors for store '{store_str}', item '{item_str}'.")
    return all_successful # Or True even if some files were missing, as the goal is to ensure they are gone.

def clear_all_cached_rf_models():
    """
    Deletes all files within the RF model cache directory.
    Returns True if successful or if the directory was already empty/missing.
    Returns False if an error occurs during deletion of the directory or its contents.
    """
    instance_path = current_app.instance_path
    cache_base_dir = os.path.join(instance_path, MODEL_CACHE_DIR_NAME)
    
    current_app.logger.info(f"Attempting to clear all cached RF models from directory: {cache_base_dir}")
    
    if not os.path.exists(cache_base_dir):
        current_app.logger.info(f"Cache directory {cache_base_dir} does not exist. Nothing to clear.")
        return True
        
    if not os.path.isdir(cache_base_dir):
        current_app.logger.error(f"Cache path {cache_base_dir} exists but is not a directory. Cannot clear.")
        return False

    cleared_files_count = 0
    errors_encountered = False
    
    for filename in os.listdir(cache_base_dir):
        file_path = os.path.join(cache_base_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                current_app.logger.info(f"Deleted cached file: {file_path}")
                cleared_files_count += 1
            elif os.path.isdir(file_path): # Should not have subdirectories based on current logic, but good to handle
                # For simplicity, we are not recursively deleting subdirectories here.
                # If subdirectories were expected, shutil.rmtree(file_path) would be used.
                current_app.logger.warning(f"Found unexpected subdirectory in cache: {file_path}. Skipping.")
        except Exception as e:
            current_app.logger.error(f"Error deleting {file_path}. Reason: {e}")
            errors_encountered = True
            
    if errors_encountered:
        current_app.logger.error(f"Finished clearing RF cache with errors. {cleared_files_count} files deleted.")
        return False
    else:
        current_app.logger.info(f"Successfully cleared {cleared_files_count} files from RF cache directory {cache_base_dir}.")
        return True

