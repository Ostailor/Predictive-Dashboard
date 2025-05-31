from flask import Blueprint, render_template, request, jsonify
from app.ml_models import (
    train_sales_forecasting_model_rf, 
    predict_future_sales_rf
)
import plotly.graph_objects as go
import plotly.utils
import json
import pandas as pd
import numpy as np
from datetime import timedelta
import os
import joblib

bp = Blueprint('main', __name__)

MAX_POINTS_TO_PLOT = 5000 
# Path for the cached, plot-ready aggregated and downsampled test data
CACHED_DAILY_TEST_PLOT_DF_PATH = 'daily_test_plot_for_graph.pkl'

def downsample_for_plot(df, date_col='date'):
    if df.empty or len(df) <= MAX_POINTS_TO_PLOT:
        return df
    print(f"DEBUG: Downsampling daily data from {len(df)} to {MAX_POINTS_TO_PLOT} days for plotting.")
    return df.sample(n=MAX_POINTS_TO_PLOT, random_state=42).sort_values(by=date_col)

@bp.route('/')
@bp.route('/index')
def index():
    return render_template('index.html', title='Welcome')

@bp.route('/sales_overview')
def sales_overview():
    graph_json = None
    force_retrain = request.args.get('retrain', 'false').lower() == 'true'
    
    MODEL_PATH = 'sales_forecaster_rf_model.joblib'
    TRAIN_DF_PATH = 'train_plot_df.pkl' # For raw training data (e.g. for last_historical_date)
    TEST_DF_PATH = 'test_plot_df.pkl'   # For raw test data (source if plot cache is missed)
    TEST_MSE_PATH = 'test_mse.json'

    rf_model = None
    raw_train_plot_df = pd.DataFrame()
    raw_test_plot_df = pd.DataFrame() # Will hold raw test data if needed
    test_mse = float('nan')
    daily_test_plot_df = pd.DataFrame() # This is what we'll use for plotting test data

    loaded_from_plot_cache = False

    if not force_retrain and os.path.exists(CACHED_DAILY_TEST_PLOT_DF_PATH) and \
       os.path.exists(MODEL_PATH) and os.path.exists(TRAIN_DF_PATH) and \
       os.path.exists(TEST_MSE_PATH):
        try:
            print(f"DEBUG: Attempting to load cached plot-ready test data from {CACHED_DAILY_TEST_PLOT_DF_PATH} and other essential artifacts...")
            daily_test_plot_df = pd.read_pickle(CACHED_DAILY_TEST_PLOT_DF_PATH)
            rf_model = joblib.load(MODEL_PATH)
            raw_train_plot_df = pd.read_pickle(TRAIN_DF_PATH) # Still needed for last_historical_date
            with open(TEST_MSE_PATH, 'r') as f:
                test_mse = json.load(f)['test_mse']
            print(f"DEBUG: Successfully loaded cached daily_test_plot_df (shape: {daily_test_plot_df.shape}) and other artifacts.")
            loaded_from_plot_cache = True
        except Exception as e:
            print(f"DEBUG: Error loading from {CACHED_DAILY_TEST_PLOT_DF_PATH} or other artifacts: {e}. Will proceed to generate.")
            # Reset to ensure regeneration path is taken
            daily_test_plot_df = pd.DataFrame() 
            rf_model = None
            loaded_from_plot_cache = False

    if not loaded_from_plot_cache:
        # Load or retrain base model and raw data
        if not force_retrain and os.path.exists(MODEL_PATH) and \
           os.path.exists(TRAIN_DF_PATH) and os.path.exists(TEST_DF_PATH) and \
           os.path.exists(TEST_MSE_PATH):
            try:
                print("DEBUG: Loading base model artifacts (model, raw_train, raw_test, mse)...")
                rf_model = joblib.load(MODEL_PATH)
                raw_train_plot_df = pd.read_pickle(TRAIN_DF_PATH)
                raw_test_plot_df = pd.read_pickle(TEST_DF_PATH) # Load raw test data
                with open(TEST_MSE_PATH, 'r') as f:
                    test_mse = json.load(f)['test_mse']
                print(f"DEBUG: Loaded base model artifacts. Test MSE: {test_mse}")
            except Exception as e:
                print(f"DEBUG: Error loading base model artifacts: {e}. Will proceed to retrain if necessary.")
                rf_model = None # Fallback to retraining if any base artifact fails to load

        if rf_model is None or force_retrain: # Condition for retraining
            print(f"DEBUG: Training new RF model (force_retrain={force_retrain}, rf_model is None={rf_model is None})...")
            rf_model, raw_train_plot_df, raw_test_plot_df, test_mse = \
                train_sales_forecasting_model_rf(data_source='db')
            
            if rf_model is None:
                print("DEBUG: Model training failed.")
                return render_template('sales_overview.html', graph_json=None, title="Sales Overview & Forecast")
            
            # If retrained, invalidate the old plot-specific cache
            if os.path.exists(CACHED_DAILY_TEST_PLOT_DF_PATH):
                try:
                    os.remove(CACHED_DAILY_TEST_PLOT_DF_PATH)
                    print(f"DEBUG: Removed stale {CACHED_DAILY_TEST_PLOT_DF_PATH} due to retraining.")
                except OSError as e_remove:
                    print(f"DEBUG: Error removing stale {CACHED_DAILY_TEST_PLOT_DF_PATH}: {e_remove}")
        
        # Generate daily_test_plot_df from raw_test_plot_df if not loaded from its own cache
        # This block runs if plot cache was missed, or after retraining.
        if not raw_test_plot_df.empty:
            print("DEBUG: Generating daily_test_plot_df from raw_test_plot_df for plotting and caching...")
            temp_aggregated_test_df = raw_test_plot_df.groupby('date').agg(
                total_daily_actuals=('actual_sales', 'sum'),
                total_daily_predictions=('predicted_sales', 'sum')
            ).reset_index()
            temp_aggregated_test_df['prediction_error'] = temp_aggregated_test_df['total_daily_actuals'] - temp_aggregated_test_df['total_daily_predictions']
            
            daily_test_plot_df = downsample_for_plot(temp_aggregated_test_df) # This is the final df for plot
            
            # Save the newly generated daily_test_plot_df to its specific cache
            try:
                daily_test_plot_df.to_pickle(CACHED_DAILY_TEST_PLOT_DF_PATH)
                print(f"DEBUG: Saved newly generated daily_test_plot_df to {CACHED_DAILY_TEST_PLOT_DF_PATH} (shape: {daily_test_plot_df.shape})")
            except Exception as e_save_cache:
                print(f"DEBUG: Error saving daily_test_plot_df to {CACHED_DAILY_TEST_PLOT_DF_PATH}: {e_save_cache}")
        elif rf_model is not None: # Model exists, but raw_test_plot_df was somehow empty
             print("DEBUG: raw_test_plot_df is empty after load/train. Cannot generate plot data.")
             daily_test_plot_df = pd.DataFrame() # Ensure it's empty for plotting logic
        # If rf_model is None here, it means training failed and we already returned.

    # --- Plotting logic starts here ---
    fig = go.Figure()
    
    if rf_model is None and daily_test_plot_df.empty : # Check if we have anything to plot
        print("DEBUG: No model or data available for plotting.")
        # Fallback to render template without graph_json or with a message
        # This case should ideally be caught earlier if training fails.
    else:
        print(f"DEBUG: Proceeding to plot. daily_test_plot_df shape: {daily_test_plot_df.shape if not daily_test_plot_df.empty else 'N/A'}")
        if not daily_test_plot_df.empty:
            fig.add_trace(go.Scatter(
                x=daily_test_plot_df['date'].tolist(),
                y=daily_test_plot_df['total_daily_actuals'].tolist(),
                mode='lines', 
                name='Test Data (Actuals)',
                line=dict(color='dodgerblue', width=2),
                hovertemplate=
                    "<b>Date</b>: %{x|%Y-%m-%d}<br>" +
                    "<b>Actual Sales</b>: %{y:,.0f}<br>" +
                    "<extra></extra>" 
            ))
            fig.add_trace(go.Scatter(
                x=daily_test_plot_df['date'].tolist(),
                y=daily_test_plot_df['total_daily_predictions'].tolist(),
                mode='lines', 
                name=f'Test Data (Predictions)<br>Original MSE: {test_mse:.2f}',
                line=dict(color='orangered', dash='dash', width=2),
                customdata=daily_test_plot_df['prediction_error'] if 'prediction_error' in daily_test_plot_df else None,
                hovertemplate=
                    "<b>Date</b>: %{x|%Y-%m-%d}<br>" +
                    "<b>Predicted Sales</b>: %{y:,.0f}<br>" +
                    ("%{customdata:,.0f}<br>" if 'prediction_error' in daily_test_plot_df else "") + # Conditional hover part
                    "<extra></extra>"
            ))
        
        # --- Future Forecast ---
        all_raw_historical_dates = []
        if not raw_train_plot_df.empty: # raw_train_plot_df should be loaded by now
            all_raw_historical_dates.append(raw_train_plot_df['date'])
        # If raw_test_plot_df was not loaded because plot cache hit, we might not have it.
        # However, last_historical_date should ideally come from the full dataset context.
        # For simplicity, if raw_test_plot_df is available, use it.
        if not raw_test_plot_df.empty: # Check if it was loaded/generated
             all_raw_historical_dates.append(raw_test_plot_df['date'])
        
        last_historical_date = pd.Timestamp.now().normalize() - timedelta(days=1) 
        if all_raw_historical_dates:
            try:
                last_historical_date = pd.to_datetime(pd.concat(all_raw_historical_dates)).max()
            except Exception as e_concat:
                print(f"DEBUG: Error concatenating dates for last_historical_date: {e_concat}")
                # Keep default last_historical_date

        future_start_date = last_historical_date + timedelta(days=1)
        num_future_periods = 30 
        future_dates_for_plot = pd.date_range(start=future_start_date, periods=num_future_periods, freq='D')

        if not future_dates_for_plot.empty and rf_model is not None: # Ensure model is loaded for prediction
            future_features_df = pd.DataFrame({
                'date': future_dates_for_plot,
                'store': 1, 
                'item': 1   
            })
            future_predictions_values = predict_future_sales_rf(future_features_df, model_path=MODEL_PATH) # Uses MODEL_PATH

            if future_predictions_values is not None:
                fig.add_trace(go.Scatter(
                    x=future_dates_for_plot.tolist(),
                    y=future_predictions_values.tolist() if hasattr(future_predictions_values, 'tolist') else future_predictions_values,
                    mode='lines', 
                    name='Future Forecast (Store 1, Item 1)', 
                    line=dict(color='green', dash='dot', width=2),
                    hovertemplate=
                        "<b>Date</b>: %{x|%Y-%m-%d}<br>" +
                        "<b>Forecasted Sales</b>: %{y:,.0f}<br>" +
                        "<extra></extra>"
                ))
            else:
                print("DEBUG: Future predictions returned None.")
        elif rf_model is None:
             print("DEBUG: RF Model not loaded, cannot make future predictions.")
        else: # future_dates_for_plot was empty
            print("DEBUG: No future dates to predict.")

        fig.update_layout(
            title_text='Daily Sales: Test Performance & RF Forecast',
            xaxis_title='Date',
            yaxis_title='Total Daily Sales',
            template='plotly_white', 
            legend_title_text='Legend',
            hovermode='x unified', 
            height=600 
        )
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        print("DEBUG: graph_json generated for RF model.")

    return render_template('sales_overview.html', graph_json=graph_json, title="Sales Overview & Forecast")

@bp.route('/get_predictions')
def get_predictions_route():
    # This route will handle AJAX requests for getting predictions
    # For now, it just returns a static example response
    return jsonify({"predictions": "example"})