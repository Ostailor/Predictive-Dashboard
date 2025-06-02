from flask import Blueprint, render_template, request, jsonify, current_app, flash, redirect, url_for
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import json
import os
from werkzeug.utils import secure_filename

from app.ml_models import (
    train_sales_forecasting_model_rf, 
    predict_future_sales_rf, 
    get_distinct_filter_options_from_db,
    get_sales_data_from_db_for_rf, # Assuming this fetches data appropriately
    perform_seasonal_decomposition, # Import the new function
    import_sales_csv_to_db # Make sure this is imported
)
from app.models.data_models import SalesData
from app.extensions import db

bp = Blueprint('main', __name__)

MAX_POINTS_TO_PLOT = 5000 
# Path for the cached, plot-ready aggregated and downsampled test data
CACHED_DAILY_TEST_PLOT_DF_PATH = 'daily_test_plot_for_graph.pkl' # Not used for this issue

# MODEL_PATH = 'sales_forecaster_rf_model.joblib' # This global can be removed or kept as a general default if needed elsewhere
                                                # but for this route, the dynamic path is key.


def downsample_for_plot(df, date_col='date'):
    if df.empty or len(df) <= MAX_POINTS_TO_PLOT:
        return df
    return df.sample(n=MAX_POINTS_TO_PLOT, random_state=42).sort_values(by=date_col)

@bp.route('/')
@bp.route('/index')
def index():
    return render_template('index.html', title='Welcome')

@bp.route('/sales_overview')
def sales_overview():
    graph_json = None
    feature_importance_plot_json = None
    raw_test_plot_df = pd.DataFrame()
    model_mse = float('nan')
    rf_model = None 
    actual_model_path_for_predict = None
    feature_importances_data = None
    decomposition_plots_json = {} # To store decomposition plots

    selected_store = request.args.get('store', 'all')
    selected_item = request.args.get('item', 'all')

    current_store_filter_for_model = selected_store if selected_store != 'all' else None
    current_item_filter_for_model = selected_item if selected_item != 'all' else None

    filter_options = get_distinct_filter_options_from_db()
    
    rf_model, raw_train_plot_df, raw_test_plot_df, model_mse, actual_model_path_for_predict, feature_importances_data = \
        train_sales_forecasting_model_rf(
            store_filter=current_store_filter_for_model,
            item_filter=current_item_filter_for_model
        )

    if rf_model and not raw_train_plot_df.empty:
        # Main forecast plot generation
        future_periods = int(request.args.get('future_periods', 90)) # Default 90 days
        future_predictions_df, _ = predict_future_sales_rf(
            rf_model, 
            raw_train_plot_df.copy(), # Pass a copy to avoid modification issues
            n_periods=future_periods, 
            store_id=current_store_filter_for_model, 
            item_id=current_item_filter_for_model
        )

        fig = go.Figure()
        # fig.add_trace(go.Scatter(x=raw_train_plot_df['date'], y=raw_train_plot_df['sales'], mode='lines', name='Historical Sales (Train)')) # Commented out this line
        if not raw_test_plot_df.empty:
             fig.add_trace(go.Scatter(x=raw_test_plot_df['date'], y=raw_test_plot_df['sales'], mode='lines', name='Actual Sales (Test)'))
             fig.add_trace(go.Scatter(x=raw_test_plot_df['date'], y=raw_test_plot_df['predictions'], mode='lines', name='Predicted Sales (Test)', line=dict(dash='dash')))
        if not future_predictions_df.empty:
            fig.add_trace(go.Scatter(x=future_predictions_df['date'], y=future_predictions_df['predicted_sales'], mode='lines', name='Future Forecast', line=dict(dash='dot')))
        
        fig.update_layout(
            title=f'Sales Forecast for Store: {selected_store}, Item: {selected_item}',
            xaxis_title='Date',
            yaxis_title='Sales',
            template='plotly_white'
        )
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Feature Importance Plot
        if feature_importances_data:
            # feature_importances_data is a list of [feature_name, importance_value] pairs
            importances_df = pd.DataFrame(feature_importances_data, columns=['feature', 'importance']).sort_values(by='importance', ascending=False)
            fig_importance = go.Figure([go.Bar(x=importances_df['feature'], y=importances_df['importance'])])
            fig_importance.update_layout(title_text='Feature Importances for RF Model', xaxis_title='Feature', yaxis_title='Importance')
            feature_importance_plot_json = json.dumps(fig_importance, cls=plotly.utils.PlotlyJSONEncoder)

        # Perform Seasonal Decomposition
        if not raw_train_plot_df.empty and 'sales' in raw_train_plot_df and 'date' in raw_train_plot_df:
            series_for_decomposition = raw_train_plot_df.set_index('date')['sales'].sort_index()
            
            data_span_days = (series_for_decomposition.index.max() - series_for_decomposition.index.min()).days
            num_points = len(series_for_decomposition)
            
            assumed_period = 0 # Default to 0, meaning not enough data or period undetermined

            # Try to determine a suitable period
            if data_span_days > 730 and num_points >= 2 * 365: # Prefer yearly if >2 years of data
                assumed_period = 365
            elif data_span_days > 60 and num_points >= 2 * 30: # Then monthly if >2 months of data
                assumed_period = 30
            elif num_points >= 2 * 7: # Lastly weekly if at least 2 weeks of data
                assumed_period = 7
            
            current_app.logger.info(f"Decomposition: Data span {data_span_days} days, {num_points} points. Tentative period: {assumed_period}")

            if assumed_period > 0:
                # The perform_seasonal_decomposition function itself checks if len(series) < 2 * period
                trend, seasonal, resid = perform_seasonal_decomposition(series_for_decomposition, period=assumed_period)

                if trend is not None:
                    current_app.logger.info("Decomposition: Trend component generated.")
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(x=trend.index, y=trend, mode='lines', name='Trend'))
                    fig_trend.update_layout(title='Trend Component', template='plotly_white', showlegend=False)
                    decomposition_plots_json['trend'] = json.dumps(fig_trend, cls=plotly.utils.PlotlyJSONEncoder)
                else:
                    current_app.logger.info("Decomposition: Trend component was None.")

                if seasonal is not None:
                    current_app.logger.info("Decomposition: Seasonal component generated.")
                    fig_seasonal = go.Figure()
                    # Plot only a segment of seasonality if it's too long (e.g., last 2 periods)
                    plot_seasonal = seasonal.iloc[-2*assumed_period:] if len(seasonal) > 2*assumed_period else seasonal
                    fig_seasonal.add_trace(go.Scatter(x=plot_seasonal.index, y=plot_seasonal, mode='lines', name='Seasonality'))
                    fig_seasonal.update_layout(title='Seasonal Component', template='plotly_white', showlegend=False)
                    decomposition_plots_json['seasonal'] = json.dumps(fig_seasonal, cls=plotly.utils.PlotlyJSONEncoder)
                else:
                    current_app.logger.info("Decomposition: Seasonal component was None.")

                if resid is not None:
                    current_app.logger.info("Decomposition: Residual component generated.")
                    fig_resid = go.Figure()
                    fig_resid.add_trace(go.Scatter(x=resid.index, y=resid, mode='lines', name='Residual'))
                    fig_resid.update_layout(title='Residual Component', template='plotly_white', showlegend=False)
                    decomposition_plots_json['residual'] = json.dumps(fig_resid, cls=plotly.utils.PlotlyJSONEncoder)
                else:
                    current_app.logger.info("Decomposition: Residual component was None.")
            else:
                current_app.logger.info("Decomposition: Skipped due to insufficient data for any suitable period.")
        else:
            current_app.logger.info("Decomposition: Skipped because raw_train_plot_df is empty or missing 'sales'/'date' columns.")

    else:
        # print("DEBUG: RF Model or training data is not available. Skipping forecast and decomposition.")
        # Handle case where model training might have failed or no data
        if raw_train_plot_df.empty:
             current_app.logger.info(f"No training data found for store '{selected_store}' and item '{selected_item}'. Cannot generate plots.")
        else:
             current_app.logger.info(f"RF model not available for store '{selected_store}' and item '{selected_item}'. Cannot generate plots.")


    return render_template('sales_overview.html',
                           graph_json=graph_json,
                           feature_importance_plot_json=feature_importance_plot_json,
                           stores=filter_options["stores"], # Kept for compatibility if other parts of template use it
                           items=filter_options["items"],   # Kept for compatibility
                           filter_options=filter_options, # Add this line
                           selected_store=selected_store,
                           selected_item=selected_item,
                           model_mse=f"{model_mse:.2f}" if model_mse is not None and not pd.isna(model_mse) else "N/A",
                           test_data_exists=not raw_test_plot_df.empty,
                           decomposition_plots_json=decomposition_plots_json)

# Define a helper for allowed extensions (optional but good for security)
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@bp.route('/data_management', methods=['GET', 'POST'])
def data_management():
    if request.method == 'POST':
        if 'sales_csv' not in request.files:
            flash('No file part selected.', 'warning')
            return redirect(request.url)
        file = request.files['sales_csv']
        if file.filename == '':
            flash('No file selected.', 'warning')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            upload_folder = current_app.config.get('UPLOAD_FOLDER')
            if not upload_folder:
                upload_folder = os.path.join(current_app.instance_path, 'uploads')
                os.makedirs(upload_folder, exist_ok=True)
            
            file_path = os.path.join(upload_folder, filename)
            
            try:
                file.save(file_path)
                current_app.logger.info(f"Uploaded file saved to {file_path}")
                
                import_status = import_sales_csv_to_db(file_path)
                
                if import_status['error_message']:
                    flash(f"Error processing file '{filename}': {import_status['error_message']}", 'danger')
                else:
                    success_message = f"File '{filename}' processed. "
                    success_message += f"Total records in CSV: {import_status['total_processed']}. "
                    success_message += f"New records imported: {import_status['imported_count']}. "
                    success_message += f"Records skipped (duplicates): {import_status['skipped_count']}."
                    flash(success_message, 'success')
                
                # Optionally, delete the file after processing if it's temporary
                # try:
                #     os.remove(file_path)
                #     current_app.logger.info(f"Temporary file {file_path} removed after processing.")
                # except OSError as e_remove:
                #     current_app.logger.error(f"Error removing temporary file {file_path}: {e_remove}")

            except Exception as e: # Catch errors related to file saving or other unexpected issues
                current_app.logger.error(f"Critical error during file upload or processing for {filename}: {e}")
                flash(f'A critical error occurred with file "{filename}": {str(e)}', 'danger')
            
            return redirect(url_for('main.data_management'))
        else:
            flash('Invalid file type. Please upload a CSV file.', 'danger')
            return redirect(request.url)

    return render_template('data_management.html', title='Data Management')

@bp.route('/get_predictions')
def get_predictions_route():
    # This route will handle AJAX requests for getting predictions
    # For now, it just returns a static example response
    return jsonify({"predictions": "example"})