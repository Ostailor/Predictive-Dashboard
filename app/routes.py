from flask import Blueprint, render_template, jsonify, request # Add request
from app.ml_models import get_daily_sales_data, train_sales_forecasting_model, predict_future_sales
import plotly
import plotly.express as px
import plotly.graph_objects as go # Import graph_objects for more control
import json
import pandas as pd # Import pandas

bp = Blueprint('main', __name__)

@bp.route('/')
@bp.route('/index')
def index():
    return render_template('index.html', title='Home')

@bp.route('/sales_overview')
def sales_overview():
    daily_sales_df_indexed = get_daily_sales_data()
    graph_json = None
    forecast_periods = request.args.get('forecast_periods', 30, type=int)
    print(f"DEBUG: Requested forecast_periods = {forecast_periods}") # Debug forecast_periods

    if not daily_sales_df_indexed.empty:
        # ... (code to get y_values_historical and model) ...
        daily_sales_df_for_plot = daily_sales_df_indexed.reset_index()
        y_values_historical = daily_sales_df_for_plot['total_sales'].tolist()

        model = train_sales_forecasting_model(daily_sales_df_indexed)
        
        forecast_df = None
        if model and not daily_sales_df_indexed.empty:
            last_historical_date = daily_sales_df_indexed.index.max()
            predicted_values, lower_bound, upper_bound = predict_future_sales(model, periods=forecast_periods)
            
            # Check if all prediction components are valid
            if predicted_values is not None and lower_bound is not None and upper_bound is not None:
                future_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1), 
                                             periods=forecast_periods, freq='D')
                
                # ---- START DEBUG PRINTS ----
                print(f"DEBUG: Length of future_dates: {len(future_dates)}")
                print(f"DEBUG: Type of predicted_values: {type(predicted_values)}, Length: {len(predicted_values) if hasattr(predicted_values, '__len__') else 'N/A'}")
                print(f"DEBUG: Type of lower_bound: {type(lower_bound)}, Length: {len(lower_bound) if hasattr(lower_bound, '__len__') else 'N/A'}")
                print(f"DEBUG: Type of upper_bound: {type(upper_bound)}, Length: {len(upper_bound) if hasattr(upper_bound, '__len__') else 'N/A'}")
                # ---- END DEBUG PRINTS ----

                # Defensive check for lengths before creating DataFrame
                if not (len(future_dates) == len(predicted_values) == len(lower_bound) == len(upper_bound)):
                    print("ERROR: Mismatch in lengths of arrays for forecast DataFrame!")
                    # Optionally, handle this case by not creating forecast_df or setting it to None
                    # For now, we'll let it proceed to hit the ValueError if there's a mismatch,
                    # but the debug prints will show us the problem.
                
                forecast_df = pd.DataFrame({
                    'date': future_dates, 
                    'forecast_sales': list(predicted_values),
                    'forecast_lower': list(lower_bound),
                    'forecast_upper': list(upper_bound)
                })
            else:
                print("DEBUG: One or more prediction components (values, lower, upper) is None.")
        else:
            if not model:
                print("DEBUG: Model training failed or model is None.")
            if daily_sales_df_indexed.empty:
                print("DEBUG: daily_sales_df_indexed is empty.")

        # ... (rest of the plotting logic) ...
        fig = go.Figure()
        # Historical Sales
        fig.add_trace(go.Scatter(
            x=daily_sales_df_for_plot['date'].tolist(),
            y=y_values_historical,
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='blue')
        ))

        if forecast_df is not None and not forecast_df.empty:
            # Upper bound of CI
            fig.add_trace(go.Scatter(
                x=forecast_df['date'].tolist(),
                y=forecast_df['forecast_upper'].tolist(),
                mode='lines',
                line=dict(width=0), 
                hoverinfo='skip', 
                showlegend=False,
                name='Upper CI'
            ))
            # Lower bound of CI - with fill to the upper bound
            fig.add_trace(go.Scatter(
                x=forecast_df['date'].tolist(),
                y=forecast_df['forecast_lower'].tolist(),
                mode='lines',
                line=dict(width=0), 
                fillcolor='rgba(255, 0, 0, 0.2)', 
                fill='tonexty', 
                hoverinfo='skip', 
                showlegend=False,
                name='Lower CI'
            ))
            # Forecasted Sales (Point Forecast) - plotted on top
            fig.add_trace(go.Scatter(
                x=forecast_df['date'].tolist(), 
                y=forecast_df['forecast_sales'].tolist(),
                mode='lines+markers',
                name='Forecasted Sales',
                line=dict(dash='dash', color='red')
            ))
        else:
            print("DEBUG: forecast_df is None or empty, not plotting forecast.")


        fig.update_layout(
            title_text='Daily Sales Volume with Forecast and 95% Confidence Interval',
            xaxis_title='Date',
            yaxis_title='Total Sales',
            legend_title_text='Legend'
        )
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    else:
        print("DEBUG: daily_sales_df_indexed is empty at the start of sales_overview.")
        fig = go.Figure() 
        fig.update_layout(title_text='No sales data available for display.')
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


    return render_template('sales_overview.html',
                           title='Sales Overview & Forecast',
                           graph_json=graph_json,
                           current_forecast_periods=forecast_periods)