# Predictive Sales Dashboard

## 1. Overview

The Predictive Sales Dashboard is a web application designed to provide insightful business analytics and forecast future sales trends. It leverages machine learning models, primarily Random Forest, to predict sales based on historical data. Users can visualize sales data, model predictions, feature importances, and time series decompositions. The application also includes data management features for uploading new sales data and managing cached machine learning models.

## 2. Features

*   **Sales Forecasting:**
    *   Predicts future sales using a Random Forest Regressor model.
    *   Supports model training for global (all stores/items), store-specific, item-specific, or store-item specific cohorts.
    *   Displays historical sales, actual sales on test data, predicted sales on test data, and future forecasted sales.
*   **Data Visualization (Interactive Plots via Plotly):**
    *   Sales forecast charts.
    *   Feature importance plots for the Random Forest model.
    *   Time series decomposition plots (trend, seasonal, residual components).
*   **Data Management:**
    *   Upload sales data via CSV files ([`app/routes.py#L214`](app/routes.py#L214)).
    *   Import CSV data into an SQL database ([`app/ml_models.py#L142`](predictive_dashboard/app/ml_models.py#L142)).
*   **Model Management:**
    *   **Caching:** Efficiently caches trained models, associated data (train/test splits), performance metrics, and feature importances to disk ([`app/ml_models.py#L22`](/Users/omtailor/predictive_dashboard/app/ml_models.py#L22), [`app/ml_models.py#L435`](predictive_dashboard/app/ml_models.py#L435)).
        *   Cached artifacts are stored in the `instance/rf_models_cache/` directory.
        *   Filename convention: `rf_model_store_<store_id>_item_<item_id>_<artifact_type>.<extension>`
    *   **Retraining:**
        *   Manually trigger retraining of the global model ([`app/routes.py#L283`](predictive_dashboard/app/routes.py#L283)).
        *   Retrain specific models for store/item combinations ([`app/routes.py#L256`](predictive_dashboard/app/routes.py#L256)).
    *   **Cache Management:**
        *   List all cached models with details ([`app/ml_models.py#L660`](predictive_dashboard/app/ml_models.py#L660), displayed on [`data_management.html`](predictive_dashboard/app/templates/data_management.html#L40)).
        *   Delete specific cached models ([`app/routes.py#L235`](/Users/omtailor/predictive_dashboard/app/routes.py#L235)).
        *   Clear the entire Random Forest model cache ([`app/routes.py#L308`](/Users/omtailor/predictive_dashboard/app/routes.py#L308)).
*   **Filtering:**
    *   Filter sales overview and model predictions by store and item ([`app/templates/sales_overview.html#L30`](/Users/omtailor/predictive_dashboard/app/templates/sales_overview.html#L30)).
*   **User Interface:**
    *   Responsive web interface built with Bootstrap.
    *   Global loading indicator for navigation and data processing ([`app/templates/base.html#L49`](/Users/omtailor/predictive_dashboard/app/templates/base.html#L49)).

## 3. Tech Stack

*   **Backend:**
    *   Python 3
    *   Flask (Web Framework)
    *   Flask-SQLAlchemy (ORM)
    *   SQLAlchemy
*   **Machine Learning & Data Processing:**
    *   Scikit-learn (for Random Forest, metrics)
    *   Pandas (Data manipulation)
    *   NumPy (Numerical operations)
    *   Statsmodels (for seasonal decomposition)
    *   Pmdarima (for AutoARIMA, though Random Forest is the primary model in use)
    *   Joblib (for model serialization/deserialization)
*   **Frontend:**
    *   HTML5
    *   CSS3 (Bootstrap 4.5)
    *   JavaScript
    *   Plotly.js (Interactive charting)
    *   Jinja2 (Templating engine)
*   **Database:**
    *   SQLite (default, configurable via SQLAlchemy)
*   **Development Tools:**
    *   Virtualenv (recommended)

(See [`requirements.txt`](/Users/omtailor/predictive_dashboard/requirements.txt) for a detailed list of Python dependencies.)

## 4. Project Structure

```
predictive_dashboard/
├── app/                      # Main application package
│   ├── models/
│   │   └── data_models.py    # SQLAlchemy database models (e.g., SalesData)
│   ├── static/               # Static assets (CSS, JS, images) - (Standard Flask structure)
│   ├── templates/            # HTML templates (Jinja2)
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── sales_overview.html
│   │   └── data_management.html
│   ├── __init__.py           # Application factory
│   ├── extensions.py         # Flask extension initializations (e.g., db)
│   ├── ml_models.py          # Core ML logic, model training, prediction, caching
│   └── routes.py             # Flask routes and view functions
├── data/                     # Directory for data files (e.g., train.csv for seeding)
│   └── train.csv
├── instance/                 # Instance-specific data (e.g., SQLite DB, model cache)
│   └── rf_models_cache/      # Cached RF models and artifacts
├── migrations/               # Database migration scripts (if using Flask-Migrate)
├── tests/                    # Unit and integration tests (if implemented)
├── .env                      # Environment variables (example: .env.example)
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── run.py                    # Script to run the Flask application
└── seed_data.py              # Script to seed initial data into the database
```

## 5. Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd predictive_dashboard
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables (Optional but Recommended):**
    Create a `.env` file in the project root (or set system environment variables):
    ```env
    FLASK_APP=run.py
    FLASK_ENV=development  # Use 'production' for deployment
    # DATABASE_URL=sqlite:///instance/your_database_name.db # Example
    ```
    The application uses `instance_relative_config=True`, so the SQLite database will typically be created in an `instance` folder.

5.  **Initialize the Database and Seed Data:**
    The [`seed_data.py`](/Users/omtailor/predictive_dashboard/seed_data.py) script handles database table creation and initial data import from `data/train.csv`.
    ```bash
    python seed_data.py
    ```
    This will:
    *   Drop existing tables (if any).
    *   Create tables based on models defined in [`app/models/data_models.py`](/Users/omtailor/predictive_dashboard/app/models/data_models.py).
    *   Import sales data from `data/train.csv` using [`app.ml_models.import_sales_csv_to_db`](/Users/omtailor/predictive_dashboard/app/ml_models.py).

6.  **Run the Application:**
    ```bash
    flask run
    ```
    The application should be accessible at `http://127.0.0.1:5000/` by default.

## 6. Usage

*   **Homepage (`/`):** Provides a welcome message and a link to the Sales Overview.
*   **Sales Overview (`/sales_overview`):**
    *   The main dashboard for viewing sales forecasts.
    *   Use the **Store** and **Item** dropdown filters to select specific cohorts or view "all".
    *   The page displays:
        *   Sales forecast plot (historical, test actuals, test predictions, future forecast).
        *   Model performance metrics (MSE, MAE, R²) for the current selection.
        *   Feature importance plot for the Random Forest model.
        *   Time series decomposition plots (trend, seasonal, residual).
    *   Model training and data loading for the selected cohort happen on page load. Cached models are used if available and `force_retrain` is not active.
*   **Data Management (`/data_management`):**
    *   **Upload Sales Data:** Upload a CSV file with columns: `date`, `store`, `item`, `sales`. This data will be imported into the database.
    *   **Retrain Global Sales Model:** Trigger a retraining of the Random Forest model for all stores and all items. This is recommended after uploading significant new data.
    *   **Cached Model Status:** Lists all currently cached Random Forest models, including their store ID, item ID, last modification time, and filename.
        *   **Delete:** Delete a specific cached model and its associated artifacts.
        *   **Retrain:** Trigger retraining for a specific cached model's cohort.
    *   **Global Cache Actions:**
        *   **Clear All Cached RF Models:** Removes all files from the `instance/rf_models_cache` directory.

## 7. Key Components and Logic

### 7.1. Sales Forecasting (Random Forest)

*   **Core Training Function:** [`train_sales_forecasting_model_rf()`](/Users/omtailor/predictive_dashboard/app/ml_models.py#L495) in `app/ml_models.py`.
    *   Handles data loading (from DB or CSV), preprocessing, train-test splitting.
    *   Trains a `RandomForestRegressor`.
    *   Calculates metrics (MSE, MAE, R²).
    *   Extracts feature importances.
    *   Manages caching:
        *   Checks for existing cached models and artifacts (model, train/test dataframes, metrics, importances) using paths from `_get_rf_cache_paths()`.
        *   Loads from cache if `force_retrain` is `False` and all artifacts exist.
        *   Saves newly trained model and artifacts to cache.
*   **Prediction Function:** [`predict_future_sales_rf()`](/Users/omtailor/predictive_dashboard/app/ml_models.py#L603) in `app/ml_models.py`.
    *   Generates future dates and features.
    *   Uses the trained model to predict sales for `n_periods`.
*   **Data Source:** Primarily uses data from the `SalesData` table ([`app/models/data_models.py`](/Users/omtailor/predictive_dashboard/app/models/data_models.py)), fetched by `get_sales_data_from_db_for_rf()`.

### 7.2. Caching Mechanism

*   **Cache Directory:** `instance/rf_models_cache/` (defined by `MODEL_CACHE_DIR_NAME` in [`app/ml_models.py`](/Users/omtailor/predictive_dashboard/app/ml_models.py)).
*   **Cached Artifacts per Cohort (Store/Item):**
    *   `rf_model_store_<S>_item_<I>.joblib`: Serialized Scikit-learn model.
    *   `rf_model_store_<S>_item_<I>_train_df.pkl`: Pickled Pandas DataFrame for training plot data.
    *   `rf_model_store_<S>_item_<I>_test_df.pkl`: Pickled Pandas DataFrame for testing plot data (includes actuals and predictions).
    *   `rf_model_store_<S>_item_<I>_metrics.json`: JSON file with model performance metrics (MSE, MAE, R²).
    *   `rf_model_store_<S>_item_<I>_importances.json`: JSON file with feature importances.
*   **Cache Management Functions (in `app/ml_models.py`):**
    *   [`_get_rf_cache_paths()`](/Users/omtailor/predictive_dashboard/app/ml_models.py#L22): Generates file paths for a given cohort.
    *   [`list_cached_rf_models()`](/Users/omtailor/predictive_dashboard/app/ml_models.py#L660): Lists models in the cache.
    *   [`delete_cached_rf_model()`](/Users/omtailor/predictive_dashboard/app/ml_models.py#L729): Deletes files for a specific cohort.
    *   [`clear_all_cached_rf_models()`](/Users/omtailor/predictive_dashboard/app/ml_models.py#L774): Clears the entire cache directory.

### 7.3. Seasonal Decomposition

*   The [`perform_seasonal_decomposition()`](/Users/omtailor/predictive_dashboard/app/ml_models.py#L652) function in `app/ml_models.py` uses `statsmodels.tsa.seasonal.seasonal_decompose` to break down the sales time series into trend, seasonal, and residual components.
*   These components are visualized on the Sales Overview page ([`app/routes.py#L124`](/Users/omtailor/predictive_dashboard/app/routes.py#L124)).

### 7.4. Routing and Views

*   Routes are defined in [`app/routes.py`](/Users/omtailor/predictive_dashboard/app/routes.py) using a Flask Blueprint.
*   Key routes:
    *   `/sales_overview`: Fetches/trains model, gets predictions, prepares plot data, and renders [`sales_overview.html`](/Users/omtailor/predictive_dashboard/app/templates/sales_overview.html).
    *   `/data_management`: Handles CSV uploads and renders [`data_management.html`](/Users/omtailor/predictive_dashboard/app/templates/data_management.html) for model management tasks.
    *   POST routes for model actions: `/trigger_rf_retrain`, `/retrain_specific_model/...`, `/delete_cached_model/...`, `/clear_all_rf_cache`.

## 8. Potential Future Enhancements

*   **Advanced Models:** Integrate and compare other forecasting models (e.g., ARIMA, Prophet, LSTMs). `pmdarima` is present but not the primary focus.
*   **User Authentication & Authorization:** Secure the application and restrict access to certain features.
*   **Background Tasks:** Use Celery or a similar task queue for long-running processes like model training to prevent HTTP timeouts and improve responsiveness.
*   **Automated Model Retraining:** Implement a scheduler to automatically retrain models when new data is available or on a regular basis.
*   **Hyperparameter Optimization:** Add functionality for tuning model hyperparameters.
*   **Comprehensive Testing:** Expand unit and integration tests for better code quality and reliability.
*   **Configuration Management:** More robust configuration for different environments (development, testing, production).
*   **Enhanced Logging and Monitoring:** Implement more detailed logging and potentially a monitoring dashboard.
