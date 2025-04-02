import logging
import sys
import os
from utils import get_current_datetime_cet
from data_describe import (
    load_data,
    summarize_data,
    clean_data,
    feature_target_split,
    correlation_matrix,
)
from plots import (
    plot_histograms,
    plot_correlations,
    plot_timeseries,
)
from models import (
    train_test_data_split,
    run_svr,
    run_random_forest,
    run_gradient_boosting,
    run_ffnn,
)
DEBUG_FOLDER = "debug"
LOGS_FOLDER = os.path.join(DEBUG_FOLDER, "logs")
PLOTS_DIR = "plots"

os.makedirs(LOGS_FOLDER, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------
# DEFINE ALL HYPERPARAMETER GRIDS
# ---------------------------------
param_grid_svr = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 10]
}

param_grid_rf = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10]
}

param_grid_gb = {
    "n_estimators": [50, 100],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5]
}

param_grid_ffnn = {
    "hidden_layer_sizes": [(50,), (100,)],
    "alpha": [0.0001, 0.001],
    "learning_rate_init": [0.001, 0.01]
}

def get_log_filename():
    """Generates a time-stamped log filename."""
    return f"{LOGS_FOLDER}/logs_{get_current_datetime_cet()}.log"

def setup_logging(log_filename=None):
    """
    Set up logging to both console and a specified file, in real time.
    If log_filename is None, generate a default time-stamped filename.
    """
    if log_filename is None:
        log_filename = get_log_filename()

    # 1) Create the main logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 2) Create Formatter for consistent logs
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 3) Create and add a StreamHandler (for console output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 4) Create and add a FileHandler (for file output)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def main():
    # 1. Set up logging
    logger = setup_logging()  # Creates logs in debug/logs/<timestamp>.log

    logger.info("=== Starting main function ===")

    # 2. Specify CSV path
    csv_file = "BC-Data-Set.csv"  # Adjust path as needed
    logger.info("CSV file path set to '%s'", csv_file)

    # 3. Load data
    df = load_data(csv_file)

    # 4. Summarize data
    summarize_data(df)

    # 5. Clean data
    df = clean_data(df)

    # 6. Correlation analysis
    corr = correlation_matrix(df)
    plot_correlations(corr, output_dir=PLOTS_DIR)

    # 7. Plot histograms for selected columns
    plot_histograms(df, columns=["BC", "PM-2.5", "NO", "TEMP"], output_dir=PLOTS_DIR)

    # 8. Plot time series for selected columns
    plot_timeseries(df, columns=["BC", "PM-2.5", "NO"], output_dir=PLOTS_DIR)

    # 9. Feature/target split
    X, y = feature_target_split(df, target_column="BC")

    # 10. Train/test split
    X_train, X_test, y_train, y_test = train_test_data_split(X, y, test_size=0.2, shuffle=True)

    # 11. Model training/evaluation
    svr_model, svr_metrics = run_svr(X_train, y_train, X_test, y_test, param_grid_svr)
    logger.info("SVR metrics: %s", svr_metrics)

    rf_model, rf_metrics = run_random_forest(X_train, y_train, X_test, y_test, param_grid_rf)
    logger.info("Random Forest metrics: %s", rf_metrics)

    gb_model, gb_metrics = run_gradient_boosting(X_train, y_train, X_test, y_test, param_grid_gb)
    logger.info("Gradient Boosting metrics: %s", gb_metrics)

    ffnn_model, ffnn_metrics = run_ffnn(X_train, y_train, X_test, y_test, param_grid_ffnn)
    logger.info("FFNN metrics: %s", ffnn_metrics)

    logger.info("=== Finished main function ===")


if __name__ == "__main__":
    main()
