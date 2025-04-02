import logging
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

logger = logging.getLogger(__name__)

def train_test_data_split(X, y, test_size=0.2, shuffle=True, random_state=42):
    logger.info("Splitting data into train/test with test_size=%.2f, shuffle=%s", test_size, shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
    )
    logger.info("Train set shape=%s, Test set shape=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test

def run_svr(X_train, y_train, X_test, y_test, param_grid):
    """
    Trains a Support Vector Regressor using GridSearchCV with a provided param_grid.
    """
    logger.info("Training Support Vector Regressor (SVR) with GridSearchCV.")
    model = SVR()
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info("Best SVR params found: %s", grid_search.best_params_)

    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info("SVR evaluation => R2=%.3f, RMSE=%.3f", r2, rmse)

    metrics = {
        "r2": r2,
        "rmse": rmse,
        "best_params": grid_search.best_params_
    }
    return best_model, metrics

def run_random_forest(X_train, y_train, X_test, y_test, param_grid):
    """
    Trains a Random Forest Regressor using GridSearchCV with a provided param_grid.
    """
    logger.info("Training Random Forest Regressor with GridSearchCV.")
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info("Best Random Forest params found: %s", grid_search.best_params_)

    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info("Random Forest evaluation => R2=%.3f, RMSE=%.3f", r2, rmse)

    metrics = {
        "r2": r2,
        "rmse": rmse,
        "best_params": grid_search.best_params_
    }
    return best_model, metrics

def run_gradient_boosting(X_train, y_train, X_test, y_test, param_grid):
    """
    Trains a Gradient Boosting Regressor using GridSearchCV with a provided param_grid.
    """
    logger.info("Training Gradient Boosting Regressor with GridSearchCV.")
    model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info("Best Gradient Boosting params: %s", grid_search.best_params_)

    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info("Gradient Boosting evaluation => R2=%.3f, RMSE=%.3f", r2, rmse)

    metrics = {
        "r2": r2,
        "rmse": rmse,
        "best_params": grid_search.best_params_
    }
    return best_model, metrics

def run_ffnn(X_train, y_train, X_test, y_test, param_grid):
    """
    Trains a Feed-Forward Neural Network (MLPRegressor) using GridSearchCV with a provided param_grid.
    """
    logger.info("Training Feed-Forward Neural Network (MLPRegressor) with GridSearchCV.")
    model = MLPRegressor(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring="neg_mean_squared_error")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logger.info("Best FFNN params: %s", grid_search.best_params_)

    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    logger.info("FFNN evaluation => R2=%.3f, RMSE=%.3f", r2, rmse)

    metrics = {
        "r2": r2,
        "rmse": rmse,
        "best_params": grid_search.best_params_
    }
    return best_model, metrics
