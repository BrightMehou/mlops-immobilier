from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.datasets import fetch_california_housing
from pandas import DataFrame
import pandas as pd
import mlflow
import mlflow.sklearn
import logging
import os
from datetime import datetime


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

random_state = 42
housing = fetch_california_housing(as_frame=True)
# Prepare the data
X = housing.data
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

def train_model(model: BaseEstimator, params_grid: dict, X_train: DataFrame, y_train: DataFrame):
    logger.info("Starting model training...")
    # Define the search
    search = GridSearchCV(model, params_grid, scoring='neg_mean_squared_error', cv=3)
    
    # Fit the search
    search.fit(X_train, y_train)
    best_index = search.best_index_
    mean_fit_time = search.cv_results_['mean_fit_time'][best_index]
    mean_score_time = search.cv_results_['mean_score_time'][best_index]
    time = {
        "mean_fit_time": mean_fit_time,
        "mean_score_time": mean_score_time,
    }
    best_estimator = search.best_estimator_
    best_estimator.fit(X_train, y_train)
    logger.info("Model training completed.")
    return best_estimator, search.best_params_, time

def evalute_model(model: BaseEstimator, X_test, y_test):
    logger.info("Starting model testing...")
    y_pred = model.predict(X_test)
    res = {
        "score_mean_squared_error": mean_squared_error(y_test, y_pred),
        "score_r2": r2_score(y_test, y_pred),
    }
    logger.info("Model testing completed.")
    return res

def log_model(run_name, model, params, metrics, experiment_name="imo"):
    logger.info(f"Logging model to MLflow with run name: {run_name}...")
    #rel_path = "///data/mlruns.db"
    #mlflow.set_tracking_uri(f"sqlite:{rel_path}")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(sk_model=model,
                                 artifact_path="sklearn-model",
                                 #signature=signature,
                                 registered_model_name="sk-learn-random-forest-reg-model")
    logger.info("Model logged to MLflow.")

# Define the grid
grid = {
        'n_estimators': [12],
        'max_depth': [2, 3],
        }

model = RandomForestRegressor(random_state=random_state)
run_name = "RandomForestRegressor"
logger.info(f"Starting the entire training and testing pipeline with run name: {run_name}...")
best_model, params, time = train_model(model, grid, X_train, y_train)
logger.info(f"Best model: {best_model}")
logger.info(f"Best parameters: {params}")
scores = evalute_model(best_model, X_test, y_test)
metrics = dict(**scores,**time)
log_model(run_name, best_model, params, metrics)
logger.info("Pipeline completed.")
