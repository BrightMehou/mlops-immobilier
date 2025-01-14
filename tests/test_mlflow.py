import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from src.train import train_model, evalute_model, log_model
from sklearn.model_selection import train_test_split
import logging
from mlflow import MlflowClient
import os

@pytest.fixture
def data():
    # Charger les données California Housing
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def model():
    return RandomForestRegressor(random_state=42)

# @pytest.fixture
# def mlflow_client():
#     client = MlflowClient()
#     return client

# @pytest.fixture
# def test_experiment(mlflow_client):
#     # Crée une expérience de test dédiée
#     experiment_name = "test_experiment"
#     experiment = mlflow_client.get_experiment_by_name(experiment_name)
#     if None == experiment:
#          return mlflow_client.create_experiment(experiment_name)
#     return experiment.experiment_id     

# def test_train_model(data, model):
#     X_train, X_test, y_train, y_test = data
#     grid = {"n_estimators": [2], "max_depth": [2, 3]}

#     best_model, best_params, times = train_model(model, grid, X_train, y_train)

#     assert isinstance(best_model, RandomForestRegressor)
#     assert "n_estimators" in best_params
#     assert "max_depth" in best_params
#     assert "mean_fit_time" in times
#     assert "mean_score_time" in times

# def test_evalute_model(data, model):
#     X_train, X_test, y_train, y_test = data
#     model.fit(X_train, y_train)

#     results = evalute_model(model, X_test, y_test)

#     assert "mean_squared_error" in results
#     assert "score_r2" in results
#     assert isinstance(results["score_mean_squared_error"], float)
#     assert isinstance(results["score_r2"], float)



# def test_log_model(mlflow_client, model, test_experiment):
#     params = {"n_estimators": '10', "max_depth": '2'}
#     metrics = {"score_mean_squared_error": 0.5, "score_r2": 0.8}

#     # Appeler la fonction de journalisation
#     log_model("TestRun", model, params, metrics, experiment_name="test_experiment")
#     run = mlflow_client.search_runs(experiment_ids=[test_experiment])[0]
#     assert run.data.params == params
#     assert run.data.metrics == metrics
