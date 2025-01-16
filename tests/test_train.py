import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from src.ml.train import train_model, evalute_model, log_model
from sklearn.model_selection import train_test_split
from mlflow import MlflowClient
import mlflow.pyfunc
from sklearn.metrics import mean_squared_error

@pytest.fixture
def data():
    # Charger les données California Housing
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def model():
    return RandomForestRegressor(n_estimators=2,max_depth=1)

@pytest.fixture
def mlflow_client():
    client = MlflowClient()
    return client

@pytest.fixture
def test_experiment(mlflow_client):
    # Crée une expérience de test dédiée
    experiment_name = "test_experiment"
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if None == experiment:
         mlflow_client.create_experiment(experiment_name) 

def test_train_model(data, model):
    X_train, X_test, y_train, y_test = data

    trained_model = train_model(model, X_train, y_train)

    assert isinstance(trained_model, RandomForestRegressor)
    

def test_evalute_model(data, model):
    X_train, X_test, y_train, y_test = data
    model.fit(X_train, y_train)

    scores = evalute_model(model, X_test, y_test)

    assert "mean_squared_error" in scores
    assert "mean_absolute_error" in scores
    assert "r2" in scores
    assert isinstance(scores["mean_squared_error"], float)
    assert isinstance(scores["mean_absolute_error"], float)
    assert isinstance(scores["r2"], float)



def test_log_model(mlflow_client, model,data):
    X_train, X_test, y_train, y_test = data
    model.fit(X_train, y_train)
    params = {"n_estimators": '2', "max_depth": '1'}
    metrics = {"mean_squared_error": 0.5,"mean_absolute_error": 122, "r2": 0.8}
    # Appeler la fonction de journalisation
    run_id = log_model(model, params, metrics, X_train, experiment_name="test_experiment", model_name="test_model")
    run = mlflow_client.get_run(run_id)
    assert run.data.params == params
    assert run.data.metrics == metrics
    




