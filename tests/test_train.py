import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from src.ml.train import train_model, evalute_model, log_model
from sklearn.model_selection import train_test_split
from mlflow import MlflowClient
from pandas import DataFrame
from sklearn.base import BaseEstimator


@pytest.fixture
def data() -> tuple[DataFrame]:
    """
    Fixture pour charger les données California Housing et effectuer un split train/test.

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    housing = fetch_california_housing(as_frame=True)
    X = housing.data
    y = housing.target
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def model() -> BaseEstimator:
    """
    Fixture pour initialiser un modèle RandomForestRegressor simple.

    Returns:
        RandomForestRegressor: Modèle avec des hyperparamètres minimaux.
    """
    return RandomForestRegressor(n_estimators=2, max_depth=1)


@pytest.fixture
def mlflow_client() -> MlflowClient:
    """
    Fixture pour initialiser un client Mlflow.

    Returns:
        MlflowClient: Instance du client Mlflow.
    """
    client = MlflowClient()
    return client


@pytest.fixture
def test_experiment(mlflow_client: MlflowClient) -> None:
    """
    Fixture pour créer une expérience MLflow de test.

    Args:
        mlflow_client (MlflowClient): Client MLflow.

    Returns:
        None
    """
    experiment_name = "test_experiment"
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if None == experiment:
        mlflow_client.create_experiment(experiment_name)


def test_train_model(data: tuple[DataFrame], model: BaseEstimator) -> None:
    """
    Teste si la fonction `train_model` entraîne correctement le modèle.

    Args:
        data (tuple): Données d'entraînement et de test (X_train, X_test, y_train, y_test).
        model (RandomForestRegressor): Modèle RandomForest à entraîner.

    Asserts:
        - Le modèle retourné est une instance de RandomForestRegressor.
    """
    X_train, X_test, y_train, y_test = data

    trained_model = train_model(model, X_train, y_train)

    assert isinstance(trained_model, RandomForestRegressor)


def test_evalute_model(data: tuple[DataFrame], model: BaseEstimator) -> None:
    """
    Teste si la fonction `evalute_model` calcule correctement les métriques d'évaluation.

    Args:
        data (tuple): Données d'entraînement et de test (X_train, X_test, y_train, y_test).
        model (RandomForestRegressor): Modèle RandomForest à évaluer.

    Asserts:
        - Les métriques retournées contiennent les clés "mean_squared_error", "mean_absolute_error", "r2".
        - Les valeurs des métriques sont de type float.
    """
    X_train, X_test, y_train, y_test = data
    model.fit(X_train, y_train)

    scores = evalute_model(model, X_test, y_test)

    assert "mean_squared_error" in scores
    assert "mean_absolute_error" in scores
    assert "r2" in scores
    assert isinstance(scores["mean_squared_error"], float)
    assert isinstance(scores["mean_absolute_error"], float)
    assert isinstance(scores["r2"], float)


def test_log_model(
    mlflow_client: MlflowClient, model: BaseEstimator, data: tuple[DataFrame]
) -> None:
    """
    Teste si la fonction `log_model` enregistre correctement un modèle dans MLflow.

    Args:
        mlflow_client (MlflowClient): Client MLflow pour récupérer les informations de run.
        model (RandomForestRegressor): Modèle RandomForest à enregistrer.
        data (tuple): Données d'entraînement et de test (X_train, X_test, y_train, y_test).

    Asserts:
        - Les paramètres et métriques sont enregistrés correctement dans MLflow.
    """
    X_train, X_test, y_train, y_test = data
    model.fit(X_train, y_train)
    params = {"n_estimators": "2", "max_depth": "1"}
    metrics = {"mean_squared_error": 0.5, "mean_absolute_error": 122, "r2": 0.8}
    run_id = log_model(
        model,
        params,
        metrics,
        X_train,
        experiment_name="test_experiment",
        model_name="test_model",
    )
    run = mlflow_client.get_run(run_id)
    assert run.data.params == params
    assert run.data.metrics == metrics
