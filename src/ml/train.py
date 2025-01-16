from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
from pandas import DataFrame
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import logging
from sklearn.ensemble import GradientBoostingRegressor


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_model(model: BaseEstimator, X_train: DataFrame, y_train: DataFrame) -> BaseEstimator:
    """
    Entraîne un modèle de machine learning avec les données fournies.

    Args:
        model (BaseEstimator): Modèle scikit-learn à entraîner.
        X_train (DataFrame): Données d'entraînement.
        y_train (DataFrame): Cibles d'entraînement.

    Returns:
        BaseEstimator: Modèle entraîné.
    """

    logger.info("Starting model training...")
    model.fit(X_train, y_train)
    logger.info("Model training completed.")
    return model


def evalute_model(model: BaseEstimator, X_test: DataFrame, y_test: DataFrame) -> dict:
    """
    Évalue un modèle de machine learning sur un ensemble de test.

    Args:
        model (BaseEstimator): Modèle scikit-learn à tester.
        X_test (DataFrame): Données de test.
        y_test (DataFrame): Cibles de test.

    Returns:
        dict: Dictionnaire contenant les métriques d'évaluation (MSE, MAE, R²).
    """
    logger.info("Starting model testing...")
    y_pred = model.predict(X_test)
    scores = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
    logger.info("Model testing completed.")
    return scores


def log_model(
    model: BaseEstimator,
    params: dict,
    metrics: dict,
    X_train: DataFrame,
    run_name: str = None,
    experiment_name: str = "Imo_production",
    model_name: str = "Production-model",
) -> None:
    """
    Enregistre un modèle et ses artefacts associés dans MLflow.

    Args:
        model (BaseEstimator): Modèle scikit-learn à enregistrer.
        params (dict): Paramètres du modèle.
        metrics (dict): Métriques d'évaluation du modèle.
        X_train (DataFrame): Données d'entraînement pour inférer la signature.
        run_name (str, optional): Nom de l'exécution. Defaults to None.
        experiment_name (str, optional): Nom de l'expérience MLflow. Defaults to "Imo_production".
        model_name (str, optional): Nom du modèle à enregistrer. Defaults to "Production-model".

    Returns:
        str: Identifiant de l'exécution dans MLflow.
    """
    logger.info(f"Logging model to MLflow with run name: {run_name}...")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            signature=signature,
            registered_model_name=model_name,
        )
    logger.info("Model logged to MLflow.")
    return run.info.run_id


def main() -> None:
    random_state = 42
    housing = fetch_california_housing(as_frame=True)

    X = housing.data
    y = housing.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    params = {"n_estimators": 150, "max_depth": 5, "random_state": random_state}

    model = GradientBoostingRegressor(**params)
    run_name = "Production-model"
    model_name = "Production-model"
    logger.info(
        f"Starting the entire training and testing pipeline with run name: {run_name}..."
    )
    model = train_model(model, X_train, y_train)
    scores = evalute_model(model, X_test, y_test)
    log_model(model, params, scores, X_train, run_name=run_name, model_name=model_name)
    logger.info("Pipeline completed.")


if __name__ == "__main__":
    main()
