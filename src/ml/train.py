"""Script d'entraînement pour un modèle de prédiction des prix des logements en Californie 🏡

Ce script utilise un GradientBoostingRegressor pour prédire les prix à partir du dataset California Housing.
Le modèle est logué avec MLflow et évalué avec génération d’un explainer pour l’interprétation des prédictions.
"""

import logging
from typing import Any

import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration MLflow
RUN_NAME: str = "Production-model"
MODEL_NAME: str = "Production-model"
EXPLAINE_NAME: str = "explainer"

RANDOM_STATE: int = 42

housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

# Paramètres du modèle
MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 150,
    "max_depth": 5,
    "learning_rate": 0.15,
    "random_state": RANDOM_STATE,
}


def train() -> None:
    """Entraîne et logue un modèle de régression avec MLflow.

    Args:
        random_state (int): Graine aléatoire pour la reproductibilité.

    """
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.sklearn.autolog(registered_model_name=MODEL_NAME)
        model = GradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(X_train, y_train)
        logger.info("Entraînement du modèle terminé.")

        model_uri: str = f"runs:/{mlflow.active_run().info.run_id}/model"

        eval_data = X_test.copy()
        eval_data["target"] = y_test

        result = mlflow.models.evaluate(
            model=model_uri,
            data=eval_data,
            targets="target",
            model_type="regressor",
            evaluators="default",
            evaluator_config={
                "log_explainer": True,
                "explainer_artifact_path": EXPLAINE_NAME,
                "explainer_type": "permutation",
            },
        )
        logger.info("Évaluation terminée. Artifacts : %s", result.artifacts)
        logger.info("Run ID : %s", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    """
    Point d’entrée du script d’entraînement.
    """
    logger.info("Démarrage du script d'entraînement...")
    train()
    logger.info("Script terminé.")
