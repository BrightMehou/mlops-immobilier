"""
API FastAPI pour déployer un modèle MLflow de prédiction des prix des logements en Californie.
Elle charge le modèle et son explainer SHAP, puis expose des endpoints pour effectuer des prédictions
et interpréter les contributions des variables.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import mlflow
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.api.db import get_all_predictions, init_db_from_csv, save_prediction_request

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_latest_run_id(model_name: str = "Production-model") -> str:
    """
    Retourne le run_id de la version la plus récente du modèle MLflow spécifié.

    Args:
        model_name (str): Nom du modèle MLflow.

    Returns:
        str: Identifiant de l'exécution (run_id).
    """
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        latest_version = max(versions, key=lambda v: int(v.version))
        return latest_version.run_id
    except Exception as e:
        logger.error(
            f"Error occurred while fetching latest run ID for model '{model_name}': {e}"
        )
        raise


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Charge la base monitoring, le modèle et l’explainer au démarrage."""

    init_db_from_csv()
    logger.info("Base monitoring initialisée")

    RUN_ID = get_latest_run_id("Production-model")
    MODEL_URI = f"runs:/{RUN_ID}/model"
    EXPLAINER_URI = f"runs:/{RUN_ID}/explainer"

    app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
    app.state.explainer = mlflow.pyfunc.load_model(EXPLAINER_URI)

    yield

    del app.state.model
    del app.state.explainer
    logger.info("🔄 Modèle et explainer déchargés")


app = FastAPI(
    title="Prédiction des prix des logements en Californie",
    description="API simple pour prédire les prix des logements en Californie avec SHAP values",
    version="0.5.0",
    lifespan=lifespan,
)


class InputFeatures(BaseModel):
    MedInc: float = Field(ge=0)
    HouseAge: float = Field(ge=0)
    AveRooms: float = Field(ge=0)
    AveBedrms: float = Field(ge=0)
    Population: float = Field(ge=1)
    AveOccup: float = Field(ge=0)
    Latitude: float = Field(ge=31, le=43)
    Longitude: float = Field(ge=-125, le=-113)


@app.get("/")
async def root() -> dict[str, str]:
    return {"msg": "API de prédiction des prix des logements opérationnelle ✅"}


@app.get("/health")
async def health_check(request: Request) -> JSONResponse:
    """
    Vérifie l'état de santé de l'API et du modèle.

    Retourne :
    -------
    JSONResponse
        Un dictionnaire contenant :
        - "status" : état de l'API ("healthy" ou "unhealthy")
    """
    model = getattr(request.app.state, "model", None)
    explainer = getattr(request.app.state, "explainer", None)
    model_loaded = model is not None and explainer is not None
    status = "healthy" if model_loaded else "unhealthy"

    return JSONResponse(
        status_code=200 if model_loaded else 503,
        content={
            "status": status,
        },
    )


@app.get("/predictions")
def get_predictions() -> list[dict[str, Any]]:
    """Retourne toutes les requêtes enregistrées en base."""
    return get_all_predictions()


@app.post("/predict")
def predict(
    request: Request,
    input_data: InputFeatures,
    background_tasks: BackgroundTasks,
) -> dict[str, list[Any]]:
    """
    Prédit le prix d'un logement en Californie à partir de ses caractéristiques.

    Cette fonction reçoit les données d'entrée sous forme d'un objet `InputFeatures`,
    les transforme en DataFrame compatible avec le modèle MLflow, puis retourne :
    - La prédiction du prix du logement
    - Les valeurs SHAP associées pour interpréter la contribution de chaque feature

    Paramètres :
    ----------
    input_data : InputFeatures
        Données d'entrée contenant les caractéristiques du logement :
        - MedInc : revenu médian
        - HouseAge : âge moyen des habitations
        - AveRooms : nombre moyen de pièces
        - AveBedrms : nombre moyen de chambres
        - Population : population du quartier
        - AveOccup : taux d'occupation moyen
        - Latitude : latitude géographique
        - Longitude : longitude géographique

    Retour :
    -------
    dict
        Un dictionnaire contenant :
        - "prediction" : liste avec le prix prédit
        - "shap_values" : liste des valeurs SHAP pour chaque feature
    """
    model = getattr(request.app.state, "model", None)
    explainer = getattr(request.app.state, "explainer", None)

    if model is None or explainer is None:
        logger.warning("Model or explainer not loaded")
        return JSONResponse(
            status_code=503,
            content={"detail": "Model not loaded"},
        )

    df = pd.DataFrame([input_data.model_dump()])

    prediction = model.predict(df)
    shap_values = explainer.predict(df)
    background_tasks.add_task(
        save_prediction_request,
        input_data.model_dump(),
        float(prediction[0]),
    )

    return {
        "prediction": prediction.tolist(),
        "shap_values": shap_values.tolist(),
    }
