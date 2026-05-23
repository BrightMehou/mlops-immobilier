"""
Script de détection de dérive de données.
Compare un jeu de référence à un jeu courant et retourne un rapport JSON.
"""

import logging
from typing import Any

from evidently import Dataset, Report
from evidently.presets import DataDriftPreset
from pandas import DataFrame
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def reference_data() -> DataFrame:
    """Données d'entraînement California Housing (même split que train.py)."""
    housing = fetch_california_housing(as_frame=True)
    X_train, _, _, _ = train_test_split(
        housing.data, housing.target, test_size=0.2, random_state=42
    )
    return X_train


def detect_drift(reference_data: DataFrame, current_data: DataFrame) -> dict[str, Any]:
    """
    Détecte la dérive entre deux jeux de données et retourne le rapport JSON.

    Args:
        reference_data: Données de référence (ex. entraînement).
        current_data: Données actuelles (ex. production en base).
    """
    reference = Dataset.from_pandas(reference_data)
    current = Dataset.from_pandas(current_data)
    report = Report([DataDriftPreset()], include_tests=True)

    logger.info("Début de la détection de dérive...")
    snapshot = report.run(current, reference)
    result = snapshot.dict()
    logger.info("Détection de dérive terminée.")

    return result
