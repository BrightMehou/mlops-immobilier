"""Persistance SQLite simple pour le monitoring des requêtes de prédiction."""

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/monitoring.db")

_CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS prediction_requests (
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        MedInc REAL,
        HouseAge REAL,
        AveRooms REAL,
        AveBedrms REAL,
        Population REAL,
        AveOccup REAL,
        Latitude REAL,
        Longitude REAL,
        prediction REAL
    )
"""

_INSERT_SQL = """
    INSERT INTO prediction_requests
    (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, prediction)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _feature_values(features: dict[str, float], prediction: float) -> list[float]:
    return [
        features["MedInc"],
        features["HouseAge"],
        features["AveRooms"],
        features["AveBedrms"],
        features["Population"],
        features["AveOccup"],
        features["Latitude"],
        features["Longitude"],
        prediction,
    ]


def _connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE_SQL)
    return conn


def save_prediction_request(
    features: dict[str, float],
    prediction: float,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Enregistre une requête de prédiction."""
    try:
        with _connect(db_path) as conn:
            conn.execute(_INSERT_SQL, _feature_values(features, prediction))
            conn.commit()
    except Exception as e:
        logger.error("Failed to save prediction request: %s", e)


def get_all_predictions(db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Retourne toutes les requêtes enregistrées, triées par date."""
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM prediction_requests ORDER BY created_at"
            ).fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error("Failed to load prediction requests: %s", e)
        return []
