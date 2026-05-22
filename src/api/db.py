"""Persistance SQLite pour le monitoring des requêtes de prédiction."""

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/monitoring.db")

_CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS prediction_requests (
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        MedInc REAL, HouseAge REAL, AveRooms REAL, AveBedrms REAL,
        Population REAL, AveOccup REAL, Latitude REAL, Longitude REAL,
        prediction REAL
    )
"""

_INSERT_SQL = """
    INSERT INTO prediction_requests
    (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, prediction)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class MonitoringDB:
    """Encapsule la connexion et les opérations sur la base de monitoring."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Ouvre la connexion persistante et initialise le schéma."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE_SQL)

    def close(self) -> None:
        """Ferme la connexion persistante."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @staticmethod
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

    def save_prediction_request(
        self, features: dict[str, float], prediction: float
    ) -> None:
        """Enregistre une requête de prédiction (connexion dédiée, sûr en tâche de fond)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(_INSERT_SQL, self._feature_values(features, prediction))
                conn.commit()
        except Exception as e:
            logger.error("Failed to save prediction request: %s", e)

    def get_all_predictions(self) -> list[dict[str, Any]]:
        """Retourne toutes les requêtes enregistrées, triées par date."""
        if self._conn is None:
            return []
        rows = self._conn.execute(
            "SELECT * FROM prediction_requests ORDER BY created_at"
        ).fetchall()
        return [dict(row) for row in rows]
