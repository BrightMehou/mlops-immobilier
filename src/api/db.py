"""Persistance SQLite simple pour le monitoring des requêtes de prédiction."""

import csv
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/monitoring/monitoring.db")
DEFAULT_PROD_CSV_PATH = Path("data/monitoring/prod_data.csv")

_CREATE_TABLE = """
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


def _connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def save_prediction_request(
    features: dict[str, float],
    prediction: float,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """Enregistre une requête de prédiction."""
    try:
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO prediction_requests
                (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    features["MedInc"],
                    features["HouseAge"],
                    features["AveRooms"],
                    features["AveBedrms"],
                    features["Population"],
                    features["AveOccup"],
                    features["Latitude"],
                    features["Longitude"],
                    prediction,
                ],
            )
            conn.commit()
    except Exception:
        logger.exception("Failed to save prediction request")


def get_all_predictions(db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Retourne toutes les requêtes enregistrées, triées par date."""
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM prediction_requests ORDER BY created_at",
            ).fetchall()
            return [dict(row) for row in rows]
    except Exception:
        logger.exception("Failed to load prediction requests")
        return []


def init_db_from_csv(
    csv_path: Path = DEFAULT_PROD_CSV_PATH,
    db_path: Path = DEFAULT_DB_PATH,
) -> None:
    """DROP + CREATE + chargement du CSV (idempotent)."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: list[list[float]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [
            [
                float(row["MedInc"]),
                float(row["HouseAge"]),
                float(row["AveRooms"]),
                float(row["AveBedrms"]),
                float(row["Population"]),
                float(row["AveOccup"]),
                float(row["Latitude"]),
                float(row["Longitude"]),
                float(row["prediction"]),
            ]
            for row in csv.DictReader(f)
        ]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS prediction_requests")
        conn.execute(_CREATE_TABLE)
        conn.executemany(
            """
            INSERT INTO prediction_requests
            (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    logger.info("Loaded %d rows from %s into %s", len(rows), csv_path, db_path)
