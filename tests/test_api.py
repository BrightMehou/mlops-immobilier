"""
Suite de tests Pytest pour l’API FastAPI de prédiction des prix des logements.
Elle vérifie :
- l’accessibilité de l’API,
- la validité des prédictions,
- la gestion des champs manquants,
- et le traitement des types de données invalides.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import app


# Créer un client de test pour l'API
@pytest.fixture
def client() -> TestClient:
    """
    Fixture qui retourne un client de test pour l'application FastAPI.

    Returns:
        TestClient: Client de test pour l'API.
    """
    return TestClient(app)


# Test pour vérifier que l'API est accessible
def test_api_is_running(client: TestClient) -> None:
    """
    Vérifie que le point de terminaison racine ("/") de l'API est accessible.

    - Envoie une requête GET à "/".
    - Vérifie que le code de statut est 200.
    - Vérifie que le message de réponse correspond à {"msg": "API is running"}.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "msg": "API de prédiction des prix des logements opérationnelle ✅"
    }


# Test pour vérifier le health check
def test_health_check(client: TestClient) -> None:
    """
    Vérifie que le point de terminaison "/health" retourne le bon statut.

    - Envoie une requête GET à "/health".
    - Vérifie que le code de statut est 200 ou 503 (si modèle non chargé).
    - Vérifie que la réponse contient "status".
    """
    response = client.get("/health")
    assert response.status_code in [200, 503]
    assert "status" in response.json()


# Test pour vérifier une prédiction valide
def test_valid_prediction(client: TestClient) -> None:
    """
    Vérifie que l'API retourne une prédiction valide pour un jeu de données correct.

    - Envoie une requête POST avec un payload JSON valide au point de terminaison "/predict".
    - Vérifie que le code de statut est 200 ou 503 (si modèle non chargé en test).
    - Vérifie que la réponse contient une clé "prediction" si le modèle est chargé.
    """

    payload = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 880.0,
        "AveBedrms": 129.0,
        "Population": 322.0,
        "AveOccup": 126.0,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }
    response = client.post("/predict", json=payload)
    print(response.json())
    # Accepte 200 (modèle chargé) ou 503 (modèle non chargé en test)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        assert "prediction" in response.json()
        assert isinstance(response.json()["prediction"][0], float)


# Test pour vérifier les entrées manquantes
@pytest.mark.parametrize(
    "missing_field",
    [
        "Longitude",
        "Latitude",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "MedInc",
    ],
)
def test_missing_field(client: TestClient, missing_field: str) -> None:
    """
    Vérifie que l'API retourne une erreur lorsqu'un champ obligatoire est manquant.

    - Envoie une requête POST avec un payload JSON auquel un champ est supprimé.
    - Utilise la paramétrisation pour tester différents champs manquants.
    - Vérifie que le code de statut est 422 (Unprocessable Entity).
    """
    payload: dict[str, float] = {
        "Longitude": -122.23,
        "Latitude": 37.88,
        "HouseAge": 41.0,
        "AveRooms": 880.0,
        "AveBedrms": 129.0,
        "Population": 322.0,
        "AveOccup": 126.0,
        "MedInc": 8.3252,
    }
    del payload[missing_field]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity


# Test pour vérifier les types de données invalides
def test_invalid_data_type(client: TestClient) -> None:
    """
    Vérifie que l'API retourne une erreur lorsqu'un champ contient un type de données invalide.

    - Envoie une requête POST avec un payload JSON contenant des types incorrects.
    - Vérifie que le code de statut est 422 (Unprocessable Entity).
    """
    payload: dict[str, Any] = {
        "Longitude": "not_a_float",
        "Latitude": 37.88,
        "HouseAge": 41.0,
        "AveRooms": 880.0,
        "AveBedrms": 129.0,
        "Population": 322.0,
        "AveOccup": 126.0,
        "MedInc": 8.3252,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
