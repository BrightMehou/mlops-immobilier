from fastapi.testclient import TestClient
import pytest
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
    - Vérifie que le message de réponse correspond à {"msg": "Hello World"}.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}


# Test pour vérifier une prédiction valide
def test_valid_prediction(client: TestClient) -> None:
    """
    Vérifie que l'API retourne une prédiction valide pour un jeu de données correct.

    - Envoie une requête POST avec un payload JSON valide au point de terminaison "/predict".
    - Vérifie que le code de statut est 200.
    - Vérifie que la réponse contient une clé "prediction".
    - Vérifie que la prédiction est de type float.
    """

    payload = {
        "medinc": 8.3252,
        "houseage": 41.0,
        "averooms": 880.0,
        "avebedrms": 129.0,
        "population": 322.0,
        "aveoccup": 126.0,
        "latitude": 37.88,
        "longitude": -122.23,
    }
    response = client.post("/predict", json=payload)
    print(response.json())
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)


# Test pour vérifier les entrées manquantes
@pytest.mark.parametrize(
    "missing_field",
    [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ],
)
def test_missing_field(client: TestClient, missing_field: str) -> None:
    """
    Vérifie que l'API retourne une erreur lorsqu'un champ obligatoire est manquant.

    - Envoie une requête POST avec un payload JSON auquel un champ est supprimé.
    - Utilise la paramétrisation pour tester différents champs manquants.
    - Vérifie que le code de statut est 422 (Unprocessable Entity).
    """
    payload = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
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
    payload = {
        "longitude": "not_a_float",
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
