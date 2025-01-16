from fastapi.testclient import TestClient
import pytest
from src.api.app import app  # Remplacez 'main' par le nom de votre fichier contenant l'API

# Créer un client de test pour l'API
client = TestClient(app)

# Test pour vérifier que l'API est accessible
def test_api_is_running():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}
    
# Test pour vérifier une prédiction valide
def test_valid_prediction():
    payload = {
        "medinc": 8.3252,
        "houseage": 41.0,
        "averooms": 880.0,
        "avebedrms": 129.0,
        "population": 322.0,
        "aveoccup": 126.0,
        "latitude": 37.88,
        "longitude": -122.23
    }
    response = client.post("/predict", json=payload)
    print(response.json())
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)

# Test pour vérifier une prédiction avec des valeurs limites
# def test_edge_case_prediction():
#     payload = {
#         "longitude": -180.0,
#         "latitude": -90.0,
#         "housing_median_age": 0.0,
#         "total_rooms": 0.0,
#         "total_bedrooms": 0.0,
#         "population": 0.0,
#         "households": 0.0,
#         "median_income": 0.0
#     }
#     response = client.post("/predict", json=payload)
#     assert response.status_code == 200
#     assert "prediction" in response.json()
#     assert response.json()["prediction"] == 0.0

# Test pour vérifier les entrées manquantes
@pytest.mark.parametrize("missing_field", [
    "longitude", "latitude", "housing_median_age", 
    "total_rooms", "total_bedrooms", "population", 
    "households", "median_income"
])
def test_missing_field(missing_field):
    payload = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252
    }
    del payload[missing_field]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

# Test pour vérifier les types de données invalides
def test_invalid_data_type():
    payload = {
        "longitude": "not_a_float",
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

# Test pour vérifier un tableau vide (aucune donnée envoyée)
def test_empty_payload():
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Unprocessable Entity
