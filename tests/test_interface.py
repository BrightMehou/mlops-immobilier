from streamlit.testing.v1 import AppTest
from src.interface.interface import model_prediction
import pytest
import requests

# Classe personnalisée pour être la valeur de retour mock
# Remplacera requests.Response retourné par requests.post
class MockResponseSuccess:
    # Ajout de l'attribut status_code pour succès
    status_code = 200

    # La méthode json() mock retourne toujours un dictionnaire de test spécifique
    @staticmethod
    def json():
        return {"prediction": 350000}

class MockResponseError:
    # Ajout de l'attribut status_code pour erreur
    status_code = 500

    # La méthode json() mock retourne un dictionnaire vide
    @staticmethod
    def json():
        return {}

def test_model_prediction_success(monkeypatch):
    def mock_post(*args, **kwargs):
        return MockResponseSuccess()

    monkeypatch.setattr(requests, "post", mock_post)
    
    input_data = { "medinc": 5.0,
                   "houseage": 15.0,
                   "averooms": 6.0,
                   "avebedrms": 1.0,
                   "population": 800.0,
                   "aveoccup": 3.0,
                   "latitude": 37.0,
                   "longitude": -122.0
                   } 
    result = model_prediction(input=input_data)
    assert result == "The predicted housing price is: 350000 dollars."

def test_model_prediction_exception(monkeypatch):
    def mock_post(*args, **kwargs):
        raise requests.exceptions.RequestException
    
    monkeypatch.setattr(requests, "post", mock_post)
    
    input_data = { "medinc": 5.0,
                   "houseage": 15.0,
                   "averooms": 6.0,
                   "avebedrms": 1.0,
                   "population": 800.0,
                   "aveoccup": 3.0,
                   "latitude": 37.0,
                   "longitude": -122.0
                   } 
    result = model_prediction(input=input_data)
    assert result == "Error: the model could not make a prediction."

def test_model_prediction_http_error(monkeypatch):
    def mock_post(*args, **kwargs):
        return MockResponseError()

    monkeypatch.setattr(requests, "post", mock_post)
    
    input_data = { "medinc": 5.0,
                   "houseage": 15.0,
                   "averooms": 6.0,
                   "avebedrms": 1.0,
                   "population": 800.0,
                   "aveoccup": 3.0,
                   "latitude": 37.0,
                   "longitude": -122.0
                   } 
    result = model_prediction(input=input_data)
    assert result == "Error: the model could not make a prediction."

@pytest.fixture
def session():
    at = AppTest.from_file("src/interface/interface.py")  # Nom de votre fichier Streamlit
    at.run(timeout=10)
    return at


def test_initial_state(session):

    # Vérifier l'étsession initial
    assert len(session.text_input) == 8  # 8 champs de texte pour les fonctionnalités
    assert len(session.button) == 1  # Bouton "Predict"
    assert session.text_input[0].label == "Median income"
    assert session.text_input[1].label == "House age"
    assert session.text_input[2].label == "Average number of rooms per household"
    assert session.text_input[3].label == "Average number of bedrooms per household"
    assert session.text_input[4].label == "Population"
    assert session.text_input[5].label == "Average number of household members"
    assert session.text_input[6].label == "Latitude"
    assert session.text_input[7].label == "Longitude"
    assert session.button[0].label == "Predict"
    assert len(session.markdown) == 2  # 2 éléments Markdown

    #assert not at.session_state.get("prediction", None)  # Pas de prédiction initialement

def test_valid_input(session,monkeypatch):
    def mock_post(*args, **kwargs):
        return MockResponseSuccess()

    monkeypatch.setattr(requests, "post", mock_post)
    # Fournir des entrées utilissessioneur valides
    session.text_input[0].input("-122.23")  
    session.text_input[1].input("37.88")  
    session.text_input[2].input("29.0")  
    session.text_input[3].input("700.0")  
    session.text_input[4].input("100.0")  
    session.text_input[5].input("500.0")  
    session.text_input[6].input("200.0")  
    session.text_input[7].input("5.0")  
    session.button[0].click().run()  # Cliquer sur "Predict"
    
    assert session.markdown[2].value == "The predicted housing price is: 350000 dollars."
  

def test_invalid_input(session):
    # Fournir des entrées invalides
    session.text_input[0].input("abc")  
    session.text_input[1].input("37.88")  
    session.text_input[2].input("29.0")  
    session.text_input[3].input("700.0")  
    session.text_input[4].input("100.0")  
    session.text_input[5].input("500.0")  
    session.text_input[6].input("200.0")  
    session.text_input[7].input("5.0")  
    session.button[0].click().run()  # Cliquer sur "Predict"
    
    assert session.markdown[2].value == "Please enter valid numbers in all fields."  # Message d'erreur affiché
