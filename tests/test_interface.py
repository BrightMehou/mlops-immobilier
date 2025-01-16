from streamlit.testing.v1 import AppTest
from src.interface.interface import model_prediction
import pytest
import requests


# Classe personnalisée pour être la valeur de retour mock
class MockResponseSuccess:
    """
    Représente une réponse simulée réussie pour le modèle.
    """

    status_code = 200

    @staticmethod
    def json():
        """
        Retourne une réponse simulée avec une prédiction.
        """
        return {"prediction": 3}


class MockResponseError:
    """
    Représente une réponse simulée avec une erreur HTTP.
    """

    status_code = 500

    @staticmethod
    def json():
        """
        Retourne un dictionnaire vide.
        """
        return {}


def test_model_prediction_success(monkeypatch) -> None:
    """
    Teste si `model_prediction` retourne correctement une prédiction
    lorsque le modèle répond avec succès.
    """

    def mock_post(*args, **kwargs) -> MockResponseSuccess:
        return MockResponseSuccess()

    monkeypatch.setattr(requests, "post", mock_post)

    input_data = {
        "medinc": 5.0,
        "houseage": 15.0,
        "averooms": 6.0,
        "avebedrms": 1.0,
        "population": 800.0,
        "aveoccup": 3.0,
        "latitude": 37.0,
        "longitude": -122.0,
    }
    result = model_prediction(input=input_data)
    assert result == "Le prix prédit pour le logement est : 300000 dollars."


def test_model_prediction_exception(monkeypatch) -> None:
    """
    Teste si `model_prediction` retourne un message d'erreur
    lorsqu'une exception est levée par la requête.
    """

    def mock_post(*args, **kwargs) -> requests.exceptions.RequestException:
        raise requests.exceptions.RequestException

    monkeypatch.setattr(requests, "post", mock_post)

    input_data = {
        "medinc": 5.0,
        "houseage": 15.0,
        "averooms": 6.0,
        "avebedrms": 1.0,
        "population": 800.0,
        "aveoccup": 3.0,
        "latitude": 37.0,
        "longitude": -122.0,
    }
    result = model_prediction(input=input_data)
    assert result == "Erreur : le modèle n'a pas pu effectuer une prédiction."


def test_model_prediction_http_error(monkeypatch):
    """
    Teste si `model_prediction` retourne un message d'erreur
    lorsqu'une erreur HTTP est retournée par le modèle.
    """

    def mock_post(*args, **kwargs) -> MockResponseError:
        return MockResponseError()

    monkeypatch.setattr(requests, "post", mock_post)

    input_data = {
        "medinc": 5.0,
        "houseage": 15.0,
        "averooms": 6.0,
        "avebedrms": 1.0,
        "population": 800.0,
        "aveoccup": 3.0,
        "latitude": 37.0,
        "longitude": -122.0,
    }
    result = model_prediction(input=input_data)
    assert result == "Erreur : le modèle n'a pas pu effectuer une prédiction."


@pytest.fixture
def session() -> AppTest:
    """
    Initialise une session de test Streamlit.
    """
    at = AppTest.from_file(
        "src/interface/interface.py"
    )  # Nom de votre fichier Streamlit
    at.run(timeout=10)
    return at


def test_initial_state(session) -> None:
    """
    Vérifie l'état initial de l'interface Streamlit.

    - Vérifie le nombre de champs de texte et de boutons.
    - Vérifie les labels des champs et boutons.
    - Vérifie le nombre d'éléments Markdown.
    """

    assert len(session.text_input) == 8  # 8 champs de texte pour les fonctionnalités
    assert len(session.button) == 1  # Bouton "Predict"
    assert session.text_input[0].label == "Revenu médian des ménages"
    assert session.text_input[1].label == "Âge moyen des maisons"
    assert session.text_input[2].label == "Nombre moyen de pièces par logement"
    assert session.text_input[3].label == "Nombre moyen de chambres par logement"
    assert session.text_input[4].label == "Population de la région"
    assert session.text_input[5].label == "Nombre moyen d'occupants par logement"
    assert session.text_input[6].label == "Latitude de la région"
    assert session.text_input[7].label == "Longitude de la région"
    assert session.button[0].label == "Prédire"
    assert len(session.markdown) == 2  # 2 éléments Markdown


def test_valid_input(session: AppTest, monkeypatch) -> None:
    """
    Vérifie que l'application Streamlit retourne une prédiction valide pour des entrées correctes.
    """

    def mock_post(*args, **kwargs) -> MockResponseSuccess:
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

    assert (
        session.markdown[2].value
        == "Le prix prédit pour le logement est : 300000 dollars."
    )


def test_invalid_input(session: AppTest) -> None:
    """
    Vérifie que l'application Streamlit retourne un message d'erreur pour des entrées invalides.
    """
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

    assert (
        session.markdown[2].value
        == "Veuillez entrer des nombres valides dans tous les champs."
    )  # Message d'erreur affiché
