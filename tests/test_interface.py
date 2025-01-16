from streamlit.testing.v1 import AppTest
from src.interface.interface import model_prediction
import pytest

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

def test_valid_input(session):

    # Fournir des entrées utilissessioneur valides
    session.text_input[0].input("-122.23")  
    session.text_input[1].input("37.88")  
    session.text_input[2].input("29.0")  
    session.text_input[3].input("700.0")  
    session.text_input[4].input("100.0")  
    session.text_input[5].input("500.0")  
    session.text_input[6].input("200.0")  
    session.text_input[7].input("5.0")  
    #session.button[0].click().run()  # Cliquer sur "Predict"

    # Vérifier l'étsession après interaction
    # assert session.session_stsessione["prediction"] == 500000  # La prédiction est stockée dans l'étsession
    # assert "The predicted housing price is: 1449.65 dollars." in session.markdown[2].value

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
