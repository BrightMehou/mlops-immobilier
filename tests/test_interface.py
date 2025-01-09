from streamlit.testing.v1 import AppTest


def test_initial_state():
    at = AppTest.from_file("interface.py")  # Nom de votre fichier Streamlit
    at.run(timeout=10)
    # Vérifier l'état initial
    assert len(at.text_input) == 8  # 8 champs de texte pour les fonctionnalités
    assert len(at.button) == 1  # Bouton "Predict"
    assert at.text_input[0].label == "Longitude"
    assert at.text_input[1].label == "Latitude"
    assert at.button[0].label == "Predict"
    assert len(at.markdown) == 2  # 2 éléments Markdown
    #assert not at.session_state.get("prediction", None)  # Pas de prédiction initialement

def test_valid_input():

    at = AppTest.from_file("interface.py")
    at.run()

    # Fournir des entrées utilisateur valides
    at.text_input[0].input("-122.23")  # Longitude
    at.text_input[1].input("37.88")  # Latitude
    at.text_input[2].input("29.0")  # Housing median age
    at.text_input[3].input("700.0")  # Total rooms
    at.text_input[4].input("100.0")  # Total bedrooms
    at.text_input[5].input("500.0")  # Population
    at.text_input[6].input("200.0")  # Households
    at.text_input[7].input("5.0")  # Median income
    #at.button[0].click().run()  # Cliquer sur "Predict"

    # Vérifier l'état après interaction
    # assert at.session_state["prediction"] == 500000  # La prédiction est stockée dans l'état
    # assert "The predicted housing price is: 1449.65 dollars." in at.markdown[2].value

def test_invalid_input():
    at = AppTest.from_file("interface.py")
    at.run()
    # Fournir des entrées invalides
    at.text_input[0].input("abc")  # Longitude (valeur non valide)
    at.text_input[1].input("xyz")  # Latitude (valeur non valide)
    at.button[0].click().run()  # Cliquer sur "Predict"
    
    assert "could not convert string to float:" in at.exception[0].value # Un message d'avertissement devrait être affiché
