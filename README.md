# Projet de Prédiction des Prix des Logements en Californie

Ce projet est une application de machine learning permettant de prédire les prix des logements en Californie à partir de différentes caractéristiques. Il comprend une API FastAPI, une interface utilisateur Streamlit, et des scripts pour l'entraînement et l'évaluation de modèles de machine learning.

## Structure du Projet

Voici la structure du projet :

```
├───.github
│   └───workflows         # Fichiers pour l'intégration et déploiement continus (CI/CD)
├───notebooks             # Notebooks Jupyter pour l'exploration et les analyses préliminaires
│   ├───Analyse_des_features    # Analyse de l'importance des caractéristiques avec SHAP
│   ├───Analyse_exploratoire    # Analyse exploratoire des données
│   ├───Data_drift_detection    # Pour détecter d'éventuelle data_drift
│   ├───Experimentations        # Tests et sélection du meilleur modèle à mettre en production
├───src                   # Code source du projet
│   ├───api               # API construite avec FastAPI (fichier app.py)
│   ├───interface         # Interface utilisateur Streamlit (fichier interface.py)
│   ├───ml                # Scripts pour l'entraînement et l'évaluation des modèles (fichier train.py)
├───tests                 # Tests unitaires et d'intégration
├───Dockerfile            # Fichier Docker pour containeriser l'API
├───docker-compose.yml    # Fichier Compose pour orchestrer les services
└───pyproject.toml        # Fichier de configuration pour Poetry
```

## Fonctionnalités

1. **API FastAPI** :
   - Permet de servir un modèle de machine learning pour les prédictions.
   - Points de terminaison pour les prédictions et la validation des données d'entrée.

2. **Interface Utilisateur Streamlit** :
   - Permet aux utilisateurs de saisir les caractéristiques d'un logement et de recevoir une prédiction de prix en temps réel.

3. **Notebooks Jupyter** :
   - **Analyse_des_features** : Utilisation de SHAP pour analyser l'importance des caractéristiques dans les prédictions.
   - **Analyse_exploratoire** : Exploration des données pour comprendre leur structure et identifier des tendances.
   - **Experimentations** : Création et évaluation de plusieurs modèles, avec journalisation des résultats dans MLflow pour sélectionner le meilleur modèle.

4. **Scripts de Machine Learning** :
   - **train.py** : Industrialise le modèle pour la production en l'enregistrant dans le registre de modèles MLflow.
   - Entraînement des modèles avec des données de logement en Californie.
   - Évaluation des modèles à l'aide de métriques telles que le MSE, MAE et R².
   - Journalisation des modèles et des résultats avec MLflow.

5. **Docker et Orchestration** :
   - Un fichier Dockerfile permet de containeriser l'API.
   - Un autre fichier Dockerfile permet de containeriser l'interface utilisateur.
   - Le fichier docker-compose.yml facilite l'orchestration des services (API, MLflow, etc.).

6. **Gestion des Dépendances avec Poetry** :
   - Utilisation de Poetry pour gérer les dépendances et les scripts.

7. **Tests Unitaires** :
   - Couverture des fonctions critiques, y compris l'entraînement, l'évaluation, et les points de terminaison de l'API.


## Installation

1. **Cloner le répertoire** :
   ```bash
   git clone <url_du_repository>
   cd <nom_du_repertoire>
   ```

2. **Installer Poetry** :
   Si Poetry n'est pas encore installé : [Poetry](https://python-poetry.org/docs/)

3. **Installer les dépendances** :
   ```bash
   poetry install
   ```

## Utilisation

### Démarrer l'API et l'Interface Utilisateur avec Docker

1. Construire et lancer les conteneurs :
   ```bash
   docker-compose up -d
   ```
2. Accéder à l'API via Swagger :
   - URL : `http://localhost:8000/docs`

3. Accéder à l'interface:
   - URL : `http://localhost:8501`

### Entraînement, Évaluation et Mise en production des Modèles

1. Exécuter le script d'entraînement :
   ```bash
   poetry run python src/ml/train.py
   ```
2. Le modèle et métriques seront journalisés dans MLflow.
   ```bash
   poetry run mlflow ui
   ```
### Accéder aux expérimentation du notebook experiments

1. Accéder au dossier notebook :
   ```bash
   cd notebooks
   ```
2. Accéder au dossier notebook :
   ```bash
   poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
3. Accéder à l'interface UI de mlfow:
   - url : `http://localhost:5000`
### Tests

1. Exécuter le script d'entraînement :
   ```bash
   poetry run python src/ml/train.py
   ```

2. Exécuter les tests avec Pytest :
   ```bash
   poetry run pytest
   ```

## Technologies Utilisées

- **Python** : Langage principal
- **FastAPI** : API backend
- **Streamlit** : Interface utilisateur
- **scikit-learn** : Entraînement et évaluation des modèles
- **MLflow** : Suivi des expériences et journalisation des modèles
- **Docker** : Containerisation des services
- **docker-compose** : Orchestration des conteneurs
- **Poetry** : Gestion des dépendances et des environnements
- **Pytest** : Framework de tests

## Licence

Ce projet est sous licence [MIT](LICENSE).