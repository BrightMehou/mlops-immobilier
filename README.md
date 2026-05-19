# Projet de Prédiction des Prix des Logements en Californie

Ce projet est une application de machine learning permettant de prédire les prix des logements en Californie à partir de différentes caractéristiques. Il comprend une API FastAPI, une interface utilisateur Streamlit, et des scripts pour l'entraînement et l'évaluation de modèles de machine learning.

## 🗂️ Structure du Projet

Voici la structure du projet :

```
├───.github/
│   └───workflows/        # Fichiers pour l'intégration continue
├───data                  # Data
├───notebooks/            # Notebooks Jupyter pour les expérimentations
├───src/                  # Code source du projet
│   ├───api/              # AppFastAPI
│   ├───ui/               # Interface utilisateur Streamlit
│   ├───ml/               # Scripts pour l'entraînement et l'évaluation des modèles
├───tests/                # Tests unitaires et d'intégration
├───Dockerfile            # Fichier Docker pour containeriser l'API
├───docker-compose.yml    # Fichier Compose pour orchestrer les services
└───pyproject.toml        # Fichier de configuration pour uv et Ruff
```

## Fonctionnalités

1. **API FastAPI** :
   - Permet de servir un modèle de machine learning pour les prédictions.
   - Points de terminaison pour les prédictions et la validation des données d'entrée.

2. **Interface Utilisateur Streamlit** :
   - Permet aux utilisateurs de saisir les caractéristiques d'un logement et de recevoir une prédiction de prix en temps réel.

3. **Notebooks Jupyter** :
   - **Experimentations** : Création et évaluation de plusieurs modèles, avec journalisation des résultats dans MLflow pour 

4. **Scripts de Machine Learning** :
   - **train.py** : Industrialise le modèle pour la production en l'enregistrant dans le registre de modèles MLflow.
   - Entraînement des modèles avec des données de logement en Californie.
   - Évaluation des modèles à l'aide de métriques telles que le MSE, MAE et R².
   - Journalisation des modèles et des résultats avec MLflow.

## 📥 Installation et utilisation

1. **Cloner le répertoire** :
   ```bash
   git clone <url_du_repository>
   cd <nom_du_repertoire>
   ```
### Démarrer l'API et l'Interface Utilisateur avec Docker

1. Construire et lancer les conteneurs :
   ```bash
   docker compose up -d
   ```
2. Accéder à l'API via Swagger :
   - URL : `http://localhost:8000/docs`

3. Accéder à l'interface:
   - URL : `http://localhost:8501`


4. Seulement pour les dev du projet
   ```bash
   uv run pre-commit install
   uv run pre-commit run --all-files
   ```