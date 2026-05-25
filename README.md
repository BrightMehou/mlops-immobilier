# Prédiction des prix des logements en Californie

Application de machine learning pour prédire les prix des logements en Californie à partir de caractéristiques socio-démographiques et géographiques. Le projet expose une **API FastAPI** (modèle MLflow + explications SHAP), une **interface Streamlit** multi-pages, d’analyse et de monitoring.

## Fonctionnalités

### API FastAPI

Cette API déploie un modèle de machine learning et son explainer SHAP pour prédire des prix et expliquer les résultats en temps réel. Elle stocke chaque requête dans une base SQLite pour suivre l'historique et génère des rapports de drift (via Evidently) pour détecter si les données reçues en production évoluent par rapport à l'entraînement.

### Interface Streamlit

| Page | Rôle |
|------|------|
| **Accueil**| Saisie des caractéristiques d’un logement, prédiction en temps réel, graphique SHAP |
| **Data exploration** | Exploration du dataset California Housing (stats, corrélations, carte) |
| **Feature analysis** | Visualisations SHAP et dépendance partielle |
| **Monitoring** | Historique des prédictions et rapport de drift |

## Structure du projet

```
├── .github/workflows/     # CI : tests Python et build Docker
├── data/
│   ├── feature_analysis/  # Résultats des anaylses de features
│   └── monitoring/        # Données et base SQLite de monitoring
├── notebooks/             # Notebook d’expérimentation
├── src/
│   ├── api/               # FastAPI, persistance SQLite
│   ├── ml/                # Entraînement, drift, analyse des features
│   └── ui/                # Streamlit (accueil + pages)
├── tests/                 # Tests API, base de données et UI
├── docker-compose.yaml
├── Dockerfile
└── pyproject.toml         # Dépendances (uv) et configuration Ruff
```

## Prérequis

- [Python](https://www.python.org/) ≥ 3.13
- [uv](https://docs.astral.sh/uv/) (gestionnaire de paquets)
- [Docker](https://www.docker.com/) et Docker Compose (optionnel, pour lancer API + UI)

## Installation

```bash
git clone <url_du_repository>
cd california-housing-prices
uv sync
```

## Utilisation

### Avec Docker (recommandé)

Au démarrage, le conteneur API entraîne le modèle puis lance le serveur :

```bash
docker compose up -d --build
```

| Service | URL |
|---------|-----|
| API (Swagger) | http://localhost:8000/docs |
| Interface Streamlit | http://localhost:8501 |

### En local (développement)

1. Entraîner et enregistrer le modèle :

   ```bash
   uv run python src/ml/train.py
   ```

2. Lancer l’API :

   ```bash
   uv run uvicorn src.api.app:app --reload --port 8000
   ```

3. Lancer l’interface :

   ```bash
   uv run streamlit run src/ui/app.py
   ```

4. Lancer les tests :

   ```bash
   uv run pytest
   ```

### Qualité de code (développeurs)

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Licence

Voir le fichier [LICENSE](LICENSE).
