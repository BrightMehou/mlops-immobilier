# projet-mlops-imo-bm
uvicorn fichier:app --reload

poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db

poetry run pytest -p no:warnings 

http://127.0.0.1:8000/redoc

http://127.0.0.1:8000/docs