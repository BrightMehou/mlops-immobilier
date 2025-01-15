from fastapi import FastAPI
import mlflow.pyfunc
import numpy as np
import pandas as pd
from src.train import Input


model_name = "Production-model"
model_version = 1
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
# Initialiser l'application FastAPI
app = FastAPI(
    title="Californie Housing price prediction",
    description="This is a simple API to predict housing prices in California",
    version="0.1.0"
)


@app.get("/")
async def read_main():
    return {"msg": "Hello World"}

# Point de terminaison pour les prédictions
@app.post("/predict")
def predict(input_data: Input):

    features = np.array([[input_data.medinc, input_data.houseage, input_data.averooms, input_data.avebedrms, input_data.population, input_data.aveoccup, input_data.latitude, input_data.longitude]])
    # Faire une prédiction
    features = pd.DataFrame(features, columns=["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"])
    prediction = model.predict(features) 
    print(prediction)
    return {"prediction": float(prediction[0])} 