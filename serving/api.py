from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Définir le modèle de données attendu en entrée de l'API
class SingleFeature(BaseModel):
    Seniority: float
    Home: float
    Time: float
    Age: float
    Marital: float
    Records: float
    Job: float
    Expenses: float
    Income: float
    Assets: float
    Debt: float
    Amount: float
    Price: float


class PredictionInput(BaseModel):
    features: list[SingleFeature]


model_path = '../artifacts/pipeline.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)


@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        data = np.array([list(d.dict().values()) for d in input_data.features])
        prediction = model.predict(data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
