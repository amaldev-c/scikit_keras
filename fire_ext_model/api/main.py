"""Contains REST API definitions built using FastAPI and Uvicorn"""
import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pandas import DataFrame
from fire_ext_model.processing import load
from fire_ext_model.processing.train import Train
from fire_ext_model.processing.predict import Predict
from fire_ext_model.api.request import PredictRequest


app = FastAPI()

if __name__ == "__main__":
    df_input = load()
    train = Train(df_input)
    train.train()
    uvicorn.run(app, port=8002)

@app.get("/")
def root() -> str:
    """Placeholder for base url api

    Returns:
        str: Hello World string
    """
    return "Hello World"

@app.post("/predict")
def predict(request:PredictRequest) -> list:
    """API for model inference

    Args:
        request (PredictRequest): Request model. Contains a single input.

    Returns:
        list: Prediction for the input
    """
    pred_obj = Predict(train)
    data = jsonable_encoder(request)
    data['FUEL']=data['FUEL'].encode()
    df_pred = DataFrame(data,[0])
    pred_array = pred_obj.predict(df_pred)
    result = [1 if x >= 0.5 else 0 for x in pred_array[:,0]]
    return result[0]
