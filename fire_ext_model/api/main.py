from fastapi import FastAPI
import uvicorn
from request import PredictInput
import json
from fastapi.encoders import jsonable_encoder
from pandas import DataFrame


app = FastAPI()

@app.get("/")
def root():
    return "Hello World"

@app.post("/predict")
def predict(input:PredictInput):
    pred_obj = Predict(train)
    data = jsonable_encoder(input)
    data['FUEL']=data['FUEL'].encode()
    df = DataFrame(data,[0])
    pred_array = pred_obj.predict(df)
    result = [1 if x >= 0.5 else 0 for x in pred_array[:,0]]
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    
    from fire_ext_model.processing import load
    from fire_ext_model.processing.train import Train
    from fire_ext_model.processing.predict import Predict

    df = load()
    train = Train(df)
    train.train()
    uvicorn.run(app, port=8002)