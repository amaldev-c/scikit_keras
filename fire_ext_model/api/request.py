from pydantic import BaseModel

class PredictInput(BaseModel):
    SIZE:int
    FUEL:str
    DISTANCE:int
    DESIBEL:int
    AIRFLOW:float
    FREQUENCY:int