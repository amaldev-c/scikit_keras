"""Request and response pydantic models used by APIs"""
from pydantic import BaseModel

class PredictRequest(BaseModel):
    """Request model for the 'predict' API"""
    SIZE:int
    FUEL:str
    DISTANCE:int
    DESIBEL:int
    AIRFLOW:float
    FREQUENCY:int
