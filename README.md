<h1>Tensorflow based DNN trained on Acoustic Fire Extinguisher tabular dataset.</h1>

This dataset contains results of the extinguishing tests of four different fuel flames with a sound wave extinguishing system. The DNN consists of an Input -> Normalization -> 2 Dense layers.

<h3>API</h3>

/predict - Predict whether fire will be extinguished

Request

{

    SIZE: int
  
    FUEL: str
  
    DISTANCE: int
  
    DESIBEL: int
  
    AIRFLOW: float
  
    FREQUENCY: int
  
}
