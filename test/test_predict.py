from pandas import DataFrame
from fire_ext_model.processing.predict import Predict
from fire_ext_model.processing.train import Train


class TestPredict:
    def test_prediction(self, trained_model: Train):
        data = [[1, b"gasoline", 30, 101, 0.4, 70]]
        df = DataFrame(
            data,
            columns=["SIZE", "FUEL", "DISTANCE", "DESIBEL", "AIRFLOW", "FREQUENCY"],
        )
        pred_obj = Predict(trained_model)

        pred_array = pred_obj.predict(df)
        result = [1 if x >= 0.5 else 0 for x in pred_array[:, 0]]
        assert result[0] == 1
