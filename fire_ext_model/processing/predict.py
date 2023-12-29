"""Contains wrappers for model inference"""
from pandas import DataFrame
from fire_ext_model.processing.train import Train

class Predict:
    """Wrapper for model inference"""

    def __init__(self,trained_model:Train):
        self.__model=trained_model

    def predict(self,df:DataFrame) -> list:
        """Makes model prediction for the given dataframe

        Args:
            df (DataFrame): Input data for which prediction is to be made

        Returns:
            list: Prediction for each input
        """
        data = self.__model.pre_processor.convert(df)
        return self.__model.model.predict(data)
