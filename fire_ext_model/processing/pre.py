"""Contains wrappers for input pre-processing to be done before training and inference"""
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder

class PreProcessor:
    """Performs various pre-processing of given input"""
    def __init__(self,df:DataFrame):
        self.oe=OrdinalEncoder()
        self.oe.fit(df[['FUEL']])

    def convert(self,df:DataFrame) -> DataFrame:
        """Convert the input to the format that is expected by the model.
        Transforms the 'categorical' values of the FUEL column of the input
        dataframe through an ordinal encoder to convert it to 'int' value. 

        Args:
            df (DataFrame): Input data to be pre-processed

        Returns:
            DataFrame: Pre-processed input
        """
        df_copy = df.copy()
        df_copy['FUEL'] = self.oe.transform(df[['FUEL']])

        return df_copy
