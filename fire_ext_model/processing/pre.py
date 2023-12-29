from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

class PreProcessor:
    def __init__(self,df:DataFrame):
        self.oe=OrdinalEncoder()
        self.oe.fit(df[['FUEL']])
        
    def convert(self,df:DataFrame) -> DataFrame:
        df_copy = df.copy()
        df_copy['FUEL'] = self.oe.transform(df[['FUEL']])
    
        return df_copy
    
