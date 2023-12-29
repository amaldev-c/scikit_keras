from sklearn.model_selection import train_test_split
import pandas as pd
from keras import Input,Model,activations,optimizers,losses,metrics
from keras.layers import Dense,Normalization
from fire_ext_model.processing.pre import PreProcessor

class Train:
    def __init__(self,df:pd.DataFrame):
        self.__pre_processor = PreProcessor(df)
        df = self.__pre_processor.convert(df)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df.drop('CLASS',axis=1), df['CLASS'], test_size=0.25)
        
        layer_norm = Normalization(axis=1)
        layer_norm.adapt(self.X_train)
    
        input = Input(6)
        norm = (layer_norm)(input)
        dense_1 = Dense(32,activations.relu)(norm)
        dense_2 = Dense(16,activations.relu)(dense_1)
        output = Dense(1,activations.sigmoid)(dense_2)

        self.__keras_fun_model = Model(input,output)
        self.__keras_fun_model.summary()
        
    def train(self):    
        self.__keras_fun_model.compile(optimizers.Adam(),losses.BinaryCrossentropy(),metrics.BinaryAccuracy())
        self.__keras_fun_model.fit(self.X_train,self.y_train.astype('int'),32,10,validation_split=0.2)

    def test(self) -> float:
        y_pred = self.__keras_fun_model.evaluate(self.X_test,self.y_test.astype('int'))
        return y_pred[1]
    
    @property
    def model(self):
        return self.__keras_fun_model
    
    @property
    def pre_processor(self):
        return self.__pre_processor
