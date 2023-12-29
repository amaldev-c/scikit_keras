"""Contains wrappers for model training"""
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from keras import Input,Model,activations,optimizers,losses,metrics
from keras.layers import Dense,Normalization
from fire_ext_model.processing.pre import PreProcessor

class Train:
    """Wrapper for model training"""

    def __init__(self,df:DataFrame):
        self.__pre_processor = PreProcessor(df)
        df = self.__pre_processor.convert(df)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            df.drop('CLASS',axis=1), df['CLASS'], test_size=0.25)

        layer_norm = Normalization(axis=1)
        layer_norm.adapt(self.x_train)

        input_layer = Input(6)
        norm = (layer_norm)(input_layer)
        dense_1 = Dense(32,activations.relu)(norm)
        dense_2 = Dense(16,activations.relu)(dense_1)
        output_layer = Dense(1,activations.sigmoid)(dense_2)

        self.__keras_fun_model = Model(input_layer,output_layer)
        self.__keras_fun_model.summary()

    def train(self):
        """Trains the model using the given input data.
        75% of the input is used for training and rest is for testing.
        Within the training records, 20% of the records are used for validation. 
        """
        self.__keras_fun_model.compile(optimizers.Adam(),losses.BinaryCrossentropy(),
                                       metrics.BinaryAccuracy())
        self.__keras_fun_model.fit(self.x_train,self.y_train.astype('int'),32,10,
                                   validation_split=0.2)

    def test(self) -> float:
        """Performs evaluation of the model using part of the given input data.
        The evaluation is expected to return an accuracy score of more than 90%.

        Returns:
            float: Accuracy score, a value between 0 and 1
        """
        y_pred = self.__keras_fun_model.evaluate(self.x_test,self.y_test.astype('int'))
        return y_pred[1]

    @property
    def model(self) -> Model:
        """Getter for the Keras model used for training

        Returns:
            Model: Keras model used for training
        """
        return self.__keras_fun_model

    @property
    def pre_processor(self) -> PreProcessor:
        """Getter for the input pre-processor used for training

        Returns:
            PreProcessor: Input pre-processor used for training
        """
        return self.__pre_processor
