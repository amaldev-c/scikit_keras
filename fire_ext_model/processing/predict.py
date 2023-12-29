from fire_ext_model.processing.train import Train

class Predict:
    def __init__(self,trained_model:Train):
        self.model=trained_model

    def predict(self,df):
        data = self.model.pre_processor.convert(df)
        return self.model.model.predict(data)