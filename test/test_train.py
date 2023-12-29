from fire_ext_model.processing.train import Train

class TestTrain:
    
    def test_training(self,trained_model:Train):
        assert trained_model.model != None
        assert trained_model.test() > 0.9