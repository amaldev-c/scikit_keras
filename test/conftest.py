import pytest
from fire_ext_model.processing import load
from fire_ext_model.processing.train import Train


@pytest.fixture(scope="package")
def df():
    return load()


@pytest.fixture(scope="package")
def trained_model(df):
    train = Train(df)
    train.train()
    return train
