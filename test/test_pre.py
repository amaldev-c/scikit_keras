from fire_ext_model.processing.pre import PreProcessor


def test_pre_process(df):
    assert isinstance(df["FUEL"][0], bytes)
    pre_processor = PreProcessor(df)
    df = pre_processor.convert(df)
    assert isinstance(df["FUEL"][0], float)
