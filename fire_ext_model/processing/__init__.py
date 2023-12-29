"""Contains utility functions for dataset initialization"""
from pathlib import Path
from scipy.io import arff
from pandas import DataFrame
import yaml

PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE_PATH = PACKAGE_DIR / "config.yml"
DATASET_DIR = PACKAGE_DIR / "dataset"


def load() -> DataFrame:
    """Loads the training data file into a dataframe.

    Returns:
        DataFrame: Training records
    """
    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
        print(config)

    data = arff.loadarff(DATASET_DIR / config["train"]["dataset_file_name"])
    df = DataFrame(data[0])
    print(df.head())

    return df
