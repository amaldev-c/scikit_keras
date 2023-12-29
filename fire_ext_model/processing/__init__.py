from scipy.io import arff
import pandas as pd
import yaml
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE_PATH = PACKAGE_DIR / "config.yml"
DATASET_DIR = PACKAGE_DIR / "dataset"

def load():
    with open(CONFIG_FILE_PATH, 'r') as file:
        config = yaml.safe_load(file)
        print(config)
        
    data = arff.loadarff(DATASET_DIR / config['train']['dataset_file_name'])
    df = pd.DataFrame(data[0])
    print(df.head())
    
    return df