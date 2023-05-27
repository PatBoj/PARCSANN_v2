import pandas as pd


def load_raw_file(path: str) -> pd.DataFrame:
    """ Read file from a given path """
    
    return pd.read_csv(path)


def load_dataset(cfg: dict) -> pd.DataFrame:
    """ Read file using dictionary """
    
    return pd.read_csv(cfg.get('file_path'))