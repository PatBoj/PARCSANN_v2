import pandas as pd


def read_raw_excel(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """ Read raw excel file """
    
    return pd.read_excel(file_path, engine='openpyxl', sheet_name=sheet_name)


def read_raw_csv(path: str, sep: str = ',') -> pd.DataFrame:
    """ Read raw csv file """
    
    if sep is None:
        sep = ','
    
    return pd.read_csv(path, sep=sep)


def read_raw_file(
    file_path: str, 
    sep: str = ',', 
    sheet_name: str = None) -> pd.DataFrame:
    """ Read raw .csv or .xlsx file """
    
    file_extension = file_path.split('.')[-1]
    
    if file_extension == 'xlsx':
        return read_raw_excel(file_path, sheet_name)
    elif file_extension == 'csv':
        return read_raw_csv(file_path, sep)
    else:
        raise NameError(f'Unsupported file format.')


def load_dataset(cfg: dict) -> pd.DataFrame:
    """ Read file using dictionary """
    
    df = read_raw_file(
        cfg.get('file_path'), 
        cfg.get('separator'), 
        cfg.get('sheet_name'))
    
    return df