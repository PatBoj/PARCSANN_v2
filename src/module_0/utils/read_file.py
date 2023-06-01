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
    
    
def create_new_columns_series(
    df: pd.DataFrame, 
    new_col_name_template: str, 
    formula_template: str) -> pd.DataFrame:
    
    df_new = df.copy()
    
    i = 0
    
    while True:
        i += 1
        
        new_col_name = new_col_name_template.replace('N', str(i))
        formula = formula_template.replace('N', str(i))
        
        try:
            df_new[new_col_name] = eval(formula)
        except AttributeError:
            break
    
    return df_new


def create_new_columns_series_all(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    
    for new_col_name, formula in cfg.items():
        df_new = create_new_columns_series(
            df=df, 
            new_col_name_template=new_col_name, 
            formula_template=formula)
        
    return df_new

    
def create_new_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """ Create new columns based on the configuration file """
    
    df_new = df.copy()
    
    if cfg is None:
        return df
    
    for new_col, formula in cfg.items():
        if new_col != 'series':
            df_new[new_col] = eval(formula)
            
    if cfg.get('series') is not None:
        df_new = create_new_columns_series_all(
            df_new, cfg.get('series'))
    
    return df_new


def load_dataset(cfg: dict) -> pd.DataFrame:
    """ Read file using dictionary """
    
    df = read_raw_file(
        cfg.get('file_path'), 
        cfg.get('separator'), 
        cfg.get('sheet_name'))
    
    df = create_new_columns(df, cfg.get('create_cols'))
    
    return df