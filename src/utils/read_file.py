import sys
import pandas as pd
from loguru import logger


def read_raw_excel(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """ Read raw .xlsx file """
    
    return pd.read_excel(file_path, engine='openpyxl', sheet_name=sheet_name)


def read_raw_csv(path: str, sep: str = ',') -> pd.DataFrame:
    """ Read raw .csv file """

    if sep is None:
        sep = ','
    
    return pd.read_csv(path, sep=sep)


@logger.catch(onerror=lambda _: sys.exit(1))
def read_raw_file(file_path: str, sep: str = ',', sheet_name: str = None) -> pd.DataFrame:
    """ Read raw .csv or .xlsx file """

    file_extension = file_path.split('.')[-1]
    
    logger.info(f'Reading a file: {file_path}.')
    if file_extension == 'xlsx':
        return read_raw_excel(file_path, sheet_name)
    elif file_extension == 'csv':
        return read_raw_csv(file_path, sep)
    else:
        raise TypeError(f'Invalid file format: {file_extension}, only .csv and .xlsx are possible.')


def create_new_multiple_columns_single_series(
    df: pd.DataFrame,
    new_col_template: str,
    formula_template: str) -> pd.DataFrame:
    """ 
    Create new multiple columns based on a single configuration line in 'multiple_cols' 
    For example in the file we have columns named 'sec1', 'sec2', 'sec3', ..., and we want to
    convert them all into minutes. In that case in 'multiple_cols' we can write:
        minN: 'df.secN / 60'
    All 'secN' columns will be divided by 60, and new columns with names 'minN' will be created
    """
    
    df_new = df.copy()
    i = 0
    
    while True:
        
        i += 1
        new_col_name = new_col_template.replace('N', str(i))
        formula = formula_template.replace('N', str(i))
        
        try:
            df_new[new_col_name] = eval(formula)
        except AttributeError:
            break
    
    return df_new


def create_new_multiple_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """ Create multiple new columns """
    
    for new_col, formula in cfg.items():
        logger.info(f'Creating multiple columns based on the template: "{new_col}".')
        df_new = create_new_multiple_columns_single_series(df, new_col, formula)
        
    return df_new


def create_new_single_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """ Create new single columns based on the configuration 'single_cols' """
    
    df_new = df.copy()
    
    for new_col, formula in cfg.items():
        logger.info(f'Creating new column: "{new_col}".')
        df_new[new_col] = eval(formula)
        
    return df_new


def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """ Fixing weird column names from excel """
    
    fix_column_names = [col.replace(u'\xa0', '') for col in df.columns]
    fix_column_names = [col.strip() for col in fix_column_names]
    fix_column_names = [col.lower() for col in fix_column_names]
    
    df_fix = df.copy()
    df_fix.columns = fix_column_names
    
    return df_fix


@logger.catch(onerror=lambda _: sys.exit(1))
def create_new_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """ Create new columns based on the configuration file """
    
    df_new = df.copy()
    
    if cfg is None:
        return df
    
    logger.info('Creating new columns.')

    for new_cols_type, cfg_new_columns in cfg.items():
        if new_cols_type == 'single_cols':
            df_new = create_new_single_columns(df_new, cfg_new_columns)

        elif new_cols_type == 'multiple_cols':
            df_new = create_new_multiple_columns(df_new, cfg_new_columns)

        else:
            raise KeyError('Invalid entry in "create_new_cols", only "single_cols" and "multiple_cols" are possible.')

    return df_new


def load_dataset(cfg: dict, sheet_name: str = None) -> pd.DataFrame:
    """ Read file using dictionary """
    
    if sheet_name is None:
        sheet_name = cfg.get('sheet_name')

    df = read_raw_file(cfg.get('file_path'), cfg.get('data_sep'), sheet_name)
    df = fix_column_names(df)
    df = create_new_columns(df, cfg.get('create_new_cols'))
    
    return df