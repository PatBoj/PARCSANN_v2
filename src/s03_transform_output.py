import sys
import pandas as pd
import numpy as np
import math
from loguru import logger

from utils.config import CFG

def apply_transformer(df: pd.DataFrame) -> pd.DataFrame:
    """ Applies custom transformer to the data """
    
    if not CFG['log_transform_output']:
        return df        

    logger.info('Applying logarithm transformer to the ouput data.')
    
    return df.apply(lambda col: col.apply(lambda x: math.log10(x)))


def reverse_transformer(ndmatrix: pd.DataFrame, column_names: list) -> pd.DataFrame:
    """ Reverses the transformer """
    
    df = pd.DataFrame(ndmatrix, columns=column_names)
    
    if not CFG['log_transform_output']:
        return df
    
    logger.info('Applying reverse logarithm transformer to the ouput data.')
    
    return df.apply(lambda col: col.apply(lambda x: 10**x))


@logger.catch(onerror=lambda _: sys.exit(1))
def transform_output(output_data: object, column_names: list = None) -> object:
    """ Transforms output data """
    
    if isinstance(output_data, pd.DataFrame):
        output_data_transform = apply_transformer(output_data)
        return output_data_transform.values

    elif isinstance(output_data, np.ndarray):
        output_data_transform = reverse_transformer(output_data, column_names)
        return output_data_transform
    
    else:
        raise TypeError(f'Invalid "output_data" type.')
        
