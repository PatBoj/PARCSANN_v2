import numpy as np

def absolute_error(y_true, y_pred) -> np.ndarray:
    """ Get absolute error between true and predictied values """
    
    return np.abs(y_true - y_pred)


def mean_absolute_error(y_true, y_pred) -> np.float:
    """ Get average absolute error from true and predicted values """
    
    abs_error = absolute_error(y_true, y_pred)
    
    return np.average(abs_error)


def relative_error(y_true, y_pred) -> np.ndarray:
    """ Get relative error between true and predicted values """
    
    return np.abs(y_true - y_pred) / y_true


def mean_realative_error(y_true, y_pred) -> np.float:
    """ Get average relative error from true and predicted values """
    
    rel_error = relative_error(y_true, y_pred)
    
    return np.average(rel_error)