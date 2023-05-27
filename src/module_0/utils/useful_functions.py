import numpy as np

def convert_lists_to_ndarrays(dictionary: dict) -> dict:
    """ Return dictionary with convertet list into ndarrays """
    
    converted_dict = {}
    
    for key, value in dictionary.items():
        if isinstance(value, list):
            converted_dict[key] = np.array(value)
        elif isinstance(value, dict):
            converted_dict[key] = convert_lists_to_ndarrays(value)
        else:
            converted_dict[key] = value

    return converted_dict