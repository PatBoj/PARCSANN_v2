import numpy as np
from omegaconf import OmegaConf


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


def load_yaml(path: str) -> dict:
    """ Return .yaml file as a dictionary """

    config = OmegaConf.load(path)
    config = OmegaConf.to_container(config, resolve=True)
    config = convert_lists_to_ndarrays(config)
    
    return config


def get_config() -> dict:
    """ Retrun config file """
    
    return load_yaml('configs/config.yaml')