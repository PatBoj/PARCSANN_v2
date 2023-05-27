from omegaconf import OmegaConf

from utils.useful_functions import convert_lists_to_ndarrays


def load_yaml(path: str) -> dict:
    """ Return .yaml file as a dictionary """

    config = OmegaConf.load(path)
    config = OmegaConf.to_container(config, resolve=True)
    config = convert_lists_to_ndarrays(config)
    
    return config


def get_config() -> dict:
    """ Retrun config file """
    
    return load_yaml('configs/config.yaml')