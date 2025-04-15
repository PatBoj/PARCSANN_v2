import sys
from abc import ABC, abstractmethod

from loguru import logger
from yaml import safe_load


def load_yaml(path: str) -> dict:
    """Reads .yaml file (configuration file).

    Args:
        path (str): path to a .yaml file.

    Returns:
        dict: configuration file read from .yaml file.
    """
    logger.info(f"Reading configuration file: {path}.")
    try:
        with open(path) as file:
            return safe_load(file)
    except FileNotFoundError:
        logger.error(f"The file: {path} was not found.")
        sys.exit(1)


def basic_check(cfg_dict: dict, cfg_entry: str, desired_type: object) -> None:
    """Checks if variable from a given dictionary file exists and have appropriate type.

    Args:
        cfg_dict (dict): dictionary to check.
        cfg_entry (str): name of the configuration field.
        desired_type (object): desired type of configuration entry.
    """
    if cfg_dict.get(cfg_entry) is None:
        logger.error(f"`{cfg_entry}` is missing in a configuration file.")
        sys.exit(1)

    if not isinstance(cfg_dict.get(cfg_entry), desired_type):
        logger.error(
            f"Variable `{cfg_entry}` is of type `{type(cfg_dict.get(cfg_entry)).__name__}`, "
            f"but it should be of type `{desired_type.__name__}`.",
        )


class Config(ABC):
    def __init__(self, cfg_path: str) -> None:
        """Abstract class that contains the config file.

        Args:
            cfg_path (str): Path to a configuration file.
        """
        self.cfg = load_yaml(cfg_path)

    def basic_check(self, cfg_entry: str, desired_type: object, cfg_dict: dict | None = None) -> None:
        """Checks if variable from a configuration file exists and have appropriate type.

        Args:
            cfg_entry (str): name of the configuration field.
            desired_type (object): desired type of configuration entry.
            cfg_dict (dict): dictionary to check.
        """
        if cfg_dict is None:
            cfg_dict = self.cfg

        basic_check(cfg_dict, cfg_entry, desired_type)

    @abstractmethod
    def check_config(self) -> None:
        pass
