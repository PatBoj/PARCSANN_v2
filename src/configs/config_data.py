from loguru import logger

from configs.config_base import Config


class ConfigData(Config):
    def __init__(self, cfg_path: str = "../configs/cfg_data.yaml") -> None:
        """Class that contains information about input files and their settings. It performs basic checks, like
        checking if the filed exists or if it has appropriate type.

        Args:
            cfg_path (str): path to a configuration file.
        """
        super().__init__(cfg_path)
        self.check_config()

        self.INPUT_OUTPUT_FILE_DETAILS = self.cfg["input_output_file_details"]
        self.MONOCORE_FILE_DETAILS = self.cfg["monocore_file_details"]
        self.MONOCORE_EVOLUTION_FILE_DETAILS = self.cfg["monocore_evolution_file_details"]

    def check_config(self) -> None:
        logger.info("Checking data configuration file.")

        self.basic_check("input_output_file_details", dict)
        self.basic_check("monocore_file_details", dict)
        self.basic_check("monocore_evolution_file_details", dict)
