import sys

from loguru import logger

from configs.config_base import Config


class ConfigModeling(Config):
    def __init__(self, cfg_path: str = "../configs/cfg_modeling_template.yaml") -> None:
        super().__init__(cfg_path)
        self.check_config()

        self.TUNING = self.cfg["tuning"]

        self.CORE_SYMMETRY = self.cfg["core_symmetry"]
        self.ONE_HOT_ENCODING = self.cfg["one_hot_encoding"]
        self.TRAIN_SPLIT = self.cfg["train_split"]
        self.OUTPUT_COLUMNS = self.cfg["output_columns"]
        self.USE_MONOCORES = self.cfg["use_monocores"]
        self.INPUT_COLUMNS = self.cfg["input_columns"]

        self.LAYERS = self.cfg["layers"]
        self.LOSS_FUNCTION = self.cfg["loss_function"]
        self.LEARNING_RATE = self.cfg["learning_rate"]
        self.EPOCHS = self.cfg["epochs"]

    def check_config(self) -> None:
        logger.info("Checking modeling configuration file.")

        self.basic_check("tuning", bool)

        self.basic_check("core_symmetry", str)
        self.basic_check("one_hot_encoding", bool)
        self.basic_check("train_split", float)
        self.basic_check("output_columns", list)
        self.basic_check("use_monocores", bool)
        self.basic_check("input_columns", list)

        self.basic_check("layers", dict)
        self.check_layers()
        self.basic_check("loss_function", str)
        self.basic_check("learning_rate", float)
        self.basic_check("epochs", int)

    def check_layers(self) -> None:
        layers = self.cfg["layers"]

        for layer_name, layer_setting in layers.items():
            if layer_name == "layer_output":
                continue

            self.basic_check("neurons", int, layer_setting)
            self.basic_check("activation", str, layer_setting)

        if "layer_output" not in layers:
            logger.error("There must be a layer called `layer_output` in the configuration file.")
            sys.exit(1)

        if layers.get("layer_output").get("neurons") is not None:
            logger.warning(
                "In the `layer_output` number of neurons is determined by the size of output data. "
                "In the configuration file, number of neurons for output layer is obsolete.",
            )
