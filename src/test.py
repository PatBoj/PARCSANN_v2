from modeling import NeuralNetwork

from configs.config_data import ConfigData
from configs.config_modeling import ConfigModeling
from src.prepare_data import CoreData

# TF_ENABLE_ONEDNN_OPTS=0 # This is for reproducibility of resets, remove it after everything is ok


def main() -> None:
    """Main ofc."""
    cfg_data = ConfigData()
    cfg_modeling = ConfigModeling()
    core_data = CoreData(cfg_data)(cfg_modeling)

    nn = NeuralNetwork(cfg_modeling, core_data)
    nn()


if __name__ == "__main__":
    main()
