from s01_get_config import get_cfg
from s02_load_data import load_dataset


def main():
    cfg = get_cfg('configs/config.yaml')
    
    data = load_dataset(cfg.data_path)
    
    print(data.info())


if __name__ == '__main__':
    main()