from read_yaml import get_config
from read_file import load_dataset


def main():
    cfg = get_config()
    
    data = load_dataset(cfg.get('data_path'))
    
    print(data.info())


if __name__ == '__main__':
    main()