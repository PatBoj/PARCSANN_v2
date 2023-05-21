from s01_read_yaml import get_config
from s02_prepare_data import prepare_input_output
from s03_modeling import get_acc

from utils.read_file import load_dataset


def main():
    cfg = get_config()
    
    X, y = prepare_input_output(cfg.get('prepare_data'))
    
    get_acc(X, y, cfg.get('modeling'))
    

if __name__ == '__main__':
    main()