from s01_read_yaml import get_config
from s02_prepare_data import prepare_input_output
from s03_modeling import get_test_predict


def main():
    cfg = get_config()
    
    X, y = prepare_input_output(cfg.get('prepare_data'))
    
    y_true, y_pred = get_test_predict(X, y, cfg.get('modeling'))
    

if __name__ == '__main__':
    main()