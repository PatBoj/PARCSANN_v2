from s01_read_yaml import get_config
from s02_prepare_data import prepare_input_output
from s03_use_monocores import apply_monocore_dictionary
from s04_modeling import get_test_predict
from s05_evaluate import evaluate_model


def main():
    cfg = get_config()
    
    X, y = prepare_input_output(cfg=cfg.get('prepare_data'))
    
    X = apply_monocore_dictionary(input_data=X, cfg=cfg.get('use_monocores'))
    
    y_true, y_pred = get_test_predict(X, y, cfg.get('modeling'))
    
    df_metrics = evaluate_model(y_true, y_pred, cfg.get('evaluate'))
    
    print(df_metrics)

if __name__ == '__main__':
    main()