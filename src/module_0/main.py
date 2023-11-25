from s01_read_yaml import get_config
from s02_prepare_data import prepare_input_output
from s03_use_monocores import apply_monocore_dictionary
from s04_modeling import get_test_model
from s05_evaluate import evaluate_model


def main():
    cfg = get_config()
    
    X, y, output_titles = prepare_input_output(cfg=cfg.get('prepare_data'))

    X = apply_monocore_dictionary(input_data=X, cfg=cfg.get('use_monocores'))

    X_test, y_test, model = get_test_model(X=X, y=y, cfg=cfg.get('modeling'))
    
    df_metrics = evaluate_model(
        y_true=y_test,
        y_pred=model.predict(X_test),
        output_titles=output_titles,
        cfg=cfg.get('evaluate'))
    
    print(df_metrics)


if __name__ == '__main__':
    main()