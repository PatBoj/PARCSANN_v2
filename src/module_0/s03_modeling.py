import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def get_acc(X, y, cfg):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42)

    classifier = MLPRegressor(
        hidden_layer_sizes=cfg.get('hidden_layer_sizes'), 
        activation=cfg.get('activation'), 
        solver='adam', 
        random_state=42)

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    abs_error = np.abs(y_test - y_pred)
    rel_error = abs_error / y_test
    
    avg_abs_error = np.mean(abs_error)
    avg_rel_error = np.mean(rel_error)
    
    print(f'''
        average absoulte error: {np.round(avg_abs_error, 2)}, 
        and mean realative error: {np.round(avg_rel_error * 100, 2)}%
        ''')