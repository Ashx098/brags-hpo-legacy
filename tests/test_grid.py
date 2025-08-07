# tests/test_grid.py
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from brags import BragsGridSearch

def test_brags_runs_and_returns_results():
    X, y = load_digits(return_X_y=True)

    param_grid = {
        'hidden_layer_sizes': [(64,), (128,)],
        'activation': ['relu'],
        'alpha': [0.0001],
    }

    model = MLPClassifier(max_iter=200, random_state=42)

    search = BragsGridSearch(estimator=model, param_grid=param_grid, cv=3, threshold=0.9)
    search.fit(X, y)

    assert search.best_params_ is not None
    assert search.best_score_ > 0.8
    assert len(search.results_) > 0
    assert isinstance(search.results_[0], dict)