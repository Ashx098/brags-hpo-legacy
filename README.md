# BRAGS: Bias-Reduced Adaptive Grid Search

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-Alpha-orange.svg)]()

**BRAGS** is a smart, resource-efficient hyperparameter tuning tool built on top of Grid and Random Search with adaptive pruning. It reduces wasted trials and accelerates model search without sacrificing performance.

---

## 🔍 What is BRAGS?

BRAGS introduces **bias-reduced pruning** into traditional search loops.  
It tracks the best score so far and **skips configurations** that fall below a specified threshold.

> 🧠 Smarter than Random  
> ⚡ Faster than GridSearchCV  
> ✅ Works with or without cross-validation

---

## ⚙️ Core Algorithm

1. Loop through all hyperparameter combinations.
2. Train a model for each configuration.
3. Score it using either a validation split or cross-validation.
4. If score is:
   - **better than current best** → update best
   - **significantly worse** → skip using threshold
5. Train final model with the best parameters.

---

## 🧪 Syntax

```python
from brags import BRAGSGridSearch
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

param_grid = {
    'hidden_layer_sizes': [(32,), (64,)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001]
}

searcher = BRAGSGridSearch(
    model_class=MLPClassifier,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=0,  # Set >0 to enable cross-validation
    prune_threshold=0.95,
    random_state=42
)

searcher.fit(X_train, y_train, X_test, y_test)

print("Best Params:", searcher.best_params_)
print("Best Score:", searcher.best_score_)
```

---------

## 📌 Key Features

- ✅ Supports both **Grid** and **Random** search strategies
- ✅ **Adaptive pruning**: skips bad trials early
- ✅ Configurable **prune threshold** (e.g. 95% of best score)
- ✅ Optional **cross-validation**
- ✅ Tracks and logs scores for each trial

---
## 📦 Installation

Coming soon via PyPI

For now, clone the repo:
```
git clone https://github.com/avinashh-ai/brags-hpo.git
cd brags-hpo
pip install -e .
```

---

## 📖 License

This project is licensed under the [MIT License](LICENSE).

© 2025 Avinash Mynampati

---
## Made with ❤️ by Avinash Mynampati
