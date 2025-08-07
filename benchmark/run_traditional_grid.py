# benchmark/run_traditional_grid.py
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import time
import json

X, y = load_digits(return_X_y=True)

param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 64)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001]
}

model = MLPClassifier(max_iter=1000, random_state=42)

grid = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=0)

start = time.time()
grid.fit(X, y)
end = time.time()

result = {
    "method": "Traditional GridSearchCV",
    "best_params": grid.best_params_,
    "best_score": grid.best_score_,
    "total_time": round(end - start, 3)
}

with open("benchmark/traditional_results.json", "w") as f:
    json.dump(result, f, indent=4)

print("✅ Traditional Grid done.")
