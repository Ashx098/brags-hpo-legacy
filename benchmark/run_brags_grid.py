# benchmark/run_brags_grid.py
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from brags import BragsGridSearch
import time
import json
import numpy as np

# Utility to recursively convert non-serializable numpy objects
def make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    else:
        return obj

# Load dataset
X, y = load_digits(return_X_y=True)

# Define param grid
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 64)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001]
}

# Define model
model = MLPClassifier(max_iter=1000, random_state=42)

# Run BRAGS search
search = BragsGridSearch(
    model,
    param_grid,
    cv=3,
    threshold=0.95,
    max_train_time=999,
    max_gpu_pct=100
)

start = time.time()
search.fit(X, y)
end = time.time()

# Prepare result dictionary
result = {
    "method": "BRAGSGridSearch",
    "best_params": search.best_params_,
    "best_score": search.best_score_,
    "total_time": round(end - start, 3),
    "all_trials": search.results_,
}

# Save as JSON (convert np types)
with open("benchmark/brags_results.json", "w") as f:
    json.dump(make_json_safe(result), f, indent=4)

# Print best result
print("✅ BRAGS Grid done.")
print("\n✅ BEST PARAMS:", search.best_params_)
print("✅ BEST ACCURACY:", search.best_score_)
