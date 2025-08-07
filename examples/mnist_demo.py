from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from brags import BragsGridSearch

# Load dataset
X, y = load_digits(return_X_y=True)

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(64,), (128,), (64, 64)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001],
}

# Initialize BRAGS search
brags = BragsGridSearch(
    estimator=MLPClassifier(max_iter=1000),
    param_grid=param_grid,
    cv=3,
    threshold=0.95,
    max_train_time=5,
    max_gpu_pct=90,
    scoring='accuracy'
)

# Run fit
brags.fit(X, y)

# Results
print("\n✅ BEST PARAMS:", brags.best_params_)
print("✅ BEST ACCURACY:", brags.best_score_)

print("\n📊 ALL TRIAL RESULTS:")
for r in brags.results_:
    print(f"- {r['params']} | Score: {r['mean_score']:.4f} | Time: {r['train_time']:.2f}s | GPU: {r['gpu_pct']}%")
