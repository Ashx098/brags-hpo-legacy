import time
import numpy as np
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.metrics import accuracy_score
from .utils import get_gpu_utilization

class BragsGridSearch:
    def __init__(self, estimator, param_grid, cv=0, threshold=0.95,
                 max_train_time=60, max_gpu_pct=100, scoring='accuracy'):
        self.estimator = estimator
        self.param_grid = list(ParameterGrid(param_grid))
        self.cv = cv
        self.threshold = threshold
        self.max_train_time = max_train_time
        self.max_gpu_pct = max_gpu_pct
        self.scoring = scoring
        self.results_ = []
        self.best_params_ = None
        self.best_score_ = -np.inf

    def fit(self, X, y):
        for i, params in enumerate(self.param_grid):
            model = self.estimator.set_params(**params)
            start_time = time.time()

            try:
                if self.cv > 1:
                    scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
                    score = np.mean(scores)
                else:
                    model.fit(X, y)
                    score = model.score(X, y)
                    scores = [score]

                duration = time.time() - start_time
                gpu_util = get_gpu_utilization()

                if duration > self.max_train_time or gpu_util > self.max_gpu_pct:
                    continue

                if score > self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params

                if score < self.threshold * self.best_score_:
                    continue

                self.results_.append({
                    "params": params,
                    "mean_score": score,
                    "scores_per_fold": scores,
                    "train_time": duration,
                    "gpu_pct": gpu_util
                })

            except Exception as e:
                print(f"🚫 Skipping {params} due to error: {e}")
                continue
