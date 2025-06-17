import numpy as np
import pandas as pd

class MyKNNReg:
    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform"):
        self.k = k
        self.train_size = None
        self.metric = metric
        self.weight = weight
    
    def __str__(self):
        return f"MyKNNReg class: k={self.k}"
    
    def fit(self,
            X: pd.DataFrame,
            y: pd.Series
            ) -> None:
        self.X_train = X.to_numpy()
        self.y_train = y.to_numpy()
        self.train_size = self.X_train.shape # (number of samples, number of features)
    
    def _predict_base(self, X_row: np.ndarray) -> float:
        # Euclidean distance from X_row to all X_train
        dists = self._calculate_dist(X_row, self.X_train, self.metric)
        sorted_k_indices = np.argsort(dists)[:self.k]
        
        # All sorted distances and labels
        sorted_dists = dists[sorted_k_indices]
        sorted_labels = self.y_train[sorted_k_indices]
        sorted_rank = np.arange(1, len(sorted_dists) + 1)
        
        if self.weight == 'uniform':
            return np.mean(sorted_labels)
        elif self.weight == 'rank':
            # Add rank weights: 1 / rank
            inv_rank = 1 / sorted_rank
            weights = inv_rank / np.sum(inv_rank)

            return np.dot(weights, sorted_labels)
        elif self.weight == "distance":
            # Add dist weights: 1 / dist
            inv_dists = 1 / (sorted_dists + 1e-12)
            weights = inv_dists / np.sum(inv_dists)

            return np.dot(weights, sorted_labels)

    def predict(self, X_pred: pd.DataFrame) -> np.ndarray:
        X_pred = X_pred.to_numpy()
        return np.array([self._predict_base(row) for row in X_pred])
    
    def _calculate_dist(self, x1: np.ndarray, x2: np.ndarray, metric: str) -> np.ndarray:
        if metric == "euclidean":
            return self._calc_euclidean(x1, x2)
        elif metric == "manhattan":
            return self._calc_manhattan(x1, x2)
        elif metric == "cosine":
            return self._calc_cosine(x1, x2)
        elif metric == "chebyshev":
            return self._calc_chebyshev(x1, x2)
        else:
            raise ValueError(f"Unknown metric: {metric}.")
            
    def _calc_euclidean(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x2 - x1, axis=1)
    
    def _calc_manhattan(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(x2 - x1), axis=1)
        
    def _calc_cosine(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        # Normalize x1 and x2
        x1_norm = np.linalg.norm(x1)
        x2_norms = np.linalg.norm(x2, axis=1)
        dot_products = x2 @ x1
        return 1 - (dot_products / (x2_norms * x1_norm + 1e-10))  # small epsilon to prevent division by zero
        
    def _calc_chebyshev(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.max(np.abs(x2 - x1), axis=1)