import numpy as np
import pandas as pd

class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform"):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.train_size = None
        self.X_train = None
        self.y_train = None
    
    def __str__(self):
        return f"MyKNNClf class: k={self.k}"

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series
            ) -> None:
        self.X_train = X.to_numpy()
        self.y_train = y.to_numpy()
        self.train_size = self.X_train.shape # (number of samples, number of features)
    
    def _predict_base(self, X_row: np.ndarray, pred_type: str = "bin") -> float:
        # distance from X_row to all X_train
        dists = self._calculate_dist(X_row, self.X_train, self.metric)
        sorted_k_indices = np.argsort(dists)[:self.k]
        k_labels = self.y_train[sorted_k_indices]
        
        # All sorted distances and labels
        sorted_dists = dists[sorted_k_indices]
        sorted_labels = self.y_train[sorted_k_indices]

        k_df = pd.DataFrame({
            "rank": np.arange(1, len(sorted_dists) + 1),
            "label": sorted_labels,
            "distance": sorted_dists
        })
        
        if self.weight == 'uniform':
            if pred_type == "bin":
                if k_labels.mean() >= 0.5:
                    return 1
                return 0
            elif pred_type == "proba":
                return k_labels.mean()
            else:
                raise ValueError(f"Unknown prediction type: {pred_type}.")
        elif self.weight == 'rank':
            # Add rank weights: 1 / rank
            k_df["inv_rank"] = 1 / k_df["rank"]

            # Weighted sum for label=1
            num = k_df.loc[k_df["label"] == 1, "inv_rank"].sum()
            denom = k_df["inv_rank"].sum()

            if pred_type == "bin":
                return 1 if num / denom >= 0.5 else 0
            elif pred_type == "proba":
                return num / denom
            else:
                raise ValueError(f"Unknown prediction type: {pred_type}.")
        elif self.weight == 'distance':
            # Add distance weights: 1 / rank
            k_df["inv_dist"] = 1 / k_df["distance"]

            # Weighted sum for label=1
            num = k_df.loc[k_df["label"] == 1, "inv_dist"].sum()
            denom = k_df["inv_dist"].sum()

            if pred_type == "bin":
                return 1 if num / denom >= 0.5 else 0
            elif pred_type == "proba":
                return num / denom
            else:
                raise ValueError(f"Unknown prediction type: {pred_type}.")
            

    def predict(self, X_pred: pd.DataFrame) -> np.ndarray:
        X_pred = X_pred.to_numpy()
        return np.array([self._predict_base(row, "bin") for row in X_pred])

    def predict_proba(self, X_pred: pd.DataFrame) -> np.ndarray:
        X_pred = X_pred.to_numpy()
        return np.array([self._predict_base(row, "proba") for row in X_pred])
           
    def _calculate_dist(self, x1: np.ndarray, x2: np.ndarray, metric: str) -> float:
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
   
