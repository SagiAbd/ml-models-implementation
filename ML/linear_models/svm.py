import numpy as np
import pandas as pd
import random

class MySVM:
    def __init__(
        self, 
        n_iter=10,
        learning_rate=0.001,
        weights=None,
        b=None,
        metric=None,
        C=1, # soft margin regularization constant
        sgd_sample=None, # int or float (0.0, 1.0),
        random_state=42
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.b = b
        self.metric = metric
        self.C = C
        self.sgd_sample = sgd_sample
        self.random_state = random_state
    
    def __str__(self):
        return f"MySVM class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(
        self, 
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = False,
    ) -> None:
        random.seed(self.random_state)
        
        # Convert DataFrame to NumPy arrays
        X = X.to_numpy()
        y = y.to_numpy()
        
        # Convert y to -1 and 1, if needed
        y = np.where(y == 0, -1, y)
        
        # Initialize weights and margin
        if self.weights is None:
            self.weights = np.ones(X.shape[1])
        if self.b is None:
            self.b = 1
        
        for iter_ in range(1, self.n_iter + 1):
            # Get batch of data (for SGD or full batch)
            X_batch, y_batch = self._get_batches(X, y)
            
            for i in range(len(X_batch)):
                xi = X_batch[i]
                yi = y_batch[i]

                margin = yi * (np.dot(self.weights, xi) + self.b)
                if margin >= 1:
                    grad_w = 2 * self.weights
                    grad_b = 0
                else:
                    grad_w = 2 * self.weights - self.C * yi * xi
                    grad_b = - self.C * yi

                self.weights -= self.learning_rate * grad_w
                self.b -= self.learning_rate * grad_b
            
            # loss
            margins = y * (X @ self.weights + self.b)
            hinge_losses = np.maximum(0, 1 - margins)
            loss_svm = np.linalg.norm(self.weights) ** 2 + self.C * np.mean(hinge_losses)
            
            # Logging
            if verbose and iter_ % verbose == 0:
                if self.metric is None:
                    print(f"{iter_} | loss: {loss_svm}")
                else:
                    print(f"{iter_} | loss: {loss_svm} | {self.metric}: {self.best_score}")
    
    def get_coef(self) -> tuple:
        return (self.weights, self.b)
    
    def predict(self, X: pd.DataFrame) -> np.array:
        X = X.to_numpy()
        
        pred = np.sign(np.dot(X, self.weights) + self.b)
        # Convert pred to 0 and 1
        pred = np.where(pred == -1, 0, 1)
        
        return pred

    def _get_batches(self, X, y):
        # Return full batch if no sampling requested
        if self.sgd_sample is None or self.sgd_sample >= X.shape[0]:
            return X, y
        else:
            # Compute sample size for mini-batch SGD
            if isinstance(self.sgd_sample, int):
                sample_size = self.sgd_sample
            elif isinstance(self.sgd_sample, float):
                sample_size = int(self.sgd_sample * X.shape[0])

            sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
            return X[sample_rows_idx], y[sample_rows_idx]
    