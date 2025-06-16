import numpy as np
import pandas as pd
import random
from typing import Union


class MySVM:
    """
    A simple implementation of a linear Support Vector Machine (SVM) using 
    stochastic gradient descent (SGD) and hinge loss.

    Parameters:
    - n_iter: Number of iterations for training
    - learning_rate: Step size for SGD updates
    - weights: Optional initial weights (will be initialized if None)
    - b: Optional initial bias (will be initialized if None)
    - metric: Metric name for evaluation logging (optional)
    - C: Regularization constant for soft-margin SVM
    - sgd_sample: Number or fraction of samples for SGD batch updates
    - random_state: Random seed for reproducibility
    """

    def __init__(
        self, 
        n_iter: int = 10,
        learning_rate: float = 0.001,
        weights: np.ndarray = None,
        b: float = None,
        metric: str = None,
        C: float = 1,
        sgd_sample: Union[int, float, None] = None,
        random_state: int = 42
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self._weights = weights
        self._b = b
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
        verbose: Union[bool, int] = False,
    ) -> None:
        """
        Train the SVM model on given data.

        Parameters:
        - X: Features as a pandas DataFrame
        - y: Target labels (0 or 1) as a pandas Series
        - verbose: If int > 0, prints loss every `verbose` iterations
        """
        random.seed(self.random_state)
        
        # Convert to NumPy arrays
        X = X.to_numpy()
        y = y.to_numpy()

        # Convert labels to -1 and 1
        y = np.where(y == 0, -1, y)

        # Initialize weights and bias if not provided
        if self._weights is None:
            self._weights = np.ones(X.shape[1])
        if self._b is None:
            self._b = 1

        for iter_ in range(1, self.n_iter + 1):
            # Get minibatch or full data
            X_batch, y_batch = self._get_batches(X, y)
            
            for i in range(len(X_batch)):
                xi = X_batch[i]
                yi = y_batch[i]

                # Calculate margin
                margin = yi * (np.dot(self._weights, xi) + self._b)

                # Compute gradients
                if margin >= 1:
                    grad_w = 2 * self._weights
                    grad_b = 0
                else:
                    grad_w = 2 * self._weights - self.C * yi * xi
                    grad_b = - self.C * yi

                # Update weights and bias
                self._weights -= self.learning_rate * grad_w
                self._b -= self.learning_rate * grad_b
            
            # Compute hinge loss
            margins = y * (X @ self._weights + self._b)
            hinge_losses = np.maximum(0, 1 - margins)
            loss_svm = np.linalg.norm(self._weights) ** 2 + self.C * np.mean(hinge_losses)

            # Logging
            if verbose and isinstance(verbose, int) and iter_ % verbose == 0:
                print(f"{iter_} | loss: {loss_svm}")

    def get_coef(self) -> tuple:
        """
        Returns the learned weight vector and bias term.
        """
        return (self._weights, self._b)

    def predict(self, X: pd.DataFrame) -> np.array:
        """
        Predict labels for given features.

        Parameters:
        - X: Input features as a DataFrame

        Returns:
        - np.array: Predicted labels (0 or 1)
        """
        X = X.to_numpy()
        pred = np.sign(np.dot(X, self._weights) + self._b)

        # Convert -1 to 0
        pred = np.where(pred == -1, 0, 1)
        return pred

    def _get_batches(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Returns a mini-batch of the data based on `sgd_sample`.

        Returns:
        - Tuple of (X_batch, y_batch)
        """
        if self.sgd_sample is None or self.sgd_sample >= X.shape[0]:
            return X, y
        else:
            # Determine sample size
            if isinstance(self.sgd_sample, int):
                sample_size = self.sgd_sample
            elif isinstance(self.sgd_sample, float):
                sample_size = int(self.sgd_sample * X.shape[0])

            sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
            return X[sample_rows_idx], y[sample_rows_idx]
