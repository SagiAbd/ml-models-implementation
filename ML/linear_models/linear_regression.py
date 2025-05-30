import pandas as pd
import numpy as np
import random


class MyLineReg:
    def __init__(self,
                 weights=None,
                 n_iter=100,
                 learning_rate=0.1,
                 metric=None,
                 reg=None,
                 l1_coef=0,
                 l2_coef=0,
                 sgd_sample=None,  # integer or float for batch sampling
                 random_state=42):
        # Initialize hyperparameters and attributes
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric
        self.best_score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state
    
    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)

        # Convert input dataframes to numpy arrays
        X = X.to_numpy()
        y = y.to_numpy()

        # Add bias term (column of 1s)
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        self.weights = np.ones(X.shape[1])  # Initialize weights

        for i in range(1, self.n_iter + 1):
            X_batch, y_batch = self.get_batches(X, y)

            # Compute predictions and loss
            y_pred_batch = np.dot(X_batch, self.weights)
            mse_loss = np.mean((y_pred_batch - y_batch) ** 2)

            # Compute gradient of the loss
            grad = 2 / X_batch.shape[0] * np.dot((y_pred_batch - y_batch), X_batch)

            # Apply regularization to loss and gradient
            if self.reg == 'l1':
                mse_loss += self.l1_coef * np.sum(np.abs(self.weights))
                grad += self.l1_coef * np.sign(self.weights)
            elif self.reg == 'l2':
                mse_loss += self.l2_coef * np.sum((self.weights) ** 2)
                grad += self.l2_coef * 2 * self.weights
            elif self.reg == 'elasticnet':
                mse_loss += self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum((self.weights) ** 2)
                grad += self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights

            # Update weights using gradient descent
            self.weights = self.weights - self.calc_lr(i) * grad

            # Compute evaluation metric if specified
            if self.metric:
                self.best_score = self.calculate_metric(X, y)

            # Logging if verbose is set
            if verbose and i % verbose == 0:
                if self.metric is None:
                    print(f"{i} | loss: {mse_loss}")
                else:
                    print(f"{i} | loss: {mse_loss} | {self.metric}: {self.best_score}")
    
    def get_batches(self, X, y):
        # Return full dataset if no SGD sampling specified
        if self.sgd_sample is None or self.sgd_sample >= X.shape[0]:
            return X, y
        else:
            # Determine sample size for SGD
            if isinstance(self.sgd_sample, int):
                sample_size = self.sgd_sample
            elif isinstance(self.sgd_sample, float):
                sample_size = int(self.sgd_sample * X.shape[0])

            sample_rows_idx = random.sample(range(X.shape[0]), sample_size)
            return X[sample_rows_idx], y[sample_rows_idx]

    def calc_lr(self, iter):
        # Support for dynamic or constant learning rate
        if callable(self.learning_rate):
            try:
                return self.learning_rate(iter)
            except TypeError:
                return self.learning_rate
        return self.learning_rate

    def predict(self, X):
        # Predict using trained weights
        X = X.to_numpy()
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        return np.dot(X, self.weights)

    def get_best_score(self):
        return self.best_score

    def calculate_metric(self, X, y):
        # Compute chosen evaluation metric
        if self.metric is None:
            raise ValueError("No metric specified for calculation.")
        if self.metric == "mae":
            return self.mae(X, y)
        elif self.metric == "mse":
            return self.mse(X, y)
        elif self.metric == "rmse":
            return self.rmse(X, y)
        elif self.metric == "mape":
            return self.mape(X, y)
        elif self.metric == "r2":
            return self.r2(X, y)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def get_coef(self):
        # Return weights without bias term
        return self.weights[1:]

    # =========================
    # Metric Implementations
    # =========================
    
    def mae(self, X, y):
        y_pred = np.dot(X, self.weights)
        return np.mean(np.abs(y_pred - y))

    def mse(self, X, y):
        y_pred = np.dot(X, self.weights)
        return np.mean((y_pred - y) ** 2)

    def rmse(self, X, y):
        return np.sqrt(self.mse(X, y))

    def mape(self, X, y):
        y_pred = np.dot(X, self.weights)
        mask = y != 0
        if not np.any(mask):
            raise ValueError("All y values are zero — MAPE undefined.")
        return np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100

    def r2(self, X, y):
        y_pred = np.dot(X, self.weights)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
