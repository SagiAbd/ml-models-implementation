import pandas as pd
import numpy as np
import random


class MyLogReg:
    def __init__(self,
                 weights=None,
                 n_iter=10,
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
        return f"MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)

        # Convert input dataframes to numpy arrays
        X = X.to_numpy()
        y = y.to_numpy()

        # Add bias term (column of 1s)
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        
        if self.weights is None:
            self.weights = np.ones(X.shape[1])

        for i in range(1, self.n_iter + 1):
            X_batch, y_batch = self.get_batches(X, y)

            # Compute predictions and loss
            y_pred_batch = 1 / (1 + np.exp(-np.dot(X_batch, self.weights)))
            log_loss = -np.mean(y_batch * np.log(y_pred_batch + 1e-15) + (1 - y_batch) * np.log(1 - y_pred_batch + 1e-15))

            # Compute gradient of the loss
            grad = 1 / X_batch.shape[0] * np.dot((y_pred_batch - y_batch), X_batch)

            # Apply regularization to loss and gradient
            if self.reg == 'l1':
                log_loss += self.l1_coef * np.sum(np.abs(self.weights))
                grad += self.l1_coef * np.sign(self.weights)
            elif self.reg == 'l2':
                log_loss += self.l2_coef * np.sum((self.weights) ** 2)
                grad += self.l2_coef * 2 * self.weights
            elif self.reg == 'elasticnet':
                log_loss += self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum((self.weights) ** 2)
                grad += self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights

            # Update weights using gradient descent
            self.weights = self.weights - self.calc_lr(i) * grad

            # Compute evaluation metric if specified
            if self.metric:
                self.best_score = self.calculate_metric(X, y)

            # Logging if verbose is set
            if verbose and i % verbose == 0:
                if self.metric is None:
                    print(f"{i} | loss: {log_loss}")
                else:
                    print(f"{i} | loss: {log_loss} | {self.metric}: {self.best_score}")
    
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
    
    def predict_proba(self, X):
        # Convert X to numpy array if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        # Check if X shape matches the expected input
        if X.shape[1] != self.weights.shape[0]:
            X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        
        return (1 / (1 + np.exp(-np.dot(X, self.weights))))
    
    def predict(self, X):
        prob_arr = self.predict_proba(X)

        binary = (prob_arr > 0.5).astype(int)
        return binary

    def get_best_score(self):
        return self.best_score

    def calculate_metric(self, X, y):
        # Compute chosen evaluation metric
        if self.metric is None:
            raise ValueError("No metric specified for calculation.")
        if self.metric == "accuracy":
            return self.accuracy(X, y)  
        elif self.metric == "precision":
            return self.precision(X, y)
        elif self.metric == "recall":
            return self.recall(X, y)
        elif self.metric == "f1":
            return self.f1(X, y)    
        elif self.metric == "roc_auc":
            return self.roc_auc(X, y)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def get_coef(self):
        # Return weights without bias term
        return self.weights[1:]

    # =========================
    # Metric Implementations
    # =========================
    def _get_confusion_matrix(self, X, y):
        y_pred_bin = self.predict(X)
        tp = sum((y_pred_bin == 1) & (y == 1))
        tn = sum((y_pred_bin == 0) & (y == 0))
        fp = sum((y_pred_bin == 1) & (y == 0))
        fn = sum((y_pred_bin == 0) & (y == 1))

        return tp, tn, fp, fn
    
    def accuracy(self, X, y):
        tp, tn, fp, fn = self._get_confusion_matrix(X, y)
        return (tp + tn) / (tp + tn + fp + fn)

    def precision(self, X, y):
        tp, tn, fp, fn = self._get_confusion_matrix(X, y)
        return tp / (tp + fp)
    
    def recall(self, X, y):
        tp, tn, fp, fn = self._get_confusion_matrix(X, y)
        return tp / (tp + fn)

    def f1(self, X, y):
        tp, tn, fp, fn = self._get_confusion_matrix(X, y)
        prec = self.precision(X, y)
        rec = self.recall(X, y)
        return 2 * prec * rec / (prec + rec + 1e-15)  # Adding small constant to avoid division by zero

    def roc_auc(self, X, y):
        y_pred_prob = self.predict_proba(X)
        y = y.to_numpy() if isinstance(y, pd.Series) else y
        
        # Create a DataFrame of predictions and labels
        df = pd.DataFrame({'prob': y_pred_prob, 'label': y})
        
        # Rank the predictions
        df['rank'] = df['prob'].rank(method='average', ascending=True)
        
        # Sum of ranks for positive class
        sum_ranks_pos = df[df['label'] == 1]['rank'].sum()
        
        P = sum(y == 1)
        N = sum(y == 0)

        # AUC formula based on ranks
        auc = (sum_ranks_pos - (P * (P + 1)) / 2) / (P * N)
        return auc





        