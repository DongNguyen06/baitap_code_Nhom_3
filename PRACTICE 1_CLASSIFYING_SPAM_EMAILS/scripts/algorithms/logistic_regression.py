"""
Thuật toán: Logistic Regression (From Scratch)
Đặc tả chi tiết: docs/algorithms/logistic_regression.md
"""
import numpy as np
from scipy import sparse


class LogisticRegressionScratch:
    """Class Logistic Regression tự cài đặt thuần túy bằng NumPy với phạt L2."""

    def __init__(self, learning_rate: float = 0.2, n_iters: int = 300, l2: float = 1e-4, threshold: float = 0.5):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.l2 = l2
        self.threshold = threshold
        self.weights = None
        self.bias = 0.0
        self.losses = []

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250.0, 250.0)))

    def _linear_predict(self, X):
        return (X.dot(self.weights) if sparse.issparse(X) else np.dot(X, self.weights)) + self.bias

    def fit(self, X, y):
        m, n = X.shape
        y = np.asarray(y, dtype=np.float64)
        self.weights = np.zeros(n, dtype=np.float64)
        self.bias = 0.0
        self.losses = []

        for _ in range(self.n_iters):
            # 1. Tính xác suất qua Sigmoid p = 1 / (1 + e^-(Xw + b))
            p = self._sigmoid(self._linear_predict(X))

            # 2. Đo sai số & Gradient (kèm phạt L2 chống Overfitting)
            err = p - y
            grad_w = (X.T.dot(err) if sparse.issparse(X) else np.dot(X.T, err)) / m + self.l2 * self.weights
            grad_b = float(np.mean(err))

            # 3. Cập nhật trọng số Gradient Descent
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            # 4. Ghi nhận Log-Loss
            eps = 1e-15
            p_safe = np.clip(p, eps, 1.0 - eps)
            loss = -np.mean(y * np.log(p_safe) + (1.0 - y) * np.log(1.0 - p_safe)) + 0.5 * self.l2 * np.sum(self.weights ** 2)
            self.losses.append(float(loss))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._sigmoid(self._linear_predict(X))

    def predict(self, X, threshold: float = None) -> np.ndarray:
        thresh = threshold if threshold is not None else self.threshold
        return np.where(self.predict_proba(X) >= thresh, 1, 0)
