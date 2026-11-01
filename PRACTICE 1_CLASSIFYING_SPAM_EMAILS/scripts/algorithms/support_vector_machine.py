"""
Thuật toán: Linear Support Vector Machine (From Scratch)
Đặc tả chi tiết: docs/algorithms/support_vector_machine_classifier.md
"""
import numpy as np
from scipy import sparse


class LinearSVMScratch:
    """Class Linear Support Vector Machine tự cài đặt thuần túy bằng NumPy với Hinge Loss và phạt L2."""

    def __init__(self, learning_rate: float = 0.01, lambda_param: float = 0.01, n_iters: int = 300):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0.0
        self.losses = []

    def decision_function(self, X) -> np.ndarray:
        return (X.dot(self.weights) if sparse.issparse(X) else np.dot(X, self.weights)) + self.bias

    def fit(self, X, y):
        m, n = X.shape
        y_signed = np.where(np.asarray(y) <= 0, -1.0, 1.0)  # Ánh xạ {0, 1} sang {-1, +1}
        self.weights = np.zeros(n, dtype=np.float64)
        self.bias = 0.0
        self.losses = []

        for _ in range(self.n_iters):
            scores = self.decision_function(X)
            margins = y_signed * scores
            violate = (margins < 1.0)  # Support Vectors vi phạm lề (lấn làn)

            # Subgradient: Phạt các điểm vi phạm + Phanh kéo lề L2
            grad_w = self.lambda_param * self.weights - (X.T.dot(y_signed * violate) if sparse.issparse(X) else np.dot(X.T, y_signed * violate)) / m
            grad_b = -float(np.mean(y_signed * violate))

            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b

            hinge_loss = np.mean(np.maximum(0, 1.0 - margins)) + 0.5 * self.lambda_param * np.sum(self.weights ** 2)
            self.losses.append(float(hinge_loss))
        return self

    def predict(self, X) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0.0, 1, 0)
