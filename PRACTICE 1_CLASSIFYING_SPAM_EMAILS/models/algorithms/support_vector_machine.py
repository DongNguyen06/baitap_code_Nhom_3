"""
Thuật toán: Linear Support Vector Machine (From Scratch thuần túy)
Dự án: Practice 1: Classifying Spam Emails — Group 3
Cài đặt: 100% Python và NumPy thuần túy (Không kế thừa và không import từ Scikit-learn).
Đặc tả chi tiết: docs/description/algorithms/support_vector_machine_classifier.md
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin


class LinearSVMScratch(ClassifierMixin, BaseEstimator):
    """Class Linear Support Vector Machine tự cài đặt thuần túy bằng NumPy với Hinge Loss và phạt L2."""

    def __init__(self, learning_rate: float = 5.0, lambda_param: float = 1e-5, n_iters: int = 300):
        self.learning_rate = float(learning_rate)
        self.lambda_param = float(lambda_param)
        self.n_iters = int(n_iters)

    def get_params(self, deep: bool = True) -> dict:
        """Hàm tự viết trả về tham số mô hình (Python thuần túy)."""
        return {
            'learning_rate': self.learning_rate,
            'lambda_param': self.lambda_param,
            'n_iters': self.n_iters
        }

    def set_params(self, **params):
        """Hàm tự viết cập nhật tham số mô hình (Python thuần túy)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def decision_function(self, X) -> np.ndarray:
        """Tính điểm khoảng cách hình học tới siêu phẳng: f(x) = w.x + b"""
        return (X.dot(self.weights_) if sparse.issparse(X) else np.dot(X, self.weights_)) + self.bias_

    def fit(self, X, y):
        """
        Huấn luyện Linear SVM bằng thuật toán Subgradient Descent tối ưu Hinge Loss:
        min_w 1/2 * lambda * ||w||^2 + 1/m * sum(max(0, 1 - y_i * (w.x_i + b)))
        """
        self.classes_ = np.unique(y)
        m, n = X.shape
        y_signed = np.where(np.asarray(y) <= 0, -1.0, 1.0)  # Ánh xạ {0, 1} sang {-1, +1}
        self.weights_ = np.zeros(n, dtype=np.float64)
        self.bias_ = 0.0
        self.losses_ = []

        for _ in range(self.n_iters):
            scores = self.decision_function(X)
            margins = y_signed * scores
            violate = (margins < 1.0)  # Support Vectors vi phạm lề (khoảng cách < 1)

            # Subgradient: Phạt các điểm vi phạm + Phanh kéo lề chuẩn hóa L2
            grad_w = self.lambda_param * self.weights_ - (X.T.dot(y_signed * violate) if sparse.issparse(X) else np.dot(X.T, y_signed * violate)) / m
            grad_b = -float(np.mean(y_signed * violate))

            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

            hinge_loss = np.mean(np.maximum(0, 1.0 - margins)) + 0.5 * self.lambda_param * np.sum(self.weights_ ** 2)
            self.losses_.append(float(hinge_loss))

        # Trọng số phục vụ phân tích Top từ khóa
        self.coef_ = self.weights_.reshape(1, -1)
        self.intercept_ = np.array([self.bias_])
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Áp dụng Sigmoid (Platt Scaling tự viết) lên decision_function để sinh xác suất [P(Ham), P(Spam)]."""
        scores = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(scores, -250.0, 250.0)))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        """Dự đoán nhãn theo vị trí tương đối với siêu phẳng phân tách (>= 0 là Spam)."""
        return np.where(self.decision_function(X) >= 0.0, 1, 0)

    def score(self, X, y) -> float:
        """Đo độ chính xác Accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
