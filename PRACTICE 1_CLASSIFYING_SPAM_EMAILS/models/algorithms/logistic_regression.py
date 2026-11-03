"""
Thuật toán: Logistic Regression (From Scratch thuần túy với phạt L2)
Dự án: Practice 1: Classifying Spam Emails — Group 3
Cài đặt: 100% Python và NumPy thuần túy (Không kế thừa và không import từ Scikit-learn).
Đặc tả chi tiết: docs/description/algorithms/logistic_regression.md
"""
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegressionScratch(ClassifierMixin, BaseEstimator):
    """Class Logistic Regression tự cài đặt thuần túy bằng NumPy với phạt chuẩn hóa L2."""

    def __init__(self, learning_rate: float = 0.2, n_iters: int = 150, l2: float = 1e-4, threshold: float = 0.5):
        self.learning_rate = float(learning_rate)
        self.n_iters = int(n_iters)
        self.l2 = float(l2)
        self.threshold = float(threshold)

    def get_params(self, deep: bool = True) -> dict:
        """Hàm tự viết trả về tham số mô hình (Python thuần túy)."""
        return {
            'learning_rate': self.learning_rate,
            'n_iters': self.n_iters,
            'l2': self.l2,
            'threshold': self.threshold
        }

    def set_params(self, **params):
        """Hàm tự viết cập nhật tham số mô hình (Python thuần túy)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _sigmoid(self, z):
        """Hàm kích hoạt Sigmoid kẹp khoảng ổn định số học tránh tràn số: 1 / (1 + e^-z)"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250.0, 250.0)))

    def _linear_predict(self, X):
        """Tích vô hướng ma trận trọng số và cộng độ lệch bias: z = X.w + b"""
        return (X.dot(self.weights_) if sparse.issparse(X) else np.dot(X, self.weights_)) + self.bias_

    def fit(self, X, y):
        """
        Huấn luyện mô hình Logistic Regression bằng thuật toán Gradient Descent:
        1. Tính xác suất dự đoán p = sigmoid(X.w + b)
        2. Tính sai số err = p - y và Gradient có phạt L2
        3. Cập nhật trọng số w và bias b theo learning_rate
        """
        self.classes_ = np.unique(y)
        m, n = X.shape
        y_arr = np.asarray(y, dtype=np.float64)
        self.weights_ = np.zeros(n, dtype=np.float64)
        self.bias_ = 0.0
        self.losses_ = []

        for _ in range(self.n_iters):
            # 1. Tính xác suất qua Sigmoid
            p = self._sigmoid(self._linear_predict(X))

            # 2. Đo sai số & Gradient (kèm phạt L2 chống Overfitting)
            err = p - y_arr
            grad_w = (X.T.dot(err) if sparse.issparse(X) else np.dot(X.T, err)) / m + self.l2 * self.weights_
            grad_b = float(np.mean(err))

            # 3. Cập nhật trọng số theo Gradient Descent
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b

            # 4. Ghi nhận Log-Loss
            eps = 1e-15
            p_safe = np.clip(p, eps, 1.0 - eps)
            loss = -np.mean(y_arr * np.log(p_safe) + (1.0 - y_arr) * np.log(1.0 - p_safe)) + 0.5 * self.l2 * np.sum(self.weights_ ** 2)
            self.losses_.append(float(loss))

        # Trọng số phục vụ trích xuất đặc trưng
        self.coef_ = self.weights_.reshape(1, -1)
        self.intercept_ = np.array([self.bias_])
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Dự đoán xác suất 2 lớp [P(Ham), P(Spam)]."""
        p1 = self._sigmoid(self._linear_predict(X))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        """Dự đoán nhãn nhị phân theo ngưỡng phân loại threshold."""
        p1 = self._sigmoid(self._linear_predict(X))
        return np.where(p1 >= self.threshold, 1, 0)

    def score(self, X, y) -> float:
        """Đo độ chính xác Accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))
