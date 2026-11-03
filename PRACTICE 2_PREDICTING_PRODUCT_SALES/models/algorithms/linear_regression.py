"""
Thuật toán: Linear Regression (From Scratch thuần túy với Normal Equation và Gradient Descent L2)
Dự án: Practice 2: Predicting Product Sales — Group 3
Cài đặt: Thuần túy bằng Python và NumPy.
Kế thừa: BaseEstimator, RegressorMixin (tương thích hoàn toàn với scikit-learn Pipeline & GridSearchCV).
Đặc tả chi tiết lý thuyết toán học: docs/description/algorithms/linear_regression.md
"""
from typing import Optional, Dict, Any
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin


class LinearRegressionScratch(BaseEstimator, RegressorMixin):
    """
    Mô hình Hồi quy Tuyến tính tự cài đặt thuần túy bằng NumPy.
    Hỗ trợ 2 phương pháp giải:
    - 'normal_equation': Nghiệm đóng OLS kèm điều chuẩn ổn định Ridge L2: theta = (X^T X + lambda I)^(-1) X^T y
    - 'gradient_descent': Tối ưu hóa hạ độ dốc lặp: theta := theta - alpha * grad
    """

    def __init__(
        self,
        method: str = "normal_equation",
        learning_rate: float = 0.01,
        n_iters: int = 1000,
        l2: float = 1e-4,
        fit_intercept: bool = True
    ):
        self.method = str(method)
        self.learning_rate = float(learning_rate)
        self.n_iters = int(n_iters)
        self.l2 = float(l2)
        self.fit_intercept = bool(fit_intercept)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Trả về tham số mô hình tương thích scikit-learn."""
        return {
            "method": self.method,
            "learning_rate": self.learning_rate,
            "n_iters": self.n_iters,
            "l2": self.l2,
            "fit_intercept": self.fit_intercept
        }

    def set_params(self, **params):
        """Cập nhật tham số mô hình tương thích scikit-learn."""
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _to_dense(self, X):
        """Chuyển đổi dữ liệu sang ma trận numpy float64 2D."""
        if sparse.issparse(X):
            X_mat = X.toarray().astype(np.float64)
        else:
            X_mat = np.asarray(X, dtype=np.float64)
        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(1, -1)
        return X_mat

    def fit(self, X, y):
        """
        Huấn luyện mô hình Linear Regression bằng NumPy.
        """
        X_mat = self._to_dense(X)
        y_vec = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_mat.shape

        if self.method == "normal_equation":
            if self.fit_intercept:
                # Ghép cột số 1 vào đầu ma trận: X_b = [1, X]
                ones = np.ones((n_samples, 1), dtype=np.float64)
                X_b = np.hstack([ones, X_mat])
                dim = n_features + 1
                reg_matrix = self.l2 * np.eye(dim, dtype=np.float64)
                reg_matrix[0, 0] = 0.0  # Không phạt bias

                XtX = np.dot(X_b.T, X_b) + reg_matrix
                Xty = np.dot(X_b.T, y_vec)
                try:
                    theta = np.linalg.solve(XtX, Xty)
                except np.linalg.LinAlgError:
                    theta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

                self.intercept_ = float(theta[0])
                self.coef_ = theta[1:]
            else:
                reg_matrix = self.l2 * np.eye(n_features, dtype=np.float64)
                XtX = np.dot(X_mat.T, X_mat) + reg_matrix
                Xty = np.dot(X_mat.T, y_vec)
                try:
                    theta = np.linalg.solve(XtX, Xty)
                except np.linalg.LinAlgError:
                    theta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

                self.intercept_ = 0.0
                self.coef_ = theta

            self.losses_ = []

        elif self.method == "gradient_descent":
            w = np.zeros(n_features, dtype=np.float64)
            b = 0.0
            self.losses_ = []

            for _ in range(self.n_iters):
                y_pred = np.dot(X_mat, w) + (b if self.fit_intercept else 0.0)
                error = y_pred - y_vec

                grad_w = (np.dot(X_mat.T, error) / n_samples) + (self.l2 * w)
                w -= self.learning_rate * grad_w

                if self.fit_intercept:
                    grad_b = float(np.mean(error))
                    b -= self.learning_rate * grad_b

                mse_loss = float(np.mean(error ** 2) + 0.5 * self.l2 * np.sum(w ** 2))
                self.losses_.append(mse_loss)

            self.coef_ = w
            self.intercept_ = float(b) if self.fit_intercept else 0.0
        else:
            raise ValueError(f"Phương thức '{self.method}' không hợp lệ. Chọn 'normal_equation' hoặc 'gradient_descent'.")

        return self

    def predict(self, X) -> np.ndarray:
        """Dự đoán giá trị doanh số thực tế y_hat."""
        X_mat = self._to_dense(X)
        return np.dot(X_mat, self.coef_) + self.intercept_

    def score(self, X, y) -> float:
        """Tính hệ số xác định R2."""
        y_true = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - (ss_res / ss_tot))
