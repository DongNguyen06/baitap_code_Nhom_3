"""
Thuật toán: Linear Regression (From Scratch thuần túy với Normal Equation và Gradient Descent L2)
Dự án: Practice 2: Predicting Product Sales — Group 3
Phân công: Cả 7 thành viên cùng tham gia nghiên cứu cơ sở lý thuyết toán học và cài đặt bằng NumPy thuần túy.
Kế thừa: BaseEstimator, RegressorMixin (để tương thích hoàn toàn với scikit-learn Pipeline & GridSearchCV).
Đặc tả chi tiết lý thuyết toán học: docs/description/algorithms/linear_regression.md
"""
from typing import Optional, Dict, Any
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score


class LinearRegressionScratch(BaseEstimator, RegressorMixin):
    """
    Mô hình Hồi quy Tuyến tính tự cài đặt thuần túy bằng NumPy.
    Hỗ trợ 2 phương pháp giải:
    - 'normal_equation': Nghiệm đóng OLS kèm điều chuẩn Ridge L2: theta = (X^T X + lambda I)^(-1) X^T y
    - 'gradient_descent': Tối ưu hóa hạ độ dốc lặp: theta := theta - alpha * grad

    Nhiệm vụ chung: Cả 7 thành viên cùng thực hiện.
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

        Thuật toán Normal Equation:
        1. Ghép bias column 1 vào ma trận X: X_b = [1, X] (nếu fit_intercept=True)
        2. Tạo ma trận phạt L2: reg_matrix = l2 * I (với reg_matrix[0,0] = 0 để không phạt bias)
        3. Giải hệ phương trình tuyến tính: (X_b^T X_b + reg_matrix) * theta = X_b^T y
        4. Tách theta thành self.intercept_ = theta[0] và self.coef_ = theta[1:]

        Nhiệm vụ chung: Cả 7 thành viên cùng tham gia hoàn thiện.
        """
        # TODO: [Cả 7 thành viên] Lập trình thuật toán Linear Regression From Scratch:
        # Xem tài liệu đặc tả toán học tại: docs/description/algorithms/linear_regression.md
        raise NotImplementedError(
            "Chưa cài đặt - Cả 7 thành viên cần cùng hoàn thiện thuật toán LinearRegressionScratch theo docs/description/algorithms/linear_regression.md"
        )

    def predict(self, X):
        """
        Dự đoán đầu ra: y_pred = X @ coef_ + intercept_

        Nhiệm vụ chung: Cả 7 thành viên cùng tham gia hoàn thiện.
        """
        if hasattr(self, "coef_") and hasattr(self, "intercept_"):
            X_mat = self._to_dense(X)
            return X_mat @ self.coef_ + self.intercept_

        # TODO: [Cả 7 thành viên] Lập trình hàm dự đoán:
        raise NotImplementedError(
            "Chưa cài đặt - Cả 7 thành viên cần cùng hoàn thiện hàm predict của LinearRegressionScratch"
        )

    def score(self, X, y):
        """Hệ số xác định R2 score."""
        y_pred = self.predict(X)
        y_true = np.asarray(y, dtype=np.float64).ravel()
        return float(r2_score(y_true, y_pred))
