"""
Thuật toán: Random Forest Regressor (From Scratch thuần túy với Bagging & Random Subspace)
Dự án: Practice 2: Predicting Product Sales — Group 3
Phân công: Cả 7 thành viên cùng tham gia nghiên cứu cơ sở lý thuyết toán học và cài đặt bằng NumPy thuần túy.
Kế thừa: BaseEstimator, RegressorMixin.
Đặc tả chi tiết lý thuyết toán học: docs/description/algorithms/random_forest_regressor.md
"""
from typing import Optional, Union, Dict, Any, List
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score
from .decision_tree import DecisionTreeRegressorScratch


class RandomForestRegressorScratch(BaseEstimator, RegressorMixin):
    """
    Rừng ngẫu nhiên hồi quy tự cài đặt bằng NumPy thuần túy.
    Áp dụng học kết hợp Bagging (Bootstrap Aggregation) và Không gian đặc trưng ngẫu nhiên (Random Subspace).

    Nhiệm vụ chung: Cả 7 thành viên cùng tham gia hoàn thiện.
    """

    def __init__(
        self,
        n_estimators: int = 20,
        max_depth: int = 8,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: Union[str, float, int] = "sqrt",
        bootstrap: bool = True,
        n_quantiles: int = 15,
        random_state: int = 42
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.bootstrap = bool(bootstrap)
        self.n_quantiles = int(n_quantiles)
        self.random_state = int(random_state)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Trả về tham số mô hình tương thích scikit-learn."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "n_quantiles": self.n_quantiles,
            "random_state": self.random_state
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
        Huấn luyện Rừng ngẫu nhiên hồi quy:
        1. Khởi tạo danh sách self.trees_ = []
        2. Với mỗi cây t = 1..n_estimators:
           a. Lấy mẫu bootstrap có hoàn lại n_samples dòng từ (X, y) (nếu bootstrap=True)
           b. Khởi tạo cây con DecisionTreeRegressorScratch với max_features = sqrt(n_features)
           c. Huấn luyện cây con trên mẫu bootstrap
           d. Thêm cây vào self.trees_
        3. Tính toán trọng số đặc trưng trung bình của toàn rừng (self.feature_importances_)

        Nhiệm vụ chung: Cả 7 thành viên cùng tham gia hoàn thiện.
        """
        # TODO: [Cả 7 thành viên] Lập trình thuật toán Random Forest Regressor From Scratch:
        # Xem tài liệu đặc tả toán học tại: docs/description/algorithms/random_forest_regressor.md
        raise NotImplementedError(
            "Chưa cài đặt - Cả 7 thành viên cần cùng hoàn thiện thuật toán RandomForestRegressorScratch theo docs/description/algorithms/random_forest_regressor.md"
        )

    def predict(self, X):
        """
        Dự đoán đầu ra bằng cách lấy trung bình cộng dự đoán từ toàn bộ các cây con trong rừng:
        y_pred = (1 / n_estimators) * sum(tree.predict(X) for tree in trees_)

        Nhiệm vụ chung: Cả 7 thành viên cùng tham gia hoàn thiện.
        """
        if hasattr(self, "trees_") and self.trees_:
            X_mat = self._to_dense(X)
            preds = np.array([tree.predict(X_mat) for tree in self.trees_])
            return np.mean(preds, axis=0)

        # TODO: [Cả 7 thành viên] Lập trình hàm dự đoán tổng hợp (Bagging aggregation):
        raise NotImplementedError(
            "Chưa cài đặt - Cả 7 thành viên cần cùng hoàn thiện hàm predict của RandomForestRegressorScratch"
        )

    def score(self, X, y):
        """Hệ số xác định R2 score."""
        y_pred = self.predict(X)
        y_true = np.asarray(y, dtype=np.float64).ravel()
        return float(r2_score(y_true, y_pred))
