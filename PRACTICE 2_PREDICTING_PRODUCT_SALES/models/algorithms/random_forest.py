"""
Thuật toán: Random Forest Regressor (From Scratch thuần túy với Bagging & Random Subspace)
Dự án: Practice 2: Predicting Product Sales — Group 3
Cài đặt: Thuần túy bằng Python và NumPy.
Kế thừa: BaseEstimator, RegressorMixin.
Đặc tả chi tiết lý thuyết toán học: docs/description/algorithms/random_forest_regressor.md
"""
from typing import Optional, Union, Dict, Any, List
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from .decision_tree import DecisionTreeRegressorScratch


class RandomForestRegressorScratch(BaseEstimator, RegressorMixin):
    """
    Rừng ngẫu nhiên hồi quy tự cài đặt bằng NumPy thuần túy.
    Áp dụng học kết hợp Bagging (Bootstrap Aggregation) và Không gian đặc trưng ngẫu nhiên (Random Subspace).
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
        """Trả về tham số mô hình."""
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
        """Cập nhật tham số mô hình."""
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
        """Huấn luyện tập hợp n_estimators cây quyết định hồi quy."""
        X_mat = self._to_dense(X)
        y_vec = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_mat.shape

        self.n_features_in_ = n_features
        self.trees_: List[DecisionTreeRegressorScratch] = []
        self.rng_ = np.random.RandomState(self.random_state)

        importances_sum = np.zeros(n_features, dtype=np.float64)

        for i in range(self.n_estimators):
            tree_seed = self.rng_.randint(0, 1_000_000)

            # Lấy mẫu có hoàn lại (Bootstrap)
            if self.bootstrap:
                boot_idx = self.rng_.choice(n_samples, size=n_samples, replace=True)
                X_b, y_b = X_mat[boot_idx], y_vec[boot_idx]
            else:
                X_b, y_b = X_mat, y_vec

            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                n_quantiles=self.n_quantiles,
                random_state=tree_seed
            )
            tree.fit(X_b, y_b)
            self.trees_.append(tree)

            importances_sum += tree.feature_importances_

        # Tính feature importances trung bình của cả rừng
        tot = np.sum(importances_sum)
        if tot > 0:
            self.feature_importances_ = importances_sum / tot
        else:
            self.feature_importances_ = np.ones(n_features, dtype=np.float64) / n_features

        return self

    def predict(self, X) -> np.ndarray:
        """Dự đoán giá trị doanh số thực tế bằng trung bình cộng các cây."""
        X_mat = self._to_dense(X)
        if not hasattr(self, "trees_") or len(self.trees_) == 0:
            raise ValueError("Mô hình chưa được fit!")

        all_preds = np.array([tree.predict(X_mat) for tree in self.trees_])
        return np.mean(all_preds, axis=0)

    def score(self, X, y) -> float:
        """Tính hệ số xác định R2."""
        y_true = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - (ss_res / ss_tot))
