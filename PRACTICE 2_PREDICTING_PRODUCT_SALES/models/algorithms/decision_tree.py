"""
Thuật toán: Decision Tree Regressor (From Scratch thuần túy với Quantile Splits & Giảm Phương sai MSE)
Dự án: Practice 2: Predicting Product Sales — Group 3
Cài đặt: Thuần túy bằng Python và NumPy.
Kế thừa: BaseEstimator, RegressorMixin.
Đặc tả chi tiết lý thuyết toán học: docs/description/algorithms/decision_tree_regressor.md
"""
from typing import Optional, Union, Dict, Any, List
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin


class TreeNode:
    """Nút trong cây quyết định hồi quy."""

    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['TreeNode'] = None,
        right: Optional['TreeNode'] = None,
        value: Optional[float] = None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTreeRegressorScratch(BaseEstimator, RegressorMixin):
    """
    Cây quyết định hồi quy tự cài đặt bằng NumPy thuần túy.
    Tối ưu nhát cắt dựa trên Giảm Phương sai (Variance Reduction / MSE)
    kết hợp Phân vị (Quantile Splits) để đạt hiệu năng tính toán cao trên tập dữ liệu lớn.
    """

    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: Optional[Union[int, float, str]] = None,
        n_quantiles: int = 20,
        random_state: int = 42
    ):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.n_quantiles = int(n_quantiles)
        self.random_state = int(random_state)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Trả về tham số mô hình."""
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
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
        """Huấn luyện cây quyết định hồi quy."""
        X_mat = self._to_dense(X)
        y_vec = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X_mat.shape

        self.n_features_in_ = n_features
        self.feature_importances_ = np.zeros(n_features, dtype=np.float64)
        self.rng_ = np.random.RandomState(self.random_state)

        self.root_ = self._build_tree(X_mat, y_vec, depth=0)

        # Chuẩn hóa feature importances về tổng = 1.0
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp

        return self

    def _get_feature_indices(self, total_features: int) -> np.ndarray:
        """Chọn tập thuộc tính con cho từng nút (hỗ trợ random subspace)."""
        if self.max_features is None:
            return np.arange(total_features)
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(total_features)))
            return self.rng_.choice(total_features, size=k, replace=False)
        elif self.max_features == "log2":
            k = max(1, int(np.log2(total_features)))
            return self.rng_.choice(total_features, size=k, replace=False)
        elif isinstance(self.max_features, float) and 0 < self.max_features <= 1.0:
            k = max(1, int(self.max_features * total_features))
            return self.rng_.choice(total_features, size=k, replace=False)
        elif isinstance(self.max_features, (int, np.integer)):
            k = min(total_features, max(1, int(self.max_features)))
            return self.rng_.choice(total_features, size=k, replace=False)
        else:
            return np.arange(total_features)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        n_samples, n_features = X.shape
        leaf_val = float(np.mean(y)) if n_samples > 0 else 0.0

        # Điều kiện dừng
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.all(y == y[0])
        ):
            return TreeNode(value=leaf_val)

        feat_indices = self._get_feature_indices(n_features)
        sum_total = float(np.sum(y))
        node_score = (sum_total ** 2) / n_samples

        best_score = -1e18
        best_feat = None
        best_thresh = None

        for feat in feat_indices:
            col = X[:, feat]
            # Quantile candidate thresholds
            if len(np.unique(col)) <= self.n_quantiles:
                candidates = np.unique(col)
            else:
                qs = np.linspace(0.05, 0.95, self.n_quantiles)
                candidates = np.unique(np.quantile(col, qs))

            for t in candidates:
                mask_left = col <= t
                n_l = int(np.count_nonzero(mask_left))
                n_r = n_samples - n_l

                if n_l < self.min_samples_leaf or n_r < self.min_samples_leaf:
                    continue

                sum_l = float(np.sum(y[mask_left]))
                sum_r = sum_total - sum_l
                score = (sum_l ** 2 / n_l) + (sum_r ** 2 / n_r)

                if score > best_score:
                    best_score = score
                    best_feat = feat
                    best_thresh = t

        # Không tìm được nhát cắt hợp lệ
        if best_feat is None:
            return TreeNode(value=leaf_val)

        # Ghi nhận mức giảm phương sai cho feature_importances_
        reduction = best_score - node_score
        if reduction > 0:
            self.feature_importances_[best_feat] += reduction

        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return TreeNode(
            feature=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            value=leaf_val
        )

    def _predict_row(self, x: np.ndarray, node: TreeNode) -> float:
        curr = node
        while not curr.is_leaf:
            if x[curr.feature] <= curr.threshold:
                curr = curr.left
            else:
                curr = curr.right
        return curr.value

    def predict(self, X) -> np.ndarray:
        """Dự đoán giá trị doanh số thực tế y_hat."""
        X_mat = self._to_dense(X)
        return np.array([self._predict_row(x, self.root_) for x in X_mat], dtype=np.float64)

    def score(self, X, y) -> float:
        """Tính hệ số xác định R2."""
        y_true = np.asarray(y, dtype=np.float64).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1.0 - (ss_res / ss_tot))
