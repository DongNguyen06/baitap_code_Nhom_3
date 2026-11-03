"""
Thuật toán: Decision Tree Regressor (From Scratch thuần túy với Quantile Splits & Giảm Phương sai MSE)
Dự án: Practice 2: Predicting Product Sales — Group 3
Phân công: Cả 7 thành viên cùng tham gia nghiên cứu cơ sở lý thuyết toán học và cài đặt bằng NumPy thuần túy.
Kế thừa: BaseEstimator, RegressorMixin.
Đặc tả chi tiết lý thuyết toán học: docs/description/algorithms/decision_tree_regressor.md
"""
from typing import Optional, Union, Dict, Any, List
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score


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
    kết hợp Phân vị (Quantile Splits) để đạt hiệu năng tính toán cao.

    Nhiệm vụ chung: Cả 7 thành viên cùng tham gia hoàn thiện.
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
        """Trả về tham số mô hình tương thích scikit-learn."""
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
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
        Huấn luyện Cây quyết định hồi quy bằng thuật toán đệ quy xây dựng cây:
        1. Tại mỗi nút, kiểm tra điều kiện dừng: depth >= max_depth, n_samples < min_samples_split
        2. Nếu dừng, tạo nút lá lưu giá trị trung bình value = mean(y)
        3. Nếu chưa dừng, quét các đặc trưng và ngưỡng phân vị (quantiles) để tìm điểm cắt tối ưu
           giảm phương sai MSE lớn nhất:
           Delta_MSE = Var(y) - [ (N_L/N) * Var(y_L) + (N_R/N) * Var(y_R) ]
        4. Chia dữ liệu thành nhánh trái và nhánh phải, đệ quy xây dựng cây con
        5. Tính toán trọng số đặc trưng quan trọng (self.feature_importances_)

        Nhiệm vụ chung: Cả 7 thành viên cùng tham gia hoàn thiện.
        """
        # TODO: [Cả 7 thành viên] Lập trình thuật toán Decision Tree Regressor From Scratch:
        # Xem tài liệu đặc tả toán học tại: docs/description/algorithms/decision_tree_regressor.md
        raise NotImplementedError(
            "Chưa cài đặt - Cả 7 thành viên cần cùng hoàn thiện thuật toán DecisionTreeRegressorScratch theo docs/description/algorithms/decision_tree_regressor.md"
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
        """
        Dự đoán giá trị mục tiêu cho từng mẫu trong X.
        Hỗ trợ suy luận từ cây đã lưu (self.root_) phục vụ demo, hoặc yêu cầu cài đặt khi khởi tạo mới.
        """
        if hasattr(self, "root_") and self.root_ is not None:
            X_mat = self._to_dense(X)
            return np.array([self._predict_row(x, self.root_) for x in X_mat], dtype=np.float64)

        # TODO: [Cả 7 thành viên] Lập trình hàm dự đoán duyệt cây:
        raise NotImplementedError(
            "Chưa cài đặt - Cả 7 thành viên cần cùng hoàn thiện hàm predict của DecisionTreeRegressorScratch"
        )

    def score(self, X, y) -> float:
        """Hệ số xác định R2 score."""
        y_pred = self.predict(X)
        y_true = np.asarray(y, dtype=np.float64).ravel()
        return float(r2_score(y_true, y_pred))
