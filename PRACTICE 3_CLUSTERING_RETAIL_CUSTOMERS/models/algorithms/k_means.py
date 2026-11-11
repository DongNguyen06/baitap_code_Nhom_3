"""
Thuật toán: K-Means Clustering (From Scratch thuần túy bằng NumPy)
Dự án: Practice 3: Clustering Retail Customers — Group 3
Cài đặt: Thuần túy bằng Python và NumPy (không dùng thư viện ngoài cho bước tính toán phân cụm cốt lõi).
Kế thừa: BaseEstimator, ClusterMixin (tương thích hoàn toàn với scikit-learn Pipeline & API).
Đặc tả chi tiết lý thuyết toán học: docs/description/algorithms/k_means_clustering.md
"""

from typing import Optional, Dict, Any, Union
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class KMeansClustering(BaseEstimator, ClusterMixin):
    """
    Thuật toán gom cụm K-Means tự cài đặt từ đầu (From Scratch).
    
    Tham số:
    ---------
    n_clusters : int, default=5
        Số lượng cụm K cần phân chia.
    init : str, default='k-means++'
        Phương thức khởi tạo trọng tâm ban đầu: 'k-means++' hoặc 'random'.
    max_iter : int, default=300
        Số vòng lặp cập nhật tối đa.
    tol : float, default=1e-4
        Ngưỡng dung sai dừng lặp khi độ dịch chuyển của trọng tâm nhỏ hơn giá trị này.
    random_state : int or None, default=42
        Seed cố định tính toán ngẫu nhiên để tái lặp kết quả.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        init: str = "k-means++",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = 42
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Thuộc tính trạng thái sau khi fit
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0
        self.n_iter_: int = 0
        self.n_features_in_: int = 0

    def _init_centroids(self, X: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Khởi tạo K tâm cụm theo cơ chế k-means++ hoặc random."""
        n_samples, n_features = X.shape

        if self.init == "random":
            random_indices = rng.choice(n_samples, size=self.n_clusters, replace=False)
            return X[random_indices].copy().astype(np.float64)

        elif self.init == "k-means++":
            centroids = np.empty((self.n_clusters, n_features), dtype=np.float64)
            # 1. Chọn tâm đầu tiên ngẫu nhiên
            first_idx = rng.randint(0, n_samples)
            centroids[0] = X[first_idx]

            # 2. Chọn các tâm tiếp theo với xác suất tỷ lệ với D(x)^2
            for c_idx in range(1, self.n_clusters):
                # Tính khoảng cách bình phương nhỏ nhất từ mỗi điểm đến các tâm đã chọn
                current_centroids = centroids[:c_idx]
                # Shape: (n_samples, c_idx)
                dist_matrix = np.sum((X[:, np.newaxis, :] - current_centroids[np.newaxis, :, :]) ** 2, axis=2)
                min_sq_dists = np.min(dist_matrix, axis=1)

                sum_dists = np.sum(min_sq_dists)
                if sum_dists > 0:
                    probs = min_sq_dists / sum_dists
                    next_idx = rng.choice(n_samples, p=probs)
                else:
                    next_idx = rng.randint(0, n_samples)
                centroids[c_idx] = X[next_idx]

            return centroids
        else:
            raise ValueError(f"Phương thức khởi tạo '{self.init}' không hợp lệ (chỉ hỗ trợ 'k-means++' hoặc 'random').")

    def _compute_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Tính ma trận bình phương khoảng cách Euclidean giữa các mẫu dữ liệu và tâm cụm.
        Output shape: (n_samples, n_clusters)
        """
        # Vectorized Euclidean squared distance: ||x - mu||^2
        return np.sum((X[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)

    def fit(self, X: Union[np.ndarray, Any], y: Optional[Any] = None) -> "KMeansClustering":
        """
        Huấn luyện thuật toán K-Means trên tập dữ liệu X.
        
        Tham số:
        ---------
        X : array-like shape (n_samples, n_features)
            Ma trận đặc trưng số đã chuẩn hóa.
        y : Bỏ qua (đảm bảo tương thích với chuẩn scikit-learn API).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_arr.shape

        if self.n_clusters > n_samples:
            raise ValueError(f"Số cụm n_clusters={self.n_clusters} không được vượt quá số mẫu n_samples={n_samples}.")

        self.n_features_in_ = n_features
        rng = np.random.RandomState(self.random_state)

        # 1. Khởi tạo tâm cụm ban đầu
        centroids = self._init_centroids(X_arr, rng)
        labels = np.zeros(n_samples, dtype=np.int64)

        # 2. Vòng lặp tối ưu hóa Expectation-Maximization
        for iteration in range(1, self.max_iter + 1):
            self.n_iter_ = iteration

            # Bước Gán nhãn (Assignment / Expectation): Tìm tâm gần nhất cho mỗi điểm
            dist_sq = self._compute_distances(X_arr, centroids)
            new_labels = np.argmin(dist_sq, axis=1)

            # Bước Cập nhật (Update / Maximization): Tính tâm mới bằng trung bình các điểm trong cụm
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                cluster_members = X_arr[new_labels == k]
                if len(cluster_members) > 0:
                    new_centroids[k] = np.mean(cluster_members, axis=0)
                else:
                    # Xử lý cụm rỗng: Gán lại tại điểm có khoảng cách xa nhất so với tâm của nó
                    farthest_idx = np.argmax(np.min(dist_sq, axis=1))
                    new_centroids[k] = X_arr[farthest_idx]

            # Kiểm tra điều kiện hội tụ qua khoảng cách dịch chuyển của tâm
            center_shift = np.sum(np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)))
            centroids = new_centroids
            labels = new_labels

            if center_shift < self.tol:
                break

        # Lưu lại kết quả sau cùng
        self.cluster_centers_ = centroids
        self.labels_ = labels

        # Tính hàm mất mát Inertia (tổng bình phương khoảng cách tới tâm gần nhất)
        final_dist_sq = self._compute_distances(X_arr, self.cluster_centers_)
        self.inertia_ = float(np.sum(np.min(final_dist_sq, axis=1)))

        return self

    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Dự đoán nhãn cụm cho các mẫu mới dựa trên các tâm cụm đã học.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Mô hình chưa được huấn luyện. Vui lòng gọi fit() trước khi predict().")
        X_arr = np.asarray(X, dtype=np.float64)
        dist_sq = self._compute_distances(X_arr, self.cluster_centers_)
        return np.argmin(dist_sq, axis=1)

    def fit_predict(self, X: Union[np.ndarray, Any], y: Optional[Any] = None) -> np.ndarray:
        """Thực hiện fit và trả về nhãn phân cụm trực tiếp."""
        return self.fit(X, y).labels_

    def transform(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Chuyển đổi dữ liệu X sang không gian khoảng cách tới từng tâm cụm.
        Output shape: (n_samples, n_clusters)
        """
        if self.cluster_centers_ is None:
            raise ValueError("Mô hình chưa được huấn luyện. Vui lòng gọi fit() trước khi transform().")
        X_arr = np.asarray(X, dtype=np.float64)
        return np.sqrt(self._compute_distances(X_arr, self.cluster_centers_))

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Lấy các tham số cấu hình của mô hình."""
        return {
            "n_clusters": self.n_clusters,
            "init": self.init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state
        }

    def set_params(self, **params: Any) -> "KMeansClustering":
        """Thiết lập các tham số cho mô hình."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
