"""
Thuật toán: DBSCAN Clustering (From Scratch thuần túy bằng NumPy)
Dự án: Practice 3: Clustering Retail Customers — Group 3
Cài đặt: Thuần túy bằng Python và NumPy (sử dụng hàng đợi BFS để lan truyền mật độ cụm).
Kế thừa: BaseEstimator, ClusterMixin (tương thích hoàn toàn với scikit-learn API).
Đặc tả chi tiết lý thuyết toán học: docs/description/algorithms/dbscan_clustering.md
"""

from collections import deque
from typing import Optional, Dict, Any, Union, List
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class DBSCANClustering(BaseEstimator, ClusterMixin):
    """
    Thuật toán gom cụm dựa trên mật độ DBSCAN tự cài đặt từ đầu (From Scratch).
    
    Tham số:
    ---------
    eps : float, default=0.5
        Bán kính lân cận cực đại giữa hai điểm để được coi là láng giềng.
    min_samples : int, default=5
        Số lượng điểm lân cận tối thiểu trong bán kính eps để một điểm trở thành Điểm lõi (Core point).
    metric : str, default='euclidean'
        Độ đo khoảng cách sử dụng ('euclidean' hoặc 'manhattan').
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean"
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        # Thuộc tính trạng thái sau khi fit
        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        self.n_clusters_: int = 0
        self.n_noise_: int = 0

    def _compute_pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Tính ma trận khoảng cách giữa mọi cặp điểm dữ liệu (Vectorized cao tốc).
        Output shape: (n_samples, n_samples)
        """
        if self.metric == "euclidean":
            # Khoảng cách Euclidean: sqrt(sum((x_i - x_j)^2))
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            return np.sqrt(np.sum(diff ** 2, axis=2))
        elif self.metric == "manhattan":
            # Khoảng cách Manhattan: sum(|x_i - x_j|)
            return np.sum(np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]), axis=2)
        else:
            raise ValueError(f"Metric '{self.metric}' không được hỗ trợ (chỉ dùng 'euclidean' hoặc 'manhattan').")

    def fit(self, X: Union[np.ndarray, Any], y: Optional[Any] = None) -> "DBSCANClustering":
        """
        Huấn luyện thuật toán DBSCAN và gán nhãn phân cụm cho tập dữ liệu X.
        
        Tham số:
        ---------
        X : array-like shape (n_samples, n_features)
            Ma trận đặc trưng số đã chuẩn hóa.
        y : Bỏ qua (đảm bảo tương thích chuẩn scikit-learn API).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        # 1. Tính toán ma trận khoảng cách giữa mọi cặp mẫu
        dist_matrix = self._compute_pairwise_distances(X_arr)

        # 2. Xác định danh sách láng giềng epsilon cho từng điểm
        # neighbors[i] chứa mảng các chỉ số j thỏa mãn dist(i, j) <= eps
        neighbors: List[np.ndarray] = [
            np.where(dist_matrix[i] <= self.eps)[0] for i in range(n_samples)
        ]

        # 3. Phân loại Điểm lõi (Core points): Có ít nhất min_samples láng giềng
        is_core = np.array([len(neighbors[i]) >= self.min_samples for i in range(n_samples)], dtype=bool)
        self.core_sample_indices_ = np.where(is_core)[0]
        self.components_ = X_arr[self.core_sample_indices_].copy()

        # 4. Mở rộng cụm (Cluster Expansion) qua hàng đợi BFS
        # Khởi tạo nhãn: -1 đại diện cho điểm chưa gán cụm hoặc nhiễu
        labels = np.full(n_samples, -1, dtype=np.int64)
        cluster_id = 0

        for i in range(n_samples):
            # Chỉ bắt đầu lan truyền cụm mới từ điểm lõi chưa được phân cụm
            if not is_core[i] or labels[i] != -1:
                continue

            # Gán điểm lõi vào cụm mới
            labels[i] = cluster_id
            
            # Hàng đợi duyệt BFS mở rộng cụm
            queue = deque()
            for neighbor_idx in neighbors[i]:
                if neighbor_idx != i:
                    queue.append(neighbor_idx)

            while queue:
                current_point = queue.popleft()

                # Nếu điểm hiện tại là nhiễu, gán vào cụm (trở thành điểm biên Border point)
                if labels[current_point] == -1:
                    labels[current_point] = cluster_id

                    # Nếu điểm hiện tại cũng là điểm lõi, đưa thêm các láng giềng của nó vào hàng đợi
                    if is_core[current_point]:
                        for n_idx in neighbors[current_point]:
                            if labels[n_idx] == -1:
                                queue.append(n_idx)

            # Đã mở rộng trọn vẹn cụm mật độ, chuyển sang cụm tiếp theo
            cluster_id += 1

        self.labels_ = labels
        self.n_clusters_ = cluster_id
        self.n_noise_ = int(np.sum(labels == -1))

        return self

    def fit_predict(self, X: Union[np.ndarray, Any], y: Optional[Any] = None) -> np.ndarray:
        """Thực hiện gom cụm và trả về mảng nhãn (bao gồm -1 cho điểm nhiễu)."""
        return self.fit(X, y).labels_

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Lấy các tham số cấu hình của mô hình."""
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": self.metric
        }

    def set_params(self, **params: Any) -> "DBSCANClustering":
        """Thiết lập các tham số cho mô hình."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
