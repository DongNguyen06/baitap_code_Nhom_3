"""
Thuật toán: Hierarchical Agglomerative Clustering (From Scratch thuần túy bằng NumPy)
Dự án: Practice 3: Clustering Retail Customers — Group 3
Cài đặt: Thuần túy bằng Python và NumPy (xuất ma trận Linkage Matrix tương thích chuẩn Scipy Dendrogram).
Kế thừa: BaseEstimator, ClusterMixin (tương thích hoàn toàn với scikit-learn API).
Đặc tả chi tiết lý thuyết toán học: docs/description/algorithms/hierarchical_clustering.md
"""

from typing import Optional, Dict, Any, Union, List
import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class HierarchicalClustering(BaseEstimator, ClusterMixin):
    """
    Thuật toán gom cụm phân cấp thứ bậc tích tụ (Bottom-up Agglomerative Clustering) tự cài đặt từ đầu.
    
    Tham số:
    ---------
    n_clusters : int, default=5
        Số lượng cụm K cần cắt lát sau khi hoàn thành quá trình tích tụ.
    linkage : str, default='ward'
        Tiêu chí tính khoảng cách gộp giữa hai cụm: 'ward', 'complete', hoặc 'average'.
    metric : str, default='euclidean'
        Độ đo khoảng cách giữa các điểm ('euclidean').
    """

    def __init__(
        self,
        n_clusters: int = 5,
        linkage: str = "ward",
        metric: str = "euclidean"
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric

        # Thuộc tính trạng thái sau khi fit
        self.labels_: Optional[np.ndarray] = None
        self.linkage_matrix_: Optional[np.ndarray] = None
        self.children_: Optional[np.ndarray] = None
        self.distances_: Optional[np.ndarray] = None
        self.n_features_in_: int = 0
        self.n_clusters_: int = n_clusters

    def _compute_cluster_distance(
        self,
        c1_info: Dict[str, Any],
        c2_info: Dict[str, Any],
        point_dist_matrix: np.ndarray
    ) -> float:
        """Tính khoảng cách liên kết giữa 2 cụm theo tiêu chí đã chọn."""
        if self.linkage == "ward":
            # Tiêu chí Ward: sqrt(2 * |A| * |B| / (|A| + |B|)) * ||mu_A - mu_B||_2
            size1 = c1_info["size"]
            size2 = c2_info["size"]
            centroid_diff = c1_info["centroid"] - c2_info["centroid"]
            dist_centers = float(np.sqrt(np.sum(centroid_diff ** 2)))
            weight = np.sqrt(2.0 * size1 * size2 / (size1 + size2))
            return float(weight * dist_centers)

        elif self.linkage == "complete":
            # Khoảng cách lớn nhất giữa mọi cặp điểm
            m1 = c1_info["members"]
            m2 = c2_info["members"]
            sub_dist = point_dist_matrix[np.ix_(m1, m2)]
            return float(np.max(sub_dist))

        elif self.linkage == "average":
            # Khoảng cách trung bình giữa mọi cặp điểm
            m1 = c1_info["members"]
            m2 = c2_info["members"]
            sub_dist = point_dist_matrix[np.ix_(m1, m2)]
            return float(np.mean(sub_dist))

        else:
            raise ValueError(f"Tiêu chí liên kết '{self.linkage}' không được hỗ trợ (chỉ dùng 'ward', 'complete', 'average').")

    def fit(self, X: Union[np.ndarray, Any], y: Optional[Any] = None) -> "HierarchicalClustering":
        """
        Xây dựng cây gom cụm thứ bậc và cắt ở n_clusters cụm.
        
        Tham số:
        ---------
        X : array-like shape (n_samples, n_features)
            Ma trận đặc trưng số đã chuẩn hóa.
        y : Bỏ qua (đảm bảo tương thích scikit-learn API).
        """
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        if self.n_clusters > n_samples or self.n_clusters < 1:
            raise ValueError(f"n_clusters={self.n_clusters} không hợp lệ cho tập dữ liệu có {n_samples} mẫu.")

        # 1. Tính toán trước ma trận khoảng cách giữa các cặp điểm
        diff = X_arr[:, np.newaxis, :] - X_arr[np.newaxis, :, :]
        point_dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

        # 2. Khởi tạo N cụm đơn lẻ ban đầu
        active_clusters: Dict[int, Dict[str, Any]] = {}
        for i in range(n_samples):
            active_clusters[i] = {
                "id": i,
                "members": [i],
                "size": 1,
                "centroid": X_arr[i].copy()
            }

        linkage_rows: List[List[float]] = []
        labels_snapshot: Optional[np.ndarray] = None

        # Nếu n_clusters == n_samples (mỗi mẫu 1 cụm)
        if self.n_clusters == n_samples:
            labels_snapshot = np.arange(n_samples, dtype=np.int64)

        # 3. Vòng lặp tích tụ N - 1 bước gộp
        next_cluster_id = n_samples
        for step in range(n_samples - 1):
            cluster_keys = list(active_clusters.keys())
            num_active = len(cluster_keys)

            # Kiểm tra thời điểm cắt cây để lưu lại nhãn phân cụm K
            if num_active == self.n_clusters and labels_snapshot is None:
                lbls = np.zeros(n_samples, dtype=np.int64)
                for cluster_idx, c_id in enumerate(cluster_keys):
                    for member_idx in active_clusters[c_id]["members"]:
                        lbls[member_idx] = cluster_idx
                labels_snapshot = lbls

            # Tìm cặp cụm có khoảng cách liên kết nhỏ nhất
            min_dist = float("inf")
            best_pair = (-1, -1)

            for i in range(num_active):
                c1_id = cluster_keys[i]
                c1_info = active_clusters[c1_id]
                for j in range(i + 1, num_active):
                    c2_id = cluster_keys[j]
                    c2_info = active_clusters[c2_id]
                    dist = self._compute_cluster_distance(c1_info, c2_info, point_dist_matrix)
                    if dist < min_dist:
                        min_dist = dist
                        best_pair = (c1_id, c2_id)

            u_id, v_id = best_pair
            u_info = active_clusters[u_id]
            v_info = active_clusters[v_id]

            # Tạo cụm mới gộp từ u và v
            new_members = u_info["members"] + v_info["members"]
            new_size = len(new_members)
            new_centroid = np.mean(X_arr[new_members], axis=0)

            # Ghi nhận thông tin vào dòng Linkage Matrix: [c1, c2, dist, size]
            # Đảm bảo c1 < c2 theo quy ước chuẩn của Scipy
            c1_rec, c2_rec = (u_id, v_id) if u_id < v_id else (v_id, u_id)
            linkage_rows.append([float(c1_rec), float(c2_rec), float(min_dist), float(new_size)])

            # Cập nhật danh sách cụm đang hoạt động
            del active_clusters[u_id]
            del active_clusters[v_id]
            active_clusters[next_cluster_id] = {
                "id": next_cluster_id,
                "members": new_members,
                "size": new_size,
                "centroid": new_centroid
            }
            next_cluster_id += 1

        # Nếu n_clusters == 1 (toàn bộ gộp làm 1)
        if labels_snapshot is None:
            labels_snapshot = np.zeros(n_samples, dtype=np.int64)

        self.labels_ = labels_snapshot
        self.linkage_matrix_ = np.array(linkage_rows, dtype=np.float64)
        self.children_ = self.linkage_matrix_[:, :2].astype(np.int64)
        self.distances_ = self.linkage_matrix_[:, 2].astype(np.float64)

        return self

    def fit_predict(self, X: Union[np.ndarray, Any], y: Optional[Any] = None) -> np.ndarray:
        """Thực hiện gom cụm và trả về mảng nhãn."""
        return self.fit(X, y).labels_

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Lấy các tham số cấu hình của mô hình."""
        return {
            "n_clusters": self.n_clusters,
            "linkage": self.linkage,
            "metric": self.metric
        }

    def set_params(self, **params: Any) -> "HierarchicalClustering":
        """Thiết lập các tham số cho mô hình."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
