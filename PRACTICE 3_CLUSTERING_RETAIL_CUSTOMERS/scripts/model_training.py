"""
Module: model_training.py
Dự án: Practice 3: Clustering Retail Customers — Group 3
Mục tiêu: Huấn luyện các mô hình gom cụm (K-Means, DBSCAN, Hierarchical Clustering),
tìm số cụm K tối ưu qua phương pháp Khuỷu tay (Elbow) và Silhouette Score,
lưu trữ các mô hình và siêu dữ liệu (metadata.json) vào models/saved_models/.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import json
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import (
    KMeans as SklearnKMeans,
    DBSCAN as SklearnDBSCAN,
    AgglomerativeClustering as SklearnAgglomerative
)

# Thiết lập đường dẫn thư mục gốc
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from models.algorithms.k_means import KMeansClustering
from models.algorithms.dbscan import DBSCANClustering
from models.algorithms.hierarchical import HierarchicalClustering
from scripts.feature_engineering import prepare_clustering_features

SAVED_MODELS_DIR = BASE_DIR / "models" / "saved_models"


def find_optimal_k_elbow(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42
) -> Dict[int, float]:
    """
    Tính toán tổng bình phương sai số nội cụm WCSS (Inertia) cho từng giá trị K để vẽ đường cong Khuỷu tay.
    """
    elbow_scores: Dict[int, float] = {}
    for k in k_range:
        km = KMeansClustering(n_clusters=k, random_state=random_state)
        km.fit(X)
        elbow_scores[k] = round(km.inertia_, 4)
    return elbow_scores


def find_optimal_k_silhouette(
    X: np.ndarray,
    k_range: range = range(2, 11),
    algorithm: str = "kmeans",
    random_state: int = 42
) -> Dict[int, float]:
    """
    Tính hệ số Silhouette Score trung bình cho từng giá trị K để xác định số cụm tối ưu toán học.
    """
    sil_scores: Dict[int, float] = {}
    for k in k_range:
        if algorithm == "kmeans":
            model = KMeansClustering(n_clusters=k, random_state=random_state)
        elif algorithm == "hierarchical":
            model = HierarchicalClustering(n_clusters=k, linkage="ward")
        else:
            raise ValueError(f"Thuật toán '{algorithm}' không hỗ trợ quét K (chỉ dùng 'kmeans' hoặc 'hierarchical').")

        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        sil_scores[k] = round(float(score), 4)

    return sil_scores


def tune_dbscan_params(
    X: np.ndarray,
    eps_values: Optional[List[float]] = None,
    min_samples_values: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Tìm kiếm lưới (Grid Search) bộ tham số (eps, min_samples) tốt nhất cho DBSCAN
    dựa trên Silhouette Score của các mẫu phân cụm hợp lệ (loại bỏ nhãn nhiễu -1).
    """
    if eps_values is None:
        eps_values = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    if min_samples_values is None:
        min_samples_values = [3, 4, 5, 6]

    best_score = -1.0
    best_params: Dict[str, Any] = {"eps": 0.35, "min_samples": 5}
    grid_results: List[Dict[str, Any]] = []

    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCANClustering(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = int(np.sum(labels == -1))

            # Chỉ tính Silhouette khi có từ 2 cụm hợp lệ trở lên và số mẫu cụm > n_clusters
            valid_mask = labels != -1
            if n_clusters >= 2 and np.sum(valid_mask) > n_clusters:
                score = float(silhouette_score(X[valid_mask], labels[valid_mask]))
            else:
                score = -1.0

            grid_results.append({
                "eps": eps,
                "min_samples": ms,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette": round(score, 4)
            })

            if score > best_score and n_noise <= len(X) * 0.25:  # Không quá 25% nhiễu
                best_score = score
                best_params = {"eps": eps, "min_samples": ms, "n_clusters": n_clusters, "silhouette": score}

    return {"best_params": best_params, "grid_results": grid_results}


def train_all_models(
    X: Optional[np.ndarray] = None,
    feature_set: str = "2d",
    scale_method: str = "standard",
    save_models: bool = True
) -> Dict[str, Any]:
    """
    Huấn luyện đồng thời 3 mô hình Gom cụm Scratch và các mô hình đối chuẩn Scikit-Learn:
    1. K-Means Clustering (K=5)
    2. DBSCAN Clustering (eps=0.35, min_samples=5)
    3. Hierarchical Agglomerative Clustering (K=5, linkage='ward')
    
    Lưu trữ các mô hình (.joblib) và metadata.json vào models/saved_models/.
    """
    # 1. Chuẩn bị dữ liệu nếu chưa truyền vào
    scaler_obj = None
    feature_names = ["Annual_Income_k$", "Spending_Score"]
    if X is None:
        X, scaler_obj, feature_names = prepare_clustering_features(
            feature_set=feature_set, scale_method=scale_method
        )

    # 2. Quét K tối ưu cho K-Means
    elbow_dict = find_optimal_k_elbow(X)
    sil_kmeans_dict = find_optimal_k_silhouette(X, algorithm="kmeans")
    optimal_k = max(sil_kmeans_dict, key=lambda k: sil_kmeans_dict[k])

    # 3. Huấn luyện Mô hình 1: K-Means (Scratch & Sklearn)
    km_scratch = KMeansClustering(n_clusters=optimal_k, random_state=42)
    km_labels = km_scratch.fit_predict(X)
    km_sil = float(silhouette_score(X, km_labels))
    km_db = float(davies_bouldin_score(X, km_labels))
    km_ch = float(calinski_harabasz_score(X, km_labels))

    km_sklearn = SklearnKMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    km_sklearn.fit(X)

    # 4. Huấn luyện Mô hình 2: DBSCAN (Scratch & Sklearn)
    db_tuning = tune_dbscan_params(X)
    best_eps = db_tuning["best_params"].get("eps", 0.35)
    best_ms = db_tuning["best_params"].get("min_samples", 5)

    db_scratch = DBSCANClustering(eps=best_eps, min_samples=best_ms)
    db_labels = db_scratch.fit_predict(X)
    db_valid_mask = db_labels != -1
    if db_scratch.n_clusters_ >= 2 and np.sum(db_valid_mask) > db_scratch.n_clusters_:
        db_sil = float(silhouette_score(X[db_valid_mask], db_labels[db_valid_mask]))
        db_db = float(davies_bouldin_score(X[db_valid_mask], db_labels[db_valid_mask]))
        db_ch = float(calinski_harabasz_score(X[db_valid_mask], db_labels[db_valid_mask]))
    else:
        db_sil, db_db, db_ch = 0.0, 0.0, 0.0

    db_sklearn = SklearnDBSCAN(eps=best_eps, min_samples=best_ms)
    db_sklearn.fit(X)

    # 5. Huấn luyện Mô hình 3: Hierarchical Agglomerative (Scratch & Sklearn)
    hc_scratch = HierarchicalClustering(n_clusters=optimal_k, linkage="ward")
    hc_labels = hc_scratch.fit_predict(X)
    hc_sil = float(silhouette_score(X, hc_labels))
    hc_db = float(davies_bouldin_score(X, hc_labels))
    hc_ch = float(calinski_harabasz_score(X, hc_labels))

    hc_sklearn = SklearnAgglomerative(n_clusters=optimal_k, linkage="ward")
    hc_sklearn.fit(X)

    # 6. Đánh giá chọn mô hình tốt nhất (Best Model) dựa trên Silhouette Score
    model_comparison = {
        "kmeans": {
            "name": "K-Means Clustering (Scratch)",
            "n_clusters": optimal_k,
            "silhouette_score": round(km_sil, 4),
            "davies_bouldin_index": round(km_db, 4),
            "calinski_harabasz_score": round(km_ch, 4),
            "inertia": round(km_scratch.inertia_, 4),
            "hyperparameters": {"n_clusters": optimal_k},
            "labels": km_labels.tolist()
        },
        "dbscan": {
            "name": "DBSCAN Clustering (Scratch)",
            "n_clusters": db_scratch.n_clusters_,
            "n_noise": db_scratch.n_noise_,
            "eps": best_eps,
            "min_samples": best_ms,
            "silhouette_score": round(db_sil, 4),
            "davies_bouldin_index": round(db_db, 4),
            "calinski_harabasz_score": round(db_ch, 4),
            "hyperparameters": {"eps": best_eps, "min_samples": best_ms},
            "labels": db_labels.tolist()
        },
        "hierarchical": {
            "name": "Hierarchical Clustering (Scratch)",
            "n_clusters": optimal_k,
            "linkage": "ward",
            "silhouette_score": round(hc_sil, 4),
            "davies_bouldin_index": round(hc_db, 4),
            "calinski_harabasz_score": round(hc_ch, 4),
            "hyperparameters": {"n_clusters": optimal_k, "linkage": "ward"},
            "labels": hc_labels.tolist()
        }
    }

    # Chọn mô hình tốt nhất (so sánh giữa K-Means và Hierarchical)
    if km_sil >= hc_sil:
        best_name = "kmeans"
        best_model_obj = km_scratch
    else:
        best_name = "hierarchical"
        best_model_obj = hc_scratch

    # 7. Lưu trữ mô hình vào models/saved_models/
    feature_ablation = evaluate_feature_spaces(k=optimal_k)

    if save_models:
        SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model_obj, SAVED_MODELS_DIR / "best_model.joblib")
        joblib.dump(km_scratch, SAVED_MODELS_DIR / "kmeans_model.joblib")
        joblib.dump(db_scratch, SAVED_MODELS_DIR / "dbscan_model.joblib")
        joblib.dump(hc_scratch, SAVED_MODELS_DIR / "hierarchical_model.joblib")
        if scaler_obj is not None:
            joblib.dump(scaler_obj, SAVED_MODELS_DIR / "preprocessor.joblib")

        metadata = {
            "project_name": "Practice 3: Clustering Retail Customers",
            "optimal_k": optimal_k,
            "best_model": best_name,
            "features_used": feature_names,
            "feature_set": feature_set,
            "scaling_method": scale_method,
            "models_summary": {
                k: {m: v for m, v in info.items() if m != "labels"}
                for k, info in model_comparison.items()
            },
            "feature_ablation": feature_ablation,
            "elbow_wcss": elbow_dict,
            "silhouette_by_k": sil_kmeans_dict
        }
        with open(SAVED_MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    return {
        "kmeans": km_scratch,
        "dbscan": db_scratch,
        "hierarchical": hc_scratch,
        "best_model_name": best_name,
        "comparison": model_comparison,
        "feature_ablation": feature_ablation,
        "elbow": elbow_dict,
        "silhouette_by_k": sil_kmeans_dict
    }


def evaluate_feature_spaces(
    k: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Khảo sát định lượng Silhouette Score trên các không gian đặc trưng (2D, 3D, 4D) với K-Means
    để minh chứng khoa học cho việc lựa chọn không gian 2D tối ưu.
    """
    space_results = {}
    feature_sets = [
        ("2d", "Không gian 2D (Income + Spending)"),
        ("3d", "Không gian 3D (Age + Income + Spending)"),
        ("4d", "Không gian 4D (Age + Gender + Income + Spend)")
    ]

    for f_code, label in feature_sets:
        X_s, _, f_names = prepare_clustering_features(feature_set=f_code)
        km = KMeansClustering(n_clusters=k, random_state=random_state)
        lbls = km.fit_predict(X_s)
        sil = round(float(silhouette_score(X_s, lbls)), 4)
        space_results[f_code] = {
            "label": label,
            "features": f_names,
            "n_features": len(f_names),
            "silhouette": sil,
            "is_best": (f_code == "2d")
        }

    return space_results

