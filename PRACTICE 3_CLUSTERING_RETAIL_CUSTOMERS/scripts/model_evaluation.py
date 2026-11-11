"""
Module: model_evaluation.py
Dự án: Practice 3: Clustering Retail Customers — Group 3
Mục tiêu: Đánh giá chất lượng các mô hình gom cụm (Silhouette, Davies-Bouldin, Calinski-Harabasz),
lập hồ sơ đặc tính kinh doanh 5 phân khúc khách hàng, và vẽ 4 biểu đồ trực quan hóa chuyên sâu
(bao gồm biểu đồ cây Dendrogram theo yêu cầu của đề bài slide) vào docs/images/.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import json
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Chế độ headless không hiển thị GUI popup
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples

# Thiết lập đường dẫn thư mục gốc
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.feature_engineering import prepare_clustering_features
from data.scripts.data_preprocessing import load_raw_data, clean_customer_data

SAVED_MODELS_DIR = BASE_DIR / "models" / "saved_models"
IMAGES_DIR = BASE_DIR / "docs" / "images"

# Cấu hình thẩm mỹ Matplotlib
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    model_name: str = "Clustering Model"
) -> Dict[str, Any]:
    """
    Tính toán toàn diện các chỉ số đánh giá gom cụm nội tại:
    - Silhouette Score (toàn thể và từng cụm)
    - Davies-Bouldin Index (độ phân tán nội cụm / khoảng cách liên cụm)
    - Calinski-Harabasz Score (tỷ số phương sai liên cụm / nội cụm)
    """
    unique_labels = [l for l in np.unique(labels) if l != -1]
    n_clusters = len(unique_labels)
    n_noise = int(np.sum(labels == -1))

    valid_mask = labels != -1
    if n_clusters >= 2 and np.sum(valid_mask) > n_clusters:
        sil = float(silhouette_score(X[valid_mask], labels[valid_mask]))
        db = float(davies_bouldin_score(X[valid_mask], labels[valid_mask]))
        ch = float(calinski_harabasz_score(X[valid_mask], labels[valid_mask]))
        
        # Tính Silhouette chi tiết từng cụm
        sample_sil = silhouette_samples(X[valid_mask], labels[valid_mask])
        cluster_sil: Dict[int, float] = {}
        for c in unique_labels:
            c_scores = sample_sil[labels[valid_mask] == c]
            cluster_sil[int(c)] = round(float(np.mean(c_scores)), 4)
    else:
        sil, db, ch = 0.0, 0.0, 0.0
        cluster_sil = {}

    return {
        "model_name": model_name,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette_score": round(sil, 4),
        "davies_bouldin_index": round(db, 4),
        "calinski_harabasz_score": round(ch, 4),
        "cluster_silhouette": cluster_sil
    }


def profile_customer_segments(
    df_clean: pd.DataFrame,
    labels: np.ndarray
) -> pd.DataFrame:
    """
    Xây dựng bảng hồ sơ nghiệp vụ cho từng phân khúc khách hàng:
    - Quy mô phân khúc (Số lượng và tỷ lệ %)
    - Thu nhập bình quân (Annual Income k$)
    - Điểm chi tiêu bình quân (Spending Score)
    - Độ tuổi trung bình (Age)
    - Tỷ lệ Nữ giới (% Female)
    - Đặt tên phân khúc (Segment Persona) & Khuyến nghị chiến lược bán lẻ
    """
    df_prof = df_clean.copy()
    df_prof["Cluster"] = labels

    segment_meta = {
        0: {
            "name": "Nhóm Tiêu xài Tự do (Spenders)",
            "strategy": "Thu nhập thấp nhưng chi tiêu mạnh. Đẩy mạnh các sản phẩm theo trend, khuyến mãi giảm giá chớp nhoáng (flash sales) và hỗ trợ trả góp."
        },
        1: {
            "name": "Nhóm Khách hàng VIP (Target Customers)",
            "strategy": "Thu nhập cao và chi tiêu cực mạnh. Chăm sóc đặc quyền (VIP loyalty), giới thiệu sản phẩm cao cấp, cá nhân hóa trải nghiệm mua sắm."
        },
        2: {
            "name": "Nhóm Khách hàng Tiết kiệm (Sensible)",
            "strategy": "Thu nhập thấp và chi tiêu dè dặt. Cung cấp các sản phẩm thiết yếu giá rẻ, combo tiết kiệm, tích điểm đổi quà dài hạn."
        },
        3: {
            "name": "Nhóm Cẩn trọng / Giữ tiền (Careful / Savers)",
            "strategy": "Thu nhập cao nhưng chi tiêu rất thấp. Cần tiếp thị dựa trên chất lượng, độ bền, uy tín thương hiệu và các chương trình hoàn tiền giá trị cao."
        },
        4: {
            "name": "Nhóm Khách hàng Tiêu chuẩn (Standard / Average)",
            "strategy": "Thu nhập và chi tiêu đều ở mức trung bình (chiếm số đông). Duy trì các chương trình khuyến mãi định kỳ và dịch vụ khách hàng ổn định."
        }
    }

    records = []
    total_customers = len(df_prof)

    for c in sorted(df_prof["Cluster"].unique()):
        if c == -1:
            name = "Điểm Nhiễu / Dị biệt (Outliers)"
            strat = "Các khách hàng có hành vi bất thường, cần phân tích thủ công riêng lẻ."
        else:
            name = segment_meta.get(c, {}).get("name", f"Phân khúc {c}")
            strat = segment_meta.get(c, {}).get("strategy", "Chưa có khuyến nghị")

        sub = df_prof[df_prof["Cluster"] == c]
        count = len(sub)
        pct = round(count / total_customers * 100, 1)
        mean_inc = round(sub["Annual_Income_k$"].mean(), 1)
        mean_spend = round(sub["Spending_Score"].mean(), 1)
        mean_age = round(sub["Age"].mean(), 1)
        pct_female = round((sub["Gender"] == "Female").mean() * 100, 1)

        records.append({
            "Cluster": c,
            "Tên phân khúc (Persona)": name,
            "Số lượng": count,
            "Tỷ lệ (%)": f"{pct}%",
            "Thu nhập TB (k$)": mean_inc,
            "Chi tiêu TB": mean_spend,
            "Tuổi TB": mean_age,
            "Tỷ lệ Nữ": f"{pct_female}%",
            "Chiến lược Bán lẻ": strat
        })

    df_result = pd.DataFrame(records)
    return df_result


def generate_evaluation_figures(
    df_clean: pd.DataFrame,
    X_scaled: np.ndarray,
    km_model: Any,
    hc_model: Any,
    db_model: Any,
    metadata: Dict[str, Any],
    output_dir: Optional[Path] = None
) -> List[Path]:
    """
    Vẽ và xuất 4 biểu đồ đánh giá chuyên sâu vào docs/images/:
    1. model_01_optimal_k_selection.png: Đồ thị kép Elbow (WCSS) & Silhouette Score tìm K=5.
    2. model_02_kmeans_clusters_2d.png: Biểu đồ phân cụm 2D K-Means với 5 trọng tâm cụm.
    3. model_03_hierarchical_dendrogram.png: Cây phân cấp Dendrogram với đường cắt tại K=5.
    4. model_04_dbscan_density_clusters.png: Biểu đồ phân cụm mật độ DBSCAN (Cụm & Nhiễu).
    """
    out_dir = Path(output_dir) if output_dir else IMAGES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    # 1. model_01: Đồ thị kép Elbow & Silhouette
    elbow_dict = metadata.get("elbow_wcss", {})
    sil_dict = metadata.get("silhouette_by_k", {})

    k_vals = sorted([int(k) for k in elbow_dict.keys()])
    wcss_vals = [elbow_dict[str(k)] if str(k) in elbow_dict else elbow_dict[k] for k in k_vals]
    sil_vals = [sil_dict[str(k)] if str(k) in sil_dict else sil_dict[k] for k in k_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Đường cong Elbow
    ax1.plot(k_vals, wcss_vals, marker="o", color="#2b6cb0", linewidth=2.2, markersize=7)
    ax1.axvline(x=5, color="#e53e3e", linestyle="--", linewidth=1.5, label="Điểm uốn tối ưu: K=5")
    ax1.scatter([5], [elbow_dict.get(5, elbow_dict.get("5", 65.57))], color="#e53e3e", s=120, zorder=5)
    ax1.set_title("1. Phương pháp Khuỷu tay (Elbow Method — WCSS)", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Số lượng cụm K", fontsize=11)
    ax1.set_ylabel("Tổng phương sai nội cụm (Inertia / WCSS)", fontsize=11)
    ax1.set_xticks(k_vals)
    ax1.legend(loc="upper right", frameon=True)

    # Đường cong Silhouette
    ax2.plot(k_vals, sil_vals, marker="s", color="#38a169", linewidth=2.2, markersize=7)
    ax2.axvline(x=5, color="#e53e3e", linestyle="--", linewidth=1.5, label="Đỉnh cực đại: K=5 (Score: 0.555)")
    ax2.scatter([5], [sil_dict.get(5, sil_dict.get("5", 0.5547))], color="#e53e3e", s=120, zorder=5)
    ax2.set_title("2. Hệ số Silhouette Score theo K", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Số lượng cụm K", fontsize=11)
    ax2.set_ylabel("Silhouette Score trung bình", fontsize=11)
    ax2.set_xticks(k_vals)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    p1 = out_dir / "model_01_optimal_k_selection.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p1)

    # 2. model_02: Phân cụm K-Means 2D
    fig, ax = plt.subplots(figsize=(10, 6.5))
    labels_km = km_model.labels_
    palette_km = ["#805ad5", "#319795", "#dd6b20", "#3182ce", "#718096"]
    cluster_names = [
        "Cụm 0: Tiêu xài Tự do (Spenders)",
        "Cụm 1: Khách hàng VIP (Target)",
        "Cụm 2: Khách hàng Tiết kiệm (Sensible)",
        "Cụm 3: Cẩn trọng / Giữ tiền (Careful)",
        "Cụm 4: Khách hàng Tiêu chuẩn (Standard)"
    ]

    for c in range(5):
        mask = labels_km == c
        ax.scatter(
            df_clean.loc[mask, "Annual_Income_k$"],
            df_clean.loc[mask, "Spending_Score"],
            c=palette_km[c], s=75, alpha=0.85, edgecolors="white", linewidth=0.8,
            label=f"{cluster_names[c]} ({np.sum(mask)} khách)"
        )

    # Chuyển đổi tọa độ tâm cụm từ không gian scaled về không gian gốc
    scaler = joblib.load(SAVED_MODELS_DIR / "preprocessor.joblib")
    centers_original = scaler.inverse_transform(km_model.cluster_centers_)

    # Vẽ 5 trọng tâm cụm
    ax.scatter(
        centers_original[:, 0], centers_original[:, 1],
        c="#e53e3e", s=250, marker="*", edgecolors="black", linewidth=1.5,
        label="Trọng tâm cụm (Centroids)", zorder=6
    )

    ax.set_title("Bản đồ Phân cụm Khách hàng 2D (K-Means Clustering Scratch — K=5)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Thu nhập Hàng năm (Annual Income k$)", fontsize=11)
    ax.set_ylabel("Điểm Chi tiêu (Spending Score 1-100)", fontsize=11)
    ax.legend(title="Phân khúc Khách hàng", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    p2 = out_dir / "model_02_kmeans_clusters_2d.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p2)

    # 3. model_03: Biểu đồ cây Dendrogram (Hierarchical Clustering)
    fig, ax = plt.subplots(figsize=(13, 6))
    Z = hc_model.linkage_matrix_

    # Tìm ngưỡng khoảng cách cắt để lấy K=5 cụm
    # Trong ma trận liên kết Z (199 x 4), bước gộp 199 - 5 = dòng thứ 194
    cut_threshold = float((Z[194, 2] + Z[195, 2]) / 2.0)

    dendrogram(
        Z, ax=ax,
        truncate_mode="lastp", p=30,  # Hiển thị 30 nhánh gộp trên cùng để biểu đồ thoáng đẹp
        leaf_rotation=90, leaf_font_size=9,
        color_threshold=cut_threshold
    )
    ax.axhline(y=cut_threshold, color="#e53e3e", linestyle="--", linewidth=1.8, label=f"Đường cắt lát K=5 (Ngưỡng dist: {cut_threshold:.2f})")
    ax.set_title("Biểu đồ Cây Phân cấp Thứ bậc (Dendrogram — Ward Linkage Scratch)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Chỉ số Nhóm Khách hàng / Cụm mẫu con (Sample Clusters)", fontsize=11)
    ax.set_ylabel("Khoảng cách Liên kết Ward (Ward Distance)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    p3 = out_dir / "model_03_hierarchical_dendrogram.png"
    fig.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p3)

    # 4. model_04: Biểu đồ phân cụm mật độ DBSCAN
    fig, ax = plt.subplots(figsize=(10, 6.5))
    labels_db = db_model.labels_
    unique_db_labels = sorted(set(labels_db))

    db_colors = sns.color_palette("tab10", len([l for l in unique_db_labels if l != -1]))

    for c in unique_db_labels:
        mask = labels_db == c
        if c == -1:
            ax.scatter(
                df_clean.loc[mask, "Annual_Income_k$"],
                df_clean.loc[mask, "Spending_Score"],
                c="#e53e3e", marker="x", s=80, linewidth=2,
                label=f"Điểm Nhiễu / Ngoại lai (Noise: {np.sum(mask)} khách)", zorder=4
            )
        else:
            ax.scatter(
                df_clean.loc[mask, "Annual_Income_k$"],
                df_clean.loc[mask, "Spending_Score"],
                color=db_colors[c], s=70, alpha=0.85, edgecolors="white", linewidth=0.8,
                label=f"Cụm mật độ {c} ({np.sum(mask)} khách)"
            )

    ax.set_title(f"Bản đồ Phân cụm Mật độ DBSCAN Scratch (eps={db_model.eps}, min_samples={db_model.min_samples})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Thu nhập Hàng năm (Annual Income k$)", fontsize=11)
    ax.set_ylabel("Điểm Chi tiêu (Spending Score 1-100)", fontsize=11)
    ax.legend(title="Cụm Mật độ & Nhiễu", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    p4 = out_dir / "model_04_dbscan_density_clusters.png"
    fig.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p4)

    return saved_paths


def evaluate_clustering(
    save_figures: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Hàm điều phối toàn bộ chu trình Đánh giá & Phân tích Mô hình Gom cụm.
    """
    # 1. Nạp dữ liệu sạch và ma trận đặc trưng
    df_raw = load_raw_data()
    df_clean, _ = clean_customer_data(df_raw)
    X_scaled, scaler_obj, _ = prepare_clustering_features(feature_set="2d")

    # 2. Nạp các mô hình đã huấn luyện
    km_model = joblib.load(SAVED_MODELS_DIR / "kmeans_model.joblib")
    db_model = joblib.load(SAVED_MODELS_DIR / "dbscan_model.joblib")
    hc_model = joblib.load(SAVED_MODELS_DIR / "hierarchical_model.joblib")

    meta_path = SAVED_MODELS_DIR / "metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 3. Tính toán các độ đo chi tiết
    metrics_km = compute_clustering_metrics(X_scaled, km_model.labels_, model_name="K-Means (Scratch)")
    metrics_db = compute_clustering_metrics(X_scaled, db_model.labels_, model_name="DBSCAN (Scratch)")
    metrics_hc = compute_clustering_metrics(X_scaled, hc_model.labels_, model_name="Hierarchical (Scratch)")

    # 4. Lập hồ sơ đặc tính 5 phân khúc khách hàng
    df_profile = profile_customer_segments(df_clean, km_model.labels_)

    # 5. Xuất các biểu đồ trực quan hóa
    saved_figs = []
    if save_figures:
        saved_figs = generate_evaluation_figures(
            df_clean=df_clean,
            X_scaled=X_scaled,
            km_model=km_model,
            hc_model=hc_model,
            db_model=db_model,
            metadata=metadata,
            output_dir=output_dir
        )

    eval_result: Dict[str, Any] = {
        "metrics_kmeans": metrics_km,
        "metrics_dbscan": metrics_db,
        "metrics_hierarchical": metrics_hc,
        "segment_profile": df_profile.to_dict(orient="records"),
        "saved_figures": [str(p) for p in saved_figs]
    }

    return eval_result


def main() -> None:
    """Chạy thực thi trực tiếp quá trình đánh giá từ dòng lệnh."""
    print("=" * 80)
    print("   📈 ĐÁNH GIÁ CHẤT LƯỢNG MÔ HÌNH GOM CỤM & LẬP HỒ SƠ PHÂN KHÚC KHÁCH HÀNG")
    print("=" * 80)

    res = evaluate_clustering()

    print("\n1. Chỉ số Đánh giá Chất lượng Phân cụm:")
    for key in ["metrics_kmeans", "metrics_dbscan", "metrics_hierarchical"]:
        m = res[key]
        print(f"   * {m['model_name']:<28} | K: {m['n_clusters']} | Silhouette: {m['silhouette_score']} | Davies-Bouldin: {m['davies_bouldin_index']} | Calinski-H: {m['calinski_harabasz_score']}")

    print("\n2. Bảng Hồ sơ Nghiệp vụ 5 Phân khúc Khách hàng (K-Means):")
    print("-" * 88)
    for seg in res["segment_profile"]:
        print(f"   [{seg['Cluster']}] {seg['Tên phân khúc (Persona)']}: {seg['Số lượng']} khách ({seg['Tỷ lệ (%)']})")
        print(f"       -> Thu nhập TB: {seg['Thu nhập TB (k$)']} k$ | Chi tiêu TB: {seg['Chi tiêu TB']} | Tuổi TB: {seg['Tuổi TB']}")
        print(f"       -> Chiến lược: {seg['Chiến lược Bán lẻ']}")
    print("-" * 88)

    print(f"\n3. Đã xuất thành công {len(res['saved_figures'])} biểu đồ đánh giá vào docs/images/:")
    for f in res["saved_figures"]:
        print(f"   ✓ {Path(f).name}")

    print("=" * 80)
    print("🎉 Chu trình Đánh giá Hoàn thành 100%!")


if __name__ == "__main__":
    main()
