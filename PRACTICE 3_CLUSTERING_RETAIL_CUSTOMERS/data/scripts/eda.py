"""
Module: eda.py
Dự án: Practice 3: Clustering Retail Customers — Group 3
Mục tiêu: Phân tích Khám phá Dữ liệu (Exploratory Data Analysis - EDA)
Nguyên tắc: Read-Only 100% trên dữ liệu thô (Mall_Customers.csv), không ghi đè dữ liệu.
Xuất các biểu đồ trực quan hóa chuyên nghiệp vào docs/images/.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Chế độ headless không mở popup GUI
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Mall_Customers.csv"
IMAGES_DIR = BASE_DIR / "docs" / "images"

# Cấu hình thẩm mỹ đồ thị
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def compute_eda_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Tính toán các chỉ số thống kê mô tả chuyên sâu trên dữ liệu thô:
    - Trung bình, độ lệch chuẩn, trung vị, Min, Max, phân vị (25%, 75%)
    - Độ lệch (Skewness) và độ nhọn (Kurtosis)
    - Phát hiện ngoại lai theo ngưỡng IQR (Interquartile Range)
    """
    numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    stats_dict: Dict[str, Any] = {}

    for col in numeric_cols:
        series = df[col].dropna()
        q25 = float(series.quantile(0.25))
        q75 = float(series.quantile(0.75))
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)].tolist()

        stats_dict[col] = {
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "median": round(float(series.median()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "skewness": round(float(series.skew()), 3),
            "kurtosis": round(float(series.kurtosis()), 3),
            "iqr": round(iqr, 2),
            "outliers_count": len(outliers),
            "outliers_values": outliers
        }

    # Phân tích giới tính
    gender_counts = df["Gender"].value_counts().to_dict()
    gender_ratio = {k: round(v / len(df) * 100, 1) for k, v in gender_counts.items()}

    # Phân tích nhóm tuổi
    age_bins = [17, 30, 45, 60, 100]
    age_labels = ["18-30 (Trẻ)", "31-45 (Trưởng thành)", "46-60 (Trung niên)", ">60 (Cao niên)"]
    age_grouped = pd.cut(df["Age"], bins=age_bins, labels=age_labels)
    age_dist = age_grouped.value_counts().to_dict()

    summary: Dict[str, Any] = {
        "n_samples": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "gender_counts": gender_counts,
        "gender_ratio_pct": gender_ratio,
        "numeric_stats": stats_dict,
        "age_distribution": {str(k): v for k, v in age_dist.items()}
    }

    return summary


def generate_eda_figures(df: pd.DataFrame, output_dir: Optional[Path] = None) -> List[Path]:
    """
    Vẽ và lưu 5 biểu đồ trực quan hóa chuyên nghiệp vào output_dir (docs/images/):
    1. eda_01_univariate_distributions.png: Phân phối Tuổi, Thu nhập và Điểm chi tiêu
    2. eda_02_gender_comparison.png: So sánh Thu nhập & Chi tiêu theo Giới tính
    3. eda_03_income_vs_spending_scatter.png: Phân tán Thu nhập vs Điểm chi tiêu (5 cụm trực quan)
    4. eda_04_age_vs_spending_trend.png: Tương quan Độ tuổi vs Điểm chi tiêu
    5. eda_05_correlation_heatmap.png: Ma trận nhiệt tương quan Pearson
    """
    out_dir = Path(output_dir) if output_dir else IMAGES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    # 1. eda_01: Phân phối đơn biến
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    cols = [
        ("Age", "#2b6cb0", "Phân phối Độ tuổi (Age)"),
        ("Annual Income (k$)", "#2f855a", "Phân phối Thu nhập (Annual Income)"),
        ("Spending Score (1-100)", "#b83280", "Phân phối Điểm chi tiêu (Spending Score)")
    ]
    for ax, (col, color, title) in zip(axes, cols):
        sns.histplot(df[col], kde=True, ax=ax, color=color, bins=15, edgecolor="white", alpha=0.7)
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="#e53e3e", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color="#d69e2e", linestyle=":", linewidth=1.8, label=f"Median: {median_val:.1f}")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Tần số (Frequency)", fontsize=10)
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    p1 = out_dir / "eda_01_univariate_distributions.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p1)

    # 2. eda_02: So sánh theo Giới tính
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 2.1 Tỷ lệ giới tính (Donut chart)
    gender_counts = df["Gender"].value_counts()
    axes[0].pie(
        gender_counts, labels=[f"{k}\n({v} khách)" for k, v in gender_counts.items()],
        autopct="%1.1f%%", startangle=90, colors=["#ed64a6", "#4299e1"],
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
    )
    axes[0].set_title("Cơ cấu Khách hàng theo Giới tính", fontsize=12, fontweight="bold", pad=10)

    # 2.2 Thu nhập theo Giới tính
    sns.boxplot(x="Gender", y="Annual Income (k$)", data=df, ax=axes[1], palette=["#4299e1", "#ed64a6"], width=0.4)
    sns.stripplot(x="Gender", y="Annual Income (k$)", data=df, ax=axes[1], color="black", alpha=0.3, jitter=0.2)
    axes[1].set_title("Thu nhập Hàng năm theo Giới tính", fontsize=12, fontweight="bold", pad=10)
    axes[1].set_ylabel("Annual Income (k$)", fontsize=10)

    # 2.3 Điểm chi tiêu theo Giới tính
    sns.boxplot(x="Gender", y="Spending Score (1-100)", data=df, ax=axes[2], palette=["#4299e1", "#ed64a6"], width=0.4)
    sns.stripplot(x="Gender", y="Spending Score (1-100)", data=df, ax=axes[2], color="black", alpha=0.3, jitter=0.2)
    axes[2].set_title("Điểm Chi tiêu theo Giới tính", fontsize=12, fontweight="bold", pad=10)
    axes[2].set_ylabel("Spending Score (1-100)", fontsize=10)

    plt.tight_layout()
    p2 = out_dir / "eda_02_gender_comparison.png"
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p2)

    # 3. eda_03: Phân tán Thu nhập vs Điểm chi tiêu (Khám phá 5 phân cụm tiềm năng)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x="Annual Income (k$)", y="Spending Score (1-100)",
        hue="Gender", style="Gender", data=df,
        palette={"Male": "#2b6cb0", "Female": "#d53f8c"},
        s=80, alpha=0.85, ax=ax
    )
    # Vẽ các vùng gợi ý 5 cụm tự nhiên
    ax.axvline(x=40, color="#a0aec0", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(x=70, color="#a0aec0", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(y=40, color="#a0aec0", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(y=60, color="#a0aec0", linestyle="--", linewidth=1, alpha=0.6)

    ax.text(25, 80, "Cụm 1: Thu nhập thấp\nChi tiêu cao (Spenders)", fontsize=9, fontweight="bold", color="#805ad5", ha="center")
    ax.text(95, 85, "Cụm 2: Thu nhập cao\nChi tiêu cao (VIP Target)", fontsize=9, fontweight="bold", color="#319795", ha="center")
    ax.text(25, 20, "Cụm 3: Thu nhập thấp\nChi tiêu thấp (Sensible)", fontsize=9, fontweight="bold", color="#dd6b20", ha="center")
    ax.text(95, 20, "Cụm 4: Thu nhập cao\nChi tiêu thấp (Careful)", fontsize=9, fontweight="bold", color="#3182ce", ha="center")
    ax.text(55, 50, "Cụm 5: Thu nhập TB\nChi tiêu TB (Standard)", fontsize=9, fontweight="bold", color="#718096", ha="center")

    ax.set_title("Biểu đồ Phân tán: Thu nhập vs Điểm chi tiêu (5 Cụm Tự nhiên)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Annual Income (k$)", fontsize=11)
    ax.set_ylabel("Spending Score (1-100)", fontsize=11)
    ax.legend(title="Giới tính", frameon=True)

    plt.tight_layout()
    p3 = out_dir / "eda_03_income_vs_spending_scatter.png"
    fig.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p3)

    # 4. eda_04: Độ tuổi vs Điểm chi tiêu
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(
        x="Age", y="Spending Score (1-100)",
        hue="Gender", data=df,
        palette={"Male": "#3182ce", "Female": "#e53e3e"},
        s=70, alpha=0.8, ax=ax
    )
    sns.regplot(
        x="Age", y="Spending Score (1-100)", data=df,
        scatter=False, ax=ax, color="#4a5568", line_kws={"linestyle": "--", "linewidth": 2}
    )
    ax.axvline(x=40, color="#e53e3e", linestyle=":", linewidth=1.5, label="Ngưỡng tuổi 40")
    ax.set_title("Mối quan hệ giữa Độ tuổi và Điểm chi tiêu (Khách hàng < 40 tuổi chi tiêu mạnh)", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Độ tuổi (Age)", fontsize=11)
    ax.set_ylabel("Spending Score (1-100)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    p4 = out_dir / "eda_04_age_vs_spending_trend.png"
    fig.savefig(p4, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p4)

    # 5. eda_05: Ma trận nhiệt tương quan Pearson
    fig, ax = plt.subplots(figsize=(7, 6))
    df_corr = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].copy()
    df_corr["Gender_Male"] = (df["Gender"] == "Male").astype(int)
    corr_matrix = df_corr.corr(method="pearson")

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=-0.5, vmax=0.5,
        linewidths=1, square=True, ax=ax, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Ma trận Nhiệt Tương quan Tuyến tính (Pearson Correlation)", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    p5 = out_dir / "eda_05_correlation_heatmap.png"
    fig.savefig(p5, dpi=200, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p5)

    return saved_paths


def run_eda(raw_path: Optional[Union[str, Path]] = None, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Hàm điều phối toàn bộ chu trình EDA:
    1. Đọc dữ liệu thô (Read-Only)
    2. Tính toán thống kê mô tả
    3. Tạo và lưu 5 tệp biểu đồ trực quan
    """
    path = Path(raw_path) if raw_path else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu thô tại: {path}")

    df_raw = pd.read_csv(path)
    summary = compute_eda_summary(df_raw)
    saved_figs = generate_eda_figures(df_raw, output_dir)
    summary["saved_figures"] = [str(p) for p in saved_figs]

    return summary

