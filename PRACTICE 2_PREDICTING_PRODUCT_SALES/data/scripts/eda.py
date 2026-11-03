"""
Module: eda.py
Dự án: Practice 2: Predicting Product Sales — Group 3
Phân công nhiệm vụ:
- Nguyễn Xuân Đông (Trưởng nhóm): Khảo sát cấu trúc (inspect_data_structure), phân tích đơn biến Amount (get_univariate_stats). [ĐÃ HOÀN THIỆN]
- Mai An Thịnh: Nhận diện ngoại lai IQR (detect_iqr_outliers), phân tích đa biến doanh số (get_bivariate_analysis). [STARTER TEMPLATE / TODO]
"""
from typing import Dict, Any
import numpy as np
import pandas as pd


# =============================================================================
# PHẦN 1: NGUYỄN XUÂN ĐÔNG (TRƯỞNG NHÓM) — ĐÃ HOÀN THIỆN SẴN SÀNG
# =============================================================================

def inspect_data_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Khảo sát cấu trúc tổng quan dữ liệu (kích thước, kiểu dữ liệu, tỷ lệ khuyết thiếu).
    Tác giả: Nguyễn Xuân Đông (Trưởng nhóm)

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu đầu vào.

    Trả về:
        Dict[str, Any]: Từ điển chứa thông tin kích thước, số dòng trùng, và chi tiết khuyết thiếu.
    """
    n_rows, n_cols = df.shape
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = {k: round(v / max(1, n_rows) * 100, 2) for k, v in missing_counts.items()}

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_counts": missing_counts,
        "missing_pct": missing_pct,
        "n_duplicates": int(df.duplicated().sum())
    }


def get_univariate_stats(df: pd.DataFrame, col: str = 'Amount') -> Dict[str, Any]:
    """
    Tính toán thống kê mô tả đơn biến cho biến số (mặc định Amount).
    Bao gồm: mean, median, std, min, max, skewness, kurtosis, q25, q75.
    Tác giả: Nguyễn Xuân Đông (Trưởng nhóm)

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu.
        col (str): Tên cột cần tính thống kê (mặc định: 'Amount').

    Trả về:
        Dict[str, Any]: Từ điển chứa các đại lượng thống kê mô tả.
    """
    if col not in df.columns:
        raise KeyError(f"Cột '{col}' không tồn tại trong DataFrame")

    s = df[col].dropna()
    skew_val = float(s.skew()) if len(s) > 2 else 0.0
    kurt_val = float(s.kurtosis()) if len(s) > 3 else 0.0

    return {
        "column": col,
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "skewness": round(skew_val, 4),
        "kurtosis": round(kurt_val, 4),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75))
    }

# Bí danh tương thích
get_amount_univariate_stats = get_univariate_stats


# =============================================================================
# PHẦN 2: MAI AN THỊNH — NHIỆM VỤ CẦN HOÀN THIỆN (STARTER TEMPLATE)
# =============================================================================

def detect_iqr_outliers(series: pd.Series) -> Dict[str, Any]:
    """
    Nhận diện giá trị ngoại lai theo phương pháp IQR (Interquartile Range).

    Tham số:
        series (pd.Series): Chuỗi dữ liệu số đầu vào.

    Trả về:
        Dict[str, Any]: Thống kê tứ phân vị, ngưỡng ranh giới và số lượng ngoại lai.
    """
    # TODO: [Mai An Thịnh] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Mai An Thịnh cần hoàn thiện hàm 'detect_iqr_outliers' theo phân công trong docs/description/TASKS.md"
    )

# Bí danh tương thích
detect_sales_outliers_iqr = detect_iqr_outliers


def get_bivariate_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Phân tích quan hệ hai biến giữa doanh số với các biến phân loại chính.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu đầu vào.

    Trả về:
        Dict[str, Any]: Thống kê doanh số theo ngành hàng, địa lý và loại khách hàng.
    """
    # TODO: [Mai An Thịnh] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Mai An Thịnh cần hoàn thiện hàm 'get_bivariate_analysis' theo phân công trong docs/description/TASKS.md"
    )

# Bí danh tương thích
analyze_bivariate_sales = get_bivariate_analysis
