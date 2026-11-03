"""
Module: eda.py
Dự án: Practice 2: Predicting Product Sales — Group 3
Phân công nhiệm vụ:
- Nguyễn Xuân Đông (Trưởng nhóm - 16%): Khảo sát cấu trúc (inspect_data_structure), phân tích đơn biến Amount (get_univariate_stats). [ĐÃ HOÀN THIỆN]
- Mai An Thịnh (14%): Nhận diện ngoại lai IQR (detect_iqr_outliers), phân tích đa biến doanh số (get_bivariate_analysis). [STARTER TEMPLATE / TODO]
"""
from typing import Dict, Any
import numpy as np
import pandas as pd


# =============================================================================
# PHẦN 1: NGUYỄN XUÂN ĐÔNG (TRƯỞNG NHÓM - 16%) — ĐÃ HOÀN THIỆN SẴN SÀNG
# =============================================================================

def inspect_data_structure(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Khảo sát cấu trúc tổng quan dữ liệu (kích thước, kiểu dữ liệu, tỷ lệ khuyết thiếu).
    Tác giả: Nguyễn Xuân Đông (Trưởng nhóm - 16%)

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
    Tác giả: Nguyễn Xuân Đông (Trưởng nhóm - 16%)

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
# PHẦN 2: MAI AN THỊNH (14%) — NHIỆM VỤ CẦN HOÀN THIỆN (STARTER TEMPLATE)
# =============================================================================

def detect_iqr_outliers(series: pd.Series) -> Dict[str, Any]:
    """
    Nhận diện giá trị ngoại lai theo công thức IQR tiêu chuẩn:
        IQR = Q3 - Q1
        [Lower, Upper] = [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]

    Phụ trách chính: Mai An Thịnh (14%)

    Tham số:
        series (pd.Series): Chuỗi dữ liệu số (ví dụ: cột Amount).

    Trả về:
        Dict[str, Any]:
            - 'q1': Tứ phân vị 25%
            - 'q3': Tứ phân vị 75%
            - 'iqr': Khoảng trải tứ phân vị (Q3 - Q1)
            - 'lower_bound': Ranh giới dưới
            - 'upper_bound': Ranh giới trên
            - 'n_outliers': Số lượng mẫu ngoại lai
            - 'outlier_pct': Tỷ lệ phần trăm ngoại lai (%)
    """
    # TODO: [Mai An Thịnh - 14%] Lập trình phần nhận diện ngoại lai IQR:
    # 1. Loại bỏ các giá trị NaN bằng s = series.dropna()
    # 2. Tính q1 = float(s.quantile(0.25)), q3 = float(s.quantile(0.75))
    # 3. Tính iqr = q3 - q1
    # 4. Tính lower_bound = q1 - 1.5 * iqr, upper_bound = q3 + 1.5 * iqr
    # 5. Xác định mặt nạ ngoại lai: outlier_mask = (s < lower_bound) | (s > upper_bound)
    # 6. Tính n_outliers = int(outlier_mask.sum()) và outlier_pct = round(n_outliers / len(s) * 100, 2)
    # 7. Trả về dict với đầy đủ các khóa trên.
    raise NotImplementedError(
        "Chưa cài đặt - Mai An Thịnh (14%) cần hoàn thiện hàm 'detect_iqr_outliers' theo phân công trong docs/description/TASKS.md"
    )

# Bí danh tương thích
detect_sales_outliers_iqr = detect_iqr_outliers


def get_bivariate_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Phân tích quan hệ hai biến giữa Amount với Category, ship-state và B2B.

    Phụ trách chính: Mai An Thịnh (14%)

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu thương mại điện tử.

    Trả về:
        Dict[str, Any]:
            - 'category_stats': dict thống kê (count, mean, median) theo ngành hàng
            - 'top_state_stats': dict thống kê theo Top 10 bang lớn nhất
            - 'b2b_stats': dict thống kê theo khách hàng sỉ B2B vs khách lẻ
    """
    # TODO: [Mai An Thịnh - 14%] Lập trình hàm phân tích hai biến:
    # 1. Khởi tạo results = {}
    # 2. Nếu 'Category' và 'Amount' có trong df:
    #    Tính df.groupby('Category')['Amount'].agg(['count', 'mean', 'median']).round(2)
    #    Lưu vào results['category_stats'] = cat_stats.to_dict(orient='index')
    # 3. Nếu 'ship-state' và 'Amount' có trong df:
    #    Lấy Top 10 bang có nhiều đơn nhất: state_order = df['ship-state'].value_counts().head(10).index
    #    Lọc df[df['ship-state'].isin(state_order)] và nhóm theo ship-state
    #    Lưu vào results['top_state_stats']
    # 4. Nếu 'B2B' và 'Amount' có trong df:
    #    Tính df.groupby('B2B')['Amount'].agg(['count', 'mean', 'median']).round(2)
    #    Lưu vào results['b2b_stats']
    # 5. Trả về results
    raise NotImplementedError(
        "Chưa cài đặt - Mai An Thịnh (14%) cần hoàn thiện hàm 'get_bivariate_analysis' theo phân công trong docs/description/TASKS.md"
    )

# Bí danh tương thích
analyze_bivariate_sales = get_bivariate_analysis
