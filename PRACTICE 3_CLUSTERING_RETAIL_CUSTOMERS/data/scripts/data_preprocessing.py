"""
Module: data_preprocessing.py
Dự án: Practice 3: Clustering Retail Customers — Group 3
Mục tiêu: Đọc dữ liệu thô, kiểm tra chất lượng (missing, duplicate, kiểu dữ liệu),
chuẩn hóa định dạng cột và làm sạch dữ liệu khách hàng.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import sys
import pandas as pd
import numpy as np

# Đường dẫn thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Mall_Customers.csv"


def load_raw_data(data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Tải dữ liệu thô Mall Customers từ đĩa.
    
    Tham số:
    ---------
    data_path : str, Path hoặc None
        Đường dẫn tới tệp CSV thô. Nếu None, dùng RAW_DATA_PATH mặc định.
        
    Trả về:
    -------
    pd.DataFrame: Bảng dữ liệu thô ban đầu.
    """
    path = Path(data_path) if data_path else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu thô tại: {path}")

    df = pd.read_csv(path)
    return df


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Kiểm tra toàn diện chất lượng dữ liệu:
    - Kích thước dòng x cột
    - Số lượng giá trị khuyết (missing values) từng cột
    - Số dòng trùng lặp (duplicate rows)
    - Trùng lặp mã khách hàng (duplicate CustomerID)
    - Thống kê biên độ giá trị các trường số
    """
    n_rows, n_cols = df.shape
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    id_col = "CustomerID" if "CustomerID" in df.columns else df.columns[0]
    duplicate_ids = int(df.duplicated(subset=[id_col]).sum())

    report: Dict[str, Any] = {
        "n_samples": n_rows,
        "n_features": n_cols,
        "total_missing": total_missing,
        "missing_by_column": missing_counts,
        "duplicate_rows": duplicate_rows,
        "duplicate_ids": duplicate_ids,
        "columns": list(df.columns)
    }

    return report


def clean_customer_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Làm sạch và chuẩn hóa dữ liệu khách hàng bán lẻ:
    1. Chuẩn hóa tên cột không chứa dấu cách và ký tự đặc biệt gây lỗi:
       - 'CustomerID' -> 'CustomerID'
       - 'Gender' -> 'Gender'
       - 'Age' -> 'Age'
       - 'Annual Income (k$)' -> 'Annual_Income_k$'
       - 'Spending Score (1-100)' -> 'Spending_Score'
    2. Chuẩn hóa chuỗi văn bản cột Gender (xóa khoảng trắng thừa, viết hoa chữ cái đầu).
    3. Ép kiểu dữ liệu nghiêm ngặt: CustomerID (int), Age (int), Annual_Income_k$ (float), Spending_Score (float).
    4. Kiểm tra phạm vi logic nghiệp vụ:
       - 18 <= Age <= 100
       - Annual_Income_k$ >= 0
       - 1 <= Spending_Score <= 100
    5. Khử trùng lặp nếu có.
    
    Trả về:
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]: (DataFrame sạch, Báo cáo tóm tắt quá trình làm sạch)
    """
    df_clean = df.copy()

    # 1. Bảng ánh xạ chuẩn hóa tên cột
    column_mapping = {
        "CustomerID": "CustomerID",
        "Gender": "Gender",
        "Genre": "Gender",
        "Age": "Age",
        "Annual Income (k$)": "Annual_Income_k$",
        "Annual_Income": "Annual_Income_k$",
        "Spending Score (1-100)": "Spending_Score",
        "Spending_Score": "Spending_Score"
    }
    df_clean = df_clean.rename(columns=lambda c: column_mapping.get(c.strip(), c.strip()))

    # 2. Xử lý khoảng trắng văn bản và chuẩn hóa giới tính
    if "Gender" in df_clean.columns:
        df_clean["Gender"] = df_clean["Gender"].astype(str).str.strip().str.capitalize()
        # Xác thực giá trị hợp lệ
        valid_genders = {"Male", "Female"}
        invalid_genders = set(df_clean["Gender"]) - valid_genders
        if invalid_genders:
            raise ValueError(f"Phát hiện giá trị giới tính không hợp lệ: {invalid_genders}")

    # 3. Ép kiểu dữ liệu số
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)
    df_clean["Age"] = df_clean["Age"].astype(int)
    df_clean["Annual_Income_k$"] = df_clean["Annual_Income_k$"].astype(float)
    df_clean["Spending_Score"] = df_clean["Spending_Score"].astype(float)

    # 4. Khử trùng lặp theo CustomerID nếu có
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["CustomerID"]).reset_index(drop=True)
    duplicates_removed = initial_count - len(df_clean)

    # 5. Kiểm tra tính hợp lệ logic biên độ
    assert (df_clean["Age"] >= 18).all() and (df_clean["Age"] <= 100).all(), "Tuổi khách hàng ngoài phạm vi 18-100"
    assert (df_clean["Annual_Income_k$"] >= 0).all(), "Thu nhập không được âm"
    assert (df_clean["Spending_Score"] >= 1).all() and (df_clean["Spending_Score"] <= 100).all(), "Điểm chi tiêu ngoài 1-100"

    report: Dict[str, Any] = {
        "initial_rows": initial_count,
        "cleaned_rows": len(df_clean),
        "duplicates_removed": duplicates_removed,
        "columns_standardized": list(df_clean.columns),
        "gender_counts": df_clean["Gender"].value_counts().to_dict(),
        "age_range": (int(df_clean["Age"].min()), int(df_clean["Age"].max())),
        "income_range_k$": (float(df_clean["Annual_Income_k$"].min()), float(df_clean["Annual_Income_k$"].max())),
        "spending_score_range": (float(df_clean["Spending_Score"].min()), float(df_clean["Spending_Score"].max()))
    }

    return df_clean, report

