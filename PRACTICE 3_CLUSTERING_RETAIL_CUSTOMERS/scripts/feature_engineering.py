"""
Module: feature_engineering.py
Dự án: Practice 3: Clustering Retail Customers — Group 3
Mục tiêu: Mã hóa biến phân loại (Gender), co giãn thang đo (StandardScaler / MinMaxScaler),
chuẩn bị ma trận đặc trưng phân cụm đa không gian (2D, 3D, 4D), và xuất dữ liệu sạch sang data/processed/.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union, List
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Thiết lập đường dẫn thư mục gốc
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from data.scripts.data_preprocessing import load_raw_data, clean_customer_data

PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "clean_customers.csv"


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mã hóa đặc trưng phân loại Gender thành nhị phân:
    - Female: 0
    - Male: 1
    Tạo cột mới 'Gender_Code' để bảo toàn cột 'Gender' nguyên bản.
    """
    df_encoded = df.copy()
    gender_map = {"Female": 0, "Male": 1}
    df_encoded["Gender_Code"] = df_encoded["Gender"].map(gender_map).astype(int)
    return df_encoded


def scale_features(
    X: Union[pd.DataFrame, np.ndarray],
    method: str = "standard"
) -> Tuple[np.ndarray, Any]:
    """
    Co giãn thang đo ma trận đặc trưng:
    - 'standard': StandardScaler (dua ve mean=0, std=1)
    - 'minmax': MinMaxScaler (dua ve doan [0, 1])
    
    Trả về:
    -------
    Tuple[np.ndarray, Any]: (Ma trận đặc trưng sau scale, Đối tượng scaler đã fit)
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Phương pháp scaler '{method}' không hợp lệ (chỉ hỗ trợ 'standard' hoặc 'minmax').")

    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def prepare_clustering_features(
    feature_set: str = "2d",
    scale_method: str = "standard",
    data_path: Optional[Union[str, Path]] = None
) -> Tuple[np.ndarray, Any, List[str]]:
    """
    Xây dựng ma trận đặc trưng sẵn sàng cho các thuật toán Gom cụm.
    
    Tham số:
    ---------
    feature_set : str, default='2d'
        Không gian đặc trưng gom cụm:
        - '2d': ['Annual_Income_k$', 'Spending_Score'] (Không gian 2D kinh điển)
        - '3d': ['Age', 'Annual_Income_k$', 'Spending_Score'] (Không gian 3 chiều)
        - '4d': ['Age', 'Annual_Income_k$', 'Spending_Score', 'Gender_Code'] (Toàn bộ đặc trưng)
    scale_method : str, default='standard'
        Phương pháp co giãn: 'standard' hoặc 'minmax'.
    data_path : str, Path hoặc None
        Đường dẫn nguồn dữ liệu thô.
        
    Trả về:
    -------
    Tuple[np.ndarray, Any, List[str]]: (Ma trận X_scaled, Scaler object, Danh sách tên đặc trưng)
    """
    # 1. Nạp và làm sạch dữ liệu
    df_raw = load_raw_data(data_path)
    df_clean, _ = clean_customer_data(df_raw)
    df_encoded = encode_gender(df_clean)

    # 2. Lựa chọn không gian đặc trưng
    if feature_set == "2d":
        feature_names = ["Annual_Income_k$", "Spending_Score"]
    elif feature_set == "3d":
        feature_names = ["Age", "Annual_Income_k$", "Spending_Score"]
    elif feature_set in ["4d", "all"]:
        feature_names = ["Age", "Annual_Income_k$", "Spending_Score", "Gender_Code"]
    else:
        raise ValueError(f"Không gian đặc trưng '{feature_set}' không hợp lệ (hỗ trợ '2d', '3d', '4d').")

    X_raw = df_encoded[feature_names].values.astype(np.float64)

    # 3. Chuẩn hóa thang đo
    X_scaled, scaler = scale_features(X_raw, method=scale_method)

    return X_scaled, scaler, feature_names


def export_processed_data(
    output_path: Optional[Union[str, Path]] = None
) -> Path:
    """
    Thực hiện làm sạch, mã hóa và xuất tệp dữ liệu sạch hoàn chỉnh vào data/processed/clean_customers.csv.
    """
    out_path = Path(output_path) if output_path else PROCESSED_DATA_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data()
    df_clean, _ = clean_customer_data(df_raw)
    df_encoded = encode_gender(df_clean)

    df_encoded.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


def main() -> None:
    """Chạy thực thi và chẩn đoán module feature_engineering."""
    print("=" * 75)
    print("   ⚙️ CHẠY MODULE KỸ THUẬT ĐẶC TRƯNG & CHUẨN HÓA THANG ĐO (FEATURE ENGINEERING)")
    print("=" * 75)

    # 1. Xuất dữ liệu sạch sang data/processed/
    out_file = export_processed_data()
    print(f"1. Đã xuất thành công tệp dữ liệu sạch tại: {out_file}")
    df_proc = pd.read_csv(out_file)
    print(f"   - Kích thước: {df_proc.shape[0]} dòng, {df_proc.shape[1]} cột.")
    print(f"   - Cột dữ liệu: {list(df_proc.columns)}")

    # 2. Kiểm tra không gian đặc trưng 2D
    X_2d, scaler_2d, names_2d = prepare_clustering_features(feature_set="2d")
    print(f"\n2. Không gian đặc trưng 2D:")
    print(f"   - Đặc trưng: {names_2d}")
    print(f"   - Kích thước ma trận X_2d: {X_2d.shape}")
    print(f"   - Mean sau chuẩn hóa: {np.mean(X_2d, axis=0).round(4)}")
    print(f"   - Std sau chuẩn hóa: {np.std(X_2d, axis=0).round(4)}")

    # 3. Kiểm tra không gian đặc trưng 3D
    X_3d, scaler_3d, names_3d = prepare_clustering_features(feature_set="3d")
    print(f"\n3. Không gian đặc trưng 3D:")
    print(f"   - Đặc trưng: {names_3d}")
    print(f"   - Kích thước ma trận X_3d: {X_3d.shape}")

    # 4. Kiểm tra không gian đặc trưng 4D
    X_4d, scaler_4d, names_4d = prepare_clustering_features(feature_set="4d")
    print(f"\n4. Không gian đặc trưng 4D:")
    print(f"   - Đặc trưng: {names_4d}")
    print(f"   - Kích thước ma trận X_4d: {X_4d.shape}")

    print("=" * 75)
    print("🎉 Module feature_engineering hoàn thành 100%!")


if __name__ == "__main__":
    main()
