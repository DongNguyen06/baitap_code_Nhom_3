"""
Module: feature_engineering.py
Dự án: Practice 2: Predicting Product Sales — Group 3
Phân công nhiệm vụ:
- Trương Thành Công: Trích xuất đặc trưng miền (extract_domain_features, transform_target_log).
- Võ Thanh Phú: Lọc đa cộng tuyến (detect_collinear_features), phân chia Train/Test 80/20 (split_train_test, export_train_test_csv), ColumnTransformer (build_column_transformer).
- Nguyễn Xuân Đông (Trưởng nhóm): Chuẩn bị mẫu suy luận demo (prepare_sample_from_dict) & Nạp dữ liệu (load_pipeline_and_splits).
"""
from pathlib import Path
import sys
from typing import Tuple, List, Dict, Any, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


TOP_10_STATES = [
    'MAHARASHTRA', 'KARNATAKA', 'TAMIL NADU', 'TELANGANA', 'UTTAR PRADESH',
    'DELHI', 'KERALA', 'WEST BENGAL', 'ANDHRA PRADESH', 'HARYANA'
]

CATEGORY_LOOKUP = {
    'set': 'Set',
    'kurta': 'kurta',
    'western dress': 'Western Dress',
    'top': 'Top',
    'ethnic dress': 'Ethnic Dress',
    'blouse': 'Blouse',
    'bottom': 'Bottom',
    'saree': 'Saree',
    'dupatta': 'Dupatta'
}


# =============================================================================
# PHẦN 1: TRƯƠNG THÀNH CÔNG — TRÍCH XUẤT ĐẶC TRƯNG MIỀN (TODO)
# =============================================================================

def extract_domain_features(
    df: pd.DataFrame,
    top_states: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Trích xuất các đặc trưng miền chuyên sâu ngành Thương mại điện tử (thời gian, địa lý, cờ nghiệp vụ).

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu đầu vào.
        top_states (List[str], optional): Danh sách các bang lớn (mặc định: TOP_10_STATES).

    Trả về:
        pd.DataFrame: DataFrame có thêm các cột đặc trưng mới.
    """
    # TODO: [Trương Thành Công] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Trương Thành Công cần hoàn thiện hàm 'extract_domain_features' theo phân công trong docs/description/TASKS.md"
    )


def transform_target_log(
    target: Union[pd.Series, np.ndarray]
) -> np.ndarray:
    """
    Biến đổi log1p trên biến mục tiêu doanh số Amount để ổn định số học và giảm độ lệch: y_log = log(1 + target).

    Tham số:
        target (pd.Series | np.ndarray): Chuỗi giá trị doanh số thực tế (Amount).

    Trả về:
        np.ndarray: Mảng giá trị sau khi áp dụng phép biến đổi log1p.
    """
    # TODO: [Trương Thành Công] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Trương Thành Công cần hoàn thiện hàm 'transform_target_log' theo phân công trong docs/description/TASKS.md"
    )


# =============================================================================
# PHẦN 2: VÕ THANH PHÚ — ĐA CỘNG TUYẾN, TRAIN/TEST SPLIT & COLUMNTRANSFORMER (TODO)
# =============================================================================

def detect_collinear_features(
    df_features: pd.DataFrame,
    threshold: float = 0.6
) -> List[Tuple[str, str, float]]:
    """
    Quét ma trận tương quan Pearson trên các biến số để phát hiện các cặp đặc trưng đa cộng tuyến (|r| > threshold).

    Tham số:
        df_features (pd.DataFrame): DataFrame chứa các đặc trưng số.
        threshold (float): Ngưỡng tương quan Pearson xem là đa cộng tuyến (mặc định: 0.6).

    Trả về:
        List[Tuple[str, str, float]]: Danh sách các bộ ba (tên_biến_1, tên_biến_2, hệ_số_r).
    """
    # TODO: [Võ Thanh Phú] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Võ Thanh Phú cần hoàn thiện hàm 'detect_collinear_features' theo phân công trong docs/description/TASKS.md"
    )


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """
    Phân chia Train / Test theo tỷ lệ 80/20 chống rò rỉ dữ liệu.

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu sau khi trích xuất đặc trưng.
        test_size (float): Tỷ lệ tập kiểm thử (mặc định: 0.2).
        random_state (int): Hạt giống ngẫu nhiên (mặc định: 42).

    Trả về:
        Tuple: (X_train, X_test, y_train, y_test, y_train_raw, y_test_raw)
    """
    # TODO: [Võ Thanh Phú] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Võ Thanh Phú cần hoàn thiện hàm 'split_train_test' theo phân công trong docs/description/TASKS.md"
    )


def build_column_transformer(
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Thiết lập ColumnTransformer chuẩn scikit-learn (StandardScaler cho biến số, OneHotEncoder cho biến phân loại).

    Tham số:
        numeric_cols (List[str]): Danh sách tên cột số.
        categorical_cols (List[str]): Danh sách tên cột định tính.

    Trả về:
        ColumnTransformer: Bộ tiền xử lý hoàn chỉnh sẵn sàng ghép vào Pipeline.
    """
    # TODO: [Võ Thanh Phú] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Võ Thanh Phú cần hoàn thiện hàm 'build_column_transformer' theo phân công trong docs/description/TASKS.md"
    )


def export_train_test_csv(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
    output_dir: Union[str, Path]
) -> Tuple[Path, Path]:
    """
    Xuất hai tệp train.csv và test.csv kèm cột Amount thực tế phục vụ trực quan hóa và kiểm tra.

    Tham số:
        X_train, X_test: DataFrame đặc trưng của tập Train và Test.
        y_train_raw, y_test_raw: Mảng giá trị Amount thực tế tương ứng.
        output_dir: Thư mục lưu tệp (data/processed/).

    Trả về:
        Tuple[Path, Path]: (đường_dẫn_train_csv, đường_dẫn_test_csv).
    """
    # TODO: [Võ Thanh Phú] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Võ Thanh Phú cần hoàn thiện hàm 'export_train_test_csv' theo phân công trong docs/description/TASKS.md"
    )


# =============================================================================
# PHẦN 3: NGUYỄN XUÂN ĐÔNG (TRƯỞNG NHÓM) — HỖ TRỢ SUY LUẬN DEMO (ĐÃ HOÀN THIỆN)
# =============================================================================

def prepare_sample_from_dict(sample_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Chuẩn bị DataFrame từ thông tin một đơn hàng thô để suy luận thời gian thực.
    Tự động chuẩn hóa casing khớp với OneHotEncoder đã học từ dữ liệu.
    Tác giả: Nguyễn Xuân Đông (Trưởng nhóm)
    """
    df_raw = pd.DataFrame([sample_dict])

    # Gọi hàm extract_domain_features của Thành Công, hoặc tự trích xuất nếu hàm đang là TODO
    try:
        df_feat = extract_domain_features(df_raw)
    except NotImplementedError:
        df_feat = df_raw.copy()
        if 'month' not in df_feat.columns:
            if 'Date' in df_feat.columns:
                dt_series = pd.to_datetime(df_feat['Date'], format='mixed', errors='coerce')
                df_feat['month'] = dt_series.dt.month.fillna(4).astype(int)
                df_feat['day_of_week'] = dt_series.dt.dayofweek.fillna(0).astype(int)
            else:
                df_feat['month'] = 4
                df_feat['day_of_week'] = 5

        if 'is_promoted' not in df_feat.columns:
            promo = df_feat.get('promotion-ids', pd.Series(['No_Promotion'])).iloc[0]
            df_feat['is_promoted'] = 1 if promo and promo != 'No_Promotion' else 0

        if 'is_b2b' not in df_feat.columns:
            b2b_val = df_feat.get('B2B', pd.Series([False])).iloc[0]
            df_feat['is_b2b'] = 1 if b2b_val else 0

        if 'state_grouped' not in df_feat.columns:
            st = str(df_feat.get('ship-state', pd.Series(['Other'])).iloc[0]).strip().upper()
            df_feat['state_grouped'] = st if st in TOP_10_STATES else 'Other'

    # Chuẩn hóa giá trị định tính (casing) để khớp OneHotEncoder
    if 'Category' in df_feat.columns and pd.notna(df_feat['Category'].iloc[0]):
        c_val = str(df_feat['Category'].iloc[0]).strip().lower()
        df_feat['Category'] = CATEGORY_LOOKUP.get(c_val, df_feat['Category'].iloc[0])

    if 'Fulfilment' in df_feat.columns and pd.notna(df_feat['Fulfilment'].iloc[0]):
        f_val = str(df_feat['Fulfilment'].iloc[0]).strip().title()
        if f_val in ['Amazon', 'Merchant']:
            df_feat['Fulfilment'] = f_val

    if 'ship-service-level' in df_feat.columns and pd.notna(df_feat['ship-service-level'].iloc[0]):
        s_val = str(df_feat['ship-service-level'].iloc[0]).strip().title()
        if s_val in ['Standard', 'Expedited']:
            df_feat['ship-service-level'] = s_val

    if 'Courier Status' in df_feat.columns and pd.notna(df_feat['Courier Status'].iloc[0]):
        cs_val = str(df_feat['Courier Status'].iloc[0]).strip().title()
        if cs_val in ['Shipped', 'Unshipped']:
            df_feat['Courier Status'] = cs_val

    if 'Size' in df_feat.columns and pd.notna(df_feat['Size'].iloc[0]):
        sz_val = str(df_feat['Size'].iloc[0]).strip().upper()
        if sz_val == 'FREE':
            sz_val = 'Free'
        df_feat['Size'] = sz_val

    feature_cols = [
        'Qty', 'month', 'day_of_week', 'is_promoted', 'is_b2b',
        'Category', 'Size', 'Fulfilment', 'ship-service-level',
        'Courier Status', 'state_grouped'
    ]
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0 if col in ['Qty', 'month', 'day_of_week', 'is_promoted', 'is_b2b'] else 'Other'

    return df_feat[feature_cols]


def load_pipeline_and_splits(
    data_path: Union[str, Path],
    preprocessor_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Nạp dữ liệu đã chia và đối tượng preprocessor từ đĩa.
    Tác giả: Nguyễn Xuân Đông (Trưởng nhóm)
    """
    data_p = Path(data_path)
    if not data_p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {data_p}")

    loaded_dict = joblib.load(data_p)

    if preprocessor_path is not None:
        prep_p = Path(preprocessor_path)
        if prep_p.exists():
            loaded_dict['preprocessor'] = joblib.load(prep_p)

    return loaded_dict
