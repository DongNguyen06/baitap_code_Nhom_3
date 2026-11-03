"""
Module: feature_engineering.py
Dự án: Practice 2: Predicting Product Sales — Group 3
Phân công nhiệm vụ:
- Trương Thành Công (14%): Trích xuất đặc trưng miền (extract_domain_features: mùa vụ, is_promoted, is_b2b, state_grouped).
- Võ Thanh Phú (14%): Phân chia Train/Test 80/20 (split_train_test, export_train_test_csv), quét đa cộng tuyến (detect_collinear_features), đóng gói ColumnTransformer (build_column_transformer).
- Nguyễn Xuân Đông (Trưởng nhóm - 16%): Chuẩn bị mẫu suy luận demo (prepare_sample_from_dict) & Nạp dữ liệu (load_pipeline_and_splits).
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
# PHẦN 1: TRƯƠNG THÀNH CÔNG (14%) — TRÍCH XUẤT ĐẶC TRƯNG MIỀN (TODO)
# =============================================================================

def extract_domain_features(
    df: pd.DataFrame,
    top_states: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Trích xuất các đặc trưng miền chuyên sâu ngành Thương mại điện tử:
    1. Thời gian & Mùa vụ: month, day_of_week, is_weekend (từ cột 'Date')
    2. Gom nhóm địa lý: ship-state thuộc Top 10 bang lớn giữ nguyên, ngoài ra gán là 'Other'
    3. Cờ nhị phân nghiệp vụ:
       - is_promoted: 1 nếu có khuyến mãi (!= 'No_Promotion' và notna), ngược lại 0
       - is_b2b: 1 nếu khách hàng sỉ B2B (True), ngược lại 0

    Phụ trách chính: Trương Thành Công (14%)

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu đầu vào.
        top_states (List[str], optional): Danh sách các bang lớn (mặc định: TOP_10_STATES).

    Trả về:
        pd.DataFrame: DataFrame có thêm các cột đặc trưng mới.
    """
    # TODO: [Trương Thành Công - 14%] Lập trình trích xuất đặc trưng miền:
    # 1. Tạo bản sao: df_feat = df.copy()
    # 2. Xử lý thời gian từ 'Date':
    #    dt_series = pd.to_datetime(df_feat['Date'], format='mixed', errors='coerce')
    #    df_feat['month'] = dt_series.dt.month.fillna(4).astype(int)
    #    df_feat['day_of_week'] = dt_series.dt.dayofweek.fillna(0).astype(int)
    #    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    # 3. Gom nhóm địa phương:
    #    valid_top = set(top_states) if top_states is not None else set(TOP_10_STATES)
    #    df_feat['state_grouped'] = df_feat['ship-state'].apply(
    #        lambda s: str(s).strip().upper() if str(s).strip().upper() in valid_top else 'Other'
    #    )
    # 4. Tạo cờ nhị phân is_promoted và is_b2b
    # 5. Trả về df_feat
    raise NotImplementedError(
        "Chưa cài đặt - Trương Thành Công (14%) cần hoàn thiện hàm 'extract_domain_features' theo phân công trong docs/description/TASKS.md"
    )


def transform_target_log(
    target: Union[pd.Series, np.ndarray]
) -> np.ndarray:
    """
    Biến đổi log1p trên biến mục tiêu doanh số Amount để ổn định số học và giảm độ lệch:
        y_log = log(1 + target)
    Phụ trách chính: Trương Thành Công (14%)

    Tham số:
        target (pd.Series | np.ndarray): Chuỗi giá trị doanh số thực tế (Amount).

    Trả về:
        np.ndarray: Mảng giá trị sau khi áp dụng phép biến đổi log1p.
    """
    # TODO: [Trương Thành Công - 14%] Lập trình biến đổi Log cho biến mục tiêu:
    # 1. Chuyển đổi target sang dạng numpy array kiểu float: y_arr = np.asarray(target, dtype=np.float64)
    # 2. Áp dụng np.log1p(y_arr)
    # 3. Trả về mảng y_log
    raise NotImplementedError(
        "Chưa cài đặt - Trương Thành Công (14%) cần hoàn thiện hàm 'transform_target_log' theo phân công trong docs/description/TASKS.md"
    )


# =============================================================================
# PHẦN 2: VÕ THANH PHÚ (14%) — ĐA CỘNG TUYẾN, TRAIN/TEST SPLIT & COLUMNTRANSFORMER (TODO)
# =============================================================================

def detect_collinear_features(
    df_features: pd.DataFrame,
    threshold: float = 0.6
) -> List[Tuple[str, str, float]]:
    """
    Kỹ thuật Cô Tiên: Quét ma trận tương quan Pearson trên các biến số
    để phát hiện các cặp đặc trưng có hiện tượng đa cộng tuyến (|r| > threshold).

    Phụ trách chính: Võ Thanh Phú (14%)

    Tham số:
        df_features (pd.DataFrame): DataFrame chứa các đặc trưng.
        threshold (float): Ngưỡng tương quan Pearson xem là đa cộng tuyến (mặc định: 0.6).

    Trả về:
        List[Tuple[str, str, float]]: Danh sách các bộ ba (tên_biến_1, tên_biến_2, hệ_số_r).
    """
    # TODO: [Võ Thanh Phú - 14%] Lập trình hàm phát hiện đa cộng tuyến:
    # 1. Lọc các cột số: num_cols = df_features.select_dtypes(include=[np.number]).columns
    # 2. Tính ma trận tương quan: corr_matrix = df_features[num_cols].corr()
    # 3. Lặp qua các cặp biến (i, j với j > i):
    #    Nếu abs(corr_matrix.iloc[i, j]) > threshold:
    #        thêm (col_i, col_j, round(r, 4)) vào danh sách kết quả
    # 4. Trả về collinear_pairs
    raise NotImplementedError(
        "Chưa cài đặt - Võ Thanh Phú (14%) cần hoàn thiện hàm 'detect_collinear_features' theo phân công trong docs/description/TASKS.md"
    )


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """
    Phân chia Train / Test theo tỷ lệ 80/20 (82.281 train / 20.571 test) làm hàng rào chống rò rỉ dữ liệu.

    Target:
    - y_train, y_test: np.log1p(Amount) dùng để huấn luyện mô hình ổn định số học
    - y_train_raw, y_test_raw: Amount gốc tính bằng INR để đối chiếu nghiệm thu thực tế

    Phụ trách chính: Võ Thanh Phú (14%)

    Tham số:
        df (pd.DataFrame): DataFrame dữ liệu sau khi trích xuất đặc trưng.
        test_size (float): Tỷ lệ tập kiểm thử (mặc định: 0.2).
        random_state (int): Hạt giống ngẫu nhiên (mặc định: 42).

    Trả về:
        Tuple: (X_train, X_test, y_train, y_test, y_train_raw, y_test_raw)
    """
    # TODO: [Võ Thanh Phú - 14%] Lập trình phân tách Train/Test 80/20:
    # 1. Định nghĩa feature_cols = [
    #        'Qty', 'month', 'day_of_week', 'is_promoted', 'is_b2b',
    #        'Category', 'Size', 'Fulfilment', 'ship-service-level',
    #        'Courier Status', 'state_grouped'
    #    ]
    # 2. Lấy X = df[valid_feature_cols].copy()
    # 3. Lấy y_raw = df['Amount'].values và y_log = np.log1p(df['Amount'])
    # 4. Phân chia bằng train_test_split(X, y_log, y_raw, test_size=test_size, random_state=random_state)
    # 5. Trả về (X_train, X_test, y_train, y_test, y_train_raw, y_test_raw)
    raise NotImplementedError(
        "Chưa cài đặt - Võ Thanh Phú (14%) cần hoàn thiện hàm 'split_train_test' theo phân công trong docs/description/TASKS.md"
    )


def build_column_transformer(
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Thiết lập ColumnTransformer chuẩn scikit-learn:
    - StandardScaler cho các biến số độc lập
    - OneHotEncoder(handle_unknown='ignore', sparse_output=False) cho biến phân loại

    Phụ trách chính: Võ Thanh Phú (14%)

    Tham số:
        numeric_cols (List[str]): Danh sách tên cột số.
        categorical_cols (List[str]): Danh sách tên cột định tính.

    Trả về:
        ColumnTransformer: Bộ tiền xử lý hoàn chỉnh sẵn sàng ghép vào Pipeline.
    """
    # TODO: [Võ Thanh Phú - 14%] Lập trình bộ ColumnTransformer:
    # 1. numeric_transformer = StandardScaler()
    # 2. categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # 3. preprocessor = ColumnTransformer(transformers=[
    #        ('num', numeric_transformer, numeric_cols),
    #        ('cat', categorical_transformer, categorical_cols)
    #    ])
    # 4. Trả về preprocessor
    raise NotImplementedError(
        "Chưa cài đặt - Võ Thanh Phú (14%) cần hoàn thiện hàm 'build_column_transformer' theo phân công trong docs/description/TASKS.md"
    )


def export_train_test_csv(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
    output_dir: Union[str, Path]
) -> Tuple[Path, Path]:
    """
    Xuất hai tệp train.csv và test.csv kèm cột Amount thực tế phục vụ mở Excel trực quan.
    Phụ trách chính: Võ Thanh Phú (14%)

    Tham số:
        X_train, X_test: DataFrame đặc trưng của tập Train và Test.
        y_train_raw, y_test_raw: Mảng giá trị Amount thực tế tương ứng.
        output_dir: Thư mục lưu tệp (data/processed/).

    Trả về:
        Tuple[Path, Path]: (đường_dẫn_train_csv, đường_dẫn_test_csv).
    """
    # TODO: [Võ Thanh Phú - 14%] Lập trình xuất dữ liệu train.csv và test.csv:
    # 1. Tạo thư mục output_dir nếu chưa có: out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # 2. Ghép cột Amount: train_df = X_train.copy(); train_df['Amount'] = y_train_raw
    # 3. Xuất train.csv: train_df.to_csv(out_dir / "train.csv", index=False, encoding='utf-8')
    # 4. Ghép cột Amount: test_df = X_test.copy(); test_df['Amount'] = y_test_raw
    # 5. Xuất test.csv: test_df.to_csv(out_dir / "test.csv", index=False, encoding='utf-8')
    # 6. Trả về (train_path, test_path)
    raise NotImplementedError(
        "Chưa cài đặt - Võ Thanh Phú (14%) cần hoàn thiện hàm 'export_train_test_csv' theo phân công trong docs/description/TASKS.md"
    )


# =============================================================================
# PHẦN 3: NGUYỄN XUÂN ĐÔNG (TRƯỞNG NHÓM - 16%) — HỖ TRỢ SUY LUẬN DEMO (ĐÃ HOÀN THIỆN)
# =============================================================================

def prepare_sample_from_dict(sample_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Chuẩn bị DataFrame từ thông tin một đơn hàng thô để suy luận thời gian thực.
    Tự động chuẩn hóa casing khớp với OneHotEncoder đã học từ dữ liệu.
    Tác giả: Nguyễn Xuân Đông (Trưởng nhóm - 16%)
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
    Tác giả: Nguyễn Xuân Đông (Trưởng nhóm - 16%)
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
