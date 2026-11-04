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


def extract_domain_features(
    df: pd.DataFrame,
    top_states: Optional[List[str]] = None
) -> pd.DataFrame:
    """Trích xuất các đặc trưng miền chuyên sâu ngành TMĐT (thời gian, địa lý, cờ nghiệp vụ)."""
    df_features = df.copy()

    # --- Đặc trưng thời gian: month, day_of_week, is_weekend ---
    if "Date" in df_features.columns and df_features["Date"].notna().any():
        dates = pd.to_datetime(df_features["Date"], format="mixed", errors="coerce")
        day_of_week = dates.dt.dayofweek
        df_features["month"] = dates.dt.month.fillna(4).astype(int)
        df_features["day_of_week"] = day_of_week.fillna(5).astype(int)
        df_features["is_weekend"] = day_of_week.isin([5, 6]).astype(int)
    else:
        # Giữ nguyên giá trị nếu đã có sẵn (từ input suy luận demo), ngược lại gán mặc định
        if "month" not in df_features.columns or df_features["month"].isna().all():
            df_features["month"] = 4
        else:
            df_features["month"] = df_features["month"].fillna(4).astype(int)

        if "day_of_week" not in df_features.columns or df_features["day_of_week"].isna().all():
            df_features["day_of_week"] = 5
        else:
            df_features["day_of_week"] = df_features["day_of_week"].fillna(5).astype(int)

        df_features["is_weekend"] = df_features["day_of_week"].isin([5, 6]).astype(int)

    # --- Cờ khuyến mãi: is_promoted ---
    if "promotion-ids" in df_features.columns:
        promotions = df_features["promotion-ids"].astype("string").str.strip()
        df_features["is_promoted"] = (
            promotions.notna()
            & promotions.ne("")
            & promotions.str.casefold().ne("no_promotion")
        ).astype(int)
    elif "is_promoted" not in df_features.columns:
        df_features["is_promoted"] = 0

    # --- Cờ khách sỉ: is_b2b ---
    if "B2B" in df_features.columns:
        b2b = df_features["B2B"].astype("string").str.strip().str.casefold()
        df_features["is_b2b"] = b2b.isin({"true", "1", "1.0", "yes", "y"}).astype(int)
    elif "is_b2b" not in df_features.columns:
        df_features["is_b2b"] = 0

    # --- Gom nhóm địa phương: Top 10 bang + 'Other' ---
    states = top_states if top_states is not None else TOP_10_STATES
    normalized_states = {str(state).strip().upper() for state in states}
    if "ship-state" in df_features.columns:
        ship_states = df_features["ship-state"].astype("string").str.strip().str.upper()
        df_features["state_grouped"] = ship_states.where(
            ship_states.isin(normalized_states), "Other"
        ).fillna("Other")
    elif "state_grouped" not in df_features.columns:
        df_features["state_grouped"] = "Other"

    return df_features


def transform_target_log(target: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Biến đổi log1p trên biến mục tiêu doanh số Amount để ổn định phương sai: y_log = log(1 + Amount)."""
    return np.log1p(np.asarray(target, dtype=np.float64))


def detect_collinear_features(
    df_features: pd.DataFrame,
    threshold: float = 0.6
) -> List[Tuple[str, str, float]]:
    """Quét ma trận tương quan Pearson trên các biến số để phát hiện các cặp đa cộng tuyến (|r| > threshold)."""
    numeric_df = df_features.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method="pearson")
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    collinear_pairs = []
    for col in upper_triangle.columns:
        for row in upper_triangle.index:
            r = upper_triangle.loc[row, col]
            if pd.notna(r) and abs(r) > threshold:
                collinear_pairs.append((row, col, round(float(r), 4)))

    return collinear_pairs


def split_train_test(
    df_features: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """Phân chia độc lập tập Train/Test theo tỷ lệ chỉ định kèm trích xuất biến mục tiêu Log."""
    target_col = "Amount"
    if target_col not in df_features.columns:
        raise KeyError(f"Thiếu cột mục tiêu '{target_col}' trong DataFrame.")

    # Loại bỏ biến đa cộng tuyến is_weekend (đã có day_of_week)
    features_to_drop = [target_col, "Status", "is_weekend"]
    drop_cols = [c for c in features_to_drop if c in df_features.columns]
    X = df_features.drop(columns=drop_cols)
    y_raw = df_features[target_col].to_numpy(dtype=np.float64)

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=test_size, random_state=random_state
    )

    y_train = pd.Series(transform_target_log(y_train_raw), index=X_train.index, name="Amount_log")
    y_test = pd.Series(transform_target_log(y_test_raw), index=X_test.index, name="Amount_log")

    return X_train, X_test, y_train, y_test, y_train_raw, y_test_raw


def build_column_transformer(
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """Thiết lập ColumnTransformer chuẩn scikit-learn (StandardScaler cho biến số, OneHotEncoder cho biến định tính)."""
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    if not transformers:
        raise ValueError("Không có cột nào để tiền xử lý.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def export_train_test_csv(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_raw: np.ndarray,
    y_test_raw: np.ndarray,
    output_dir: Union[str, Path]
) -> Tuple[Path, Path]:
    """Xuất hai tệp train.csv và test.csv kèm cột Amount thực tế phục vụ trực quan hóa."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = X_train.copy()
    test_data = X_test.copy()
    train_data["Amount"] = np.asarray(y_train_raw).reshape(-1)
    test_data["Amount"] = np.asarray(y_test_raw).reshape(-1)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    return train_path, test_path


def load_pipeline_and_splits(
    data_path: Union[str, Path],
    preprocessor_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Nạp bundle dữ liệu train/test và đối tượng tiền xử lý ColumnTransformer từ đĩa."""
    data_p = Path(data_path)
    if not data_p.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {data_p}")

    loaded_dict = joblib.load(data_p)
    if preprocessor_path is not None:
        prep_p = Path(preprocessor_path)
        if prep_p.exists():
            loaded_dict['preprocessor'] = joblib.load(prep_p)

    return loaded_dict


def prepare_sample_from_dict(sample_dict: Dict[str, Any]) -> pd.DataFrame:
    """Chuẩn bị DataFrame từ thông tin một đơn hàng thô để suy luận thời gian thực."""
    df_raw = pd.DataFrame([sample_dict])
    df_feat = extract_domain_features(df_raw)

    # Chuẩn hóa casing cho các thuộc tính danh mục để khớp OneHotEncoder
    if 'Category' in df_feat.columns and pd.notna(df_feat['Category'].iloc[0]):
        c_val = str(df_feat['Category'].iloc[0]).strip().lower()
        df_feat['Category'] = CATEGORY_LOOKUP.get(c_val, df_feat['Category'].iloc[0])

    if 'Fulfilment' in df_feat.columns and pd.notna(df_feat['Fulfilment'].iloc[0]):
        df_feat['Fulfilment'] = str(df_feat['Fulfilment'].iloc[0]).strip().title()

    if 'ship-service-level' in df_feat.columns and pd.notna(df_feat['ship-service-level'].iloc[0]):
        df_feat['ship-service-level'] = str(df_feat['ship-service-level'].iloc[0]).strip().title()

    if 'Courier Status' in df_feat.columns and pd.notna(df_feat['Courier Status'].iloc[0]):
        df_feat['Courier Status'] = str(df_feat['Courier Status'].iloc[0]).strip().title()

    if 'Size' in df_feat.columns and pd.notna(df_feat['Size'].iloc[0]):
        sz_val = str(df_feat['Size'].iloc[0]).strip().upper()
        df_feat['Size'] = 'Free' if sz_val == 'FREE' else sz_val

    feature_cols = [
        'Qty', 'month', 'day_of_week', 'is_promoted', 'is_b2b',
        'Category', 'Size', 'Fulfilment', 'ship-service-level',
        'Courier Status', 'state_grouped'
    ]
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0 if col in ['Qty', 'month', 'day_of_week', 'is_promoted', 'is_b2b'] else 'Other'

    return df_feat[feature_cols]
