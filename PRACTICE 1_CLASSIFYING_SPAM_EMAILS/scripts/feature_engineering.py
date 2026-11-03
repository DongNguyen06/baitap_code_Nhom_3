from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from data.scripts.data_preprocessing import clean_text

BASE_DIR = Path(__file__).resolve().parent.parent


def load_clean_data(clean_path: str = None) -> pd.DataFrame:
    """Nạp tệp dữ liệu email sạch clean_emails.csv."""
    path = Path(clean_path) if clean_path else BASE_DIR / "data" / "processed" / "clean_emails.csv"
    if not path.exists():
        path = Path("data/processed/clean_emails.csv")
    return pd.read_csv(path)


def extract_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trích xuất 7 đặc trưng miền nghiệp vụ:
    1. text_len: Chiều dài ký tự chuỗi văn bản sạch
    2. log_text_len: ln(1 + text_len) nén phân phối lệch phải
    3. caps_ratio: Tỷ lệ ký tự viết hoa trong văn bản gốc
    4. exclamation_count: Số dấu chấm than (!)
    5. dollar_count: Số ký hiệu tiền tệ ($)
    6. has_re_prefix: Có tiền tố đối thoại 'Re:' ở tiêu đề hay không
    7. subject_exclamation: Tiêu đề có chứa dấu (!) hay không
    """
    df_feat = df.copy()
    subj = df_feat["subject"].fillna("").astype(str)
    body = df_feat["body"].fillna("").astype(str)
    raw_text = subj + " " + body
    raw_len = raw_text.str.len().replace(0, 1)

    clean_series = df_feat["text_clean"].fillna("").astype(str) if "text_clean" in df_feat.columns else raw_text.apply(clean_text)
    df_feat["text_clean"] = clean_series
    df_feat["text_len"] = clean_series.str.len()
    df_feat["log_text_len"] = np.log1p(df_feat["text_len"])

    caps_count = raw_text.apply(lambda s: sum(1 for c in s if c.isupper()))
    df_feat["caps_ratio"] = (caps_count / raw_len).clip(0.0, 1.0)
    df_feat["exclamation_count"] = raw_text.str.count("!")
    df_feat["dollar_count"] = raw_text.str.count(r"\$")
    df_feat["has_re_prefix"] = subj.str.contains(r"^\s*re:", case=False, regex=True).astype(int)
    df_feat["subject_exclamation"] = subj.str.contains("!", regex=False).astype(int)

    return df_feat


def split_stratified_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Phân chia Train/Test 80/20 phân tầng (Stratified Split) bảo toàn tỷ lệ nhãn."""
    feature_cols = [
        "text_clean", "caps_ratio", "exclamation_count", "dollar_count",
        "text_len", "log_text_len", "has_re_prefix", "subject_exclamation"
    ]
    X = df[feature_cols]
    y = df["label"].values
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_column_transformer(max_features: int = 5000, k_best: int = 2000, ngram_range: tuple = (1, 2)) -> ColumnTransformer:
    """
    Xây dựng Pipeline ColumnTransformer gồm:
    - Nhánh Text: TfidfVectorizer(max_features=5000, ngram=(1, 2)) + SelectKBest(chi2, k=2000)
    - Nhánh Số: MinMaxScaler trên 7 đặc trưng miền (bao gồm log_text_len)
    """
    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words="english")),
        ("select", SelectKBest(score_func=chi2, k=k_best))
    ])

    numeric_cols = [
        "caps_ratio", "exclamation_count", "dollar_count",
        "text_len", "log_text_len", "has_re_prefix", "subject_exclamation"
    ]
    numeric_pipeline = Pipeline([("scaler", MinMaxScaler())])

    return ColumnTransformer([
        ("text", text_pipeline, "text_clean"),
        ("num", numeric_pipeline, numeric_cols)
    ])


def export_train_test_csv(X_train: pd.DataFrame, X_test: pd.DataFrame,
                          y_train: pd.Series, y_test: pd.Series,
                          output_dir: str = None) -> tuple:
    """
    Xuất độc lập 2 tệp train.csv (80% - 26.128 mẫu) và test.csv (20% - 6.532 mẫu)
    kèm cột nhãn 'label', có thể mở trực tiếp bằng Excel/Notepad để kiểm tra trực quan.
    """
    out_dir = Path(output_dir) if output_dir else BASE_DIR / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df["label"] = np.asarray(y_train)

    test_df = X_test.copy()
    test_df["label"] = np.asarray(y_test)

    train_csv_path = out_dir / "train.csv"
    test_csv_path = out_dir / "test.csv"

    train_df.to_csv(train_csv_path, index=False, encoding="utf-8")
    test_df.to_csv(test_csv_path, index=False, encoding="utf-8")

    return str(train_csv_path), str(test_csv_path)


def save_pipeline_and_splits(preprocessor: ColumnTransformer, X_train, X_test, y_train, y_test,
                             prep_path: str = None, data_path: str = None,
                             export_csv: bool = True) -> tuple:
    """Lưu preprocessor.joblib, train_test_data.joblib và tự động xuất train.csv, test.csv vào data/processed."""
    p_path = Path(prep_path) if prep_path else BASE_DIR / "models" / "saved_models" / "preprocessor.joblib"
    d_path = Path(data_path) if data_path else BASE_DIR / "data" / "processed" / "train_test_data.joblib"

    p_path.parent.mkdir(parents=True, exist_ok=True)
    d_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, p_path)
    joblib.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}, d_path)

    if export_csv:
        export_train_test_csv(X_train, X_test, y_train, y_test, output_dir=str(d_path.parent))

    return str(p_path), str(d_path)


def load_pipeline_and_splits(prep_path: str = None, data_path: str = None) -> tuple:
    """Nạp preprocessor.joblib và bộ dữ liệu train_test_data.joblib."""
    p_path = Path(prep_path) if prep_path else BASE_DIR / "models" / "saved_models" / "preprocessor.joblib"
    d_path = Path(data_path) if data_path else BASE_DIR / "data" / "processed" / "train_test_data.joblib"
    if not p_path.exists():
        candidates_p = [Path("models/saved_models/preprocessor.joblib"), Path("../models/saved_models/preprocessor.joblib")]
        for c in candidates_p:
            if c.exists():
                p_path = c
                break
    if not d_path.exists():
        candidates_d = [Path("data/processed/train_test_data.joblib"), Path("../data/processed/train_test_data.joblib")]
        for c in candidates_d:
            if c.exists():
                d_path = c
                break

    preprocessor = joblib.load(p_path)
    data = joblib.load(d_path)
    return preprocessor, data["X_train"], data["X_test"], data["y_train"], data["y_test"]


def prepare_email_from_text(raw_text: str) -> pd.DataFrame:
    """Bóc tách chuỗi email thô (Subject & Body) và trích xuất trọn vẹn 7 đặc trưng miền."""
    raw_stripped = raw_text.strip()
    subj_match = re.search(r"(?i)^\s*subject:\s*(.*)$", raw_stripped, re.MULTILINE)
    if subj_match:
        subject = subj_match.group(1).strip()
        body = re.sub(r"(?i)^\s*subject:\s*.*$", "", raw_stripped, count=1, flags=re.MULTILINE).strip()
    else:
        lines = [line.strip() for line in raw_stripped.splitlines() if line.strip()]
        subject = lines[0] if lines else ""
        body = " ".join(lines[1:]) if len(lines) > 1 else ""

    combined = f"{subject} {body}".strip()
    df_raw = pd.DataFrame([{
        "subject": subject,
        "body": body,
        "text_clean": clean_text(combined)
    }])
    return extract_domain_features(df_raw)
