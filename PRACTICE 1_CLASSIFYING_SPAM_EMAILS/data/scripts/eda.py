import os
import re
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def load_raw_data(data_path: str = None) -> pd.DataFrame:
    """Nạp tập dữ liệu email thô CEAS_08.csv."""
    if data_path is None:
        # Tự động dò tìm đường dẫn CEAS_08.csv
        candidates = [
            Path(__file__).resolve().parent.parent / "raw" / "CEAS_08.csv",
            Path("data/raw/CEAS_08.csv"),
            Path("../data/raw/CEAS_08.csv"),
            Path("../../data/raw/CEAS_08.csv")
        ]
        for p in candidates:
            if p.exists():
                data_path = str(p)
                break
        if data_path is None:
            data_path = str(candidates[0])
    df = pd.read_csv(data_path)
    return df


def inspect_data_structure(df: pd.DataFrame) -> dict:
    """Khảo sát quy mô, kiểu dữ liệu, khuyết thiếu và trùng lặp kỹ thuật ban đầu."""
    return {
        "shape": df.shape,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_values": df.isnull().sum(),
        "exact_duplicates": int(df.duplicated().sum()),
        "subject_body_duplicates": int(df.duplicated(subset=['subject', 'body']).sum())
    }


def get_univariate_stats(df: pd.DataFrame) -> dict:
    """Thống kê đơn biến: phân phối nhãn, tỷ lệ url và độ dài subject."""
    label_counts = df['label'].value_counts()
    url_counts = df['urls'].value_counts()
    subj_lens = df['subject'].fillna('').astype(str).str.len()

    return {
        "ham_count": int(label_counts.get(0, 0)),
        "spam_count": int(label_counts.get(1, 0)),
        "ham_ratio": float(label_counts.get(0, 0) / len(df) * 100),
        "spam_ratio": float(label_counts.get(1, 0) / len(df) * 100),
        "url_present_ratio": float((df['urls'] == 1).mean() * 100),
        "subject_len_mean": float(subj_lens.mean()),
        "subject_len_median": float(subj_lens.median())
    }


def detect_iqr_outliers(series: pd.Series) -> dict:
    """Xác định số lượng và tỷ lệ ngoại lai theo quy tắc khoảng tứ phân vị IQR (1.5 * IQR)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    outliers_mask = series > upper_bound
    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "upper_bound": float(upper_bound),
        "n_outliers": int(outliers_mask.sum()),
        "outlier_ratio": float(outliers_mask.mean() * 100)
    }


def extract_keywords(series: pd.Series, top_n: int = 10) -> list:
    """Trích xuất danh sách top_n từ khóa xuất hiện nhiều nhất từ Series văn bản."""
    words = []
    for text in series.dropna().astype(str):
        tokens = [w for w in re.sub(r'[^a-zA-Z\s]', ' ', text).lower().split() if w not in STOP_WORDS and len(w) > 2]
        words.extend(tokens)
    return Counter(words).most_common(top_n)


def get_top_keywords_comparison(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Tạo bảng DataFrame đối sánh từ khóa tiêu đề phổ biến giữa Ham (0) và Spam (1)."""
    top_ham = pd.DataFrame(extract_keywords(df[df['label'] == 0]['subject'], top_n=top_n), columns=['Từ khóa Ham', 'Tần suất'])
    top_spam = pd.DataFrame(extract_keywords(df[df['label'] == 1]['subject'], top_n=top_n), columns=['Từ khóa Spam', 'Tần suất'])
    return pd.concat([top_ham, top_spam], axis=1)


def analyze_re_prefix_and_duplicates(df: pd.DataFrame) -> dict:
    """Điều tra hành vi phát tán lặp lại và dấu hiệu hội thoại qua tiền tố 'Re:'."""
    dup_mask = df.duplicated(subset=['sender', 'subject'], keep='first')
    dup_ham = int((dup_mask & (df['label'] == 0)).sum())
    dup_spam = int((dup_mask & (df['label'] == 1)).sum())

    subj_lower = df['subject'].fillna('').str.lower()
    has_re = subj_lower.str.contains(r'\bre:')
    re_ham = float((has_re & (df['label'] == 0)).sum() / (df['label'] == 0).sum() * 100)
    re_spam = float((has_re & (df['label'] == 1)).sum() / (df['label'] == 1).sum() * 100)

    return {
        "n_duplicates": int(dup_mask.sum()),
        "dup_ham": dup_ham,
        "dup_spam": dup_spam,
        "re_ham_ratio": re_ham,
        "re_spam_ratio": re_spam
    }
