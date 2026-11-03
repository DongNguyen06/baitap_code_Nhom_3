import os
import re
import html
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Xử lý giá trị khuyết thiếu an toàn cho các cột subject, body và receiver."""
    df_clean = df.copy()
    df_clean['subject'] = df_clean['subject'].fillna('')
    df_clean['body'] = df_clean['body'].fillna('')
    df_clean['receiver'] = df_clean['receiver'].fillna('unknown')
    return df_clean


def clean_text(text: str) -> str:
    """
    Hàm chuẩn hóa NLP: Bóc tách mã HTML unescape, xóa URL, xóa ký tự đặc biệt,
    chuyển thành chữ thường và lọc từ dừng Stopwords.
    """
    if not isinstance(text, str) or not text.strip():
        return ''
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    words = [w for w in text.lower().split() if w not in STOP_WORDS and len(w) > 1]
    return ' '.join(words)


def process_and_save_clean_data(df: pd.DataFrame, output_path: str = None) -> tuple:
    """
    Thực hiện làm sạch văn bản toàn bộ email, khử trùng lặp nội dung sạch
    (Nguyên tắc Zero Duplicate Leakage) và lưu sang clean_emails.csv.
    """
    df_proc = handle_missing_values(df)
    if 'text_clean' not in df_proc.columns:
        df_proc['text_clean'] = (df_proc['subject'] + ' ' + df_proc['body']).apply(clean_text)

    # Khử trùng lặp nội dung văn bản sạch
    n_before = len(df_proc)
    df_proc = df_proc.drop_duplicates(subset=['text_clean']).reset_index(drop=True)
    n_removed = n_before - len(df_proc)

    if output_path is None:
        candidates = [
            Path(__file__).resolve().parent.parent / "processed" / "clean_emails.csv",
            Path("data/processed/clean_emails.csv"),
            Path("../data/processed/clean_emails.csv"),
            Path("../../data/processed/clean_emails.csv")
        ]
        for p in candidates:
            if p.parent.exists():
                output_path = str(p)
                break
        if output_path is None:
            output_path = str(candidates[0])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cols = ['sender', 'receiver', 'date', 'subject', 'body', 'text_clean', 'urls', 'label']
    df_proc[cols].to_csv(output_path, index=False, encoding='utf-8')
    return df_proc, n_removed, output_path
