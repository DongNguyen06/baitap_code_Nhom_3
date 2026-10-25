"""
CLI Inference Script - Classifying Spam Emails (SpamGuard-ML)
Hệ thống: Học máy Phân loại Thư rác Tự động
"""

import sys
import os
import re
import html
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Cấu hình UTF-8 cho console Windows tránh lỗi Unicode
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

STOP_WORDS = set(ENGLISH_STOP_WORDS)
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = str(BASE_DIR / "models" / "best_model.joblib")


def clean_text(text: str) -> str:
    """Tiền xử lý làm sạch văn bản NLP theo chuẩn Pipeline."""
    text = html.unescape(str(text))
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return ' '.join(tokens)


def extract_features(raw_text: str) -> pd.DataFrame:
    """Trích xuất DataFrame chứa text_clean và 6 đặc trưng miền giá trị (domain features)."""
    raw_text_stripped = raw_text.strip()
    
    # Tự động tách subject và body
    subj_match = re.search(r'(?i)^\s*subject:\s*(.*)$', raw_text_stripped, re.MULTILINE)
    if subj_match:
        subject = subj_match.group(1).strip()
        body = re.sub(r'(?i)^\s*subject:\s*.*$', '', raw_text_stripped, count=1, flags=re.MULTILINE).strip()
    else:
        lines = [line.strip() for line in raw_text_stripped.splitlines() if line.strip()]
        if len(lines) > 1:
            subject = lines[0]
            body = ' '.join(lines[1:])
        elif len(lines) == 1:
            subject = lines[0]
            body = ''
        else:
            subject = ''
            body = ''

    combined_text = f"{subject} {body}".strip()
    text_clean_val = clean_text(combined_text)
    raw_len = len(combined_text)
    caps_count = sum(1 for c in combined_text if c.isupper())
    caps_ratio = (caps_count / raw_len) if raw_len > 0 else 0.0
    exclamation_count = combined_text.count('!')
    dollar_count = combined_text.count('$')
    text_len_val = len(text_clean_val)
    has_re_prefix = 1 if bool(re.search(r'^\s*re:', str(subject), re.IGNORECASE)) else 0
    subject_exclamation = 1 if '!' in str(subject) else 0

    return pd.DataFrame([{
        'text_clean': text_clean_val,
        'caps_ratio': float(caps_ratio),
        'exclamation_count': int(exclamation_count),
        'dollar_count': int(dollar_count),
        'text_len': int(text_len_val),
        'has_re_prefix': int(has_re_prefix),
        'subject_exclamation': int(subject_exclamation)
    }])


def predict_email(email_text: str, model_path: str = DEFAULT_MODEL_PATH):
    """
    Dự đoán nhãn email (Spam / Ham) từ chuỗi văn bản với cơ chế Guardrail phòng vệ.
    """
    raw_stripped = email_text.strip()
    # Guardrail phòng vệ cho văn bản rỗng hoặc quá ngắn
    if len(raw_stripped) < 5:
        return {
            "status": "warning",
            "message": "Văn bản quá ngắn hoặc rỗng, không đủ thông tin tin cậy để phân loại.",
            "label": "UNCERTAIN / UNKNOWN",
            "spam_probability": 0.0
        }

    df_features = extract_features(raw_stripped)
    row = df_features.iloc[0]

    # Guardrail: nếu không chứa từ vựng và không có tín hiệu nguy cơ ngoại quan ($ ! Viết hoa)
    if not row['text_clean'] and row['dollar_count'] == 0 and row['exclamation_count'] == 0 and row['caps_ratio'] == 0.0:
        return {
            "status": "warning",
            "message": "Văn bản không chứa từ khóa có nghĩa sau khi lọc từ dừng và không có tín hiệu nguy cơ.",
            "label": "HAM (0)",
            "spam_probability": 0.05,
            "features": row.to_dict()
        }

    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy mô hình tại: {model_path}. Hãy chạy Notebook 3 trước!")
        return None

    model = joblib.load(model_path)

    # Dự đoán qua Pipeline
    pred = model.predict(df_features)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df_features)[0]
        spam_prob = float(proba[1])
    else:
        spam_prob = 0.95 if pred == 1 else 0.05

    label = "SPAM (1)" if pred == 1 else "HAM (0)"
    return {
        "status": "success",
        "label": label,
        "spam_probability": spam_prob,
        "features": df_features.to_dict(orient="records")[0]
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CLI Phân loại Email Spam (SpamGuard-ML)")
    parser.add_argument("email", nargs="?", default=None, help="Văn bản email cần kiểm tra")
    parser.add_argument("--text", "-t", type=str, default=None, help="Văn bản email cần kiểm tra")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL_PATH, help="Đường dẫn tới file model (.joblib)")
    args = parser.parse_args()

    input_text = args.text or args.email or "Congratulations! You won a free lottery prize of $50,000. Click http://claim.now to receive your reward immediately!!!"
    print("=== CLI PHÂN LOẠI EMAIL SPAM (SPAMGUARD-ML) ===")
    print(f"Email kiểm tra: {input_text}\n")

    result = predict_email(input_text, model_path=args.model)
    if result:
        print(f"🎯 Kết quả dự đoán: {result.get('label', 'N/A')}")
        if result.get('status') == 'success':
            print(f"📊 Xác suất Spam: {result['spam_probability'] * 100:.2f}%")
            print(f"🔍 Đặc trưng trích xuất: {result.get('features', {})}")
        else:
            print(f"⚠️ Thông báo: {result.get('message', '')}")
