"""
CLI Inference Script - Classifying Spam Emails (SpamGuard-ML)
Hệ thống: Học máy Phân loại Thư rác Tự động
"""

import sys
import os
from pathlib import Path

# Tự động cấu hình chuẩn UTF-8 cho console Windows tránh UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import pandas as pd

# Tự động xác định đường dẫn tuyệt đối đến file mô hình
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = str(BASE_DIR / "models" / "best_model.joblib")

def predict_email(email_text: str, model_path: str = DEFAULT_MODEL_PATH):
    """
    Dự đoán nhãn email (Spam / Ham) từ chuỗi văn bản.
    """
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy mô hình tại: {model_path}. Hãy chạy Notebook 3 trước!")
        return None
    
    model = joblib.load(model_path)
    pred = model.predict([email_text])[0]
    proba = model.predict_proba([email_text])[0]
    
    label = "SPAM (1)" if pred == 1 else "HAM (0)"
    return {
        "label": label,
        "spam_probability": float(proba[1])
    }

if __name__ == "__main__":
    print("=== CLI PHÂN LOẠI EMAIL SPAM (SPAMGUARD-ML) ===")
    sample_email = "Congratulations! You won a free lottery prize. Click http://claim.now to receive your reward."
    print(f"Email mẫu: {sample_email}\n")
    
    result = predict_email(sample_email)
    if result:
        print(f"✅ Kết quả dự đoán: {result['label']}")
        print(f"📊 Xác suất Spam: {result['spam_probability'] * 100:.2f}%")
