"""
CLI Inference Script - Classifying Spam Emails (SpamGuard-ML)
Nhóm 3 - Học máy (Đại học Giao thông Vận tải TP.HCM)
"""

import sys
import os
import joblib
import pandas as pd

def predict_email(email_text: str, model_path: str = "models/best_model.joblib"):
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
