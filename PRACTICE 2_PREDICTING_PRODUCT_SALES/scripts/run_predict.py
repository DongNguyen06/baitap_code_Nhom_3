"""
CLI Inference Script - Predicting Product Sales
Nhóm 3 - Học máy (Đại học Giao thông Vận tải)
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np

def predict(input_data: dict, model_path: str = "models/best_model.joblib"):
    """
    Dự đoán doanh số bán hàng từ input dictionary.
    """
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy mô hình tại: {model_path}. Hãy chạy notebook huấn luyện trước!")
        return None
    
    model = joblib.load(model_path)
    df_input = pd.DataFrame([input_data])
    prediction = model.predict(df_input)
    return float(prediction[0])

if __name__ == "__main__":
    print("=== CLI DỰ ĐOÁN DOANH SỐ BÁN HÀNG SẢN PHẨM ===")
    sample_input = {
        "Price": 150.0,
        "Marketing_Spend": 500.0,
        "Category": "Electronics"
    }
    pred = predict(sample_input)
    if pred is not None:
        print(f"✅ Doanh số dự đoán: {pred:.2f}")
