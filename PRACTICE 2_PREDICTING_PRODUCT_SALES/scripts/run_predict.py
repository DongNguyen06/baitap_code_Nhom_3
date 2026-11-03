"""
CLI Inference Script - Predicting Product Sales
Nhóm 3 - Học máy (Đại học Giao thông Vận tải)
"""
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.feature_engineering import prepare_sample_from_dict


def predict(input_data: dict, model_path: Path = None) -> float:
    """
    Dự đoán doanh số bán hàng từ input dictionary.
    """
    if model_path is None:
        model_path = BASE_DIR / "models" / "saved_models" / "best_model.joblib"
        if not model_path.exists():
            model_path = BASE_DIR / "models" / "best_model.joblib"

    if not model_path.exists():
        print(f"❌ Không tìm thấy mô hình tại: {model_path}. Hãy chạy scripts/run_pipeline.py trước!")
        return None

    model = joblib.load(model_path)
    df_sample = prepare_sample_from_dict(input_data)
    pred_log = model.predict(df_sample)[0]
    pred_inr = float(np.clip(np.expm1(pred_log), 0.0, None))
    return pred_inr


if __name__ == "__main__":
    print("=== CLI DỰ ĐOÁN DOANH SỐ BÁN HÀNG SẢN PHẨM ===")
    sample_input = {
        "Category": "Kurta",
        "Size": "M",
        "Qty": 1,
        "Fulfilment": "Merchant",
        "ship-service-level": "Standard",
        "Courier Status": "Shipped",
        "ship-state": "MAHARASHTRA",
        "month": 4,
        "day_of_week": 5,
        "is_promoted": 0,
        "is_b2b": 0
    }
    pred = predict(sample_input)
    if pred is not None:
        print(f"✅ Doanh số dự đoán: {pred:,.2f} INR")
