"""
Module: model_evaluation.py
Dự án: Practice 2: Predicting Product Sales — Group 3
Phụ trách chính: Mai Hoàng Danh
Nhiệm vụ: Đánh giá độc lập trên tập Test với biến đổi nghịch đảo expm1,
vẽ biểu đồ Thực tế vs Dự đoán, Residual Plot, trích xuất Feature Importances và xuất metadata.json.
"""
import json
from pathlib import Path
import sys
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def evaluate_holdout_test(
    pipelines: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test_log: pd.Series,
    y_test_raw: np.ndarray
) -> pd.DataFrame:
    """
    Đánh giá kiểm thử độc lập trên tập Test với biến đổi nghịch đảo np.expm1 về thang đo tiền tệ thực tế INR.

    Tham số:
        pipelines (Dict[str, Pipeline]): Từ điển chứa các Pipeline đã huấn luyện.
        X_test (pd.DataFrame): Dữ liệu đặc trưng tập kiểm thử.
        y_test_log (pd.Series): Biến mục tiêu không gian log np.log1p(Amount).
        y_test_raw (np.ndarray): Biến mục tiêu thực tế theo Rupee (INR).

    Trả về:
        pd.DataFrame: Bảng tổng kết hiệu năng với các chỉ số R2 (Thực tế), R2 (Log), RMSE (INR), MAE (INR), MSE.
    """
    # TODO: [Mai Hoàng Danh] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Mai Hoàng Danh cần hoàn thiện hàm 'evaluate_holdout_test' theo phân công trong docs/description/TASKS.md"
    )


def extract_feature_importances(
    pipeline: Pipeline,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Trích xuất trọng số đặc trưng quan trọng nhất từ mô hình hồi quy.

    Tham số:
        pipeline (Pipeline): Pipeline hoàn chỉnh đã huấn luyện.
        top_n (int): Số lượng đặc trưng hàng đầu cần trích xuất (mặc định: 15).

    Trả về:
        pd.DataFrame: Bảng chứa cột 'Feature' và 'Importance' sắp xếp giảm dần.
    """
    # TODO: [Mai Hoàng Danh] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Mai Hoàng Danh cần hoàn thiện hàm 'extract_feature_importances' theo phân công trong docs/description/TASKS.md"
    )


def export_evaluation_artifacts(
    model_name: str,
    metrics_dict: Dict[str, Any],
    top_features: pd.DataFrame,
    output_dir: Path,
    dataset_summary: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Xuất tệp metadata.json nghiệm thu toàn diện các chỉ số kiểm định.

    Tham số:
        model_name (str): Tên mô hình quán quân được nghiệm thu.
        metrics_dict (dict): Các chỉ số test đã tính toán.
        top_features (pd.DataFrame): Bảng đặc trưng quan trọng nhất.
        output_dir (Path): Thư mục lưu file (models/saved_models/).
        dataset_summary (dict, optional): Tóm tắt số mẫu train/test.

    Trả về:
        Path: Đường dẫn đến tệp metadata.json đã xuất.
    """
    # TODO: [Mai Hoàng Danh] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Mai Hoàng Danh cần hoàn thiện hàm 'export_evaluation_artifacts' theo phân công trong docs/description/TASKS.md"
    )


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int = 1000
):
    """
    Trực quan hóa biểu đồ phân tán Thực tế vs Dự đoán (Actual vs Predicted) kèm đường lý tưởng y = ŷ.

    Tham số:
        y_true (np.ndarray): Giá trị thực tế.
        y_pred (np.ndarray): Giá trị dự đoán.
        max_points (int): Giới hạn số điểm hiển thị để biểu đồ gọn gàng (mặc định: 1000).
    """
    # TODO: [Mai Hoàng Danh] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Mai Hoàng Danh cần hoàn thiện hàm 'plot_actual_vs_predicted' theo phân công trong docs/description/TASKS.md"
    )


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int = 1000
):
    """
    Trực quan hóa biểu đồ phân tích phần dư (Residual Plot: ŷ vs e) kèm trục chuẩn e = 0.

    Tham số:
        y_true (np.ndarray): Giá trị thực tế.
        y_pred (np.ndarray): Giá trị dự đoán.
        max_points (int): Giới hạn số điểm hiển thị (mặc định: 1000).
    """
    # TODO: [Mai Hoàng Danh] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Mai Hoàng Danh cần hoàn thiện hàm 'plot_residuals' theo phân công trong docs/description/TASKS.md"
    )
