"""
Module: model_evaluation.py
Dự án: Practice 2: Predicting Product Sales — Group 3
Phụ trách chính: Mai Hoàng Danh (14%)
Nhiệm vụ: Đánh giá độc lập trên tập Test (20.571 mẫu) với biến đổi nghịch đảo expm1,
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
    Đánh giá kiểm thử độc lập trên tập Test (20.571 mẫu).
    Áp dụng biến đổi nghịch đảo np.expm1 để chuyển đổi dự đoán log về giá trị tiền tệ thực tế (INR):
        y_pred_inr = np.clip(np.expm1(y_pred_log), 0.0, None)
    Tính toán 4 chỉ số thực tế: MSE, RMSE (INR), MAE (INR), R2 Score.

    Phụ trách chính: Mai Hoàng Danh (14%)

    Tham số:
        pipelines (Dict[str, Pipeline]): Từ điển chứa các Pipeline đã huấn luyện.
        X_test (pd.DataFrame): Dữ liệu đặc trưng tập kiểm thử.
        y_test_log (pd.Series): Biến mục tiêu không gian log np.log1p(Amount).
        y_test_raw (np.ndarray): Biến mục tiêu thực tế theo Rupee (INR).

    Trả về:
        pd.DataFrame: Bảng tổng kết hiệu năng với các cột ['Mô hình', 'R2 (Thực tế)', 'R2 (Log)', 'RMSE (INR)', 'MAE (INR)', 'MSE'].
    """
    # TODO: [Mai Hoàng Danh - 14%] Lập trình đánh giá độc lập Holdout Test:
    # 1. Khởi tạo results = []
    # 2. Duyệt qua từng (name, pipe) trong pipelines:
    #    - Dự đoán log: y_pred_log = pipe.predict(X_test)
    #    - Nghịch đảo ra tiền tệ: y_pred_inr = np.clip(np.expm1(y_pred_log), 0.0, None)
    #    - Tính toán metrics: mse_inr, rmse_inr = sqrt(mse), mae_inr, r2_inr, r2_log
    #    - Đóng gói vào dict và append vào results
    # 3. Trả về pd.DataFrame(results)
    raise NotImplementedError(
        "Chưa cài đặt - Mai Hoàng Danh (14%) cần hoàn thiện hàm 'evaluate_holdout_test' theo phân công trong docs/description/TASKS.md"
    )


def extract_feature_importances(
    pipeline: Pipeline,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Kỹ thuật Cô Tiên: Trích xuất trọng số đặc trưng quan trọng từ mô hình cây / rừng
    kết hợp tên đặc trưng sau khi biến đổi qua ColumnTransformer.

    Phụ trách chính: Mai Hoàng Danh (14%)

    Tham số:
        pipeline (Pipeline): Pipeline hoàn chỉnh đã huấn luyện.
        top_n (int): Số lượng đặc trưng hàng đầu cần trích xuất (mặc định: 15).

    Trả về:
        pd.DataFrame: Bảng chứa cột 'Feature' và 'Importance' sắp xếp giảm dần.
    """
    # TODO: [Mai Hoàng Danh - 14%] Lập trình trích xuất Feature Importances:
    # 1. Lấy preprocessor và regressor từ pipeline:
    #    preprocessor = pipeline.named_steps.get("preprocessor")
    #    regressor = pipeline.named_steps.get("regressor")
    # 2. Lấy tên đặc trưng qua preprocessor.get_feature_names_out()
    # 3. Lấy độ quan trọng qua regressor.feature_importances_ (hoặc abs(coef_) nếu Linear Regression)
    # 4. Rút gọn tên tiền tố 'cat__' và 'num__'
    # 5. Đóng gói vào DataFrame, sort descending theo Importance, lấy head(top_n)
    # 6. Trả về df_imp
    raise NotImplementedError(
        "Chưa cài đặt - Mai Hoàng Danh (14%) cần hoàn thiện hàm 'extract_feature_importances' theo phân công trong docs/description/TASKS.md"
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
    Phụ trách chính: Mai Hoàng Danh (14%)

    Tham số:
        model_name (str): Tên mô hình quán quân được nghiệm thu.
        metrics_dict (dict): Các chỉ số test đã tính toán.
        top_features (pd.DataFrame): Bảng đặc trưng quan trọng nhất.
        output_dir (Path): Thư mục lưu file (models/saved_models/).
        dataset_summary (dict, optional): Tóm tắt số mẫu train/test.

    Trả về:
        Path: Đường dẫn đến tệp metadata.json đã xuất.
    """
    # TODO: [Mai Hoàng Danh - 14%] Lập trình xuất siêu dữ liệu nghiệm thu:
    # 1. output_dir.mkdir(parents=True, exist_ok=True)
    # 2. Xây dựng dictionary meta_content gồm model_name, test_metrics, dataset_summary, top_features
    # 3. Ghi ra tệp metadata.json (indent=2, ensure_ascii=False)
    # 4. Trả về meta_path
    raise NotImplementedError(
        "Chưa cài đặt - Mai Hoàng Danh (14%) cần hoàn thiện hàm 'export_evaluation_artifacts' theo phân công trong docs/description/TASKS.md"
    )


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int = 1000
):
    """
    Trực quan hóa biểu đồ phân tán Thực tế vs Dự đoán (Actual vs Predicted) kèm đường lý tưởng y = ŷ.
    Phụ trách chính: Mai Hoàng Danh (14%)
    """
    # TODO: [Mai Hoàng Danh - 14%] Vẽ biểu đồ tán xạ Actual vs Predicted bằng Matplotlib/Seaborn
    raise NotImplementedError(
        "Chưa cài đặt - Mai Hoàng Danh (14%) cần hoàn thiện hàm 'plot_actual_vs_predicted' theo phân công trong docs/description/TASKS.md"
    )


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_points: int = 1000
):
    """
    Trực quan hóa biểu đồ phân tích phần dư (Residual Plot: ŷ vs e) kèm trục chuẩn e = 0.
    Phụ trách chính: Mai Hoàng Danh (14%)
    """
    # TODO: [Mai Hoàng Danh - 14%] Vẽ biểu đồ phần dư Residual Plot
    raise NotImplementedError(
        "Chưa cài đặt - Mai Hoàng Danh (14%) cần hoàn thiện hàm 'plot_residuals' theo phân công trong docs/description/TASKS.md"
    )
