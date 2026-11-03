"""
Module: model_training.py
Dự án: Practice 2: Predicting Product Sales — Group 3
Phụ trách chính: Võ Khôi Nguyên
Nhiệm vụ: Khởi tạo mô hình Scratch, đóng gói Pipeline chuẩn chống rò rỉ dữ liệu, đánh giá 5-Fold Cross-Validation và huấn luyện.
"""
from pathlib import Path
import sys
from typing import Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
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

from models.algorithms import (
    LinearRegressionScratch,
    DecisionTreeRegressorScratch,
    RandomForestRegressorScratch
)


def create_scratch_models() -> Dict[str, Any]:
    """
    Khởi tạo từ điển 3 mô hình hồi quy From Scratch tự viết bằng NumPy thuần túy.

    Trả về:
        Dict[str, Any]: Từ điển chứa 3 đối tượng mô hình Scratch ('Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor').
    """
    # TODO: [Võ Khôi Nguyên] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Võ Khôi Nguyên cần hoàn thiện hàm 'create_scratch_models' theo phân công trong docs/description/TASKS.md"
    )


def build_training_pipelines(
    preprocessor: Any,
    models: Optional[Dict[str, Any]] = None
) -> Dict[str, Pipeline]:
    """
    Đóng gói Pipeline chuẩn chống rò rỉ dữ liệu kết hợp tiền xử lý và mô hình hồi quy.

    Tham số:
        preprocessor: Đối tượng ColumnTransformer tiền xử lý dữ liệu.
        models (Dict[str, Any], optional): Từ điển mô hình. Mặc định gọi create_scratch_models().

    Trả về:
        Dict[str, Pipeline]: Từ điển chứa các Pipeline huấn luyện hoàn chỉnh.
    """
    # TODO: [Võ Khôi Nguyên] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Võ Khôi Nguyên cần hoàn thiện hàm 'build_training_pipelines' theo phân công trong docs/description/TASKS.md"
    )


def evaluate_cv_models(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    sample_size: Optional[int] = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Đánh giá chéo 5-Fold Cross Validation trên tập Train để đo lường độ ổn định mô hình.

    Tham số:
        pipelines (Dict[str, Pipeline]): Từ điển các Pipeline cần đánh giá.
        X_train (pd.DataFrame): Dữ liệu đặc trưng huấn luyện.
        y_train (pd.Series): Biến mục tiêu log1p(Amount).
        cv (int): Số fold chia tập kiểm định chéo (mặc định: 5).
        sample_size (int, optional): Kích thước mẫu lấy ngẫu nhiên để CV nhanh (mặc định: 10.000).
        random_state (int): Hạt giống ngẫu nhiên (mặc định: 42).

    Trả về:
        pd.DataFrame: Bảng tổng hợp CV R2 (Mean, Std) và CV RMSE (Mean, Std).
    """
    # TODO: [Võ Khôi Nguyên] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Võ Khôi Nguyên cần hoàn thiện hàm 'evaluate_cv_models' theo phân công trong docs/description/TASKS.md"
    )


def tune_hyperparameters(
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 3
) -> GridSearchCV:
    """
    Tinh chỉnh siêu tham số tối ưu thông qua GridSearchCV.

    Tham số:
        pipeline (Pipeline): Pipeline cần tinh chỉnh.
        param_grid (dict): Lưới không gian siêu tham số.
        X_train (pd.DataFrame): Dữ liệu đặc trưng huấn luyện.
        y_train (pd.Series): Biến mục tiêu log.
        cv (int): Số fold kiểm định chéo khi tinh chỉnh (mặc định: 3).

    Trả về:
        GridSearchCV: Đối tượng grid search đã fit.
    """
    # TODO: [Võ Khôi Nguyên] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Võ Khôi Nguyên cần hoàn thiện hàm 'tune_hyperparameters' theo phân công trong docs/description/TASKS.md"
    )


def train_and_save_all_models(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path
) -> Dict[str, Pipeline]:
    """
    Huấn luyện toàn bộ các Pipeline trên dữ liệu tập Train và lưu trữ mô hình tối ưu.

    Tham số:
        pipelines (Dict[str, Pipeline]): Từ điển các Pipeline.
        X_train (pd.DataFrame): Dữ liệu đặc trưng huấn luyện.
        y_train (pd.Series): Biến mục tiêu log.
        output_dir (Path): Thư mục lưu mô hình (models/saved_models/).

    Trả về:
        Dict[str, Pipeline]: Từ điển các Pipeline đã fit hoàn chỉnh.
    """
    # TODO: [Võ Khôi Nguyên] Hoàn thiện hàm này theo phân công trong docs/description/TASKS.md
    raise NotImplementedError(
        "Chưa cài đặt - Võ Khôi Nguyên cần hoàn thiện hàm 'train_and_save_all_models' theo phân công trong docs/description/TASKS.md"
    )
