"""
Module: model_training.py
Dự án: Practice 2: Predicting Product Sales — Group 3
Phụ trách chính: Võ Khôi Nguyên (14%)
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
    Khởi tạo từ điển 3 mô hình hồi quy From Scratch tự viết bằng NumPy thuần túy:
    1. 'Linear Regression': LinearRegressionScratch(method="normal_equation", l2=1e-4, fit_intercept=True)
    2. 'Decision Tree Regressor': DecisionTreeRegressorScratch(max_depth=8, min_samples_split=20, min_samples_leaf=10, n_quantiles=15, random_state=42)
    3. 'Random Forest Regressor': RandomForestRegressorScratch(n_estimators=20, max_depth=10, min_samples_split=20, min_samples_leaf=10, max_features="sqrt", n_quantiles=15, random_state=42)

    Phụ trách chính: Võ Khôi Nguyên (14%) (Cùng toàn nhóm nghiên cứu lý thuyết toán học)

    Trả về:
        Dict[str, Any]: Từ điển chứa 3 đối tượng mô hình Scratch.
    """
    # TODO: [Võ Khôi Nguyên - 14%] Lập trình hàm khởi tạo 3 mô hình hồi quy Scratch:
    # 1. Khởi tạo LinearRegressionScratch với các siêu tham số phù hợp
    # 2. Khởi tạo DecisionTreeRegressorScratch với max_depth, min_samples_split, n_quantiles
    # 3. Khởi tạo RandomForestRegressorScratch với n_estimators, max_depth, max_features
    # 4. Đóng gói vào dict và trả về
    raise NotImplementedError(
        "Chưa cài đặt - Võ Khôi Nguyên (14%) cần hoàn thiện hàm 'create_scratch_models' theo phân công trong docs/description/TASKS.md"
    )


def build_training_pipelines(
    preprocessor: Any,
    models: Optional[Dict[str, Any]] = None
) -> Dict[str, Pipeline]:
    """
    Đóng gói Pipeline chuẩn chống rò rỉ dữ liệu (Anti-Leakage Pipeline):
    Từng mô hình được ghép nối trực tiếp với preprocessor (ColumnTransformer) qua sklearn Pipeline.
    Pipeline = [('preprocessor', preprocessor), ('regressor', model)]

    Phụ trách chính: Võ Khôi Nguyên (14%)

    Tham số:
        preprocessor: Đối tượng ColumnTransformer đã định nghĩa.
        models (Dict[str, Any], optional): Từ điển mô hình. Mặc định gọi create_scratch_models().

    Trả về:
        Dict[str, Pipeline]: Từ điển chứa các Pipeline huấn luyện.
    """
    # TODO: [Võ Khôi Nguyên - 14%] Lập trình đóng gói Pipeline chống rò rỉ:
    # 1. Nếu models is None: gọi create_scratch_models()
    # 2. Tạo dict pipelines = {}
    # 3. Duyệt qua từng (name, model) trong models:
    #    pipelines[name] = Pipeline([('preprocessor', preprocessor), ('regressor', model)])
    # 4. Trả về pipelines
    raise NotImplementedError(
        "Chưa cài đặt - Võ Khôi Nguyên (14%) cần hoàn thiện hàm 'build_training_pipelines' theo phân công trong docs/description/TASKS.md"
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
    Đánh giá chéo 5-Fold Cross Validation trên tập Train.
    Sử dụng mẫu đại diện (mặc định 10.000 dòng) để tăng tốc độ kiểm thử chéo mà vẫn bảo toàn tính khái quát.

    Phụ trách chính: Võ Khôi Nguyên (14%)

    Tham số:
        pipelines (Dict[str, Pipeline]): Từ điển các Pipeline.
        X_train (pd.DataFrame): Dữ liệu đặc trưng huấn luyện.
        y_train (pd.Series): Biến mục tiêu log1p(Amount).
        cv (int): Số fold chia tập kiểm định chéo (mặc định: 5).
        sample_size (int, optional): Kích thước mẫu lấy ngẫu nhiên để CV nhanh (mặc định: 10.000).
        random_state (int): Hạt giống ngẫu nhiên (mặc định: 42).

    Trả về:
        pd.DataFrame: Bảng tổng hợp CV R2 (Mean, Std) và CV RMSE (Mean, Std).
    """
    # TODO: [Võ Khôi Nguyên - 14%] Lập trình kiểm định chéo 5-Fold Cross-Validation:
    # 1. Nếu sample_size hợp lệ và len(X_train) > sample_size:
    #    Lấy mẫu ngẫu nhiên không hoàn lại sample_size dòng
    # 2. Khởi tạo KFold(n_splits=cv, shuffle=True, random_state=random_state)
    # 3. Duyệt qua từng pipeline, chạy cross_validate với scoring={'r2': 'r2', 'neg_rmse': 'neg_root_mean_squared_error'}
    # 4. Đóng gói kết quả: Mô hình, CV R2 (Mean/Std), CV RMSE (Mean/Std)
    # 5. Trả về pd.DataFrame(records)
    raise NotImplementedError(
        "Chưa cài đặt - Võ Khôi Nguyên (14%) cần hoàn thiện hàm 'evaluate_cv_models' theo phân công trong docs/description/TASKS.md"
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
    Phụ trách chính: Võ Khôi Nguyên (14%)

    Tham số:
        pipeline (Pipeline): Pipeline cần tinh chỉnh.
        param_grid (dict): Lưới không gian siêu tham số.
        X_train (pd.DataFrame): Dữ liệu đặc trưng huấn luyện.
        y_train (pd.Series): Biến mục tiêu log.
        cv (int): Số fold kiểm định chéo khi tinh chỉnh (mặc định: 3).

    Trả về:
        GridSearchCV: Đối tượng grid search đã fit.
    """
    # TODO: [Võ Khôi Nguyên - 14%] Lập trình tinh chỉnh siêu tham số:
    # 1. grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='r2', n_jobs=-1)
    # 2. grid.fit(X_train, y_train)
    # 3. Trả về grid
    raise NotImplementedError(
        "Chưa cài đặt - Võ Khôi Nguyên (14%) cần hoàn thiện hàm 'tune_hyperparameters' theo phân công trong docs/description/TASKS.md"
    )


def train_and_save_all_models(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    output_dir: Path
) -> Dict[str, Pipeline]:
    """
    Huấn luyện toàn bộ các Pipeline trên 82.281 dòng tập Train và lưu trữ mô hình tối ưu.
    Phụ trách chính: Võ Khôi Nguyên (14%)

    Tham số:
        pipelines (Dict[str, Pipeline]): Từ điển các Pipeline.
        X_train, y_train: Dữ liệu huấn luyện.
        output_dir: Thư mục lưu mô hình (models/saved_models/).

    Trả về:
        Dict[str, Pipeline]: Từ điển các Pipeline đã fit.
    """
    # TODO: [Võ Khôi Nguyên - 14%] Lập trình huấn luyện và lưu trữ mô hình:
    # 1. output_dir.mkdir(parents=True, exist_ok=True)
    # 2. Duyệt qua từng pipeline, gọi pipe.fit(X_train, y_train)
    # 3. Chọn mô hình tốt nhất (ví dụ: Decision Tree Regressor hoặc Random Forest)
    # 4. Lưu best_model.joblib vào output_dir
    # 5. Trả về trained_pipelines
    raise NotImplementedError(
        "Chưa cài đặt - Võ Khôi Nguyên (14%) cần hoàn thiện hàm 'train_and_save_all_models' theo phân công trong docs/description/TASKS.md"
    )
