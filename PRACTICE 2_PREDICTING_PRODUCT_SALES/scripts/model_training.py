from pathlib import Path
import sys
from typing import Dict, Any, Optional, Union
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
    """Khởi tạo từ điển 3 mô hình hồi quy From Scratch tự viết bằng NumPy thuần túy."""
    return {
        "Linear Regression": LinearRegressionScratch(method="normal_equation"),
        "Decision Tree Regressor": DecisionTreeRegressorScratch(
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        "Random Forest Regressor": RandomForestRegressorScratch(
            n_estimators=20,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features=None,
            random_state=42
        )
    }


def build_training_pipelines(
    preprocessor: Any,
    models: Optional[Dict[str, Any]] = None
) -> Dict[str, Pipeline]:
    """Đóng gói Pipeline chuẩn chống rò rỉ dữ liệu kết hợp ColumnTransformer và mô hình hồi quy."""
    if models is None:
        models = create_scratch_models()

    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
    return pipelines


def evaluate_cv_models(
    pipelines: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    sample_size: Optional[int] = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """Đánh giá độ ổn định của các Pipeline qua K-Fold Cross Validation trên tập Train."""
    if sample_size is not None and len(X_train) > sample_size:
        X_eval = X_train.sample(n=sample_size, random_state=random_state)
        y_eval = y_train.loc[X_eval.index]
    else:
        X_eval = X_train
        y_eval = y_train

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    records = []

    for name, pipe in pipelines.items():
        scores = cross_validate(
            pipe,
            X_eval,
            y_eval,
            cv=kf,
            scoring={"r2": "r2", "neg_rmse": "neg_root_mean_squared_error"},
            n_jobs=1
        )
        r2_scores = scores["test_r2"]
        rmse_scores = -scores["test_neg_rmse"]

        records.append({
            "Mô hình": name,
            "CV R² (Mean)": f"{np.mean(r2_scores):.4f}",
            "CV R² (Std)": f"± {np.std(r2_scores):.4f}",
            "CV RMSE (Mean)": f"{np.mean(rmse_scores):.4f}",
            "CV RMSE (Std)": f"± {np.std(rmse_scores):.4f}"
        })

    return pd.DataFrame(records)


def get_default_param_grids() -> Dict[str, Dict[str, Any]]:
    """Trả về lưới tham số tinh chỉnh tối ưu cho các mô hình hồi quy Scratch."""
    return {
        "Linear Regression": {
            "regressor__l2": [0.0, 0.0001, 0.001, 0.01]
        },
        "Decision Tree Regressor": {
            "regressor__max_depth": [5, 6, 7],
            "regressor__min_samples_split": [5, 10]
        },
        "Random Forest Regressor": {
            "regressor__n_estimators": [15, 20],
            "regressor__max_depth": [6, 8]
        }
    }


def tune_grid_search(
    pipelines: Dict[str, Pipeline],
    param_grids: Optional[Dict[str, Dict[str, Any]]] = None,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    cv: int = 3,
    subsample_size: Optional[int] = 10000,
    random_state: int = 42,
    models_dir: Optional[Union[str, Path]] = None,
    return_summary: bool = False
):
    """Tinh chỉnh siêu tham số qua GridSearchCV trên mẫu đại diện, sau đó huấn luyện mô hình tối ưu trên toàn bộ Train set."""
    if param_grids is None:
        param_grids = get_default_param_grids()

    save_dir = Path(models_dir) if models_dir else BASE_DIR / "models" / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    if subsample_size is not None and len(X_train) > subsample_size:
        X_sub = X_train.sample(n=subsample_size, random_state=random_state)
        y_sub = y_train.loc[X_sub.index]
    else:
        X_sub = X_train
        y_sub = y_train

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    tuned_models: Dict[str, Pipeline] = {}
    summary_records = []

    for name, pipe in pipelines.items():
        grid = param_grids.get(name, {})
        if grid:
            gs = GridSearchCV(pipe, param_grid=grid, cv=kf, scoring="r2", n_jobs=1)
            gs.fit(X_sub, y_sub)
            best_p = gs.best_params_
            best_score = gs.best_score_
            pipe.set_params(**best_p)
        else:
            best_p = {}
            best_score = float("nan")

        pipe.fit(X_train, y_train)
        tuned_models[name] = pipe

        summary_records.append({
            "Mô hình": name,
            "Best CV R² (Subsample 10k)": f"{best_score:.4f}" if not np.isnan(best_score) else "N/A",
            "Siêu tham số tối ưu (Best Params)": str(best_p) if best_p else "Mặc định"
        })
        print(f"  ✓ [{name}] Tinh chỉnh hoàn tất | R² = {best_score:.4f} | Params: {best_p}")

        # Lưu riêng từng mô hình phục vụ suy luận độc lập
        file_map = {
            "Linear Regression": "linear_regression.joblib",
            "Decision Tree Regressor": "decision_tree.joblib",
            "Random Forest Regressor": "random_forest.joblib"
        }
        if name in file_map:
            joblib.dump(pipe, save_dir / file_map[name])

    summary_df = pd.DataFrame(summary_records)
    if return_summary:
        return tuned_models, summary_df
    return tuned_models
