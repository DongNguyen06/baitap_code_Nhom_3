import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV

from models.algorithms.naive_bayes import MultinomialNaiveBayesScratch
from models.algorithms.logistic_regression import LogisticRegressionScratch
from models.algorithms.support_vector_machine import LinearSVMScratch

BASE_DIR = Path(__file__).resolve().parent.parent


def create_scratch_models(random_state: int = 42) -> dict:
    """Khởi tạo 3 mô hình phân loại tự cài đặt (From Scratch) tối ưu tốc độ và độ chính xác."""
    return {
        "naive_bayes": MultinomialNaiveBayesScratch(alpha=1.0),
        "logistic_regression": LogisticRegressionScratch(learning_rate=12.0, n_iters=300, l2=1e-4),
        "linear_svm": LinearSVMScratch(learning_rate=5.0, lambda_param=1e-5, n_iters=300)
    }


def build_training_pipelines(base_preprocessor, models: dict = None) -> dict:
    """Ghép nối bộ tiền xử lý và 3 mô hình Scratch thành các Pipeline độc lập."""
    if models is None:
        models = create_scratch_models()
    return {
        name: Pipeline([("preprocessor", clone(base_preprocessor)), ("classifier", model)])
        for name, model in models.items()
    }


def run_cross_validation(pipelines: dict, X_train, y_train, cv=None, random_state: int = 42) -> pd.DataFrame:
    """
    Thực hiện kiểm định chéo 5-Fold Stratified CV trên tập Train.
    Trả về bảng DataFrame tổng hợp: Accuracy, Precision, Recall, F1-Score.
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    cv_summary = []
    for name, pipe in pipelines.items():
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring=["accuracy", "precision", "recall", "f1"],
            n_jobs=-1
        )
        cv_summary.append({
            "Mô hình": name,
            "Accuracy (%)": f"{scores['test_accuracy'].mean()*100:.2f} ± {scores['test_accuracy'].std()*100:.2f}",
            "Precision (%)": f"{scores['test_precision'].mean()*100:.2f} ± {scores['test_precision'].std()*100:.2f}",
            "Recall (%)": f"{scores['test_recall'].mean()*100:.2f} ± {scores['test_recall'].std()*100:.2f}",
            "F1-Score (%)": f"{scores['test_f1'].mean()*100:.2f} ± {scores['test_f1'].std()*100:.2f}"
        })
    return pd.DataFrame(cv_summary)


def get_default_param_grids() -> dict:
    """Trả về lưới tham số tinh chỉnh toàn diện, bao gồm chọn lọc đặc trưng và siêu tham số mô hình."""
    return {
        "naive_bayes": {
            "preprocessor__text__select__k": [1500, 2000],
            "classifier__alpha": [0.05, 0.1, 0.5, 1.0]
        },
        "logistic_regression": {
            "preprocessor__text__select__k": [1500, 2000],
            "classifier__learning_rate": [10.0, 15.0]
        },
        "linear_svm": {
            "preprocessor__text__select__k": [1500, 2000],
            "classifier__learning_rate": [5.0, 8.0]
        }
    }


def tune_grid_search(pipelines: dict, param_grids: dict = None, X_train=None, y_train=None,
                     cv=None, subsample_size: int = 6000, random_state: int = 42,
                     models_dir: str = None, return_summary: bool = False):
    """
    Thực hiện tinh chỉnh siêu tham số qua GridSearchCV trên tập mẫu đại diện (subsample 6.000 mẫu),
    sau đó áp dụng siêu tham số tối ưu huấn luyện toàn diện trên toàn bộ tập Train (26.128 mẫu)
    và lưu trữ mô hình tối ưu vào models/saved_models/.
    
    Tham số:
        pipelines (dict): Từ điển chứa các Pipeline huấn luyện.
        param_grids (dict, optional): Lưới không gian siêu tham số.
        X_train, y_train: Dữ liệu huấn luyện.
        cv: Chiến lược phân chia Cross Validation (mặc định 3-Fold Stratified CV).
        subsample_size (int): Kích thước mẫu đại diện để quét siêu tham số nhanh (mặc định 6.000).
        random_state (int): Hạt giống ngẫu nhiên (mặc định 42).
        models_dir (str, optional): Thư mục lưu trữ mô hình.
        return_summary (bool): Nếu True, trả về tuple (tuned_models, summary_df).
        
    Trả về:
        dict hoặc tuple: Mô hình tối ưu (hoặc kèm DataFrame tổng hợp tham số).
    """
    if param_grids is None:
        param_grids = get_default_param_grids()
    if cv is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    save_dir = Path(models_dir) if models_dir else BASE_DIR / "models" / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Lấy mẫu đại diện để quét nhanh qua GridSearchCV
    if subsample_size is not None and len(X_train) > subsample_size:
        X_sub = X_train.iloc[:subsample_size]
        y_sub = y_train[:subsample_size]
    else:
        X_sub = X_train
        y_sub = y_train

    tuned_models = {}
    summary_records = []
    for name, pipe in pipelines.items():
        grid = param_grids.get(name, {})
        if grid:
            gs = GridSearchCV(pipe, param_grid=grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0)
            gs.fit(X_sub, y_sub)
            best_p = gs.best_params_
            best_f1 = gs.best_score_
            pipe.set_params(**best_p)
        else:
            best_p = {}
            best_f1 = 0.0

        # Huấn luyện hoàn chỉnh mô hình tối ưu trên TOÀN BỘ 26.128 mẫu Train
        pipe.fit(X_train, y_train)
        tuned_models[name] = pipe

        f1_col = f"Best CV F1 (Mẫu {subsample_size:,})" if (subsample_size is not None and subsample_size < len(X_train)) else "Best CV F1 (Toàn bộ Train)"
        summary_records.append({
            "Mô hình": name,
            f1_col: f"{best_f1 * 100:.2f}%",
            "Siêu tham số tối ưu (Best Params)": str(best_p)
        })
        logging.getLogger("pipeline").info(f"  ✓ [{name}] Best F1 = {best_f1 * 100:.2f}% | Params: {best_p}")

        # Lưu độc quyền vào models/saved_models/
        joblib.dump(pipe, save_dir / f"{name}.joblib")

    summary_df = pd.DataFrame(summary_records)
    if return_summary:
        return tuned_models, summary_df
    return tuned_models
