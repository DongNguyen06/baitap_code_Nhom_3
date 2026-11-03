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
    """Trả về lưới tham số tinh chỉnh toàn diện, tối ưu hiệu năng cao nhất cho các mô hình Scratch."""
    return {
        "naive_bayes": {
            "classifier__alpha": [0.01, 0.1, 0.5, 1.0]
        },
        "logistic_regression": {
            "classifier__learning_rate": [5.0, 10.0, 15.0],
            "classifier__n_iters": [250, 300]
        },
        "linear_svm": {
            "classifier__learning_rate": [2.0, 5.0, 8.0],
            "classifier__lambda_param": [1e-5, 1e-4, 1e-3],
            "classifier__n_iters": [200, 300]
        }
    }


def tune_grid_search(pipelines: dict, param_grids: dict = None, X_train=None, y_train=None,
                     cv=None, models_dir: str = None) -> dict:
    """Thực hiện tinh chỉnh siêu tham số qua GridSearchCV và lưu trữ mô hình tối ưu vào models/saved_models/."""
    if param_grids is None:
        param_grids = get_default_param_grids()
    if cv is None:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    save_dir = Path(models_dir) if models_dir else BASE_DIR / "models" / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    tuned_models = {}
    for name, pipe in pipelines.items():
        gs = GridSearchCV(pipe, param_grid=param_grids[name], cv=cv, scoring="f1", n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        best_est = gs.best_estimator_
        tuned_models[name] = best_est

        # Lưu độc quyền vào models/saved_models/
        joblib.dump(best_est, save_dir / f"{name}.joblib")

    return tuned_models
