from datetime import datetime
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent


def evaluate_holdout_test(tuned_models: dict, X_test, y_test) -> tuple:
    """Đo lường 4 chỉ số hiệu năng và ma trận nhầm lẫn trên tập kiểm tra độc lập."""
    test_results = []
    cms = {}

    for name, model in tuned_models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred) * 100
        rec = recall_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100

        cm = confusion_matrix(y_test, y_pred)
        cms[name] = cm
        tn, fp, fn, tp = cm.ravel()

        test_results.append({
            "Mô hình": name,
            "Accuracy (%)": round(float(acc), 2),
            "Precision (%)": round(float(prec), 2),
            "Recall (%)": round(float(rec), 2),
            "F1-Score (%)": round(float(f1), 2),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        })
    return test_results, cms


def get_top_spam_keywords(best_model, top_n: int = 15) -> pd.DataFrame:
    """Trích xuất top_n từ khóa/đặc trưng có trọng số báo hiệu thư rác mạnh nhất."""
    prep = best_model.named_steps["preprocessor"]
    clf = best_model.named_steps["classifier"]

    # Làm sạch tiền tố text__ và num__ từ get_feature_names_out()
    feature_names = [f.replace("text__", "").replace("num__", "") for f in prep.get_feature_names_out()]
    feature_names = np.array(feature_names)

    if hasattr(clf, "coef_"):
        weights = clf.coef_[0]
    elif hasattr(clf, "calibrated_classifiers_"):
        weights = np.mean([c.estimator.coef_[0] for c in clf.calibrated_classifiers_], axis=0)
    else:
        weights = getattr(clf, "weights_", None)

    top_idx = np.argsort(weights)[-top_n:][::-1]
    return pd.DataFrame({
        "Từ khóa": feature_names[top_idx],
        "Trọng số": weights[top_idx]
    })


def select_best_model(models: dict, test_results: list) -> tuple:
    """Xác định mô hình quán quân có F1-Score cao nhất từ test_results."""
    best_res = max(test_results, key=lambda item: item["F1-Score (%)"])
    best_name = best_res["Mô hình"]
    return best_name, models[best_name]


def export_model_artifacts(model_or_dict, test_results: list, top15_df: pd.DataFrame,
                           models_dir: str = None) -> tuple:
    """
    Đóng gói best_model.joblib và xuất metadata.json dựa trên mô hình có F1-Score cao nhất.
    Hỗ trợ truyền trực tiếp đối tượng mô hình hoặc từ điển các mô hình (tuned_models).
    Lưu trữ chuẩn vào models/saved_models/ và đồng bộ models/.
    """
    save_dir = Path(models_dir) if models_dir else BASE_DIR / "models" / "saved_models"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Xác định mô hình quán quân động dựa trên F1-Score cao nhất
    best_metrics = max(test_results, key=lambda item: item["F1-Score (%)"])
    if isinstance(model_or_dict, dict):
        best_model = model_or_dict[best_metrics["Mô hình"]]
    else:
        best_model = model_or_dict

    # 2. Lưu best_model.joblib độc quyền vào models/saved_models/
    best_path = save_dir / "best_model.joblib"
    joblib.dump(best_model, best_path)
    model_name_map = {
        "linear_svm": "Support Vector Machine (Linear SVM)",
        "logistic_regression": "Logistic Regression (L2)",
        "naive_bayes": "Naive Bayes"
    }
    best_display_name = model_name_map.get(best_metrics["Mô hình"], best_metrics["Mô hình"])
    clf_class_name = best_model.named_steps["classifier"].__class__.__name__

    metadata = {
        "project_name": "Practice 1: Classifying Spam Emails — Group 3",
        "team": "Nhóm 3",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "best_model": best_display_name,
        "test_metrics": {
            "accuracy_percent": best_metrics["Accuracy (%)"],
            "precision_percent": best_metrics["Precision (%)"],
            "recall_percent": best_metrics["Recall (%)"],
            "f1_score_percent": best_metrics["F1-Score (%)"]
        },
        "models_comparison": [
            {
                "Model": model_name_map.get(r["Mô hình"], r["Mô hình"]),
                "Accuracy (%)": r["Accuracy (%)"],
                "Precision (%)": r["Precision (%)"],
                "Recall (%)": r["Recall (%)"],
                "F1-Score (%)": r["F1-Score (%)"],
                "FP": r.get("FP", 0),
                "FN": r.get("FN", 0)
            }
            for r in test_results
        ],
        "pipeline_steps": [
            "preprocessor (ColumnTransformer: TF-IDF + Chi-Square + MinMaxScaler)",
            f"classifier ({clf_class_name})"
        ],
        "top_15_keywords": [
            {"rank": i + 1, "keyword": str(feat), "weight": round(float(w), 4)}
            for i, (feat, w) in enumerate(zip(top15_df["Từ khóa"], top15_df["Trọng số"]))
        ]
    }

    meta_path = save_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return str(best_path), str(meta_path)
