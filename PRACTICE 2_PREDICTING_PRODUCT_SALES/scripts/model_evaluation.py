import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

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
    """Đánh giá kiểm thử độc lập trên tập Test với biến đổi nghịch đảo np.expm1 về thang đo INR."""
    records = []
    y_true_raw = np.asarray(y_test_raw, dtype=np.float64)
    y_true_log = np.asarray(y_test_log, dtype=np.float64)

    for name, pipe in pipelines.items():
        y_pred_log = pipe.predict(X_test)
        pred_clipped = np.clip(y_pred_log, -10.0, 15.0)
        y_pred_raw = np.clip(np.expm1(pred_clipped), 0.0, None)

        r2_log = r2_score(y_true_log, y_pred_log)
        r2_real = r2_score(y_true_raw, y_pred_raw)
        rmse_val = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))
        mae_val = mean_absolute_error(y_true_raw, y_pred_raw)
        mse_val = mean_squared_error(y_true_raw, y_pred_raw)

        records.append({
            "Mô hình": name,
            "R2 (Log)": round(float(r2_log), 4),
            "R2 (Thực tế)": round(float(r2_real), 4),
            "RMSE (INR)": round(float(rmse_val), 2),
            "MAE (INR)": round(float(mae_val), 2),
            "MSE": round(float(mse_val), 2)
        })

    return pd.DataFrame(records)


def extract_feature_importances(
    pipeline: Pipeline,
    top_n: int = 15
) -> pd.DataFrame:
    """Trích xuất danh sách Top N đặc trưng chi phối doanh số mạnh nhất từ mô hình Quán quân."""
    regressor = pipeline.named_steps.get("regressor", pipeline)
    preprocessor = pipeline.named_steps.get("preprocessor", None)

    if hasattr(regressor, "feature_importances_") and regressor.feature_importances_ is not None:
        importances = np.asarray(regressor.feature_importances_)
    elif hasattr(regressor, "coef_") and regressor.coef_ is not None:
        importances = np.abs(np.asarray(regressor.coef_))
    else:
        return pd.DataFrame(columns=["Đặc trưng", "Tầm quan trọng"])

    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    df_imp = pd.DataFrame({
        "Đặc trưng": feature_names,
        "Tầm quan trọng": importances
    }).sort_values("Tầm quan trọng", ascending=False).reset_index(drop=True)

    return df_imp.head(top_n)


def export_evaluation_artifacts(*args, **kwargs) -> Path:
    """Đóng gói mô hình Quán quân vào best_model.joblib và xuất siêu dữ liệu nghiệm thu metadata.json."""
    best_name = kwargs.get("model_name", kwargs.get("best_name", None))
    metrics_data = kwargs.get("metrics_dict", kwargs.get("metrics_df", None))
    top_features_df = kwargs.get("top_features", None)
    output_dir = kwargs.get("output_dir", None)
    models_comp_raw = kwargs.get("models_comparison", None)
    best_pipeline = kwargs.get("best_pipeline", kwargs.get("pipeline", kwargs.get("best_model", None)))

    if args:
        if len(args) >= 1 and best_name is None:
            best_name = args[0]
        if len(args) == 4 and output_dir is None:
            if metrics_data is None:
                metrics_data = args[1]
            if top_features_df is None:
                top_features_df = args[2]
            if output_dir is None:
                output_dir = args[3]
        elif len(args) >= 5:
            if best_pipeline is None:
                best_pipeline = args[1]
            if metrics_data is None:
                metrics_data = args[2]
            if top_features_df is None:
                top_features_df = args[3]
            if output_dir is None:
                output_dir = args[4]

    out_p = Path(output_dir) if output_dir else BASE_DIR / "models" / "saved_models"
    out_p.mkdir(parents=True, exist_ok=True)

    if best_pipeline is not None:
        joblib.dump(best_pipeline, out_p / "best_model.joblib")

    if models_comp_raw is not None:
        if hasattr(models_comp_raw, "to_dict"):
            models_comp = models_comp_raw.to_dict(orient="records")
        else:
            models_comp = models_comp_raw
    elif metrics_data is not None:
        if hasattr(metrics_data, "to_dict"):
            models_comp = metrics_data.to_dict(orient="records")
        elif isinstance(metrics_data, list):
            models_comp = metrics_data
        else:
            models_comp = [metrics_data]
    else:
        models_comp = []

    if hasattr(top_features_df, "to_dict"):
        top_feats = top_features_df.to_dict(orient="records")
    elif isinstance(top_features_df, list):
        top_feats = top_features_df
    else:
        top_feats = [top_features_df] if top_features_df is not None else []

    best_metrics = models_comp[-1] if models_comp else {}
    if models_comp and isinstance(models_comp, list):
        for m in models_comp:
            if isinstance(m, dict) and "Random Forest" in str(m.get("Mô hình", "")):
                best_metrics = m
                break

    metadata = {
        "project_name": "Practice 2: Predicting Product Sales — Group 3",
        "team": "Nhóm 3",
        "created_at": datetime.now().isoformat(),
        "model_name": str(best_name),
        "best_model": str(best_name),
        "test_metrics": best_metrics,
        "models_comparison": models_comp,
        "top_features": top_feats
    }

    meta_path = out_p / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return meta_path


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Quán quân",
    save_path: Optional[Union[str, Path]] = None,
    max_points: Optional[int] = None,
    **kwargs
):
    """Trực quan hóa biểu đồ đối chiếu Doanh số Thực tế vs Dự đoán kèm đường lý tưởng y = ŷ."""
    plt.figure(figsize=(7, 5))
    y_t = y_true[:max_points] if max_points else y_true
    y_p = y_pred[:max_points] if max_points else y_pred
    plt.scatter(y_t, y_p, alpha=0.3, color="#2563eb", s=18, label="Dự báo kiểm thử")
    max_val = min(max(np.max(y_t), np.max(y_p)), 3000)
    plt.plot([0, max_val], [0, max_val], color="#dc2626", linestyle="--", linewidth=1.8, label="Đường lý tưởng (y = ŷ)")
    plt.xlabel("Doanh số thực tế (INR)", fontsize=11)
    plt.ylabel("Doanh số dự đoán (INR)", fontsize=11)
    plt.title(f"Thực tế vs Dự đoán — {model_name}", fontsize=12, fontweight="bold")
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Quán quân",
    save_path: Optional[Union[str, Path]] = None,
    max_points: Optional[int] = None,
    **kwargs
):
    """Trực quan hóa biểu đồ phần dư Residual Plot (ŷ vs e) khảo sát phương sai sai số."""
    y_t = y_true[:max_points] if max_points else y_true
    y_p = y_pred[:max_points] if max_points else y_pred
    residuals = y_t - y_p
    plt.figure(figsize=(7, 5))
    plt.scatter(y_p, residuals, alpha=0.3, color="#059669", s=18)
    plt.axhline(0, color="#dc2626", linestyle="--", linewidth=1.8)
    plt.xlabel("Doanh số dự đoán (INR)", fontsize=11)
    plt.ylabel("Phần dư sai số e = y - ŷ (INR)", fontsize=11)
    plt.title(f"Biểu đồ phần dư (Residual Plot) — {model_name}", fontsize=12, fontweight="bold")
    plt.xlim(0, min(np.max(y_p), 3000))
    plt.ylim(-1500, 1500)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_performance_comparison(
    metrics_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs
):
    """Trực quan hóa biểu đồ cột so sánh trực diện R² (Thực tế), RMSE và MAE của 3 mô hình."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    models = metrics_df["Mô hình"].tolist()
    short_names = [m.replace(" Regressor", "").replace(" Regression", "") for m in models]
    colors = ["#2563eb", "#059669", "#d97706"]

    # 1. R² Thực tế
    r2_vals = metrics_df["R2 (Thực tế)"].tolist()
    axes[0].bar(short_names, r2_vals, color=colors, width=0.55)
    axes[0].set_title("R² Thực tế (Càng cao càng tốt)", fontweight="bold")
    axes[0].set_ylabel("Hệ số xác định R²")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(r2_vals):
        axes[0].text(i, v + (0.02 if v >= 0 else -0.08), f"{v:.4f}", ha="center", fontweight="bold", fontsize=10)

    # 2. RMSE
    rmse_vals = metrics_df["RMSE (INR)"].tolist()
    axes[1].bar(short_names, rmse_vals, color=colors, width=0.55)
    axes[1].set_title("RMSE (INR - Càng thấp càng tốt)", fontweight="bold")
    axes[1].set_ylabel("Sai số RMSE (INR)")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(rmse_vals):
        axes[1].text(i, v + 8, f"{v:.1f}", ha="center", fontweight="bold", fontsize=10)

    # 3. MAE
    mae_vals = metrics_df["MAE (INR)"].tolist()
    axes[2].bar(short_names, mae_vals, color=colors, width=0.55)
    axes[2].set_title("MAE (INR - Càng thấp càng tốt)", fontweight="bold")
    axes[2].set_ylabel("Sai số MAE (INR)")
    axes[2].grid(True, axis="y", linestyle=":", alpha=0.6)
    for i, v in enumerate(mae_vals):
        axes[2].text(i, v + 3, f"{v:.1f}", ha="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_importances(
    top_features_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    top_n: Optional[int] = None,
    **kwargs
):
    """Trực quan hóa biểu đồ thanh ngang Top 15 đặc trưng chi phối doanh số mạnh nhất."""
    plt.figure(figsize=(9, 5.5))
    df = top_features_df.head(top_n) if top_n else top_features_df
    df_sorted = df.sort_values("Tầm quan trọng", ascending=True)
    clean_labels = [c.replace("cat__", "").replace("num__", "") for c in df_sorted["Đặc trưng"]]
    plt.barh(clean_labels, df_sorted["Tầm quan trọng"], color="#1d4ed8", height=0.65)
    plt.xlabel("Trọng số đóng góp / Độ giảm phương sai MSE", fontsize=11)
    plt.title("Top 15 Đặc Trưng Chi Phối Doanh Số Mạnh Nhất (Random Forest)", fontsize=12, fontweight="bold")
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()
