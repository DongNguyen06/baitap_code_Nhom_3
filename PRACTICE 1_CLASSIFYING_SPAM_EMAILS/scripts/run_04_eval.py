import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập đường dẫn gốc cho dự án
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    TRAIN_TEST_JOBLIB,
    SAVED_MODELS_DIR,
    IMAGES_DIR,
    LOGS_DIR,
    require_files,
    save_figure,
    save_stage_result,
)
from scripts.logger_utils import setup_logger, timed, log_environment
from scripts.model_evaluation import (
    evaluate_holdout_test,
    get_top_spam_keywords,
    select_best_model,
    export_model_artifacts,
)
import scripts.report_builder as report_builder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chặng 04: Đánh giá độc lập trên tập Hold-out Test, vẽ biểu đồ và xuất best_model.joblib."
    )
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="Chạy nhanh ở chế độ demo (đánh giá trên tập mẫu con) để kiểm tra luồng."
    )
    return parser.parse_args()


def plot_confusion_matrices(cms: dict) -> Path:
    """Vẽ 3 Ma trận nhầm lẫn (Heatmaps) cạnh nhau trong 1 biểu đồ duy nhất."""
    model_labels = [
        ("naive_bayes", "Naive Bayes"),
        ("logistic_regression", "Logistic Regression"),
        ("linear_svm", "Support Vector Machine (Linear SVM)")
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Ma Trận Nhầm Lẫn (Confusion Matrix) — Kiểm Thử Độc Lập Hold-out Test Set", fontsize=15, fontweight="bold", y=1.02)

    for idx, (key, title) in enumerate(model_labels):
        ax = axes[idx]
        cm = cms.get(key)
        if cm is not None:
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Dự đoán Ham", "Dự đoán Spam"],
                yticklabels=["Thực tế Ham", "Thực tế Spam"],
                cbar=False, annot_kws={"size": 13, "weight": "bold"}
            )
            ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        else:
            ax.text(0.5, 0.5, "Không có dữ liệu", ha="center", va="center")
        ax.set_xlabel("Nhãn Dự Đoán", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Nhãn Thực Tế", fontsize=11)
        else:
            ax.set_ylabel("")

    return save_figure(fig, "model_01_confusion_matrices.png", dpi=150)


def plot_top_keywords(top15_df: pd.DataFrame, best_name: str) -> Path:
    """Vẽ biểu đồ thanh ngang Top 15 từ khóa/đặc trưng báo hiệu thư rác mạnh nhất."""
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(top15_df))
    bars = ax.barh(
        y_pos, top15_df["Trọng số"],
        color="#2563eb", edgecolor="#1d4ed8", alpha=0.9, height=0.65
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top15_df["Từ khóa"], fontsize=11, fontweight="500")
    ax.invert_yaxis()  # Đưa từ khóa có trọng số cao nhất lên đỉnh
    ax.set_xlabel("Trọng số phân loại (Feature Weight / Coefficient)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Top 15 Đặc Trưng / Từ Khóa Báo Hiệu Thư Rác Mạnh Nhất ({best_name.upper()})",
        fontsize=13, fontweight="bold", pad=12
    )

    # Hiển thị số liệu trực tiếp trên đầu thanh
    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"{w:.3f}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=10, fontweight="bold", color="#1e293b"
        )

    ax.set_xlim(0, max(top15_df["Trọng số"]) * 1.15)
    return save_figure(fig, "model_02_top_15_spam_keywords.png", dpi=150)


def plot_performance_comparison(test_results: list) -> Path:
    """Vẽ biểu đồ cột nhóm đối đầu 4 chỉ số (Accuracy, Precision, Recall, F1) của 3 mô hình."""
    metrics = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
    model_map = {
        "naive_bayes": "Naive Bayes",
        "logistic_regression": "Logistic Reg",
        "linear_svm": "Linear SVM"
    }

    df_plot = pd.DataFrame(test_results)
    df_plot["Display_Name"] = df_plot["Mô hình"].map(lambda x: model_map.get(x, x))

    df_melt = pd.melt(
        df_plot,
        id_vars=["Display_Name"],
        value_vars=metrics,
        var_name="Chỉ số",
        value_name="Điểm số (%)"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    palette = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]

    sns.barplot(
        data=df_melt,
        x="Display_Name",
        y="Điểm số (%)",
        hue="Chỉ số",
        palette=palette,
        ax=ax
    )

    ax.set_title("Đối Đầu Hiệu Năng 3 Thuật Toán Trên Hold-out Test Set", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mô Hình", fontsize=11, fontweight="bold")
    ax.set_ylabel("Điểm Số (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(85, 101)  # Giới hạn trục Y làm nổi bật sự khác biệt nhỏ giữa các mô hình đạt điểm cao
    ax.legend(loc="lower right", frameon=True)

    # Hiển thị số liệu trên từng cột
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.1f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center", va="bottom",
                fontsize=8.5, xytext=(0, 2),
                textcoords="offset points"
            )

    return save_figure(fig, "model_03_performance_comparison.png", dpi=150)


def run_stage_04(quick_mode: bool = False, extra_handler=None):
    start_total = time.time()
    started_at = datetime.now().isoformat()

    # 1. Khởi tạo Logger ghi nhật ký console và file logs/stage_04_*.log (hỗ trợ extra_handler)
    logger, log_path = setup_logger(log_dir=LOGS_DIR, prefix="run_04_eval", extra_handler=extra_handler)
    logger.info("=" * 80)
    logger.info("   🚀 BẮT ĐẦU CHẶNG 04: ĐÁNH GIÁ ĐỘC LẬP TẬP TEST & XUẤT HIỆN VẬT TỐI ƯU")
    logger.info(f"   Thời điểm bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Chế độ chạy: {'⚡ Rút gọn (Quick Mode)' if quick_mode else '🏆 Đầy đủ (Full Benchmark)'}")
    logger.info("=" * 80)

    log_environment(logger)
    timings = {}

    # 2. Kiểm tra điều kiện tiên quyết
    nb_path = SAVED_MODELS_DIR / "naive_bayes.joblib"
    lr_path = SAVED_MODELS_DIR / "logistic_regression.joblib"
    svm_path = SAVED_MODELS_DIR / "linear_svm.joblib"

    require_files(
        logger,
        [TRAIN_TEST_JOBLIB, nb_path, lr_path, svm_path],
        hint="Hãy chạy 'python scripts/run_03_train.py' trước để huấn luyện và lưu trữ 3 mô hình .joblib."
    )

    # 3. Nạp dữ liệu Test set và 3 mô hình đã huấn luyện
    with timed(logger, "Nạp 'train_test_data.joblib' và 3 mô hình .joblib"):
        t0 = time.time()
        splits = joblib.load(TRAIN_TEST_JOBLIB)
        X_test = splits["X_test"]
        y_test = splits["y_test"]

        tuned_models = {
            "naive_bayes": joblib.load(nb_path),
            "logistic_regression": joblib.load(lr_path),
            "linear_svm": joblib.load(svm_path)
        }
        timings["load_artifacts_sec"] = round(time.time() - t0, 3)

    if quick_mode and len(X_test) > 1000:
        logger.info("  ⚡ Quick Mode: Thu nhỏ tập Test xuống 1.000 mẫu để đánh giá nhanh.")
        X_test = X_test.iloc[:1000]
        y_test = y_test[:1000]

    logger.info(f"  ✓ Sẵn sàng đánh giá trên tập kiểm thử độc lập: {len(X_test):,} mẫu (Tỷ lệ Spam: {float(y_test.mean())*100:.2f}%).")

    # 4. Đánh giá Hold-out Test
    with timed(logger, f"Dự đoán và đo lường 4 chỉ số trên {len(X_test):,} mẫu Test set"):
        t0 = time.time()
        test_results, cms = evaluate_holdout_test(tuned_models, X_test, y_test)
        timings["evaluation_sec"] = round(time.time() - t0, 3)

    # In bảng kết quả Hold-out Test
    logger.info("\n" + "=" * 80)
    logger.info("   🏆 BẢNG TỔNG KẾT HIỆU NĂNG KIỂM THỬ ĐỘC LẬP (HOLD-OUT TEST SET)")
    logger.info("=" * 80)
    header = f"{'Mô hình':<22} | {'Accuracy (%)':<12} | {'Precision (%)':<13} | {'Recall (%)':<10} | {'F1-Score (%)':<12} | {'TP':<5} | {'FP':<5} | {'FN':<5}"
    logger.info(header)
    logger.info("-" * 80)
    for r in test_results:
        logger.info(
            f"{r['Mô hình']:<22} | {r['Accuracy (%)']:<12.2f} | {r['Precision (%)']:<13.2f} | {r['Recall (%)']:<10.2f} | {r['F1-Score (%)']:<12.2f} | {r['TP']:<5} | {r['FP']:<5} | {r['FN']:<5}"
        )
    logger.info("=" * 80)

    # 5. Xác định Quán quân & Trích xuất Top 15 từ khóa Spam
    best_name, best_model = select_best_model(tuned_models, test_results)
    best_metrics = max(test_results, key=lambda x: x["F1-Score (%)"])
    logger.info(f"\n🎉 Mô hình Quán quân được vinh danh: [{best_name.upper()}] với F1-Score = {best_metrics['F1-Score (%)']:.2f}%")

    with timed(logger, f"Trích xuất Top 15 từ khóa spam từ mô hình [{best_name}]"):
        t0 = time.time()
        top15_df = get_top_spam_keywords(best_model, top_n=15)
        timings["feature_weights_sec"] = round(time.time() - t0, 3)

    # 6. Đo tốc độ suy luận (Inference Latency)
    sample_bench = X_test.iloc[:100]
    t_inf_start = time.time()
    best_model.predict(sample_bench)
    inf_duration = time.time() - t_inf_start
    inf_ms_per_email = round((inf_duration / len(sample_bench)) * 1000, 2)
    inference_time_str = f"{inf_ms_per_email:.2f} ms"
    logger.info(f"  ⚡ Tốc độ suy luận thực tế: {inference_time_str} / 1 email.")

    # 7. Đóng gói hiện vật best_model.joblib & metadata.json
    with timed(logger, "Đóng gói 'best_model.joblib' và xuất 'metadata.json'"):
        t0 = time.time()
        best_path, meta_path = export_model_artifacts(
            best_model, test_results, top15_df, models_dir=str(SAVED_MODELS_DIR)
        )
        timings["export_artifacts_sec"] = round(time.time() - t0, 3)

    logger.info(f"  ✓ Đã xuất: {Path(best_path).name} và {Path(meta_path).name}")

    # 8. Vẽ và lưu 3 biểu đồ phục vụ báo cáo
    figures = []
    with timed(logger, "Khởi tạo và xuất 3 biểu đồ trực quan hóa"):
        f1 = plot_confusion_matrices(cms)
        f2 = plot_top_keywords(top15_df, best_name)
        f3 = plot_performance_comparison(test_results)
        figures = [str(f1), str(f2), str(f3)]
        logger.info(f"  ✓ Đã sinh biểu đồ 1: {f1.name}")
        logger.info(f"  ✓ Đã sinh biểu đồ 2: {f2.name}")
        logger.info(f"  ✓ Đã sinh biểu đồ 3: {f3.name}")

    total_duration = round(time.time() - start_total, 2)
    finished_at = datetime.now().isoformat()
    timings["total_stage_sec"] = total_duration

    # 9. Đóng gói phong bì kết quả outputs/stage_04.json
    stage_data = {
        "status": "success",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": total_duration,
        "quick_mode": quick_mode,
        "log_file": str(log_path),
        "config": {
            "test_samples": len(X_test),
            "best_model_name": best_name
        },
        "metrics": {
            "test_samples": len(X_test),
            "best_model": best_name,
            "best_accuracy": best_metrics["Accuracy (%)"],
            "best_precision": best_metrics["Precision (%)"],
            "best_recall": best_metrics["Recall (%)"],
            "best_f1": best_metrics["F1-Score (%)"],
            "missed_spam": best_metrics["FN"],
            "false_alarms": best_metrics["FP"],
            "inference_time_ms": inf_ms_per_email,
            "inference_time_str": inference_time_str
        },
        "tables": {
            "test_results": test_results,
            "top_15_keywords": top15_df.to_dict(orient="records"),
            "confusion_matrices": {k: v.tolist() for k, v in cms.items()}
        },
        "figures": figures,
        "timings": timings
    }

    envelope_path = save_stage_result("04", stage_data)
    logger.info(f"\n✔ Đã lưu phong bì kết quả Chặng 4: {envelope_path.name}")

    # 10. Cập nhật Báo cáo HTML tự động (Báo cáo 04, Báo cáo Nhật ký và Bảng điều khiển Index)
    try:
        report_builder.build_report_04()
        report_builder.build_run_log_page()
        report_builder.build_index()
        logger.info("✔ Đã cập nhật 04_model_evaluation_report.html, run_log.html & index.html")
    except Exception as e:
        logger.warning(f"⚠️ Cảnh báo khi cập nhật báo cáo HTML: {e}")

    logger.info("=" * 80)
    logger.info(f"🎉 HOÀN THÀNH CHẶNG 04 THÀNH CÔNG TRONG {total_duration:.2f} GIÂY!")
    logger.info(f"👉 Mời xem báo cáo nghiệm thu trực quan tại: docs/report/index.html")
    logger.info(f"👉 Trải nghiệm thử nghiệm email thời gian thực tại: notebooks/demo.ipynb")
    logger.info("=" * 80)

    return stage_data


if __name__ == "__main__":
    args = parse_args()
    run_stage_04(quick_mode=args.quick)
