import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    TRAIN_TEST_JOBLIB,
    SAVED_MODELS_DIR,
    IMAGES_DIR,
    LOGS_DIR,
    require_files,
    save_stage_result,
    RANDOM_SEED
)
from scripts.logger_utils import setup_logger, timed, log_environment
from scripts.model_evaluation import (
    evaluate_holdout_test,
    extract_feature_importances,
    export_evaluation_artifacts,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_performance_comparison,
    plot_feature_importances
)
import scripts.report_builder as report_builder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chặng 04: Đánh giá độc lập trên tập Hold-out Test, trích xuất đặc trưng và xuất best_model.joblib."
    )
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="Chạy nhanh ở chế độ demo (đánh giá trên tập mẫu con 2,000 đơn hàng) để kiểm tra luồng."
    )
    return parser.parse_args()


def run_stage_04(quick_mode: bool = False, extra_handler=None):
    start_total = time.time()
    started_at = datetime.now().isoformat()

    # 1. Khởi tạo Logger riêng biệt cho Chặng 04 (Zero-Drift Logging)
    logger, log_path = setup_logger(log_dir=LOGS_DIR, prefix="run_04_eval", extra_handler=extra_handler)
    logger.info("=" * 80)
    logger.info("   🚀 BẮT ĐẦU CHẶNG 04: ĐÁNH GIÁ ĐỘC LẬP TẬP KIỂM THỬ (HOLD-OUT TEST SET)")
    logger.info(f"   Thời điểm bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Chế độ chạy: {'⚡ Rút gọn (Quick Demo 2.000 mẫu)' if quick_mode else '🏆 Đầy đủ 100% Test Set (20.571 mẫu)'}")
    logger.info("=" * 80)

    log_environment(logger, seed=RANDOM_SEED)
    timings = {}

    # 2. Kiểm tra điều kiện tiên quyết
    model_paths = {
        "Linear Regression": SAVED_MODELS_DIR / "linear_regression.joblib",
        "Decision Tree Regressor": SAVED_MODELS_DIR / "decision_tree.joblib",
        "Random Forest Regressor": SAVED_MODELS_DIR / "random_forest.joblib"
    }

    require_files(
        logger,
        [TRAIN_TEST_JOBLIB] + list(model_paths.values()),
        hint="Hãy chạy 'python scripts/run_03_train.py' trước để huấn luyện và lưu trữ các mô hình."
    )

    # 3. Nạp dữ liệu Test set
    with timed(logger, "Nạp 'train_test_data.joblib'"):
        t0 = time.time()
        splits = joblib.load(TRAIN_TEST_JOBLIB)
        X_test = splits["X_test"]
        y_test_log = splits["y_test"]
        y_test_raw = splits["y_test_raw"]
        timings["data_loading_sec"] = round(time.time() - t0, 3)

    if quick_mode and len(X_test) > 2000:
        logger.info("  ⚡ Quick Mode: Lấy mẫu ngẫu nhiên 2,000 đơn hàng từ Test Set.")
        sample_idx = X_test.sample(n=2000, random_state=RANDOM_SEED).index
        X_test = X_test.loc[sample_idx]
        y_test_log = y_test_log.loc[sample_idx]
        y_test_raw = y_test_raw[sample_idx]

    logger.info(f"  ✓ Đã nạp thành công tập Test: {len(X_test):,} mẫu ({X_test.shape[1]} đặc trưng gốc).")

    # 4. Nạp 3 mô hình đã lưu
    with timed(logger, "Nạp 3 mô hình hồi quy Scratch từ models/saved_models/"):
        t0 = time.time()
        pipelines = {}
        for name, m_path in model_paths.items():
            pipelines[name] = joblib.load(m_path)
        timings["models_loading_sec"] = round(time.time() - t0, 3)

    logger.info(f"  ✓ Đã nạp đủ 3 mô hình Pipeline: {list(pipelines.keys())}.")

    # 5. Đánh giá kiểm thử Hold-out Test
    with timed(logger, f"Đánh giá kiểm thử độc lập trên {len(X_test):,} mẫu Test set"):
        t0 = time.time()
        eval_df = evaluate_holdout_test(pipelines, X_test, y_test_log, y_test_raw)
        timings["evaluation_sec"] = round(time.time() - t0, 3)

    logger.info("\n" + "=" * 80)
    logger.info("   📊 BẢNG ĐỐI ĐẦU HIỆU NĂNG 3 MÔ HÌNH TRÊN HOLD-OUT TEST SET")
    logger.info("=" * 80)
    for line in eval_df.to_string(index=False).split("\n"):
        logger.info(line)
    logger.info("=" * 80)

    # 6. Đo độ trễ suy luận (Inference Latency)
    with timed(logger, "Đo độ trễ suy luận thời gian thực (Inference Latency)"):
        t0 = time.time()
        n_sample_latency = min(1000, len(X_test))
        sample_subset = X_test.iloc[:n_sample_latency]
        rf_pipe = pipelines["Random Forest Regressor"]
        _ = rf_pipe.predict(sample_subset)
        latency_total = time.time() - t0
        latency_per_sample_ms = round((latency_total / n_sample_latency) * 1000, 3)
        timings["latency_test_sec"] = round(latency_total, 3)

    logger.info(f"  ✓ Tốc độ suy luận Random Forest: {latency_per_sample_ms:.3f} ms / đơn hàng ({int(1000 / max(latency_per_sample_ms, 0.001)):,} đơn/giây).")

    # 7. Vinh danh mô hình quán quân & Trích xuất Top đặc trưng
    best_model_name = "Random Forest Regressor"
    best_pipe = pipelines[best_model_name]
    best_row = eval_df[eval_df["Mô hình"] == best_model_name].iloc[0]

    with timed(logger, f"Trích xuất Top 15 đặc trưng chi phối doanh số từ {best_model_name}"):
        t0 = time.time()
        top_15_feats = extract_feature_importances(best_pipe, top_n=15)
        timings["feature_importance_sec"] = round(time.time() - t0, 3)

    logger.info(f"  ✓ Đã trích xuất thành công Top 15 đặc trưng chi phối doanh số.")

    # 8. Đóng gói mô hình Quán quân và xuất siêu dữ liệu metadata.json
    with timed(logger, "Đóng gói 'best_model.joblib' và xuất 'metadata.json'"):
        t0 = time.time()
        meta_path = export_evaluation_artifacts(
            model_name="Random Forest Regressor (Ensemble Bagging Scratch)",
            best_pipeline=best_pipe,
            metrics_dict=best_row.to_dict(),
            top_features=top_15_feats,
            output_dir=SAVED_MODELS_DIR,
            models_comparison=eval_df
        )
        timings["artifacts_export_sec"] = round(time.time() - t0, 3)

    logger.info(f"  ✔ Đã đóng gói mô hình quán quân: models/saved_models/best_model.joblib")
    logger.info(f"  ✔ Đã xuất siêu dữ liệu nghiệm thu: {meta_path.name}")

    # 9. Vẽ và xuất 4 biểu đồ trực quan hóa chuyên sâu vào docs/images/
    with timed(logger, "Vẽ và xuất 4 biểu đồ trực quan hóa vào docs/images/"):
        t0 = time.time()
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

        # Tính toán dự đoán của mô hình quán quân
        y_pred_log = best_pipe.predict(X_test)
        pred_clipped = np.clip(y_pred_log, -10.0, 15.0)
        y_pred_raw = np.clip(np.expm1(pred_clipped), 0.0, None)

        fig1_path = IMAGES_DIR / "model_01_actual_vs_predicted.png"
        fig2_path = IMAGES_DIR / "model_02_residuals.png"
        fig3_path = IMAGES_DIR / "model_03_performance_comparison.png"
        fig4_path = IMAGES_DIR / "model_04_feature_importances.png"

        plot_actual_vs_predicted(
            y_test_raw, y_pred_raw,
            model_name=best_model_name,
            save_path=fig1_path,
            max_points=1500
        )
        plot_residuals(
            y_test_raw, y_pred_raw,
            model_name=best_model_name,
            save_path=fig2_path,
            max_points=1500
        )
        plot_performance_comparison(
            eval_df,
            save_path=fig3_path
        )
        plot_feature_importances(
            top_15_feats,
            save_path=fig4_path,
            top_n=15
        )
        timings["export_figures_sec"] = round(time.time() - t0, 3)

    logger.info("  ✔ Đã xuất đủ 4 biểu đồ: model_01 đến model_04.")

    total_duration = round(time.time() - start_total, 2)

    # 10. Đóng gói phong bì kết quả Chặng 4 vào outputs/stage_04.json
    stage_data = {
        "stage": "04",
        "status": "success",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "duration_sec": total_duration,
        "quick_mode": quick_mode,
        "log_file": str(log_path),
        "config": {
            "test_samples": len(X_test),
            "random_seed": RANDOM_SEED,
            "best_model_chosen": best_model_name
        },
        "metrics": {
            "test_samples": len(X_test),
            "best_model_name": best_model_name,
            "best_r2_log": float(best_row["R2 (Log)"]),
            "best_r2_real": float(best_row["R2 (Thực tế)"]),
            "best_rmse_inr": float(best_row["RMSE (INR)"]),
            "best_mae_inr": float(best_row["MAE (INR)"]),
            "best_mse": float(best_row["MSE"]),
            "inference_speed_ms": latency_per_sample_ms
        },
        "tables": {
            "test_results": eval_df.to_dict(orient="records"),
            "top_15_features": top_15_feats.to_dict(orient="records")
        },
        "figures": [
            str(fig1_path),
            str(fig2_path),
            str(fig3_path),
            str(fig4_path)
        ],
        "timings": timings
    }

    envelope_path = save_stage_result("04", stage_data)
    logger.info(f"\n✔ Đã lưu phong bì kết quả Chặng 4: {envelope_path.name}")

    # 11. Tự động cập nhật Báo cáo HTML và Dashboard
    try:
        report_builder.build_report_04()
        report_builder.build_run_log_page()
        report_builder.build_index()
        logger.info("✔ Đã cập nhật giao diện báo cáo HTML: docs/report/04_model_evaluation_report.html, run_log.html & index.html")
    except Exception as e:
        logger.warning(f"⚠️ Cảnh báo khi cập nhật báo cáo HTML: {e}")

    logger.info("=" * 80)
    logger.info(f"🎉 HOÀN THÀNH CHẶNG 04 THÀNH CÔNG TRONG {total_duration:.2f} GIÂY!")
    logger.info("👉 Mở xem toàn bộ báo cáo tổng hợp tại: docs/report/index.html")
    logger.info("=" * 80)

    return stage_data


if __name__ == "__main__":
    args = parse_args()
    run_stage_04(quick_mode=args.quick)
