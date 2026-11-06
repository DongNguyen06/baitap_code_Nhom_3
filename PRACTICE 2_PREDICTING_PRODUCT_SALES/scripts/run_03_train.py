import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    RANDOM_SEED,
    TRAIN_TEST_JOBLIB,
    PREPROCESSOR_JOBLIB,
    SAVED_MODELS_DIR,
    LOGS_DIR,
    require_files,
    save_stage_result,
)
from scripts.logger_utils import setup_logger, timed, log_environment
from scripts.model_training import (
    create_scratch_models,
    build_training_pipelines,
    evaluate_cv_models,
    tune_grid_search,
)
import scripts.report_builder as report_builder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chặng 03: Huấn luyện 3 mô hình hồi quy Scratch & 5-Fold Cross Validation."
    )
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="Chạy nhanh ở chế độ demo (CV 2-Fold, subsample 2.000 mẫu) để kiểm tra luồng."
    )
    return parser.parse_args()


def run_stage_03(quick_mode: bool = False, extra_handler=None):
    start_total = time.time()
    started_at = datetime.now().isoformat()

    # 1. Khởi tạo Logger ghi nhật ký console và file logs/run_03_train_*.log (hỗ trợ extra_handler)
    logger, log_path = setup_logger(log_dir=LOGS_DIR, prefix="run_03_train", extra_handler=extra_handler)
    logger.info("=" * 80)
    logger.info("   🚀 BẮT ĐẦU CHẶNG 03: HUẤN LUYỆN 3 MÔ HÌNH HỒI QUY SCRATCH & 5-FOLD CV")
    logger.info(f"   Thời điểm bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Chế độ chạy: {'⚡ Rút gọn (Quick Mode)' if quick_mode else '🏆 Đầy đủ 100% dữ liệu (Full Benchmark: 82k mẫu)'}")
    logger.info("=" * 80)

    log_environment(logger)
    timings = {}

    # 2. Kiểm tra điều kiện tiên quyết
    require_files(
        logger,
        [PREPROCESSOR_JOBLIB, TRAIN_TEST_JOBLIB],
        hint="Hãy chạy 'python scripts/run_02_fe.py' trước để chuẩn bị preprocessor và dữ liệu Train/Test."
    )

    # 3. Nạp dữ liệu và preprocessor
    with timed(logger, "Nạp 'preprocessor.joblib' và 'train_test_data.joblib'"):
        t0 = time.time()
        splits = joblib.load(TRAIN_TEST_JOBLIB)
        preprocessor = joblib.load(PREPROCESSOR_JOBLIB)
        X_train = splits["X_train"]
        y_train = splits["y_train"]
        timings["data_loading_sec"] = round(time.time() - t0, 3)

    logger.info(f"  ✓ Đã nạp thành công tập Train: {len(X_train):,} mẫu ({X_train.shape[1]} đặc trưng gốc).")

    # 4. Khởi tạo 3 mô hình Scratch & Pipelines
    with timed(logger, "Khởi tạo 3 mô hình Scratch và ghép nối Pipeline đa tầng"):
        t0 = time.time()
        models = create_scratch_models()
        pipelines = build_training_pipelines(preprocessor, models=models)
        timings["init_models_sec"] = round(time.time() - t0, 3)

    logger.info(f"  ✓ Đã ghép nối thành công {len(pipelines)} Pipeline: {list(pipelines.keys())}.")

    # 5. Đánh giá kiểm định chéo K-Fold Cross Validation
    cv_splits = 2 if quick_mode else 5
    sample_size = 2000 if quick_mode else None
    eval_samples = sample_size if sample_size else len(X_train)

    with timed(logger, f"Đánh giá độ ổn định {cv_splits}-Fold Cross Validation trên {eval_samples:,} mẫu Train"):
        t0 = time.time()
        cv_df = evaluate_cv_models(
            pipelines,
            X_train=X_train,
            y_train=y_train,
            cv=cv_splits,
            sample_size=sample_size,
            random_state=RANDOM_SEED
        )
        timings["cross_validation_sec"] = round(time.time() - t0, 3)

    logger.info("\n" + "=" * 80)
    logger.info(f"   📊 BẢNG KẾT QUẢ {cv_splits}-FOLD CROSS VALIDATION (TẬP TRAIN)")
    logger.info("=" * 80)
    for line in cv_df.to_string(index=False).split("\n"):
        logger.info(line)
    logger.info("=" * 80)

    # 6. Tinh chỉnh siêu tham số qua GridSearchCV và lưu trữ mô hình
    subsample = 2000 if quick_mode else None
    grid_cv = 2 if quick_mode else 3
    tune_samples = subsample if subsample else len(X_train)

    with timed(logger, f"Tinh chỉnh siêu tham số (GridSearchCV {grid_cv}-Fold trên {tune_samples:,} mẫu)"):
        t0 = time.time()
        trained_pipelines, gs_summary = tune_grid_search(
            pipelines,
            X_train=X_train,
            y_train=y_train,
            cv=grid_cv,
            subsample_size=subsample,
            random_state=RANDOM_SEED,
            models_dir=SAVED_MODELS_DIR,
            return_summary=True
        )
        timings["grid_search_sec"] = round(time.time() - t0, 3)

    logger.info("\n" + "=" * 80)
    logger.info("   🎯 BẢNG KẾT QUẢ TINH CHỈNH SIÊU THAM SỐ (GRIDSEARCHCV)")
    logger.info("=" * 80)
    for line in gs_summary.to_string(index=False).split("\n"):
        logger.info(line)
    logger.info("=" * 80)

    # Tìm mô hình có Best CV R² cao nhất
    best_row = gs_summary.sort_values(by="Best CV R²", ascending=False).iloc[0]
    best_cv_model = str(best_row["Mô hình"])
    best_cv_score = str(best_row["Best CV R²"])

    total_duration = round(time.time() - start_total, 2)

    # 7. Lưu phong bì kết quả Chặng 3 vào outputs/stage_03.json
    cv_records = cv_df.to_dict(orient="records")
    gs_records = gs_summary.to_dict(orient="records")

    stage_data = {
        "stage": "03",
        "status": "success",
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "duration_sec": total_duration,
        "quick_mode": quick_mode,
        "log_file": str(log_path),
        "config": {
            "cv_splits": cv_splits,
            "train_samples_used": eval_samples,
            "random_seed": RANDOM_SEED,
            "optimization": "GridSearchCV"
        },
        "metrics": {
            "models_count": len(pipelines),
            "models_trained": list(pipelines.keys()),
            "best_cv_model": best_cv_model,
            "best_cv_r2": best_cv_score,
            "train_samples_used": eval_samples
        },
        "tables": {
            "cv_summary": cv_records,
            "grid_search_summary": gs_records
        },
        "figures": [],
        "timings": timings
    }

    envelope_path = save_stage_result("03", stage_data)
    logger.info(f"\n✔ Đã lưu phong bì kết quả Chặng 3: {envelope_path.name}")

    # 8. Cập nhật Báo cáo HTML tự động
    try:
        report_builder.build_report_03()
        report_builder.build_index()
        logger.info("✔ Đã cập nhật giao diện báo cáo HTML: docs/report/03_model_training_report.html & index.html")
    except Exception as e:
        logger.warning(f"⚠️ Cảnh báo khi cập nhật báo cáo HTML: {e}")

    logger.info("=" * 80)
    logger.info(f"🎉 HOÀN THÀNH CHẶNG 03 THÀNH CÔNG TRONG {total_duration:.2f} GIÂY!")
    logger.info(f"👉 Tiếp theo, hãy chạy 'python scripts/run_04_eval.py' để đánh giá Test set và xuất best_model.")
    logger.info("=" * 80)

    return stage_data


if __name__ == "__main__":
    args = parse_args()
    run_stage_03(quick_mode=args.quick)
