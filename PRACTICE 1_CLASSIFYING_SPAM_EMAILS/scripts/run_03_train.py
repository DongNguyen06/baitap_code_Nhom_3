import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# Thiết lập đường dẫn gốc cho dự án
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
from scripts.feature_engineering import load_pipeline_and_splits
from scripts.model_training import (
    create_scratch_models,
    build_training_pipelines,
    run_cross_validation,
    tune_grid_search,
)
import scripts.report_builder as report_builder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chặng 03: Huấn luyện 3 mô hình Scratch & 5-Fold Stratified Cross Validation."
    )
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="Chạy nhanh ở chế độ demo (CV 2-Fold, subsample 1.500 mẫu) để kiểm tra luồng."
    )
    return parser.parse_args()


def run_stage_03(quick_mode: bool = False, extra_handler=None):
    start_total = time.time()
    started_at = datetime.now().isoformat()

    # 1. Khởi tạo Logger ghi nhật ký console và file logs/stage_03_*.log (hỗ trợ extra_handler)
    logger, log_path = setup_logger(log_dir=LOGS_DIR, prefix="run_03_train", extra_handler=extra_handler)
    logger.info("=" * 80)
    logger.info("   🚀 BẮT ĐẦU CHẶNG 03: HUẤN LUYỆN 3 MÔ HÌNH SCRATCH & KIỂM ĐỊNH CHÉO")
    logger.info(f"   Thời điểm bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Chế độ chạy: {'⚡ Rút gọn (Quick Mode)' if quick_mode else '🏆 Đầy đủ (Full Benchmark)'}")
    logger.info("=" * 80)

    log_environment(logger)
    timings = {}

    # 2. Kiểm tra điều kiện tiên quyết
    require_files(
        logger,
        [PREPROCESSOR_JOBLIB, TRAIN_TEST_JOBLIB],
        hint="Hãy chạy 'python scripts/run_02_fe.py' trước để chuẩn bị preprocessor và tập dữ liệu phân tầng."
    )

    # 3. Nạp dữ liệu và preprocessor
    with timed(logger, "Nạp 'preprocessor.joblib' và 'train_test_data.joblib'"):
        t0 = time.time()
        preprocessor, X_train, X_test, y_train, y_test = load_pipeline_and_splits(
            str(PREPROCESSOR_JOBLIB), str(TRAIN_TEST_JOBLIB)
        )
        timings["data_loading_sec"] = round(time.time() - t0, 3)

    logger.info(f"  ✓ Đã nạp thành công: X_train={len(X_train):,} mẫu, X_test={len(X_test):,} mẫu.")
    logger.info(f"  ✓ Tỷ lệ Spam trong tập Train: {float(y_train.mean()) * 100:.2f}%.")

    # 4. Khởi tạo 3 mô hình Scratch & Pipelines
    with timed(logger, "Khởi tạo 3 mô hình Scratch và ghép nối Pipeline đa tầng"):
        t0 = time.time()
        models = create_scratch_models(random_state=RANDOM_SEED)
        pipelines = build_training_pipelines(preprocessor, models)
        timings["pipeline_init_sec"] = round(time.time() - t0, 3)

    for name in pipelines.keys():
        logger.info(f"  • Sẵn sàng pipeline: [{name}]")

    # 5. Chạy Kiểm định chéo (Stratified CV)
    cv_splits = 2 if quick_mode else 5
    logger.info(f"\n[Bước 1/2] Tiến hành kiểm định chéo {cv_splits}-Fold Stratified CV trên {len(X_train):,} mẫu Train...")
    cv_strat = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_SEED)

    with timed(logger, f"Thực thi {cv_splits}-Fold Stratified CV song song"):
        t0 = time.time()
        cv_summary_df = run_cross_validation(
            pipelines, X_train, y_train, cv=cv_strat, random_state=RANDOM_SEED
        )
        timings["cross_validation_sec"] = round(time.time() - t0, 3)

    logger.info("\n--- BẢNG KẾT QUẢ STRATIFIED CV TRÊN TẬP HUẤN LUYỆN (TRAIN SET) ---")
    logger.info("\n" + cv_summary_df.to_string(index=False))

    # 6. Tinh chỉnh siêu tham số và huấn luyện toàn diện trên toàn bộ tập Train
    subsample_size = 1500 if quick_mode else 6000
    tune_splits = 2 if quick_mode else 3
    cv_tune = StratifiedKFold(n_splits=tune_splits, shuffle=True, random_state=RANDOM_SEED)

    tune_desc = f"subsample {subsample_size:,} mẫu" if subsample_size is not None else f"toàn bộ 100% ({len(X_train):,} mẫu Train)"
    logger.info(f"\n[Bước 2/2] Quét siêu tham số ({tune_desc}) và huấn luyện trên {len(X_train):,} mẫu Train...")
    with timed(logger, "Quét GridSearchCV và huấn luyện toàn diện 3 mô hình tối ưu"):
        t0 = time.time()
        tuned_models, gs_summary_df = tune_grid_search(
            pipelines,
            X_train=X_train,
            y_train=y_train,
            cv=cv_tune,
            subsample_size=subsample_size,
            random_state=RANDOM_SEED,
            models_dir=str(SAVED_MODELS_DIR),
            return_summary=True
        )
        timings["grid_search_train_sec"] = round(time.time() - t0, 3)

    logger.info("\n--- BẢNG KẾT QUẢ TINH CHỈNH SIÊU THAM SỐ (GRID SEARCH) ---")
    logger.info("\n" + gs_summary_df.to_string(index=False))
    logger.info(f"  ✓ Đã lưu trữ 3 mô hình .joblib độc lập vào: {SAVED_MODELS_DIR.resolve()}")

    total_duration = round(time.time() - start_total, 2)
    finished_at = datetime.now().isoformat()
    timings["total_stage_sec"] = total_duration

    # 7. Đóng gói phong bì kết quả outputs/stage_03.json
    cv_records = cv_summary_df.to_dict(orient="records")
    gs_records = gs_summary_df.to_dict(orient="records")

    stage_data = {
        "status": "success",
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": total_duration,
        "quick_mode": quick_mode,
        "log_file": str(log_path),
        "config": {
            "random_seed": RANDOM_SEED,
            "cv_splits": cv_splits,
            "subsample_size": subsample_size if subsample_size is not None else len(X_train),
            "models": list(pipelines.keys())
        },
        "metrics": {
            "train_samples": len(X_train),
            "cv_splits": cv_splits,
            "models_count": len(pipelines),
            "models_trained": list(pipelines.keys())
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
