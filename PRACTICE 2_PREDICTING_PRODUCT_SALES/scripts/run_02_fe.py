import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    RAW_CSV,
    CLEAN_CSV,
    TRAIN_CSV,
    TEST_CSV,
    TRAIN_TEST_JOBLIB,
    PREPROCESSOR_JOBLIB,
    PROCESSED_DIR,
    SAVED_MODELS_DIR,
    LOGS_DIR,
    RANDOM_SEED,
    require_files,
    save_stage_result
)
from scripts.logger_utils import setup_logger, timed, log_environment
from scripts.feature_engineering import (
    clean_raw_sales_data,
    extract_domain_features,
    detect_collinear_features,
    split_train_test,
    build_column_transformer,
    export_train_test_csv
)
from scripts.report_builder import build_report_02, build_index


def main(raw_csv_path: str = None, clean_csv_path: str = None, extra_handler=None, logger=None) -> dict:
    """
    Kịch bản thực thi Chặng 02: Kỹ thuật đặc trưng & Pipeline tiền xử lý khép kín 6 bước:
    1. [B1 Làm sạch & Khử trùng lặp]: Đọc Amazon Sale Report.csv thô, lọc bỏ đơn Cancelled, âm/null, khử trùng lặp.
    2. [B2 Tạo biến miền TMĐT]: Trích xuất month, day_of_week, cờ khuyến mãi, B2B, phân nhóm địa lý.
    3. [B3 Chia Train/Test 80/20]: Tách độc lập tập Train/Test, biến đổi log1p mục tiêu, kiểm tra Zero Data Leakage.
    4. [B4 Chuẩn hóa biến số]: Khởi tạo StandardScaler, huấn luyện (.fit) DUY NHẤT trên Train set.
    5. [B5 Mã hóa biến định tính]: Khởi tạo OneHotEncoder, huấn luyện (.fit) DUY NHẤT trên Train set.
    6. [B6 Đóng gói Pipeline]: Đóng gói ColumnTransformer, xuất hiện vật (train.csv, test.csv, *.joblib).
    7. Lưu phong bì kết quả vào outputs/stage_02.json.
    8. Tự động sinh docs/report/02_feature_pipeline_report.html và cập nhật index.html.
    """
    start_time_iso = datetime.now().isoformat()
    t_start = time.time()
    timings = {}

    # 1. Khởi tạo Logger riêng cho Chặng 02
    if logger is None:
        logger, log_file = setup_logger(log_dir=LOGS_DIR, prefix="run_02_fe", extra_handler=extra_handler)
    else:
        log_file = getattr(logger, "log_file", "")

    logger.info("=" * 80)
    logger.info("🚀 BẮT ĐẦU CHẶNG 02: KỸ THUẬT ĐẶC TRƯNG & DÂY CHUYỀN PIPELINE KHÉP KÍN 6 BƯỚC")
    logger.info("=" * 80)
    log_environment(logger, seed=RANDOM_SEED)

    # 2. Kiểm tra điều kiện tiên quyết
    raw_path = Path(raw_csv_path) if raw_csv_path else RAW_CSV
    require_files(
        logger,
        [raw_path],
        hint=f"Tệp '{raw_path.name}' chưa được tìm thấy. Vui lòng kiểm tra thư mục '{raw_path.parent}'."
    )

    # =========================================================================
    # [BƯỚC 1: LÀM SẠCH DỮ LIỆU & KHỬ TRÙNG LẶP THÔ (DATA CLEANING & DEDUPLICATION)]
    # =========================================================================
    logger.info("--- [BƯỚC 1/6]: LÀM SẠCH DỮ LIỆU THÔ & KHỬ TRÙNG LẶP TRIỆT ĐỂ ---")
    with timed(logger, f"Nạp dữ liệu thô từ {raw_path.name}") as b:
        df_raw = pd.read_csv(raw_path, low_memory=False)
    timings["load_raw_data"] = b.elapsed
    logger.info(f"  • Đã nạp thành công {len(df_raw):,} dòng dữ liệu thô ban đầu ({df_raw.shape[1]} thuộc tính).")

    with timed(logger, "Lọc đơn Cancelled, đơn âm/null và khử trùng lặp (Zero Duplicate Leakage)") as b:
        df_clean, cleaning_stats = clean_raw_sales_data(df_raw)
        df_clean.to_csv(CLEAN_CSV, index=False)
    timings["clean_raw_data"] = b.elapsed
    logger.info(f"  • Loại bỏ đơn hủy Status 'Cancelled': {cleaning_stats['cancelled_removed']:,} đơn.")
    logger.info(f"  • Loại bỏ đơn khuyết thiếu / âm Amount hoặc Qty: {cleaning_stats['invalid_amount_qty_removed']:,} đơn.")
    logger.info(f"  • Loại bỏ đơn trùng lặp bản ghi: {cleaning_stats['duplicates_removed']:,} đơn.")
    logger.info(f"  ✔ Dữ liệu sạch thu được: {cleaning_stats['clean_rows']:,} đơn hàng hợp lệ.")
    logger.info(f"  ✔ Đã xuất tệp dữ liệu sạch: data/processed/{CLEAN_CSV.name}")

    # =========================================================================
    # [BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG MIỀN NGHIỆP VỤ TMĐT (DOMAIN FEATURES)]
    # =========================================================================
    logger.info("--- [BƯỚC 2/6]: TRÍCH XUẤT CÁC ĐẶC TRƯNG MIỀN NGHIỆP VỤ TMĐT ---")
    with timed(logger, "Trích xuất đặc trưng thời gian, tiếp thị, B2B và địa lý") as b:
        df_feat = extract_domain_features(df_clean)
    timings["extract_features"] = b.elapsed
    logger.info(f"  • Bảng đặc trưng mới: {df_feat.shape[1]} cột (đã tạo: month, day_of_week, is_weekend, is_promoted, is_b2b, state_grouped).")

    # =========================================================================
    # [BƯỚC 3: PHÂN CHIA TRAIN/TEST 80/20 & CHỐNG RÒ RỈ (SPLIT & ANTI-LEAKAGE)]
    # =========================================================================
    logger.info("--- [BƯỚC 3/6]: PHÂN CHIA TRAIN/TEST 80/20 & CƠ CHẾ CHỐNG RÒ RỈ DỮ LIỆU ---")
    with timed(logger, "Quét phát hiện các cặp đặc trưng đa cộng tuyến (|r| > 0.6)") as b:
        collinear_pairs = detect_collinear_features(df_feat, threshold=0.6)
    timings["collinear_check"] = b.elapsed
    logger.info(f"  • Phát hiện {len(collinear_pairs)} cặp đa cộng tuyến (|r| > 0.6): {collinear_pairs}.")
    logger.info("  • Đã chủ động loại bỏ 'is_weekend' để tránh trùng lặp thông tin với 'day_of_week'.")

    with timed(logger, "Phân chia Train/Test 80/20 và biến đổi log1p biến mục tiêu") as b:
        X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = split_train_test(
            df_feat, test_size=0.2, random_state=RANDOM_SEED
        )
    timings["split_train_test"] = b.elapsed
    logger.info(f"  • Phân chia thành công: Train={len(X_train):,} mẫu (80%), Test={len(X_test):,} mẫu (20%).")
    logger.info(f"  • Doanh số TB: Train={np.mean(y_train_raw):.2f} INR, Test={np.mean(y_test_raw):.2f} INR.")

    with timed(logger, "Kiểm tra rò rỉ dữ liệu (Zero Data Leakage Check)") as b:
        train_idx_set = set(X_train.index)
        test_idx_set = set(X_test.index)
        overlap_count = len(train_idx_set.intersection(test_idx_set))
    timings["leakage_check"] = b.elapsed
    logger.info(f"  ✔ Trùng lặp giữa Train và Test: {overlap_count} mẫu (Zero Data Leakage đạt 100%).")

    # =========================================================================
    # [BƯỚC 4 & 5: THIẾT LẬP VÀ KHỚP DÂY CHUYỀN COLUMNTRANSFORMER]
    # =========================================================================
    logger.info("--- [BƯỚC 4 & 5/6]: CHUẨN HÓA & MÃ HÓA QUA DÂY CHUYỀN COLUMNTRANSFORMER ---")
    with timed(logger, "Cấu hình ColumnTransformer và huấn luyện (.fit) DUY NHẤT trên Train set") as b:
        num_cols = ['Qty', 'month', 'day_of_week', 'is_promoted', 'is_b2b']
        cat_cols = ['Category', 'Size', 'Fulfilment', 'ship-service-level', 'Courier Status', 'state_grouped']

        preprocessor = build_column_transformer(num_cols, cat_cols)
        preprocessor.fit(X_train)

        X_train_trans = preprocessor.transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        total_dim = X_train_trans.shape[1]
    timings["build_column_transformer"] = b.elapsed
    logger.info(f"  • Không gian đặc trưng sau One-Hot: {total_dim} chiều ({len(num_cols)} số + {total_dim - len(num_cols)} One-Hot).")

    # =========================================================================
    # [BƯỚC 6: ĐÓNG GÓI PIPELINE & XUẤT HIỆN VẬT BÀN GIAO]
    # =========================================================================
    logger.info("--- [BƯỚC 6/6]: ĐÓNG GÓI PIPELINE & XUẤT HIỆN VẬT BÀN GIAO ---")
    with timed(logger, "Lưu trữ các tệp train.csv, test.csv, train_test_data.joblib & preprocessor.joblib") as b:
        export_train_test_csv(X_train, X_test, y_train_raw, y_test_raw, PROCESSED_DIR)

        splits_data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_raw': y_train_raw,
            'y_test_raw': y_test_raw,
            'num_cols': num_cols,
            'cat_cols': cat_cols
        }
        joblib.dump(splits_data, TRAIN_TEST_JOBLIB)
        joblib.dump(preprocessor, PREPROCESSOR_JOBLIB)
    timings["export_artifacts"] = b.elapsed
    logger.info(f"  ✔ Đã lưu: {TRAIN_TEST_JOBLIB.name} và {PREPROCESSOR_JOBLIB.name}")
    logger.info("  ✔ Đã xuất: data/processed/train.csv & test.csv")

    total_duration = round(time.time() - t_start, 2)

    # Đóng gói phong bì kết quả vào outputs/stage_02.json
    feature_descriptions = [
        {"feature": "Qty", "type": "Numeric (Số lượng)", "description": "Số lượng sản phẩm đặt mua trong đơn hàng"},
        {"feature": "month", "type": "Numeric (Thời gian)", "description": "Tháng phát sinh giao dịch (Tháng 4, 5, 6)"},
        {"feature": "day_of_week", "type": "Numeric (Thời gian)", "description": "Thứ trong tuần (0: Thứ 2 đến 6: Chủ nhật)"},
        {"feature": "is_promoted", "type": "Binary Flag (Tiếp thị)", "description": "Cờ 1 nếu đơn hàng áp dụng voucher/mã khuyến mãi, 0 nếu không"},
        {"feature": "is_b2b", "type": "Binary Flag (Nghiệp vụ)", "description": "Cờ 1 nếu là khách hàng doanh nghiệp sỉ B2B, 0 nếu khách lẻ"},
        {"feature": "Category", "type": "Categorical (Ngành hàng)", "description": "Nhóm ngành hàng thời trang (Set, Kurta, Top, Western Dress...)"},
        {"feature": "Size", "type": "Categorical (Kích cỡ)", "description": "Kích cỡ sản phẩm (XS, S, M, L, XL, XXL, 3XL...)"},
        {"feature": "Fulfilment", "type": "Categorical (Vận hành)", "description": "Kênh hoàn tất đơn hàng (Merchant tự giao vs Amazon FBA)"},
        {"feature": "ship-service-level", "type": "Categorical (Dịch vụ)", "description": "Mức độ dịch vụ giao hàng (Standard vs Expedited)"},
        {"feature": "Courier Status", "type": "Categorical (Trạng thái)", "description": "Tình trạng chuyển phát qua bưu tá (Shipped, Unshipped)"},
        {"feature": "state_grouped", "type": "Categorical (Địa lý)", "description": "Phân nhóm địa lý Top 10 bang lớn tại Ấn Độ hoặc 'Other'"}
    ]

    stage_data = {
        "stage": "02",
        "status": "success",
        "started_at": start_time_iso,
        "finished_at": datetime.now().isoformat(),
        "duration_sec": total_duration,
        "quick_mode": False,
        "log_file": str(log_file),
        "config": {
            "test_size": 0.2,
            "random_seed": RANDOM_SEED,
            "numeric_scaler": "StandardScaler",
            "categorical_encoder": "OneHotEncoder(handle_unknown='ignore')",
            "target_transform": "ln(1 + Amount)"
        },
        "metrics": {
            "raw_rows": cleaning_stats["raw_rows"],
            "cancelled_removed": cleaning_stats["cancelled_removed"],
            "invalid_amount_qty_removed": cleaning_stats["invalid_amount_qty_removed"],
            "duplicates_removed": cleaning_stats["duplicates_removed"],
            "clean_rows": cleaning_stats["clean_rows"],
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_amount_mean": round(float(np.mean(y_train_raw)), 2),
            "test_amount_mean": round(float(np.mean(y_test_raw)), 2),
            "train_amount_std": round(float(np.std(y_train_raw)), 2),
            "test_amount_std": round(float(np.std(y_test_raw)), 2),
            "data_leakage_overlap": overlap_count,
            "total_features_dim": total_dim,
            "num_features_dim": len(num_cols),
            "cat_encoded_dim": total_dim - len(num_cols)
        },
        "tables": {
            "domain_features_summary": feature_descriptions,
            "collinear_features": [f"{pair[0]} - {pair[1]} (|r|={pair[2]:.2f})" for pair in collinear_pairs]
        },
        "figures": [],
        "timings": timings
    }
    save_stage_result("02", stage_data)
    logger.info("  ✔ Đã lưu phong bì kết quả: outputs/stage_02.json")

    # Tự động sinh báo cáo HTML
    with timed(logger, "Tự động sinh báo cáo HTML Chặng 2 và cập nhật index.html") as b:
        build_report_02()
        build_index()
    timings["build_html_reports"] = b.elapsed
    logger.info("  ✔ Đã tự động tạo: docs/report/02_feature_pipeline_report.html")
    logger.info("  ✔ Đã cập nhật bảng trạng thái: docs/report/index.html")

    logger.info("=" * 80)
    logger.info(f"🎉 HOÀN THÀNH CHẶNG 02 THÀNH CÔNG TRONG {total_duration:.2f} GIÂY!")
    logger.info(f"👉 Mở xem báo cáo trực quan tại: docs/report/02_feature_pipeline_report.html")
    logger.info("=" * 80)

    return stage_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chặng 02: Kỹ thuật đặc trưng & Pipeline tiền xử lý khép kín 6 bước (Feature Engineering)")
    parser.add_argument("--raw-csv", type=str, default=str(RAW_CSV), help="Đường dẫn tới tệp dữ liệu thô Amazon Sale Report.csv")
    parser.add_argument("--clean-csv", type=str, default=str(CLEAN_CSV), help="Đường dẫn tới tệp dữ liệu sạch clean_amazon_sales.csv")
    args = parser.parse_args()

    main(raw_csv_path=args.raw_csv, clean_csv_path=args.clean_csv)
