import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    CLEAN_CSV,
    TRAIN_TEST_JOBLIB,
    PREPROCESSOR_JOBLIB,
    LOGS_DIR,
    RANDOM_SEED,
    require_files,
    save_stage_result
)
from scripts.logger_utils import setup_logger, timed, log_environment
from scripts.feature_engineering import (
    load_clean_data,
    extract_domain_features,
    split_stratified_train_test,
    build_column_transformer,
    save_pipeline_and_splits
)
from scripts.report_builder import build_report_02, build_index


def main(clean_csv_path: str = None, extra_handler=None, logger=None) -> dict:
    """
    Kịch bản thực thi Chặng 02: Kỹ thuật đặc trưng & Pipeline phân tách tập dữ liệu:
    1. Đọc tệp dữ liệu sạch clean_emails.csv (32.660 mẫu).
    2. Trích xuất 7 đặc trưng miền nghiệp vụ và tính thống kê trung bình theo nhãn.
    3. Phân chia Train/Test 80/20 phân tầng (Stratified Split) với seed=42.
    4. Quét kiểm tra rò rỉ dữ liệu (Zero Data Leakage: số văn bản trùng lặp = 0).
    5. Khởi tạo cấu hình ColumnTransformer 2.007 chiều (TF-IDF + Chi2 + MinMaxScaler).
    6. Đóng gói lưu trữ preprocessor.joblib, train_test_data.joblib, train.csv, test.csv.
    7. Lưu phong bì kết quả vào outputs/stage_02.json.
    8. Tự động cập nhật báo cáo docs/report/02_feature_pipeline_report.html và index.html.
    """
    start_time_iso = datetime.now().isoformat()
    t_start = time.time()
    timings = {}

    # 1. Khởi tạo Logger riêng cho Chặng 02 (hỗ trợ extra_handler từ pipeline tổng)
    if logger is None:
        logger, log_file = setup_logger(log_dir=LOGS_DIR, prefix="run_02_fe", extra_handler=extra_handler)
    else:
        log_file = getattr(logger, "log_file", "")

    logger.info("=" * 75)
    logger.info("🚀 BẮT ĐẦU CHẶNG 02: KỸ THUẬT ĐẶC TRƯNG & DÂY CHUYỀN PIPELINE (FEATURE ENGINEERING)")
    logger.info("=" * 75)
    log_environment(logger, seed=RANDOM_SEED)

    # 2. Kiểm tra điều kiện tiên quyết
    clean_path = Path(clean_csv_path) if clean_csv_path else CLEAN_CSV
    require_files(
        logger,
        [clean_path],
        hint=f"Tệp '{clean_path.name}' chưa được tạo. Vui lòng chạy 'python scripts/run_01_eda.py' trước."
    )

    # 3. Nạp dữ liệu sạch
    with timed(logger, f"Nạp dữ liệu sạch từ {clean_path.name}") as b:
        df_clean = load_clean_data(str(clean_path))
    timings["load_clean_data"] = b.elapsed
    logger.info(f"  • Đã nạp thành công {len(df_clean):,} email sạch.")

    # 4. Trích xuất 7 đặc trưng miền nghiệp vụ
    with timed(logger, "Trích xuất 7 đặc trưng miền nghiệp vụ (Domain Features)") as b:
        df_feat = extract_domain_features(df_clean)
        if "label" not in df_feat.columns and "label" in df_clean.columns:
            df_feat["label"] = df_clean["label"]
    timings["extract_domain_features"] = b.elapsed

    domain_cols = [
        "caps_ratio", "exclamation_count", "dollar_count",
        "has_re_prefix", "subject_exclamation", "text_len", "log_text_len"
    ]
    desc_map = {
        "caps_ratio": "Tỷ lệ ký tự IN HOA trong toàn bộ email",
        "exclamation_count": "Số lượng dấu chấm than (!)",
        "dollar_count": "Số lượng ký hiệu tiền tệ ($)",
        "has_re_prefix": "Có tiền tố trả lời 'Re:' ở tiêu đề",
        "subject_exclamation": "Tiêu đề có chứa dấu chấm than (!)",
        "text_len": "Chiều dài ký tự chuỗi văn bản sạch",
        "log_text_len": "Biến đổi ln(1 + text_len) nén độ lệch"
    }
    feature_stats = []
    for col in domain_cols:
        h_mean = float(df_feat[df_feat["label"] == 0][col].mean())
        s_mean = float(df_feat[df_feat["label"] == 1][col].mean())
        feature_stats.append({
            "feature": col,
            "ham_mean": round(h_mean, 4),
            "spam_mean": round(s_mean, 4),
            "description": desc_map.get(col, "")
        })
    logger.info(f"  • Đã tính toán xong thống kê phân biệt 7 đặc trưng miền Ham vs Spam.")
    logger.info(f"    - Tiền tố 'Re:': Ham={feature_stats[3]['ham_mean']*100:.2f}%, Spam={feature_stats[3]['spam_mean']*100:.2f}%")
    logger.info(f"    - Dấu chấm than (!): Ham={feature_stats[1]['ham_mean']:.2f}, Spam={feature_stats[1]['spam_mean']:.2f}")

    # 5. Phân chia Train/Test 80/20 phân tầng
    with timed(logger, "Phân chia Train/Test 80/20 phân tầng (Stratified Split)") as b:
        X_train, X_test, y_train, y_test = split_stratified_train_test(
            df_feat, test_size=0.2, random_state=RANDOM_SEED
        )
    timings["train_test_split"] = b.elapsed

    train_ham = int((y_train == 0).sum())
    train_spam = int((y_train == 1).sum())
    test_ham = int((y_test == 0).sum())
    test_spam = int((y_test == 1).sum())
    train_spam_ratio = float(train_spam / len(y_train) * 100)
    test_spam_ratio = float(test_spam / len(y_test) * 100)

    logger.info(f"  • Tập Huấn luyện (Train set 80%): {len(X_train):,} mẫu (Ham={train_ham:,}, Spam={train_spam:,} ~ {train_spam_ratio:.2f}%)")
    logger.info(f"  • Tập Kiểm thử (Test set 20%): {len(X_test):,} mẫu (Ham={test_ham:,}, Spam={test_spam:,} ~ {test_spam_ratio:.2f}%)")

    # 6. Kiểm tra rò rỉ dữ liệu (Zero Data Leakage Check)
    with timed(logger, "Kiểm tra rò rỉ dữ liệu ranh giới Train/Test (Leakage Check)") as b:
        train_texts = set(X_train["text_clean"].dropna().astype(str))
        test_texts = set(X_test["text_clean"].dropna().astype(str))
        overlap_count = len(train_texts.intersection(test_texts))
    timings["leakage_check"] = b.elapsed

    if overlap_count == 0:
        logger.info(f"  ✔ Hàng rào chống rò rỉ: Số lượng văn bản sạch trùng giữa Train và Test = 0 mẫu (PASS 100%).")
    else:
        logger.warning(f"  ⚠️ Cảnh báo: Phát hiện {overlap_count} mẫu văn bản trùng giữa Train và Test!")

    # 7. Xây dựng cấu hình ColumnTransformer
    with timed(logger, "Thiết lập cấu hình dây chuyền tiền xử lý đa tầng (ColumnTransformer)") as b:
        preprocessor = build_column_transformer(max_features=5000, k_best=2000, ngram_range=(1, 2))
        total_dim = 2000 + 7
    timings["build_preprocessor"] = b.elapsed
    logger.info(f"  • Cấu trúc Pipeline: Nhánh Text TF-IDF + Chi2 (2,000 chiều) + Nhánh Số MinMaxScaler (7 chiều) = {total_dim:,} chiều.")
    logger.info(f"  • Tuân thủ nghiêm ngặt Zero Data Leakage: Pipeline chỉ được fit trên tập Train ở Chặng 3, không fit tại đây.")

    # 8. Đóng gói và lưu hiện vật
    with timed(logger, "Đóng gói preprocessor.joblib, train_test_data.joblib và xuất train.csv, test.csv") as b:
        save_pipeline_and_splits(
            preprocessor,
            X_train, X_test, y_train, y_test,
            prep_path=str(PREPROCESSOR_JOBLIB),
            data_path=str(TRAIN_TEST_JOBLIB),
            export_csv=True
        )
    timings["save_artifacts"] = b.elapsed
    logger.info(f"  ✔ Đã lưu: {PREPROCESSOR_JOBLIB.name}")
    logger.info(f"  ✔ Đã lưu: {TRAIN_TEST_JOBLIB.name}")
    logger.info(f"  ✔ Đã xuất độc lập: data/processed/train.csv & test.csv")

    total_duration = round(time.time() - t_start, 2)

    # 9. Đóng gói phong bì kết quả vào outputs/stage_02.json
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
            "tfidf_max_features": 5000,
            "k_best": 2000,
            "ngram_range": [1, 2],
            "scaler": "MinMaxScaler"
        },
        "metrics": {
            "total_samples": len(df_clean),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "train_ham_count": train_ham,
            "train_spam_count": train_spam,
            "train_spam_ratio": round(train_spam_ratio, 2),
            "test_ham_count": test_ham,
            "test_spam_count": test_spam,
            "test_spam_ratio": round(test_spam_ratio, 2),
            "text_overlap_count": overlap_count,
            "text_features_dim": 2000,
            "numeric_features_dim": 7,
            "total_features_dim": total_dim
        },
        "tables": {
            "domain_features_stats": feature_stats
        },
        "figures": [],
        "timings": timings
    }
    save_stage_result("02", stage_data)
    logger.info("  ✔ Đã lưu phong bì kết quả: outputs/stage_02.json")

    # 10. Tự động sinh báo cáo HTML
    with timed(logger, "Tự động sinh báo cáo HTML Chặng 2 và cập nhật index.html") as b:
        build_report_02()
        build_index()
    timings["build_html_reports"] = b.elapsed
    logger.info("  ✔ Đã tự động tạo: docs/report/02_feature_pipeline_report.html")
    logger.info("  ✔ Đã cập nhật bảng trạng thái: docs/report/index.html")

    logger.info("=" * 75)
    logger.info(f"🎉 HOÀN THÀNH CHẶNG 02 THÀNH CÔNG TRONG {total_duration:.2f} GIÂY!")
    logger.info(f"👉 Mở xem báo cáo trực quan tại: docs/report/02_feature_pipeline_report.html")
    logger.info("=" * 75)

    return stage_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chặng 02: Kỹ thuật đặc trưng & Pipeline phân tách tập dữ liệu")
    parser.add_argument("--clean-csv", type=str, default=str(CLEAN_CSV), help="Đường dẫn tới tệp dữ liệu sạch clean_emails.csv")
    args = parser.parse_args()

    main(clean_csv_path=args.clean_csv)
