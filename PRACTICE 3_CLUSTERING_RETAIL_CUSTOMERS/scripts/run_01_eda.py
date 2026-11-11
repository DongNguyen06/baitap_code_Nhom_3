"""
Kịch bản Điều phối: run_01_eda.py
Dự án: Practice 3: Clustering Retail Customers — Group 3
Chặng 1: Khám phá Dữ liệu Khách hàng Bán lẻ (EDA — Read-Only trên Raw Data)
Đầu ra: docs/images/eda_01 -> eda_05.png, outputs/stage_01.json, logs/run_01_eda.log, docs/report/01_data_report.html
Đồng bộ định dạng đường dẫn tương đối và đa luồng log chuẩn Practice 1 & 2.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Đảm bảo đường dẫn import gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    RAW_CSV,
    LOGS_DIR,
    IMAGES_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
    to_rel_path,
    require_files,
    save_stage_result
)
from scripts.logger_utils import setup_logger, timed, log_environment
from data.scripts.eda import run_eda


def main(extra_handler=None, logger=None) -> Dict[str, Any]:
    t_start = time.time()

    # 1. Khởi tạo Logger riêng cho Chặng 01 (hỗ trợ extra_handler từ pipeline tổng)
    if logger is None:
        logger, log_file = setup_logger(log_dir=LOGS_DIR, prefix="run_01_eda", extra_handler=extra_handler)
    else:
        log_file = getattr(logger, "log_file", "")

    logger.info("=" * 80)
    logger.info("🚀 BẮT ĐẦU CHẶNG 01: KHÁM PHÁ DỮ LIỆU KHÁCH HÀNG BÁN LẺ (EDA READ-ONLY)")
    logger.info("=" * 80)
    log_environment(logger, seed=RANDOM_SEED)

    # 2. Kiểm tra tệp tiền đề bắt buộc
    require_files(RAW_CSV)
    logger.info(f"Đã xác minh nguồn dữ liệu thô: {to_rel_path(RAW_CSV)}")

    # 3. Thực thi quy trình EDA (Read-Only)
    with timed(logger, f"Khám phá dữ liệu thô và xuất 5 biểu đồ EDA vào {to_rel_path(IMAGES_DIR)}") as b:
        eda_summary = run_eda(raw_path=RAW_CSV, output_dir=IMAGES_DIR)

    logger.info(f"  • Quy mô dữ liệu: {eda_summary['n_samples']} khách hàng ({eda_summary['n_columns']} trường thuộc tính).")
    logger.info(f"  • Cơ cấu giới tính: {eda_summary['gender_counts']} (Tỷ lệ: {eda_summary['gender_ratio_pct']}).")
    logger.info(f"  ✔ Đã xuất đủ {len(eda_summary['saved_figures'])} biểu đồ EDA vào: {to_rel_path(IMAGES_DIR)}")

    # 4. Lưu hiện vật kết quả chặng vào outputs/stage_01.json
    stage_data = {
        "stage_name": "Stage 01: Exploratory Data Analysis (EDA)",
        "status": "SUCCESS",
        "elapsed_seconds": round(b.elapsed, 3),
        "data_path": to_rel_path(RAW_CSV),
        "n_samples": eda_summary["n_samples"],
        "n_columns": eda_summary["n_columns"],
        "columns": eda_summary["columns"],
        "gender_counts": eda_summary["gender_counts"],
        "gender_ratio_pct": eda_summary["gender_ratio_pct"],
        "numeric_stats": eda_summary["numeric_stats"],
        "age_distribution": eda_summary["age_distribution"],
        "figures": [to_rel_path(p) for p in eda_summary["saved_figures"]]
    }
    stage_file = save_stage_result(1, stage_data)
    logger.info(f"  ✔ Đã lưu phong bì kết quả: {to_rel_path(stage_file)}")

    # 5. Kích hoạt biên dịch Báo cáo HTML Chặng 1 và cập nhật index.html
    report_path = REPORTS_DIR / "01_data_report.html"
    with timed(logger, "Tự động sinh báo cáo HTML Chặng 1 và cập nhật index.html"):
        try:
            from scripts.report_builder import build_eda_report, build_index
            report_path = build_eda_report()
            build_index()
            logger.info(f"  ✔ Đã tự động tạo: {to_rel_path(report_path)}")
            logger.info(f"  ✔ Đã cập nhật bảng trạng thái: {to_rel_path(REPORTS_DIR / 'index.html')}")
        except Exception as e:
            logger.warning(f"  Chưa tạo được báo cáo HTML: {e}")

    elapsed = time.time() - t_start
    logger.info("=" * 80)
    logger.info(f"🎉 HOÀN THÀNH CHẶNG 01 THÀNH CÔNG TRONG {elapsed:.2f} GIÂY!")
    logger.info(f"👉 Mở xem báo cáo trực quan tại: {to_rel_path(report_path)}")
    logger.info("=" * 80)

    return stage_data


if __name__ == "__main__":
    main()
