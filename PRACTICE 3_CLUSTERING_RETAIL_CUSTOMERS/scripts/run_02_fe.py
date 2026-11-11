"""
Kịch bản Điều phối: run_02_fe.py
Dự án: Practice 3: Clustering Retail Customers — Group 3
Chặng 2: Tiền xử lý & Kỹ thuật Đặc trưng (Feature Engineering & Scaling)
Đầu ra: data/processed/clean_customers.csv, outputs/stage_02.json, logs/run_02_fe.log, docs/report/02_feature_pipeline_report.html
Đồng bộ định dạng đường dẫn tương đối và đa luồng log chuẩn Practice 1 & 2.
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

# Đảm bảo đường dẫn import gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    RAW_CSV,
    PROCESSED_CSV,
    LOGS_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
    to_rel_path,
    require_files,
    save_stage_result
)
from scripts.logger_utils import setup_logger, timed, log_environment
from scripts.feature_engineering import export_processed_data, prepare_clustering_features


def main(extra_handler=None, logger=None) -> Dict[str, Any]:
    t_start = time.time()

    # 1. Khởi tạo Logger riêng cho Chặng 02 (hỗ trợ extra_handler từ pipeline tổng)
    if logger is None:
        logger, log_file = setup_logger(log_dir=LOGS_DIR, prefix="run_02_fe", extra_handler=extra_handler)
    else:
        log_file = getattr(logger, "log_file", "")

    logger.info("=" * 80)
    logger.info("🚀 BẮT ĐẦU CHẶNG 02: TIỀN XỬ LÝ & KỸ THUẬT ĐẶC TRƯNG (FE & SCALING)")
    logger.info("=" * 80)
    log_environment(logger, seed=RANDOM_SEED)

    # 2. Kiểm tra tệp tiền đề bắt buộc
    require_files(RAW_CSV)
    logger.info(f"Đã xác minh nguồn dữ liệu thô: {to_rel_path(RAW_CSV)}")

    # 3. Xuất tệp dữ liệu sạch sang data/processed/
    with timed(logger, f"Làm sạch, mã hóa Gender và xuất dữ liệu sạch sang {to_rel_path(PROCESSED_CSV)}") as b1:
        out_path = export_processed_data(PROCESSED_CSV)
    logger.info(f"  ✔ Đã xuất tệp dữ liệu sạch thành công: {to_rel_path(out_path)}")

    # 4. Chuẩn bị ma trận đặc trưng chuẩn hóa cho các không gian 2D, 3D, 4D
    with timed(logger, "Chuẩn hóa Thang đo (StandardScaler) cho các không gian gom cụm (2D, 3D, 4D)") as b2:
        X_2d, scaler_2d, names_2d = prepare_clustering_features(feature_set="2d")
        X_3d, scaler_3d, names_3d = prepare_clustering_features(feature_set="3d")
        X_4d, scaler_4d, names_4d = prepare_clustering_features(feature_set="4d")

    df_proc = pd.read_csv(out_path)
    logger.info(f"  • Dữ liệu sạch hoàn chỉnh: {df_proc.shape[0]} dòng, {df_proc.shape[1]} cột.")
    logger.info(f"  • Không gian 2D (Chính): {names_2d} — Kích thước: {list(X_2d.shape)}")
    logger.info(f"  • Không gian 3D (Mở rộng): {names_3d} — Kích thước: {list(X_3d.shape)}")
    logger.info(f"  • Không gian 4D (Toàn diện): {names_4d} — Kích thước: {list(X_4d.shape)}")

    # 5. Lưu hiện vật kết quả chặng vào outputs/stage_02.json
    stage_data = {
        "stage_name": "Stage 02: Preprocessing & Feature Engineering",
        "status": "SUCCESS",
        "elapsed_seconds": round(b1.elapsed + b2.elapsed, 3),
        "raw_data_path": to_rel_path(RAW_CSV),
        "processed_data_path": to_rel_path(out_path),
        "n_samples": len(df_proc),
        "n_features_processed": len(df_proc.columns),
        "processed_columns": list(df_proc.columns),
        "feature_sets": {
            "2d": {"features": names_2d, "shape": list(X_2d.shape)},
            "3d": {"features": names_3d, "shape": list(X_3d.shape)},
            "4d": {"features": names_4d, "shape": list(X_4d.shape)}
        },
        "scaling_summary": {
            "method": "StandardScaler",
            "features_2d_mean": np.mean(X_2d, axis=0).round(4).tolist(),
            "features_2d_std": np.std(X_2d, axis=0).round(4).tolist()
        }
    }
    stage_file = save_stage_result(2, stage_data)
    logger.info(f"  ✔ Đã lưu phong bì kết quả: {to_rel_path(stage_file)}")

    # 6. Kích hoạt biên dịch Báo cáo HTML Chặng 2 và cập nhật index.html
    report_path = REPORTS_DIR / "02_feature_pipeline_report.html"
    with timed(logger, "Tự động sinh báo cáo HTML Chặng 2 và cập nhật index.html"):
        try:
            from scripts.report_builder import build_fe_report, build_index
            report_path = build_fe_report()
            build_index()
            logger.info(f"  ✔ Đã tự động tạo: {to_rel_path(report_path)}")
            logger.info(f"  ✔ Đã cập nhật bảng trạng thái: {to_rel_path(REPORTS_DIR / 'index.html')}")
        except Exception as e:
            logger.warning(f"  Chưa tạo được báo cáo HTML: {e}")

    elapsed = time.time() - t_start
    logger.info("=" * 80)
    logger.info(f"🎉 HOÀN THÀNH CHẶNG 02 THÀNH CÔNG TRONG {elapsed:.2f} GIÂY!")
    logger.info(f"👉 Mở xem báo cáo trực quan tại: {to_rel_path(report_path)}")
    logger.info("=" * 80)

    return stage_data


if __name__ == "__main__":
    main()
