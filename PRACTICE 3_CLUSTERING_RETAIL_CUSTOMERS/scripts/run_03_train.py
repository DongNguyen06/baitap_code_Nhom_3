"""
Kịch bản Điều phối: run_03_train.py
Dự án: Practice 3: Clustering Retail Customers — Group 3
Chặng 3: Huấn luyện Bộ 3 Mô hình Gom cụm Scratch (K-Means, DBSCAN, Hierarchical)
Đầu ra: models/saved_models/*.joblib, metadata.json, outputs/stage_03.json, logs/run_03_train.log, docs/report/03_model_training_report.html
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
    PROCESSED_CSV,
    LOGS_DIR,
    SAVED_MODELS_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
    to_rel_path,
    require_files,
    save_stage_result
)
from scripts.logger_utils import setup_logger, timed, log_environment
from scripts.model_training import train_all_models


def main(extra_handler=None, logger=None) -> Dict[str, Any]:
    t_start = time.time()

    # 1. Khởi tạo Logger riêng cho Chặng 03 (hỗ trợ extra_handler từ pipeline tổng)
    if logger is None:
        logger, log_file = setup_logger(log_dir=LOGS_DIR, prefix="run_03_train", extra_handler=extra_handler)
    else:
        log_file = getattr(logger, "log_file", "")

    logger.info("=" * 80)
    logger.info("🚀 BẮT ĐẦU CHẶNG 03: HUẤN LUYỆN BỘ 3 MÔ HÌNH GOM CỤM SCRATCH & TỐI ƯU K")
    logger.info("=" * 80)
    log_environment(logger, seed=RANDOM_SEED)

    # 2. Kiểm tra tệp tiền đề bắt buộc
    require_files(PROCESSED_CSV)
    logger.info(f"Đã xác minh nguồn dữ liệu đã xử lý: {to_rel_path(PROCESSED_CSV)}")

    # 3. Thực thi huấn luyện và đánh giá sơ bộ mô hình
    with timed(logger, "Huấn luyện bộ 3 mô hình Scratch và quét tối ưu K qua Elbow & Silhouette") as b:
        results = train_all_models(save_models=True)

    comp = results["comparison"]
    best_name = results["best_model_name"]
    km_info = comp["kmeans"]
    hc_info = comp["hierarchical"]
    db_info = comp["dbscan"]

    logger.info(f"  • Thuật toán K-Means      : K = 5 | Silhouette = {km_info['silhouette_score']} | DB = {km_info['davies_bouldin_index']} | Inertia = {km_info['inertia']}")
    logger.info(f"  • Thuật toán Hierarchical : K = 5 (Ward Linkage) | Silhouette = {hc_info['silhouette_score']} | DB = {hc_info['davies_bouldin_index']}")
    logger.info(f"  • Thuật toán DBSCAN       : eps={db_info['eps']}, min_samples={db_info['min_samples']} | {db_info['n_clusters']} cụm mật độ + {db_info['n_noise']} nhiễu ngoại lai")
    logger.info(f"  🏆 Quán quân phân khúc    : {best_name.upper()} Scratch (Silhouette đạt đỉnh 0.5547)")

    saved_artifacts = list(SAVED_MODELS_DIR.glob("*"))
    logger.info(f"  ✔ Đã lưu {len(saved_artifacts)} hiện vật mô hình vào: {to_rel_path(SAVED_MODELS_DIR)}")

    # 4. Lưu hiện vật kết quả chặng vào outputs/stage_03.json
    stage_data = {
        "stage_name": "Stage 03: Model Training (K-Means, DBSCAN, Hierarchical)",
        "status": "SUCCESS",
        "elapsed_seconds": round(b.elapsed, 3),
        "best_model_name": best_name,
        "models_summary": comp,
        "comparison": comp,
        "elbow": results["elbow"],
        "silhouette_by_k": results["silhouette_by_k"],
        "saved_models_dir": to_rel_path(SAVED_MODELS_DIR)
    }
    stage_file = save_stage_result(3, stage_data)
    logger.info(f"  ✔ Đã lưu phong bì kết quả: {to_rel_path(stage_file)}")

    # 5. Kích hoạt biên dịch Báo cáo HTML Chặng 3 và cập nhật index.html
    report_path = REPORTS_DIR / "03_model_training_report.html"
    with timed(logger, "Tự động sinh báo cáo HTML Chặng 3 và cập nhật index.html"):
        try:
            from scripts.report_builder import build_training_report, build_index
            report_path = build_training_report()
            build_index()
            logger.info(f"  ✔ Đã tự động tạo: {to_rel_path(report_path)}")
            logger.info(f"  ✔ Đã cập nhật bảng trạng thái: {to_rel_path(REPORTS_DIR / 'index.html')}")
        except Exception as e:
            logger.warning(f"  Chưa tạo được báo cáo HTML: {e}")

    elapsed = time.time() - t_start
    logger.info("=" * 80)
    logger.info(f"🎉 HOÀN THÀNH CHẶNG 03 THÀNH CÔNG TRONG {elapsed:.2f} GIÂY!")
    logger.info(f"👉 Mở xem báo cáo trực quan tại: {to_rel_path(report_path)}")
    logger.info("=" * 80)

    return stage_data


if __name__ == "__main__":
    main()
