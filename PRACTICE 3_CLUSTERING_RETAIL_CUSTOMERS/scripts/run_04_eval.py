"""
Kịch bản Điều phối: run_04_eval.py
Dự án: Practice 3: Clustering Retail Customers — Group 3
Chặng 4: Đánh giá Toàn diện Chất lượng Gom cụm, Lập Hồ sơ Phân khúc Khách hàng & Xuất Báo cáo HTML
Đầu ra: docs/images/model_01 -> model_04.png, outputs/stage_04.json, logs/run_04_eval.log, docs/report/04_model_evaluation_report.html, index.html, run_log.html
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
    SAVED_MODELS_DIR,
    IMAGES_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
    to_rel_path,
    require_files,
    save_stage_result
)
from scripts.logger_utils import setup_logger, timed, log_environment
from scripts.model_evaluation import evaluate_clustering


def main(extra_handler=None, logger=None) -> Dict[str, Any]:
    t_start = time.time()

    # 1. Khởi tạo Logger riêng cho Chặng 04 (hỗ trợ extra_handler từ pipeline tổng)
    if logger is None:
        logger, log_file = setup_logger(log_dir=LOGS_DIR, prefix="run_04_eval", extra_handler=extra_handler)
    else:
        log_file = getattr(logger, "log_file", "")

    logger.info("=" * 80)
    logger.info("🚀 BẮT ĐẦU CHẶNG 04: ĐÁNH GIÁ GOM CỤM & LẬP HỒ SƠ 5 PHÂN KHÚC KHÁCH HÀNG")
    logger.info("=" * 80)
    log_environment(logger, seed=RANDOM_SEED)

    # 2. Kiểm tra các tệp tiền đề bắt buộc từ Chặng 2 và Chặng 3
    require_files(
        PROCESSED_CSV,
        SAVED_MODELS_DIR / "kmeans_model.joblib",
        SAVED_MODELS_DIR / "hierarchical_model.joblib",
        SAVED_MODELS_DIR / "dbscan_model.joblib",
        SAVED_MODELS_DIR / "metadata.json"
    )
    logger.info(f"Đã xác minh đầy đủ dữ liệu sạch và các mô hình tại: {to_rel_path(SAVED_MODELS_DIR)}")

    # 3. Thực thi đánh giá phân cụm, hồ sơ kinh doanh và vẽ 4 đồ thị chuyên sâu
    with timed(logger, f"Đánh giá chỉ số gom cụm, lập hồ sơ phân khúc & xuất 4 đồ thị vào {to_rel_path(IMAGES_DIR)}") as b:
        eval_res = evaluate_clustering(save_figures=True, output_dir=IMAGES_DIR)

    km_m = eval_res["metrics_kmeans"]
    logger.info(f"  • K-Means Scratch      : K=5 | Silhouette={km_m['silhouette_score']} | DB={km_m['davies_bouldin_index']} | CH={km_m['calinski_harabasz_score']}")
    
    hc_m = eval_res["metrics_hierarchical"]
    logger.info(f"  • Hierarchical Scratch : K=5 | Silhouette={hc_m['silhouette_score']} | DB={hc_m['davies_bouldin_index']} | CH={hc_m['calinski_harabasz_score']}")
    
    db_m = eval_res["metrics_dbscan"]
    logger.info(f"  • DBSCAN Scratch       : Silhouette={db_m['silhouette_score']} | DB={db_m['davies_bouldin_index']} | Nhiễu ngoại lai={db_m['n_noise']} mẫu")

    logger.info(f"  ✔ Đã lập hồ sơ thành công cho {len(eval_res['segment_profile'])} phân khúc khách hàng mục tiêu.")
    logger.info(f"  ✔ Đã xuất sinh {len(eval_res['saved_figures'])} biểu đồ đánh giá vào: {to_rel_path(IMAGES_DIR)}")

    # 4. Lưu hiện vật kết quả chặng vào outputs/stage_04.json
    stage_data = {
        "stage_name": "Stage 04: Model Evaluation & Customer Segmentation Profiling",
        "status": "SUCCESS",
        "elapsed_seconds": round(b.elapsed, 3),
        "metrics_kmeans": km_m,
        "metrics_dbscan": db_m,
        "metrics_hierarchical": hc_m,
        "segment_profile": eval_res["segment_profile"],
        "saved_figures": [to_rel_path(p) for p in eval_res["saved_figures"]]
    }
    stage_file = save_stage_result(4, stage_data)
    logger.info(f"  ✔ Đã lưu phong bì kết quả: {to_rel_path(stage_file)}")

    # 5. Kích hoạt biên dịch Báo cáo HTML Chặng 4 và Cập nhật Dashboard Tổng quan
    report_path = REPORTS_DIR / "04_model_evaluation_report.html"
    with timed(logger, "Tự động sinh báo cáo HTML Chặng 4 và cập nhật Dashboard/Run-log"):
        try:
            from scripts.report_builder import build_evaluation_report, build_index, build_run_log_page
            report_path = build_evaluation_report()
            build_index()
            build_run_log_page()
            logger.info(f"  ✔ Đã tự động tạo: {to_rel_path(report_path)}")
            logger.info(f"  ✔ Đã cập nhật Dashboard: {to_rel_path(REPORTS_DIR / 'index.html')}")
            logger.info(f"  ✔ Đã cập nhật Nhật ký: {to_rel_path(REPORTS_DIR / 'run_log.html')}")
        except Exception as e:
            logger.warning(f"  Chưa tạo được báo cáo HTML: {e}")

    elapsed = time.time() - t_start
    logger.info("=" * 80)
    logger.info(f"🎉 HOÀN THÀNH CHẶNG 04 THÀNH CÔNG TRONG {elapsed:.2f} GIÂY!")
    logger.info(f"👉 Mở xem báo cáo trực quan tại: {to_rel_path(report_path)}")
    logger.info("=" * 80)

    return stage_data


if __name__ == "__main__":
    main()
