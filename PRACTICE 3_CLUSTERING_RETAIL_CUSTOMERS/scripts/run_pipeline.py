"""
Kịch bản Điều phối Tổng: run_pipeline.py
Dự án: Practice 3: Clustering Retail Customers — Group 3
Mục tiêu: Tự động hóa 1-chạm toàn bộ quy trình Học máy không giám sát qua 4 chặng:
  - Chặng 1: Khám phá Dữ liệu (EDA Read-Only)
  - Chặng 2: Tiền xử lý & Kỹ thuật Đặc trưng (Data Cleaning & Feature Engineering)
  - Chặng 3: Huấn luyện Bộ 3 Mô hình Gom cụm Scratch (K-Means, DBSCAN, Hierarchical)
  - Chặng 4: Đánh giá Chất lượng, Lập Hồ sơ 5 Phân khúc Khách hàng & Xuất 6 Báo cáo HTML
Hỗ trợ chia sẻ file handler xuất đồng thời 5 file log (1 master log + 4 stage logs) chuẩn Practice 1 & 2.
"""

import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

# Thiết lập đường dẫn gốc cho dự án
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import LOGS_DIR, REPORTS_DIR, RANDOM_SEED, to_rel_path, load_stage_result
from scripts.logger_utils import setup_logger, timed, log_environment
import scripts.run_01_eda as stage_01
import scripts.run_02_fe as stage_02
import scripts.run_03_train as stage_03
import scripts.run_04_eval as stage_04
import scripts.report_builder as report_builder


def parse_args():
    parser = argparse.ArgumentParser(
        description="🚀 BỘ ĐIỀU PHỐI QUY TRÌNH HỌC MÁY TOÀN DIỆN (END-TO-END PIPELINE) — NHÓM 3 (PRACTICE 3)"
    )
    return parser.parse_args()


def run_full_pipeline() -> None:
    start_total = time.time()

    # 1. Khởi tạo Logger chính cho toàn bộ quy trình và lấy file_handler để chia sẻ (Multi-logging)
    logger, log_path = setup_logger(log_dir=LOGS_DIR, prefix="run_pipeline")
    pipeline_handler = getattr(logger, "file_handler", None)

    logger.info("=" * 85)
    logger.info("   🚀 KHỞI ĐỘNG QUY TRÌNH HỌC MÁY TOÀN DIỆN (END-TO-END PIPELINE) — NHÓM 3 (UTH)")
    logger.info("      ĐỀ TÀI: PHÂN CỤM KHÁCH HÀNG BÁN LẺ (PRACTICE 3 — CLUSTERING RETAIL CUSTOMERS)")
    logger.info(f"   Thời điểm bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Tập dữ liệu: data/raw/Mall_Customers.csv (200 khách hàng)")
    logger.info("=" * 85)

    log_environment(logger, seed=RANDOM_SEED)
    timings = {}

    # =========================================================================
    # CHẶNG 1: KHÁM PHÁ DỮ LIỆU KHÁCH HÀNG BÁN LẺ (EDA)
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info(">>> [CHẶNG 1/4] KHÁM PHÁ DỮ LIỆU KHÁCH HÀNG BÁN LẺ (EXPLORATORY DATA ANALYSIS - EDA)...")
    logger.info("=" * 85)
    with timed(logger, "Thực thi Chặng 01 (EDA Read-Only)"):
        t0 = time.time()
        stage_01.main(extra_handler=pipeline_handler)
        timings["stage_01_sec"] = round(time.time() - t0, 2)

    # =========================================================================
    # CHẶNG 2: TIỀN XỬ LÝ & KỸ THUẬT ĐẶC TRƯNG (FEATURE ENGINEERING)
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info(">>> [CHẶNG 2/4] LÀM SẠCH, MÃ HÓA GENDER & THIẾT LẬP CÁC KHÔNG GIAN ĐẶC TRƯNG...")
    logger.info("=" * 85)
    with timed(logger, "Thực thi Chặng 02 (Data Preprocessing & Feature Engineering)"):
        t0 = time.time()
        stage_02.main(extra_handler=pipeline_handler)
        timings["stage_02_sec"] = round(time.time() - t0, 2)

    # =========================================================================
    # CHẶNG 3: HUẤN LUYỆN BỘ 3 MÔ HÌNH GOM CỤM SCRATCH & TỐI ƯU K
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info(">>> [CHẶNG 3/4] HUẤN LUYỆN BỘ 3 MÔ HÌNH GOM CỤM SCRATCH & TỐI ƯU K (ELBOW/SILHOUETTE)...")
    logger.info("=" * 85)
    with timed(logger, "Thực thi Chặng 03 (Scratch Models Training & Hyperparameter Tuning)"):
        t0 = time.time()
        stage_03.main(extra_handler=pipeline_handler)
        timings["stage_03_sec"] = round(time.time() - t0, 2)

    # =========================================================================
    # CHẶNG 4: ĐÁNH GIÁ CHẤT LƯỢNG GOM CỤM & LẬP HỒ SƠ 5 PHÂN KHÚC KHÁCH HÀNG
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info(">>> [CHẶNG 4/4] ĐÁNH GIÁ GOM CỤM, LẬP HỒ SƠ PHÂN KHÚC & XUẤT ĐỒ THỊ CHUYÊN SÂU...")
    logger.info("=" * 85)
    with timed(logger, "Thực thi Chặng 04 (Clustering Evaluation & Persona Profiling)"):
        t0 = time.time()
        stage_04.main(extra_handler=pipeline_handler)
        timings["stage_04_sec"] = round(time.time() - t0, 2)

    total_time = round(time.time() - start_total, 2)
    logger.info("\n" + "=" * 85)
    logger.info(f"🎉 TOÀN BỘ QUY TRÌNH HỌC MÁY ĐÃ HOÀN TẤT THÀNH CÔNG TRONG {total_time:.2f} GIÂY ({total_time / 60:.2f} PHÚT)!")
    logger.info("=" * 85)
    logger.info(f"  • Thời lượng Chặng 01 (Khám phá dữ liệu EDA)    : {timings.get('stage_01_sec', 0)}s")
    logger.info(f"  • Thời lượng Chặng 02 (Tiền xử lý & Đặc trưng)  : {timings.get('stage_02_sec', 0)}s")
    logger.info(f"  • Thời lượng Chặng 03 (Huấn luyện 3 mô hình)    : {timings.get('stage_03_sec', 0)}s")
    logger.info(f"  • Thời lượng Chặng 04 (Đánh giá & Phân khúc)    : {timings.get('stage_04_sec', 0)}s")
    logger.info("-" * 85)
    logger.info("👉 MỜI XEM BÁO CÁO NGHIỆM THU TỔNG HỢP TRỰC QUAN TẠI:")
    logger.info("   docs/report/index.html")
    logger.info("👉 MỜI XEM NHẬT KÝ THỰC THI TOÀN TRÌNH ZERO-DRIFT TẠI:")
    logger.info("   docs/report/run_log.html")
    logger.info("👉 TRẢI NGHIỆM SUY LUẬN TƯƠNG TÁC THỜI GIAN THỰC TẠI:")
    logger.info("   notebooks/demo.ipynb")
    logger.info("=" * 85)

    # Đảm bảo flush log ra file trước khi dựng trang run_log.html
    for h in logger.handlers:
        try:
            h.flush()
        except Exception:
            pass

    # =========================================================================
    # ĐỒNG BỘ TOÀN BỘ BÁO CÁO HTML
    # =========================================================================
    logger.info("\n>>> ĐỒNG BỘ TOÀN BỘ BÁO CÁO HTML VÀ BẢNG ĐIỀU KHIỂN TỔNG HỢP...")
    with timed(logger, "Tái tạo đồng bộ toàn bộ 6 báo cáo HTML Base64 Standalone"):
        report_builder.build_all()


if __name__ == "__main__":
    parse_args()
    run_full_pipeline()
