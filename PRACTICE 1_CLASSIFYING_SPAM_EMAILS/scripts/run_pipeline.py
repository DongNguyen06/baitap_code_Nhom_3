import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

# Thiết lập đường dẫn gốc cho dự án
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.logger_utils import setup_logger, timed, log_environment
from scripts.common import LOGS_DIR, REPORT_DIR
import scripts.run_01_eda as stage_01
import scripts.run_02_fe as stage_02
import scripts.run_03_train as stage_03
import scripts.run_04_eval as stage_04
import scripts.report_builder as report_builder


def parse_args():
    parser = argparse.ArgumentParser(
        description="🚀 BỘ ĐIỀU PHỐI QUY TRÌNH HỌC MÁY TOÀN DIỆN (END-TO-END PIPELINE ORCHESTRATOR) — NHÓM 3"
    )
    parser.add_argument(
        "--quick", "-q", action="store_true",
        help="Chạy ở chế độ rút gọn (Quick Mode) để demo hoặc kiểm tra luồng nhanh chóng."
    )
    return parser.parse_args()


def run_full_pipeline(quick_mode: bool = False):
    start_total = time.time()

    # 1. Khởi tạo Logger chính cho toàn bộ quy trình và lấy file_handler để chia sẻ
    logger, log_path = setup_logger(log_dir=LOGS_DIR, prefix="run_pipeline")
    pipeline_handler = getattr(logger, "file_handler", None)

    logger.info("=" * 85)
    logger.info("   🚀 KHỞI ĐỘNG QUY TRÌNH HỌC MÁY TOÀN DIỆN (END-TO-END PIPELINE) — NHÓM 3 (UTH)")
    logger.info(f"   Thời điểm bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Chế độ chạy: {'⚡ Rút gọn (Quick Mode)' if quick_mode else '🏆 Huấn luyện toàn diện (Full Benchmark)'}")
    logger.info("=" * 85)

    log_environment(logger)
    timings = {}

    # =========================================================================
    # CHẶNG 1: KHÁM PHÁ & LÀM SẠCH DỮ LIỆU (EDA & DATA CLEANING)
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info(">>> [CHẶNG 1/4] KHÁM PHÁ & LÀM SẠCH DỮ LIỆU THÔ (EDA & DATA CLEANING)...")
    logger.info("=" * 85)
    with timed(logger, "Thực thi Chặng 01 (EDA & Data Cleaning)"):
        t0 = time.time()
        stage_01.main(extra_handler=pipeline_handler)
        timings["stage_01_sec"] = round(time.time() - t0, 2)

    # =========================================================================
    # CHẶNG 2: KỸ THUẬT ĐẶC TRƯNG & PHÂN TÁCH DỮ LIỆU (FEATURE ENGINEERING)
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info(">>> [CHẶNG 2/4] KỸ THUẬT ĐẶC TRƯNG & THIẾT LẬP PIPELINE TIỀN XỬ LÝ...")
    logger.info("=" * 85)
    with timed(logger, "Thực thi Chặng 02 (Feature Engineering & Train/Test Split)"):
        t0 = time.time()
        stage_02.main(extra_handler=pipeline_handler)
        timings["stage_02_sec"] = round(time.time() - t0, 2)

    # =========================================================================
    # CHẶNG 3: HUẤN LUYỆN 3 MÔ HÌNH SCRATCH & KIỂM ĐỊNH CHÉO (TRAINING & CV)
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info(">>> [CHẶNG 3/4] HUẤN LUYỆN 3 MÔ HÌNH SCRATCH & KIỂM ĐỊNH CHÉO STRATIFIED CV...")
    logger.info("=" * 85)
    with timed(logger, "Thực thi Chặng 03 (Scratch Models Training & GridSearchCV)"):
        t0 = time.time()
        stage_03.run_stage_03(quick_mode=quick_mode, extra_handler=pipeline_handler)
        timings["stage_03_sec"] = round(time.time() - t0, 2)

    # =========================================================================
    # CHẶNG 4: ĐÁNH GIÁ ĐỘC LẬP TẬP TEST & XUẤT HIỆN VẬT TỐI ƯU (EVALUATION)
    # =========================================================================
    logger.info("\n" + "=" * 85)
    logger.info(">>> [CHẶNG 4/4] ĐÁNH GIÁ ĐỘC LẬP TRÊN TEST SET & XUẤT BÁO CÁO NGHIỆM THU...")
    logger.info("=" * 85)
    with timed(logger, "Thực thi Chặng 04 (Hold-out Test Evaluation & Artifacts Export)"):
        t0 = time.time()
        stage_04.run_stage_04(quick_mode=quick_mode, extra_handler=pipeline_handler)
        timings["stage_04_sec"] = round(time.time() - t0, 2)

    total_time = round(time.time() - start_total, 2)
    logger.info("\n" + "=" * 85)
    logger.info(f"🎉 TOÀN BỘ QUY TRÌNH HỌC MÁY ĐÃ HOÀN TẤT THÀNH CÔNG TRONG {total_time:.2f} GIÂY!")
    logger.info("=" * 85)
    logger.info(f"  • Thời lượng Chặng 01 (EDA & Làm sạch)      : {timings.get('stage_01_sec', 0)}s")
    logger.info(f"  • Thời lượng Chặng 02 (Đặc trưng & Pipeline): {timings.get('stage_02_sec', 0)}s")
    logger.info(f"  • Thời lượng Chặng 03 (Huấn luyện & CV)     : {timings.get('stage_03_sec', 0)}s")
    logger.info(f"  • Thời lượng Chặng 04 (Đánh giá & Xuất model): {timings.get('stage_04_sec', 0)}s")
    logger.info("-" * 85)
    logger.info("👉 MỜI XEM BÁO CÁO NGHIỆM THU TỔNG HỢP TRỰC QUAN TẠI:")
    logger.info(f"   file:///{REPORT_DIR.resolve() / 'index.html'}")
    logger.info("👉 TRẢI NGHIỆM THỬ NGHIỆM EMAIL THỜI GIAN THỰC TẠI:")
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
    with timed(logger, "Tái tạo đồng bộ toàn bộ báo cáo HTML"):
        report_builder.build_all()


if __name__ == "__main__":
    args = parse_args()
    run_full_pipeline(quick_mode=args.quick)
