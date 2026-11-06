import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    RAW_CSV,
    LOGS_DIR,
    IMAGES_DIR,
    RANDOM_SEED,
    require_files,
    save_figure,
    save_stage_result
)
from scripts.logger_utils import setup_logger, timed, log_environment
from data.scripts.eda import (
    inspect_data_structure,
    get_univariate_stats,
    detect_iqr_outliers,
    get_bivariate_analysis,
    plot_amount_distribution,
    plot_bivariate_category_fulfilment,
    plot_outliers_boxplot,
    plot_temporal_sales_trend,
    plot_geographic_sales_distribution,
    plot_correlation_heatmap
)
from scripts.report_builder import build_report_01, build_index


def main(raw_csv_path: str = None, extra_handler=None, logger=None) -> dict:
    """
    Kịch bản thực thi Chặng 01: Khám phá Dữ liệu Bán hàng TMĐT (Exploratory Data Analysis - EDA):
    1. Đọc dữ liệu thô Amazon Sale Report.csv (128.975 dòng) - Duy trì nguyên tắc Read-Only 100%.
    2. Khảo sát cấu trúc bảng, kiểu dữ liệu và tỷ lệ khuyết thiếu kỹ thuật.
    3. Thống kê đơn biến doanh số, kiểm định phân phối lệch phải và định hướng biến đổi Log1p.
    4. Quét ngoại lai doanh số bằng phương pháp khoảng tứ phân vị IQR.
    5. Phân tích quan hệ hai biến với ngành hàng, kênh hoàn tất và tính toán chỉ số eta bình phương.
    6. Xuất 6 biểu đồ trực quan hóa chuyên sâu vào docs/images/.
    7. Lưu phong bì kết quả vào outputs/stage_01.json.
    8. Tự động sinh docs/report/01_data_report.html và cập nhật index.html.
    """
    start_time_iso = datetime.now().isoformat()
    t_start = time.time()
    timings = {}

    # 1. Khởi tạo Logger riêng cho Chặng 01 (hỗ trợ extra_handler từ pipeline tổng)
    if logger is None:
        logger, log_file = setup_logger(log_dir=LOGS_DIR, prefix="run_01_eda", extra_handler=extra_handler)
    else:
        log_file = getattr(logger, "log_file", "")

    logger.info("=" * 80)
    logger.info("🚀 BẮT ĐẦU CHẶNG 01: KHÁM PHÁ DỮ LIỆU BÁN HÀNG TMĐT (EXPLORATORY DATA ANALYSIS - EDA)")
    logger.info("=" * 80)
    log_environment(logger, seed=RANDOM_SEED)

    # 2. Kiểm tra điều kiện tiên quyết
    raw_path = Path(raw_csv_path) if raw_csv_path else RAW_CSV
    require_files(
        logger,
        [raw_path],
        hint=f"Vui lòng đặt tệp Amazon Sale Report.csv vào '{raw_path.parent}'."
    )

    # 3. Nạp dữ liệu thô
    with timed(logger, f"Nạp dữ liệu thô từ {raw_path.name}") as b:
        df_raw = pd.read_csv(raw_path, low_memory=False)
    timings["load_raw_data"] = b.elapsed
    logger.info(f"  • Quy mô dữ liệu thô: {len(df_raw):,} dòng ({df_raw.shape[1]} thuộc tính).")

    # 4. Khảo sát cấu trúc và giá trị khuyết thiếu
    with timed(logger, "Khảo sát cấu trúc bảng và khuyết thiếu kỹ thuật") as b:
        struct_stats = inspect_data_structure(df_raw)
    timings["inspect_structure"] = b.elapsed
    logger.info(f"  • Thuộc tính: {len(struct_stats['columns'])} cột | Số dòng trùng lặp thô: {struct_stats['n_duplicates']:,}.")

    # 5. Thống kê đơn biến doanh số Amount
    with timed(logger, "Thống kê đơn biến doanh số Amount") as b:
        amount_stats = get_univariate_stats(df_raw, col="Amount")
    timings["univariate_stats"] = b.elapsed
    logger.info(f"  • Doanh số Amount (INR): Mean={amount_stats['mean']:.2f}, Median={amount_stats['median']:.2f}, Std={amount_stats['std']:.2f}, Skewness={amount_stats['skewness']:.2f}.")

    # 6. Phân tích ngoại lai IQR
    with timed(logger, "Phân tích ngoại lai doanh số bằng khoảng tứ phân vị IQR") as b:
        amount_valid = pd.to_numeric(df_raw["Amount"], errors="coerce").dropna()
        amount_valid = amount_valid[amount_valid > 0]
        iqr_outliers = detect_iqr_outliers(amount_valid)
    timings["iqr_outliers"] = b.elapsed
    logger.info(f"  • Ngoại lai IQR: Ngưỡng trên={iqr_outliers['upper_bound']:.2f} INR | Số lượng={iqr_outliers['n_outliers']:,} ({iqr_outliers['outlier_pct']:.2f}%).")

    # 7. Phân tích hai biến doanh số với các yếu tố kinh doanh
    with timed(logger, "Phân tích hai biến doanh số với ngành hàng, kênh vận chuyển & địa lý") as b:
        bivariate_res = get_bivariate_analysis(df_raw.dropna(subset=["Amount"]), target="Amount")
    timings["bivariate_analysis"] = b.elapsed
    logger.info(f"  • Xếp hạng ảnh hưởng (Eta-squared): {bivariate_res.get('ranking', [])[:4]}.")

    # 8. Xuất 6 biểu đồ trực quan hóa chuyên sâu vào docs/images/
    with timed(logger, "Vẽ và xuất 6 biểu đồ phân tích EDA vào docs/images/") as b:
        fig1_path = IMAGES_DIR / "eda_01_amount_distribution.png"
        fig2_path = IMAGES_DIR / "eda_02_bivariate_category_fulfilment.png"
        fig3_path = IMAGES_DIR / "eda_03_outliers_boxplot.png"
        fig4_path = IMAGES_DIR / "eda_04_temporal_sales_trend.png"
        fig5_path = IMAGES_DIR / "eda_05_geographic_sales_distribution.png"
        fig6_path = IMAGES_DIR / "eda_06_correlation_heatmap.png"

        plot_amount_distribution(df_raw, save_path=fig1_path)
        plot_bivariate_category_fulfilment(df_raw, save_path=fig2_path)
        plot_outliers_boxplot(df_raw, save_path=fig3_path)
        plot_temporal_sales_trend(df_raw, save_path=fig4_path)
        plot_geographic_sales_distribution(df_raw, save_path=fig5_path)
        plot_correlation_heatmap(df_raw, save_path=fig6_path)
    timings["export_figures"] = b.elapsed
    logger.info("  ✔ Đã xuất đủ 6 biểu đồ EDA: eda_01 đến eda_06.")

    total_duration = round(time.time() - t_start, 2)

    # 9. Đóng gói phong bì kết quả vào outputs/stage_01.json (Read-Only)
    stage_data = {
        "stage": "01",
        "status": "success",
        "started_at": start_time_iso,
        "finished_at": datetime.now().isoformat(),
        "duration_sec": total_duration,
        "quick_mode": False,
        "log_file": str(log_file),
        "config": {
            "raw_csv": str(raw_path),
            "random_seed": RANDOM_SEED
        },
        "metrics": {
            "raw_rows": len(df_raw),
            "raw_cols": df_raw.shape[1],
            "raw_duplicates": struct_stats["n_duplicates"],
            "amount_mean": round(amount_stats["mean"], 2),
            "amount_median": round(amount_stats["median"], 2),
            "amount_std": round(amount_stats["std"], 2),
            "amount_skewness": round(amount_stats["skewness"], 2),
            "amount_kurtosis": round(amount_stats["kurtosis"], 2),
            "amount_outliers_count": iqr_outliers["n_outliers"],
            "amount_outliers_ratio": round(iqr_outliers["outlier_pct"], 2)
        },
        "tables": {
            "missing_counts": struct_stats["missing_counts"],
            "missing_pct": struct_stats["missing_pct"],
            "category_stats": bivariate_res.get("category_stats", pd.DataFrame()).head(10).to_dict(orient="records"),
            "fulfilment_stats": bivariate_res.get("fulfilment_stats", pd.DataFrame()).to_dict(orient="records"),
            "state_stats": bivariate_res.get("ship_state_stats", pd.DataFrame()).head(10).to_dict(orient="records"),
            "eta_squared": bivariate_res.get("eta_squared", {})
        },
        "figures": [
            str(fig1_path),
            str(fig2_path),
            str(fig3_path),
            str(fig4_path),
            str(fig5_path),
            str(fig6_path)
        ],
        "timings": timings
    }
    save_stage_result("01", stage_data)
    logger.info("  ✔ Đã lưu phong bì kết quả: outputs/stage_01.json")

    # 10. Tự động sinh báo cáo HTML
    with timed(logger, "Tự động sinh báo cáo HTML Chặng 1 và cập nhật index.html") as b:
        build_report_01()
        build_index()
    timings["build_html_reports"] = b.elapsed
    logger.info("  ✔ Đã tự động tạo: docs/report/01_data_report.html")
    logger.info("  ✔ Đã cập nhật bảng trạng thái: docs/report/index.html")

    logger.info("=" * 80)
    logger.info(f"🎉 HOÀN THÀNH CHẶNG 01 THÀNH CÔNG TRONG {total_duration:.2f} GIÂY!")
    logger.info(f"👉 Mở xem báo cáo trực quan tại: docs/report/01_data_report.html")
    logger.info("=" * 80)

    return stage_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chặng 01: Khám phá Dữ liệu Bán hàng TMĐT (Exploratory Data Analysis - EDA)")
    parser.add_argument("--raw-csv", type=str, default=str(RAW_CSV), help="Đường dẫn tới tệp dữ liệu thô Amazon Sale Report.csv")
    args = parser.parse_args()

    main(raw_csv_path=args.raw_csv)
