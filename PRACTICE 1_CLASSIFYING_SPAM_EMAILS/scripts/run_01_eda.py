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
    CLEAN_CSV,
    LOGS_DIR,
    RANDOM_SEED,
    require_files,
    save_figure,
    save_stage_result
)
from scripts.logger_utils import setup_logger, timed, log_environment
from data.scripts.eda import (
    load_raw_data,
    inspect_data_structure,
    get_univariate_stats,
    detect_iqr_outliers,
    get_top_keywords_comparison,
    analyze_re_prefix_and_duplicates
)
from data.scripts.data_preprocessing import process_and_save_clean_data
from scripts.report_builder import build_report_01, build_index


def main(raw_csv_path: str = None, extra_handler=None, logger=None) -> dict:
    """
    Kịch bản thực thi Chặng 01: Khám phá & Làm sạch dữ liệu (EDA & Data Cleaning):
    1. Đọc dữ liệu thô CEAS_08.csv.
    2. Khảo sát cấu trúc, thống kê đơn biến, ngoại lai IQR và từ khóa đặc trưng.
    3. Vẽ và lưu 2 biểu đồ PNG vào docs/images/.
    4. Làm sạch NLP và khử trùng lặp (Zero Duplicate Leakage) -> clean_emails.csv.
    5. Lưu kết quả phong bì vào outputs/stage_01.json.
    6. Tự động sinh docs/report/01_data_report.html và cập nhật index.html.
    """
    start_time_iso = datetime.now().isoformat()
    t_start = time.time()
    timings = {}

    # 1. Khởi tạo Logger riêng cho Chặng 01 (hỗ trợ extra_handler từ pipeline tổng)
    if logger is None:
        logger, log_file = setup_logger(log_dir=LOGS_DIR, prefix="run_01_eda", extra_handler=extra_handler)
    else:
        log_file = getattr(logger, "log_file", "")

    logger.info("=" * 75)
    logger.info("🚀 BẮT ĐẦU CHẶNG 01: KHÁM PHÁ & LÀM SẠCH DỮ LIỆU (EDA & NLP CLEANING)")
    logger.info("=" * 75)
    log_environment(logger, seed=RANDOM_SEED)

    # 2. Kiểm tra điều kiện tiên quyết
    raw_path = Path(raw_csv_path) if raw_csv_path else RAW_CSV
    require_files(
        logger,
        [raw_path],
        hint=f"Vui lòng đặt tệp CEAS_08.csv vào '{raw_path.parent}' hoặc chỉ định qua --raw-csv PATH."
    )

    # 3. Nạp dữ liệu thô
    with timed(logger, f"Nạp dữ liệu thô từ {raw_path.name}") as b:
        df_raw = load_raw_data(str(raw_path))
    timings["load_raw_data"] = b.elapsed
    logger.info(f"  • Quy mô dữ liệu thô: {len(df_raw):,} email ({df_raw.shape[1]} thuộc tính).")

    # 4. Khảo sát cấu trúc và giá trị khuyết thiếu
    with timed(logger, "Khảo sát cấu trúc bảng và khuyết thiếu kỹ thuật") as b:
        struct_stats = inspect_data_structure(df_raw)
    timings["inspect_structure"] = b.elapsed
    logger.info(f"  • Khuyết thiếu: subject={struct_stats['missing_values'].get('subject', 0)}, receiver={struct_stats['missing_values'].get('receiver', 0)}.")

    # 5. Thống kê đơn biến
    with timed(logger, "Thống kê đơn biến và tỷ lệ nhãn mục tiêu") as b:
        uni_stats = get_univariate_stats(df_raw)
    timings["univariate_stats"] = b.elapsed
    logger.info(f"  • Phân phối nhãn: Ham={uni_stats['ham_count']:,} ({uni_stats['ham_ratio']:.1f}%), Spam={uni_stats['spam_count']:,} ({uni_stats['spam_ratio']:.1f}%).")
    logger.info(f"  • Sự hiện diện của URLs: {uni_stats['url_present_ratio']:.1f}% email có liên kết URL.")

    # 6. Phân tích ngoại lai IQR
    with timed(logger, "Phân tích ngoại lai IQR độ dài tiêu đề & nội dung") as b:
        subj_len = df_raw["subject"].fillna("").astype(str).str.len()
        body_len = df_raw["body"].fillna("").astype(str).str.len()
        subj_outliers = detect_iqr_outliers(subj_len)
        body_outliers = detect_iqr_outliers(body_len)
    timings["iqr_outliers"] = b.elapsed
    logger.info(f"  • Ngoại lai IQR: Tiêu đề={subj_outliers['n_outliers']:,} ({subj_outliers['outlier_ratio']:.1f}%), Nội dung={body_outliers['n_outliers']:,} ({body_outliers['outlier_ratio']:.1f}%).")

    # 7. Trích xuất từ khóa so sánh và phân tích tiền tố Re:
    with timed(logger, "So sánh từ khóa Ham vs Spam và phân tích tiền tố Re:") as b:
        top_kw_df = get_top_keywords_comparison(df_raw, top_n=10)
        re_dup_stats = analyze_re_prefix_and_duplicates(df_raw)
    timings["keywords_and_re"] = b.elapsed
    logger.info(f"  • Tiền tố 'Re:' trong tiêu đề: Ham={re_dup_stats['re_ham_ratio']:.1f}%, Spam={re_dup_stats['re_spam_ratio']:.1f}%.")

    # 8. Vẽ và xuất 2 biểu đồ phân tích EDA vào docs/images/
    with timed(logger, "Vẽ và xuất 2 biểu đồ phân tích EDA vào docs/images/") as b:
        # Hình 1: eda_01_univariate_analysis.png (Lưới 2x2 từ Cell 9 Notebook 01)
        fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9))
        df_raw["label"].value_counts().plot.pie(
            labels=["Spam (1)", "Ham (0)"],
            autopct="%1.1f%%",
            colors=["#e74c3c", "#2ecc71"],
            ax=axes1[0, 0],
            startangle=90,
            explode=(0.03, 0)
        )
        axes1[0, 0].set_title("Tỷ lệ Spam vs Ham (Dữ liệu thô)", fontweight="bold")
        axes1[0, 0].set_ylabel("")

        sns.countplot(data=df_raw, x="urls", hue="urls", palette=["#3498db", "#f39c12"], legend=False, ax=axes1[0, 1])
        axes1[0, 1].set_title("Sự hiện diện của liên kết URLs", fontweight="bold")

        sns.histplot(subj_len[subj_len < 120], bins=30, kde=True, ax=axes1[1, 0], color="#9b59b6")
        axes1[1, 0].set_title("Phân phối Độ dài Tiêu đề (< 120 ký tự)", fontweight="bold")

        domains = df_raw["sender"].astype(str).apply(
            lambda x: x.split("@")[-1].lower() if "@" in str(x) else "unknown"
        )
        domains.value_counts().head(10).plot.barh(ax=axes1[1, 1], color="#1abc9c").invert_yaxis()
        axes1[1, 1].set_title("Top 10 Tên miền Người gửi", fontweight="bold")

        fig1.tight_layout()
        fig1_path = save_figure(fig1, "eda_01_univariate_analysis.png")

        # Hình 2: eda_02_length_distribution_boxplot.png (Boxplot từ Cell 13 Notebook 01)
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
        df_plot = df_raw.copy()
        df_plot["subject_len"] = subj_len
        df_plot["body_len"] = body_len

        sns.boxplot(
            data=df_plot,
            x="label",
            y="subject_len",
            hue="label",
            palette=["#2ecc71", "#e74c3c"],
            legend=False,
            ax=axes2[0],
            showfliers=False
        )
        axes2[0].set_xticks([0, 1])
        axes2[0].set_xticklabels(["Ham (0)", "Spam (1)"])
        axes2[0].set_title("Độ dài Tiêu đề theo Nhãn", fontweight="bold")

        sns.boxplot(
            data=df_plot,
            x="label",
            y=np.log1p(df_plot["body_len"]),
            hue="label",
            palette=["#2ecc71", "#e74c3c"],
            legend=False,
            ax=axes2[1]
        )
        axes2[1].set_xticks([0, 1])
        axes2[1].set_xticklabels(["Ham (0)", "Spam (1)"])
        axes2[1].set_title("Log(Độ dài Nội dung + 1) theo Nhãn", fontweight="bold")

        fig2.tight_layout()
        fig2_path = save_figure(fig2, "eda_02_length_distribution_boxplot.png")
    timings["export_figures"] = b.elapsed
    logger.info(f"  ✔ Đã xuất biểu đồ: {fig1_path.name}")
    logger.info(f"  ✔ Đã xuất biểu đồ: {fig2_path.name}")

    # 9. Làm sạch văn bản NLP và khử trùng lặp (Zero Duplicate Leakage)
    with timed(logger, "Làm sạch văn bản NLP và khử trùng lặp (Zero Duplicate Leakage)") as b:
        df_clean, n_removed, out_clean_path = process_and_save_clean_data(df_raw, str(CLEAN_CSV))
        clean_ham = int((df_clean["label"] == 0).sum())
        clean_spam = int((df_clean["label"] == 1).sum())
        clean_ham_ratio = float(clean_ham / len(df_clean) * 100)
        clean_spam_ratio = float(clean_spam / len(df_clean) * 100)
    timings["nlp_cleaning_and_dedup"] = b.elapsed
    logger.info(f"  • Đã loại bỏ {n_removed:,} mẫu email trùng lặp nội dung sạch.")
    logger.info(f"  ✔ Đã xuất: {CLEAN_CSV.name} ({len(df_clean):,} email sạch: Ham={clean_ham:,} ({clean_ham_ratio:.1f}%), Spam={clean_spam:,} ({clean_spam_ratio:.1f}%))")

    top_kw_records = [
        {
            "rank": i + 1,
            "ham_word": str(top_kw_df.iloc[i, 0]),
            "ham_count": int(top_kw_df.iloc[i, 1]),
            "spam_word": str(top_kw_df.iloc[i, 2]),
            "spam_count": int(top_kw_df.iloc[i, 3])
        }
        for i in range(len(top_kw_df))
    ]

    total_duration = round(time.time() - t_start, 2)

    # 10. Đóng gói phong bì kết quả vào outputs/stage_01.json
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
            "clean_csv": str(out_clean_path),
            "random_seed": RANDOM_SEED
        },
        "metrics": {
            "raw_rows": len(df_raw),
            "raw_cols": df_raw.shape[1],
            "clean_rows": len(df_clean),
            "clean_cols": df_clean.shape[1],
            "duplicates_removed": n_removed,
            "raw_ham_count": uni_stats["ham_count"],
            "raw_spam_count": uni_stats["spam_count"],
            "raw_ham_ratio": round(uni_stats["ham_ratio"], 2),
            "raw_spam_ratio": round(uni_stats["spam_ratio"], 2),
            "clean_ham_count": clean_ham,
            "clean_spam_count": clean_spam,
            "clean_ham_ratio": round(clean_ham_ratio, 2),
            "clean_spam_ratio": round(clean_spam_ratio, 2),
            "url_present_ratio": round(uni_stats["url_present_ratio"], 2),
            "subject_len_mean": round(uni_stats["subject_len_mean"], 2),
            "subject_len_median": round(uni_stats["subject_len_median"], 2),
            "subject_outliers_count": subj_outliers["n_outliers"],
            "subject_outliers_ratio": round(subj_outliers["outlier_ratio"], 2),
            "body_outliers_count": body_outliers["n_outliers"],
            "body_outliers_ratio": round(body_outliers["outlier_ratio"], 2)
        },
        "tables": {
            "missing_values": struct_stats["missing_values"],
            "top_keywords": top_kw_records,
            "re_duplicates": re_dup_stats
        },
        "figures": [
            str(fig1_path),
            str(fig2_path)
        ],
        "timings": timings
    }
    save_stage_result("01", stage_data)
    logger.info("  ✔ Đã lưu phong bì kết quả: outputs/stage_01.json")

    # 11. Tự động sinh báo cáo HTML
    with timed(logger, "Tự động sinh báo cáo HTML Chặng 1 và cập nhật index.html") as b:
        build_report_01()
        build_index()
    timings["build_html_reports"] = b.elapsed
    logger.info("  ✔ Đã tự động tạo: docs/report/01_data_report.html")
    logger.info("  ✔ Đã cập nhật bảng trạng thái: docs/report/index.html")

    logger.info("=" * 75)
    logger.info(f"🎉 HOÀN THÀNH CHẶNG 01 THÀNH CÔNG TRONG {total_duration:.2f} GIÂY!")
    logger.info(f"👉 Mở xem báo cáo trực quan tại: docs/report/01_data_report.html")
    logger.info("=" * 75)

    return stage_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chặng 01: Khám phá & Làm sạch dữ liệu (EDA & Data Cleaning)")
    parser.add_argument("--raw-csv", type=str, default=str(RAW_CSV), help="Đường dẫn tới tệp dữ liệu thô CEAS_08.csv")
    args = parser.parse_args()

    main(raw_csv_path=args.raw_csv)
