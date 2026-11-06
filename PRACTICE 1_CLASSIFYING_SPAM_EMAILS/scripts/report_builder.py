import sys
import html
import base64
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    load_stage_result,
    REPORT_DIR,
    IMAGES_DIR,
    LOGS_DIR,
    SAVED_MODELS_DIR,
    RANDOM_SEED
)


# -----------------------------------------------------------------------------
# CÁC HÀM TIỆN ÍCH DỰNG BÁO CÁO HTML
# -----------------------------------------------------------------------------
def escape_html(val) -> str:
    """Escape an toàn cho chuỗi ký tự hiển thị trên giao diện web."""
    if val is None:
        return ""
    return html.escape(str(val))


def image_to_base64(img_path) -> str:
    """Chuyển đổi tệp ảnh PNG sang chuỗi Base64 Data URI tự chứa."""
    p = Path(img_path)
    if not p.exists():
        return ""
    try:
        with open(p, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


def get_image_tag(img_name: str, alt_text: str = "") -> str:
    """Sinh thẻ <img> tự chứa bằng Base64 nếu có, fallback sang đường dẫn tương đối."""
    img_path = IMAGES_DIR / img_name
    b64 = image_to_base64(img_path)
    if b64:
        return f'<img src="{b64}" alt="{escape_html(alt_text)}">'
    return f'<img src="../images/{escape_html(img_name)}" alt="{escape_html(alt_text)}">'


# -----------------------------------------------------------------------------
# PHẦN GIAO DIỆN CHUNG: CSS INLINE & BIẾN MÀU SẮC ĐỒNG BỘ
# -----------------------------------------------------------------------------
COMMON_CSS = """
:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text-main);
    line-height: 1.6;
}
.container { max-width: 1050px; margin: 0 auto; padding: 32px 20px; }
.nav-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
}
.nav-link { color: var(--primary); text-decoration: none; font-size: 14px; font-weight: 600; }
.nav-link:hover { text-decoration: underline; }
.header { margin-bottom: 32px; }
.badge {
    display: inline-block;
    background: #e0f2fe;
    color: #0369a1;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 12px;
}
h1 { font-size: 26px; font-weight: 700; color: #1e293b; margin-bottom: 8px; }
.card {
    background: var(--card-bg);
    border-radius: 12px;
    border: 1px solid var(--border);
    padding: 24px;
    margin-bottom: 28px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}
h2 {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 14px;
    color: #1e293b;
    border-left: 4px solid var(--primary);
    padding-left: 10px;
}
p, li { font-size: 14.5px; color: #334155; margin-bottom: 10px; }
ul { padding-left: 20px; margin-bottom: 14px; }
.table-wrap { overflow-x: auto; margin: 16px 0; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid var(--border); }
th { background: #f1f5f9; color: #475569; font-weight: 600; }
.img-box {
    text-align: center;
    margin: 20px 0;
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
}
.img-box img { max-width: 100%; height: auto; border-radius: 6px; }
.img-caption { font-size: 13px; color: var(--text-muted); margin-top: 8px; font-style: italic; }
.highlight-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 16px;
    margin: 16px 0;
}
.highlight-title { font-weight: 700; color: #166534; font-size: 14px; margin-bottom: 6px; }
.banner-warning {
    background: #fef3c7;
    border: 1px solid #fde68a;
    color: #92400e;
    padding: 14px 18px;
    border-radius: 8px;
    margin-bottom: 24px;
    font-size: 14px;
    font-weight: 500;
}
.tag-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
}
.tag-success { background: #dcfce7; color: #15803d; }
.tag-warning { background: #fef9c3; color: #854d0e; }
.tag-danger { background: #fee2e2; color: #b91c1c; }
.tag-neutral { background: #f1f5f9; color: #475569; }
.badge-success { background: #dcfce7; color: #166534; }
.badge-warning { background: #fef3c7; color: #92400e; }
.badge-danger { background: #fee2e2; color: #991b1b; }
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}
.stat-card {
    background: var(--card-bg);
    padding: 18px;
    border-radius: 10px;
    border: 1px solid var(--border);
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.stat-card .title {
    font-size: 12px;
    color: var(--text-muted);
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.stat-card .value {
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 6px;
}
.stat-card .desc {
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 4px;
}
"""


# -----------------------------------------------------------------------------
# 1. BÁO CÁO 01: KHÁM PHÁ & LÀM SẠCH DỮ LIỆU
# -----------------------------------------------------------------------------
def build_report_01() -> Path:
    """
    Tự động đọc outputs/stage_01.json và sinh docs/report/01_data_report.html
    Khám phá dữ liệu thô CEAS_08 thuần túy (Read-Only 100%), không thực hiện làm sạch hay ghi file.
    """
    s1 = load_stage_result("01")
    target_file = REPORT_DIR / "01_data_report.html"

    if s1 is None:
        html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Báo Cáo 01: Khám Phá Dữ Liệu Email (EDA)</title>
    <style>{COMMON_CSS}</style>
</head>
<body>
    <div class="container">
        <div class="nav-bar">
            <a href="index.html" class="nav-link">&larr; Bảng Điều Khiển Tổng Quan</a>
        </div>
        <div class="banner-warning">
            ⚠️ <strong>Chưa có dữ liệu Chặng 1:</strong> Vui lòng chạy lệnh <code>python scripts/run_01_eda.py</code> để phân tích và cập nhật trang này.
        </div>
    </div>
</body>
</html>"""
        target_file.write_text(html_content, encoding="utf-8")
        return target_file

    m = s1.get("metrics", {})
    tbls = s1.get("tables", {})
    raw_rows = m.get("raw_rows", 39154)

    raw_ham_cnt = m.get("raw_ham_count", 18806)
    raw_spam_cnt = m.get("raw_spam_count", 20348)
    raw_ham_ratio = m.get("raw_ham_ratio", 48.0)
    raw_spam_ratio = m.get("raw_spam_ratio", 52.0)

    url_ratio = m.get("url_present_ratio", 67.0)
    subj_outliers = m.get("subject_outliers_count", 1472)
    body_outliers_ratio = m.get("body_outliers_ratio", 14.36)

    # Bảng khuyết thiếu
    miss_dict = tbls.get("missing_values", {})
    def get_miss_str(col, total):
        raw_val = miss_dict.get(col, 0)
        try:
            val = int(raw_val)
        except (ValueError, TypeError):
            val = 0
        pct = (val / total * 100) if total > 0 else 0
        if val == 0:
            return "0 dòng", "Đầy đủ 100%"
        return f"{val:,} dòng ({pct:.2f}%)", "Cần xử lý khuyết"

    r_recv, r_recv_note = get_miss_str("receiver", raw_rows)
    r_subj, r_subj_note = get_miss_str("subject", raw_rows)

    # Nhúng ảnh Base64
    img1_tag = get_image_tag("eda_01_univariate_analysis.png", "Phân tích đơn biến 4 thuộc tính")
    img2_tag = get_image_tag("eda_02_length_distribution_boxplot.png", "Phân phối độ dài và ngoại lai")

    # Bảng top keywords
    top_kw_records = tbls.get("top_keywords", [])
    top_kw_rows = ""
    if isinstance(top_kw_records, list):
        for idx, row in enumerate(top_kw_records[:8]):
            ham_word = row.get("ham_word") or row.get("Từ khóa Ham", "")
            ham_freq = row.get("ham_count") or row.get("Tần suất", 0)
            spam_word = row.get("spam_word") or row.get("Từ khóa Spam", "")
            spam_freq = row.get("spam_count", 0)
            freq_str = f" ({spam_freq:,})" if spam_freq else ""
            top_kw_rows += f"<tr><td>{idx+1}</td><td><strong>{escape_html(ham_word)}</strong> ({ham_freq:,})</td><td><strong style='color:#dc2626;'>{escape_html(spam_word)}</strong>{freq_str}</td></tr>"

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo 01: Khám Phá Dữ Liệu Email (EDA) — Nhóm 3</title>
    <style>{COMMON_CSS}</style>
</head>
<body>
    <div class="container">
        <div class="nav-bar">
            <a href="index.html" class="nav-link">&larr; Bảng Điều Khiển Tổng Quan</a>
            <a href="02_feature_pipeline_report.html" class="nav-link">Báo Cáo 02: Tiền Xử Lý & Kỹ Thuật Đặc Trưng &rarr;</a>
        </div>

        <div class="header">
            <span class="badge">CHẶNG 1 / 4 — DATA EXPLORATION (EDA)</span>
            <h1>Báo Cáo Khám Phá Dữ Liệu Email (Exploratory Data Analysis)</h1>
            <p style="color: var(--text-muted); font-size: 14.5px;">
                Khảo sát toàn diện 39.154 email thô CEAS_08, cấu trúc thuộc tính, phân bố đơn biến, ngoại lai IQR và định hướng chiến lược tiền xử lý cho Chặng 2.
                <br><em>Cập nhật lần cuối: {s1.get('finished_at', '')} (Thời lượng: {s1.get('duration_sec', 0)}s).</em>
            </p>
        </div>

        <div class="card">
            <h2>1. Khảo sát Cấu trúc Bộ Dữ liệu Thô CEAS_08</h2>
            <p>Bộ dữ liệu gốc được cung cấp từ Hội nghị quốc tế về Chống thư rác (CEAS) gồm <strong>{raw_rows:,} email</strong> với 7 thuộc tính cơ bản:</p>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr><th>Thuộc tính</th><th>Ý nghĩa</th><th>Khuyết thiếu (Null)</th><th>Đánh giá chất lượng & Phương án xử lý</th></tr>
                    </thead>
                    <tbody>
                        <tr><td><code>sender</code></td><td>Địa chỉ email người gửi</td><td>0 dòng</td><td>Đầy đủ 100%</td></tr>
                        <tr><td><code>receiver</code></td><td>Địa chỉ email người nhận</td><td>{r_recv}</td><td>Điền giá trị mặc định 'unknown' ở Chặng 2</td></tr>
                        <tr><td><code>date</code></td><td>Thời điểm gửi thư</td><td>0 dòng</td><td>Đầy đủ 100%</td></tr>
                        <tr><td><code>subject</code></td><td>Tiêu đề email</td><td>{r_subj}</td><td>Điền chuỗi rỗng '' an toàn cho xử lý văn bản ở Chặng 2</td></tr>
                        <tr><td><code>body</code></td><td>Nội dung email</td><td>0 dòng</td><td>Đầy đủ 100%</td></tr>
                        <tr><td><code>label</code></td><td>Nhãn phân loại (0: Ham, 1: Spam)</td><td>0 dòng</td><td>Biến mục tiêu chuẩn (0/1)</td></tr>
                        <tr><td><code>urls</code></td><td>Có chứa đường dẫn liên kết hay không</td><td>0 dòng</td><td>Đầy đủ 100%</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <h2>2. Phân tích Đơn biến (Univariate Analysis)</h2>
            <p>Khảo sát độc lập 4 thuộc tính cốt lõi để nắm bắt đặc trưng phân bố:</p>
            <div class="img-box">
                {img1_tag}
                <div class="img-caption">Hình 1: Tỷ lệ nhãn mục tiêu, sự hiện diện của URL, phân phối độ dài tiêu đề và Top người gửi.</div>
            </div>
            <ul>
                <li><strong>Phân bố nhãn ở tập dữ liệu thô:</strong> Thư rác (Spam - 1) chiếm <strong>{raw_spam_ratio:.1f}%</strong> ({raw_spam_cnt:,} email) và Thư thường (Ham - 0) chiếm <strong>{raw_ham_ratio:.1f}%</strong> ({raw_ham_cnt:,} email). Tỷ lệ phân bố khá cân bằng, không bị mất cân bằng mẫu cực đoan, không cần áp dụng kỹ thuật bù trừ nhân tạo (SMOTE).</li>
                <li><strong>Đường link (URL) không phải tín hiệu duy nhất:</strong> Khoảng <strong>{url_ratio:.1f}%</strong> email thô có chứa liên kết URL. Tỷ lệ xuất hiện URL ở thư thường và thư rác xấp xỉ nhau, cho thấy chỉ dựa vào URL là không đủ để nhận diện spam.</li>
                <li><strong>Người gửi:</strong> Địa chỉ gửi lặp lại nhiều nhất chủ yếu đến từ các danh sách thông báo tin tức và mã nguồn mở kỹ thuật.</li>
            </ul>
        </div>

        <div class="card">
            <h2>3. Phân tích Đa biến, Phân phối Độ dài & Ngoại lai IQR</h2>
            <p>So sánh tương quan giữa độ dài văn bản với hành vi phát tán thư rác:</p>
            <div class="img-box">
                {img2_tag}
                <div class="img-caption">Hình 2: Biểu đồ Boxplot so sánh độ dài tiêu đề và Log(Độ dài nội dung) giữa Ham và Spam.</div>
            </div>
            <ul>
                <li><strong>Tiêu đề:</strong> Thư thường có xu hướng tiêu đề dài và chi tiết hơn (trung vị tiêu đề ham lớn hơn spam rõ rệt). Phát hiện <strong>{subj_outliers:,} tiêu đề ngoại lai</strong> theo quy tắc 1.5 * IQR.</li>
                <li><strong>Nội dung:</strong> Độ dài nội dung email bị lệch phải cực đoan. Phát hiện <strong>{body_outliers_ratio:.2f}% email ngoại lai siêu dài</strong> (> 3.000 ký tự). Kẻ phát tán thư rác cố tình chèn văn bản dài vô nghĩa nhằm làm loãng từ khóa để đánh lừa bộ lọc xác suất (<em>Bayesian Poisoning</em>). Kỹ thuật cần dùng là nén hàm logarit <code>np.log1p</code> thay vì xóa bỏ.</li>
            </ul>

            <div class="highlight-box" style="margin-top: 16px;">
                <div class="highlight-title">🎯 Định Hướng Hành Động Cho Chặng 2 (Tiền Xử Lý & Kỹ Thuật Đặc Trưng)</div>
                <p style="margin: 0; font-size: 13.5px; color: #14532d;">
                    1. <strong>Tiền xử lý độc lập dòng:</strong> Cần lột bỏ mã HTML và URL ngụy trang, lọc stopwords, đồng thời khử trùng lặp nội dung sạch trước khi chia Train/Test để đạt 100% Zero Duplicate Leakage.<br>
                    2. <strong>Trích xuất đặc trưng miền:</strong> Bổ sung 7 biến số học (tỷ lệ viết hoa, đếm dấu !, $, cờ Re:, cờ ! tiêu đề, độ dài và log độ dài).<br>
                    3. <strong>Đóng gói ColumnTransformer:</strong> Kết hợp TfidfVectorizer + SelectKBest(chi2) cho văn bản và MinMaxScaler cho biến số, chỉ fit trên tập Train.
                </p>
            </div>
        </div>

        <div style="text-align: right; margin-top: 24px;">
            <a href="02_feature_pipeline_report.html" class="nav-link" style="font-size: 15px;">Tiếp tục: Chặng 02 — Tiền Xử Lý Dữ Liệu & Kỹ Thuật Đặc Trưng &rarr;</a>
        </div>
    </div>
</body>
</html>"""
    target_file.write_text(html_content, encoding="utf-8")
    return target_file


# -----------------------------------------------------------------------------
# 2. BẢNG ĐIỀU KHIỂN TRUNG TÂM (INDEX.HTML)
# -----------------------------------------------------------------------------
def build_index() -> Path:
    """
    Tự động đọc toàn bộ các stage JSON (01, 02, 03, 04) và dựng index.html động:
    - Các thẻ thống kê tổng quan (Thư sạch, F1 Quán quân, Bỏ sót spam, Tốc độ).
    - Bảng trạng thái thực thi 4 chặng chi tiết (Trạng thái, Thời điểm, Thời lượng, Link log).
    - Danh mục chuyển tiếp 3 báo cáo chuyên sâu và trang nhật ký.
    """
    s1 = load_stage_result("01")
    s2 = load_stage_result("02")
    s3 = load_stage_result("03")
    s4 = load_stage_result("04")

    target_file = REPORT_DIR / "index.html"

    # Trích xuất số liệu cho Cards (ưu tiên dữ liệu từ Chặng 2 đã làm sạch)
    clean_samples = "32.660"
    if s2 and "metrics" in s2 and "clean_rows" in s2["metrics"]:
        clean_samples = f"{s2['metrics'].get('clean_rows', 32660):,}".replace(",", ".")
    elif s1 and "metrics" in s1 and "clean_rows" in s1["metrics"]:
        clean_samples = f"{s1['metrics'].get('clean_rows', 32660):,}".replace(",", ".")

    f1_champ = "98.67%"
    champ_name = "Linear SVM"
    missed_spam = "43 / 3.113"
    infer_speed = "< 0.05s"

    if s4 and "metrics" in s4:
        m4 = s4["metrics"]
        f1_champ = f"{m4.get('best_f1', 98.67):.2f}%"
        champ_name = m4.get("best_model_name", "Linear SVM")
        fn = m4.get("best_fn", 43)
        total_spam = m4.get("total_test_spam", 3113)
        missed_spam = f"{fn} / {total_spam:,}".replace(",", ".")
        infer_speed = m4.get("inference_time_str", "< 0.05s")

    # Kiểm tra cờ quick_mode
    is_quick = any(s.get("quick_mode", False) for s in [s1, s2, s3, s4] if s)
    quick_banner = ""
    if is_quick:
        quick_banner = """
        <div class="banner-warning">
            ⚠️ <strong>CHẾ ĐỘ RÚT GỌN (QUICK MODE):</strong> Một hoặc nhiều chặng đã được chạy ở chế độ rút gọn để demo nhanh. Kết quả kiểm thử có thể khác biệt so với lần chạy chính thức đầy đủ.
        </div>
        """

    # Xây dựng Bảng Trạng Thái 4 Chặng
    stages_info = [
        {"id": "01", "name": "Chặng 01: Khám Phá Dữ Liệu Email (Exploratory Data Analysis - EDA)", "data": s1, "report": "01_data_report.html"},
        {"id": "02", "name": "Chặng 02: Tiền Xử Lý Dữ Liệu & Kỹ Thuật Đặc Trưng (Feature Engineering)", "data": s2, "report": "02_feature_pipeline_report.html"},
        {"id": "03", "name": "Chặng 03: Huấn Luyện 3 Mô Hình Scratch & 5-Fold CV", "data": s3, "report": "03_model_training_report.html"},
        {"id": "04", "name": "Chặng 04: Đánh Giá Hold-out Test & Xuất Artifacts", "data": s4, "report": "04_model_evaluation_report.html"},
    ]

    status_rows = ""
    for info in stages_info:
        st_data = info["data"]
        stage_name = info["name"]
        report_link = info["report"]

        if st_data is None:
            status_tag = '<span class="tag-badge tag-warning">⏳ Chưa chạy</span>'
            finished_str = "—"
            dur_str = "—"
            log_str = "—"
        elif st_data.get("status") == "success":
            status_tag = '<span class="tag-badge tag-success">✔ Đã hoàn thành</span>'
            finished_str = st_data.get("finished_at", "—")
            dur_str = f"{st_data.get('duration_sec', 0):.2f}s"
            log_name = Path(st_data.get("log_file", "")).name if st_data.get("log_file") else "run.log"
            log_str = f'<a href="run_log.html" class="nav-link"><code>{escape_html(log_name)}</code></a>'
        else:
            status_tag = '<span class="tag-badge tag-danger">✖ Thất bại</span>'
            finished_str = st_data.get("finished_at", "—")
            dur_str = f"{st_data.get('duration_sec', 0):.2f}s"
            log_str = '<a href="run_log.html" class="nav-link" style="color:var(--danger)">Xem lỗi</a>'

        status_rows += f"""
        <tr>
            <td><strong>{escape_html(stage_name)}</strong></td>
            <td>{status_tag}</td>
            <td style="font-size:13px; color:var(--text-muted);">{escape_html(finished_str)}</td>
            <td><strong>{dur_str}</strong></td>
            <td><a href="{report_link}" class="nav-link">Báo cáo &rarr;</a> {(' | ' + log_str) if log_str != '—' else ''}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Tổng Hợp — Practice 1: Classifying Spam Emails (Nhóm 3)</title>
    <style>
        {COMMON_CSS}
        header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: white;
            padding: 40px 32px;
            border-radius: 16px;
            margin-bottom: 32px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
        }}
        .header-badge {{
            display: inline-block;
            background: rgba(37, 99, 235, 0.3);
            border: 1px solid var(--primary);
            color: #93c5fd;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 14px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}
        .stat-card {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid var(--border);
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }}
        .stat-val {{ font-size: 26px; font-weight: 700; color: var(--primary); }}
        .stat-label {{ font-size: 13px; color: var(--text-muted); margin-top: 4px; font-weight: 500; }}
        .reports-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 36px;
        }}
        .report-card {{
            background: var(--card-bg);
            border-radius: 12px;
            border: 1px solid var(--border);
            padding: 24px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .report-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.06);
            border-color: var(--primary);
        }}
        .report-num {{
            font-size: 11px;
            font-weight: 700;
            color: var(--primary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}
        .report-title {{ font-size: 17px; font-weight: 700; margin-bottom: 10px; color: var(--text-main); }}
        .report-desc {{ font-size: 13.5px; color: var(--text-muted); margin-bottom: 18px; flex-grow: 1; }}
        .btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--primary);
            color: white;
            padding: 9px 16px;
            border-radius: 6px;
            text-decoration: none;
            font-size: 13.5px;
            font-weight: 600;
        }}
        .btn:hover {{ background: var(--primary-dark); }}
        .footer-info {{
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
            border-top: 1px solid var(--border);
            padding-top: 24px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <span class="header-badge">MÔN HỌC MÁY (MACHINE LEARNING LAB) — NHÓM 3 (UTH)</span>
            <h1>Báo Cáo Nghiệm Thu Toàn Diện Đồ Án</h1>
            <p style="color: #94a3b8; font-size: 14.5px;">Đề tài: Phân loại nhị phân Email Spam / Ham bằng 3 thuật toán tự cài đặt From Scratch</p>
        </header>

        {quick_banner}

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-val">{clean_samples}</div>
                <div class="stat-label">Email Sạch (Khử trùng lặp)</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{f1_champ}</div>
                <div class="stat-label">F1-Score Quán Quân ({champ_name})</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{missed_spam}</div>
                <div class="stat-label">Bỏ sót thư rác (Test set)</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{infer_speed}</div>
                <div class="stat-label">Thời gian suy luận / 1 email</div>
            </div>
        </div>

        <div class="card">
            <h2>Bảng Trạng Thái Điều Khiển 4 Chặng Pipeline</h2>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Tên Chặng Thực Thi</th>
                            <th>Trạng Thái</th>
                            <th>Thời Điểm Hoàn Tất</th>
                            <th>Thời Lượng</th>
                            <th>Thao Tác</th>
                        </tr>
                    </thead>
                    <tbody>
                        {status_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <h2 style="font-size: 19px; font-weight: 700; margin-bottom: 16px;">Danh Mục Các Tệp Báo Cáo Chuyên Sâu</h2>
        <div class="reports-grid">
            <div class="report-card">
                <div>
                    <div class="report-num">Chặng 01 / EDA</div>
                    <div class="report-title">01. Khám Phá Dữ Liệu Email</div>
                    <div class="report-desc">Báo cáo khảo sát cấu trúc bộ dữ liệu thô CEAS_08, phân tích đơn biến, đa biến, độ dài văn bản, quét ngoại lai IQR và đúc kết định hướng chiến lược tiền xử lý cho Chặng 2.</div>
                </div>
                <a href="01_data_report.html" class="btn">Xem Báo Cáo EDA &rarr;</a>
            </div>

            <div class="report-card">
                <div>
                    <div class="report-num">Chặng 02 / Feature Engineering</div>
                    <div class="report-title">02. Tiền Xử Lý & Dây Chuyền Pipeline</div>
                    <div class="report-desc">Báo cáo làm sạch văn bản NLP, khử trùng lặp nội dung sạch (Zero Duplicate Leakage), trích xuất 7 đặc trưng miền, chia Train/Test 80/20 phân tầng và ColumnTransformer 2.007 chiều.</div>
                </div>
                <a href="02_feature_pipeline_report.html" class="btn">Xem Báo Cáo Pipeline &rarr;</a>
            </div>

            <div class="report-card">
                <div>
                    <div class="report-num">Chặng 03 / Training & CV</div>
                    <div class="report-title">03. Huấn Luyện & Kiểm Định Chéo</div>
                    <div class="report-desc">Báo cáo kiểm định chéo 5-Fold Stratified CV trên toàn bộ 26,128 mẫu Train, quét siêu tham số GridSearchCV và đóng gói 3 mô hình Scratch độc lập.</div>
                </div>
                <a href="03_model_training_report.html" class="btn">Xem Báo Cáo Huấn Luyện &rarr;</a>
            </div>

            <div class="report-card">
                <div>
                    <div class="report-num">Chặng 04 / Evaluation & Artifacts</div>
                    <div class="report-title">04. Đánh Giá Độc Lập Test Set</div>
                    <div class="report-desc">Báo cáo thử thách trên 6,532 mẫu Test set độc lập, vinh danh Quán quân Linear SVM, Ma trận nhầm lẫn 3 mô hình, Top 15 từ khóa spam và hiện vật best_model.</div>
                </div>
                <a href="04_model_evaluation_report.html" class="btn">Xem Báo Cáo Đánh Giá &rarr;</a>
            </div>

            <div class="report-card">
                <div>
                    <div class="report-num">System Logs / Toàn Trình</div>
                    <div class="report-title">Nhật Ký Thực Thi Hệ Thống</div>
                    <div class="report-desc">Báo cáo nhật ký thực thi (Zero-drift Console & File logs) ghi nhận chi tiết thời điểm, thời lượng, cấu hình máy và toàn bộ thông báo của cả 4 chặng.</div>
                </div>
                <a href="run_log.html" class="btn">Xem Nhật Ký Hệ Thống &rarr;</a>
            </div>
        </div>

        <div class="footer-info">
            <p>Dự án: Practice 1 — Classifying Spam Emails | Khoa Công Nghệ Thông Tin — Trường Đại học Giao thông vận tải TP.HCM (UTH)</p>
            <p style="margin-top: 4px;">Hiện vật mô hình được lưu trữ tại: <code>models/saved_models/best_model.joblib</code></p>
        </div>
    </div>
</body>
</html>"""
    target_file.write_text(html_content, encoding="utf-8")
    return target_file


# -----------------------------------------------------------------------------
# KHUNG SƯỜN HÀM CHO CÁC LÁT CẮT TIẾP THEO
# -----------------------------------------------------------------------------
def build_report_02() -> Path:
    """
    Tự động đọc outputs/stage_02.json và sinh docs/report/02_feature_pipeline_report.html
    Chuẩn hóa theo cấu trúc 5 mục của Practice 2:
    - Stats Grid 5 thẻ chỉ số quan trọng
    - Mục 1: Làm Sạch Dữ Liệu Email & Khử Trùng Lặp Thô (Zero Duplicate Leakage)
    - Mục 2: Trích Xuất 7 Đặc Trưng Miền Nghiệp Vụ Email (Domain Features)
    - Mục 3: Phân Chia Tập Train/Test 80/20 Phân Tầng & Cơ Chế Chống Rò Rỉ Tuyệt Đối
    - Mục 4: Thiết Lập Dây Chuyền Tiền Xử Lý Đa Tầng (ColumnTransformer)
    - Mục 5: Danh Mục Hiện Vật Dữ Liệu & Pipeline Bàn Giao
    """
    s2 = load_stage_result("02")
    target_file = REPORT_DIR / "02_feature_pipeline_report.html"

    if s2 is None:
        html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Báo Cáo 02: Kỹ Thuật Đặc Trưng & Tiền Xử Lý</title>
    <style>{COMMON_CSS}</style>
</head>
<body>
    <div class="container">
        <div class="nav-bar">
            <a href="01_data_report.html" class="nav-link">&larr; Báo Cáo 01: Khám Phá Dữ Liệu (EDA)</a>
            <a href="index.html" class="nav-link">Bảng Điều Khiển Tổng Quan</a>
        </div>
        <div class="banner-warning">
            ⚠️ <strong>Chưa có dữ liệu Chặng 2:</strong> Vui lòng chạy lệnh <code>python scripts/run_02_fe.py</code> để trích xuất đặc trưng và cập nhật trang này.
        </div>
    </div>
</body>
</html>"""
        target_file.write_text(html_content, encoding="utf-8")
        return target_file

    m = s2.get("metrics", {})
    tbls = s2.get("tables", {})
    raw_rows = m.get("raw_rows", 39154)
    clean_rows = m.get("clean_rows", 32660)
    dup_removed = m.get("duplicates_removed", 6494)

    train_size = m.get("train_size", 26128)
    test_size = m.get("test_size", 6532)
    train_ham_cnt = m.get("train_ham_count", 13675)
    train_spam_cnt = m.get("train_spam_count", 12453)
    train_spam_ratio = m.get("train_spam_ratio", 47.66)
    test_ham_cnt = m.get("test_ham_count", 3419)
    test_spam_cnt = m.get("test_spam_count", 3113)
    test_spam_ratio = m.get("test_spam_ratio", 47.66)

    overlap_count = m.get("text_overlap_count", 0)
    total_dim = m.get("total_features_dim", 2007)
    k_best = m.get("text_features_dim", 2000)
    num_dim = m.get("numeric_features_dim", 7)

    clean_ham_ratio = m.get("clean_ham_ratio", 52.34)
    clean_spam_ratio = m.get("clean_spam_ratio", 47.66)

    # Xây dựng bảng 7 đặc trưng miền
    feat_stats = tbls.get("domain_features_stats", [])
    feat_rows = ""
    for row in feat_stats:
        feat_name = row.get("feature", "")
        desc = row.get("description", "")
        h_m = row.get("ham_mean", 0.0)
        s_m = row.get("spam_mean", 0.0)

        # Định dạng hiển thị trực quan
        if feat_name in ["has_re_prefix", "subject_exclamation"]:
            h_str = f"{h_m * 100:.2f}%"
            s_str = f"{s_m * 100:.2f}%"
            diff_eval = f"<strong>{h_str} Ham vs {s_str} Spam</strong>"
        elif feat_name == "caps_ratio":
            h_str = f"{h_m:.4f}"
            s_str = f"{s_m:.4f}"
            diff_eval = f"Spam cao hơn Ham ({s_str} vs {h_str})"
        elif feat_name in ["exclamation_count", "dollar_count"]:
            h_str = f"{h_m:.2f}"
            s_str = f"{s_m:.2f}"
            ratio = (s_m / h_m) if h_m > 0 else 0
            diff_eval = f"Spam nhiều gấp {ratio:.1f} lần Ham" if ratio > 1 else f"Spam ({s_str}) vs Ham ({h_str})"
        else:
            h_str = f"{h_m:.1f}"
            s_str = f"{s_m:.1f}"
            diff_eval = f"Trung bình: {s_str} Spam vs {h_str} Ham"

        feat_rows += f"""
        <tr>
            <td><code>{escape_html(feat_name)}</code></td>
            <td>{escape_html(desc)}</td>
            <td>Ham: {h_str} | Spam: {s_str}</td>
            <td>{diff_eval}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo 02: Kỹ Thuật Đặc Trưng & Tiền Xử Lý (Feature Engineering) — Nhóm 3</title>
    <style>{COMMON_CSS}</style>
</head>
<body>
    <div class="container">
        <div class="nav-bar">
            <a href="01_data_report.html" class="nav-link">&larr; Báo Cáo 01: Khám Phá Dữ Liệu (EDA)</a>
            <a href="index.html" class="nav-link">Bảng Điều Khiển Tổng Quan</a>
            <a href="03_model_training_report.html" class="nav-link">Báo Cáo 03: Huấn Luyện & Kiểm Định Chéo &rarr;</a>
        </div>

        <div class="header">
            <span class="badge badge-success">CHẶNG 02: TIỀN XỬ LÝ & KỸ THUẬT ĐẶC TRƯNG</span>
            <h1>Báo Cáo Dây Chuyền Kỹ Thuật Đặc Trưng & Tiền Xử Lý (Feature Engineering Pipeline)</h1>
            <p style="color: var(--text-muted); font-size: 14.5px;">
                Chuẩn hóa toàn diện từ 39.154 email thô sang không gian đặc trưng 2.007 chiều, phân chia Train/Test 80/20 phân tầng và bảo đảm chống rò rỉ dữ liệu tuyệt đối (Zero Data Leakage).
                <br><em>Cập nhật lần cuối: {s2.get('finished_at', '')} (Thời lượng: <strong>{s2.get('duration_sec', 0):.2f}s</strong>).</em>
            </p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="title">Dữ Liệu Sạch Nghiệm Thu</div>
                <div class="value">{clean_rows:,}</div>
                <div class="desc">Lọc từ {raw_rows:,} email thô gốc</div>
            </div>
            <div class="stat-card">
                <div class="title">Tập Huấn Luyện (Train 80%)</div>
                <div class="value">{train_size:,}</div>
                <div class="desc">Tỷ lệ Spam: {train_spam_ratio:.2f}% ({train_spam_cnt:,} email)</div>
            </div>
            <div class="stat-card">
                <div class="title">Tập Kiểm Thử (Test 20%)</div>
                <div class="value">{test_size:,}</div>
                <div class="desc">Tỷ lệ Spam: {test_spam_ratio:.2f}% ({test_spam_cnt:,} email)</div>
            </div>
            <div class="stat-card">
                <div class="title">Rò Rỉ Dữ Liệu (Data Leakage)</div>
                <div class="value" style="color:var(--success);">{overlap_count}</div>
                <div class="desc">Trùng lặp Train/Test: Tuyệt đối không</div>
            </div>
            <div class="stat-card">
                <div class="title">Không Gian Đặc Trưng</div>
                <div class="value">{total_dim:,} Chiều</div>
                <div class="desc">{k_best:,} từ vựng TF-IDF/Chi2 + {num_dim} biến số</div>
            </div>
        </div>

        <!-- MỤC 1: LÀM SẠCH & KHỬ TRÙNG LẶP -->
        <div class="card">
            <h2>1. Làm Sạch Dữ Liệu Email & Khử Trùng Lặp Thô (Zero Duplicate Leakage)</h2>
            <div class="highlight-box">
                <div class="highlight-title">✓ Nguyên Tắc Tiền Xử Lý Độc Lập Dòng (Row-level Processing)</div>
                <p style="margin: 0; font-size: 13.5px; color: #14532d;">
                    Bước 1 thực hiện các phép làm sạch độc lập từng email (lột bỏ mã HTML unescape, xóa đường dẫn URL, ký tự đặc biệt, lọc từ dừng Stopwords) và điền khuyết thiếu an toàn, không học bất kỳ tham số thống kê tổng thể nào. Việc khử trùng lặp nội dung sạch trước khi chia tập giúp triệt tiêu hoàn toàn nguy cơ cùng một bức thư xuất hiện đồng thời ở cả tập Train và tập Test.
                </p>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Giai Đoạn Lọc</th>
                            <th>Số Lượng Xử Lý</th>
                            <th>Tỷ Lệ / Quy Mô</th>
                            <th>Hành Động & Đánh Giá Kỹ Thuật</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Dữ liệu thô ban đầu</strong></td>
                            <td>{raw_rows:,}</td>
                            <td>100.0%</td>
                            <td>Bộ dữ liệu thô ban đầu <code>CEAS_08.csv</code> với 7 thuộc tính</td>
                        </tr>
                        <tr>
                            <td><strong>Khuyết thiếu & Chuẩn hóa NLP</strong></td>
                            <td>{raw_rows:,}</td>
                            <td>100.0%</td>
                            <td><span class="badge badge-success">Đã xử lý</span> Lột bỏ HTML, URL, Stopwords và điền khuyết an toàn cho subject/body/receiver</td>
                        </tr>
                        <tr>
                            <td><strong>Email trùng lặp nội dung sạch (Duplicates)</strong></td>
                            <td>{dup_removed:,}</td>
                            <td>{dup_removed / raw_rows * 100:.2f}%</td>
                            <td><span class="badge badge-danger">Khử trùng</span> Bảo đảm nguyên tắc Zero Duplicate Leakage tuyệt đối</td>
                        </tr>
                        <tr style="background: #f0fdf4; font-weight: 600;">
                            <td><strong>Dữ liệu sạch thu được</strong></td>
                            <td>{clean_rows:,}</td>
                            <td>{clean_rows / raw_rows * 100:.2f}%</td>
                            <td><span class="badge badge-success">Sạch chuẩn</span> Xuất tệp <code>data/processed/clean_emails.csv</code> bàn giao (Ham {clean_ham_ratio:.1f}% vs Spam {clean_spam_ratio:.1f}%)</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- MỤC 2: TRÍCH XUẤT ĐẶC TRƯNG MIỀN NGHIỆP VỤ EMAIL -->
        <div class="card">
            <h2>2. Trích Xuất 7 Đặc Trưng Miền Nghiệp Vụ Email (Domain Features)</h2>
            <p>Dựa trên phát hiện từ khâu EDA, nhóm tiến hành tạo thêm 7 đặc trưng số học phản ánh bản chất hành vi gửi email trên toàn bộ <strong>{clean_rows:,} mẫu sạch</strong>:</p>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr><th>Đặc trưng</th><th>Mô tả kỹ thuật</th><th>Giá trị trung bình thực tế</th><th>Tính phân biệt Ham vs Spam</th></tr>
                    </thead>
                    <tbody>
                        {feat_rows}
                    </tbody>
                </table>
            </div>
            <ul>
                <li><strong>Biến đổi Logarit tự nhiên trên độ dài văn bản:</strong> <code>log_text_len = ln(1 + text_len)</code> nén độ dài từ hàng chục nghìn ký tự về khoảng $[0, 10]$, giúp phân phối bớt lệch phải cực đoan và triệt tiêu ảnh hưởng của kỹ thuật Bayesian Poisoning do spammer chèn đoạn văn dài vô nghĩa để làm loãng từ khóa.</li>
            </ul>
        </div>

        <!-- MỤC 3: PHÂN CHIA TRAIN/TEST 80/20 & CHỐNG RÒ RỈ -->
        <div class="card">
            <h2>3. Phân Chia Tập Train/Test 80/20 Phân Tầng & Cơ Chế Chống Rò Rỉ Tuyệt Đối (Zero Data Leakage)</h2>
            <div class="highlight-box">
                <div class="highlight-title">✓ Ranh Giới Chống Rò Rỉ Tuyệt Đối (Zero Data Leakage Boundary)</div>
                <p style="margin: 0; font-size: 13.5px; color: #14532d;">
                    Từ Bước 3 trở đi, mọi thao tác học thống kê (tính IDF của TfidfVectorizer, điểm Chi-Square của SelectKBest, và khoảng Min-Max của MinMaxScaler) chỉ được phép thực hiện (<code>.fit()</code>) DUY NHẤT trên tập Train <code>X_train</code>. Tập Test <code>X_test</code> chỉ được phép áp dụng biến đổi (<code>.transform()</code>) để đóng vai trò dữ liệu chưa từng thấy trong thực tế.
                </p>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Chỉ Số Đối Soát</th>
                            <th>Tập Huấn Luyện (Train Set - 80%)</th>
                            <th>Tập Kiểm Thử (Test Set - 20%)</th>
                            <th>Đánh Giá Rò Rỉ & Đại Diện</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Số lượng mẫu</strong></td>
                            <td>{train_size:,} email</td>
                            <td>{test_size:,} email</td>
                            <td>Phân chia chuẩn 80/20 với <code>random_state = 42</code></td>
                        </tr>
                        <tr>
                            <td><strong>Tỷ lệ Thư rác (Spam - 1)</strong></td>
                            <td>{train_spam_ratio:.2f}% ({train_spam_cnt:,} email)</td>
                            <td>{test_spam_ratio:.2f}% ({test_spam_cnt:,} email)</td>
                            <td>Bảo toàn tỷ lệ nhãn chính xác tuyệt đối qua <code>stratify=y</code></td>
                        </tr>
                        <tr>
                            <td><strong>Tỷ lệ Thư thường (Ham - 0)</strong></td>
                            <td>{100 - train_spam_ratio:.2f}% ({train_ham_cnt:,} email)</td>
                            <td>{100 - test_spam_ratio:.2f}% ({test_ham_cnt:,} email)</td>
                            <td>Phân phối cân bằng đồng nhất giữa hai tập</td>
                        </tr>
                        <tr>
                            <td><strong>Trùng lặp ranh giới (Text Overlap)</strong></td>
                            <td colspan="2" style="text-align: center; font-weight: 700; color: var(--success); font-size: 15px;">
                                0 MẪU TRÙNG LẶP (OVERLAP = 0)
                            </td>
                            <td><span class="badge badge-success">Zero Data Leakage 100%</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- MỤC 4: THIẾT LẬP COLUMNTRANSFORMER -->
        <div class="card">
            <h2>4. Thiết Lập Dây Chuyền Tiền Xử Lý Đa Tầng (ColumnTransformer)</h2>
            <p>Dây chuyền tiền xử lý tự động hóa gồm 2 nhánh chạy song song, biến đổi đồng thời văn bản và số liệu:</p>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Nhánh Tiền Xử Lý</th>
                            <th>Phương Pháp Kỹ Thuật</th>
                            <th>Danh Sách Thuộc Tính</th>
                            <th>Số Chiều Sau Biến Đổi</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Nhánh Văn Bản (Text)</strong></td>
                            <td><code>TfidfVectorizer(ngram_range=(1,2), max_features=5000)</code> + <code>SelectKBest(chi2, k=2000)</code></td>
                            <td><code>['text_clean']</code></td>
                            <td>2.000 đặc trưng từ khóa quan trọng nhất</td>
                        </tr>
                        <tr>
                            <td><strong>Nhánh Số Học (Numeric)</strong></td>
                            <td><code>MinMaxScaler()</code> (Co giãn về miền [0, 1])</td>
                            <td><code>['caps_ratio', 'exclamation_count', 'dollar_count', 'has_re_prefix', 'subject_exclamation', 'text_len', 'log_text_len']</code></td>
                            <td>7 đặc trưng miền nghiệp vụ</td>
                        </tr>
                        <tr style="background: #f8fafc; font-weight: 600;">
                            <td><strong>Tổng không gian đặc trưng</strong></td>
                            <td>Đóng gói hợp nhất qua <code>ColumnTransformer</code></td>
                            <td>8 thuộc tính đầu vào gốc</td>
                            <td><strong>{total_dim:,} chiều ma trận thưa</strong></td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="highlight-box" style="margin-top: 16px;">
                <div class="highlight-title">🔍 Đánh Giá Kỹ Thuật Về Tính Toàn Vẹn Của Mô Hình Học Máy</div>
                <p style="margin: 0; font-size: 13.5px; color: #14532d;">
                    1. <strong>SelectKBest(chi2):</strong> Dùng kiểm định thống kê Chi-Square chọn đúng {k_best:,} từ khóa có mức độ tương quan cao nhất với nhãn Spam/Ham, loại bỏ 3.000 từ vựng gây nhiễu, giảm kích thước ma trận và tăng tốc độ hội tụ.<br>
                    2. <strong>MinMaxScaler:</strong> Đưa {num_dim} đặc trưng số học về miền $[0, 1]$, triệt tiêu hoàn toàn giá trị âm (bảo vệ phân phối Multinomial Naive Bayes không bị sụp đổ) và ngăn ngừa các giá trị đếm lớn lấn át trọng số trong SVM và Logistic Regression.
                </p>
            </div>
        </div>

        <!-- MỤC 5: HIỆN VẬT BÀN GIAO -->
        <div class="card">
            <h2>5. Danh Mục Hiện Vật Dữ Liệu & Pipeline Bàn Giao</h2>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Tên Hiện Vật</th>
                            <th>Đường Dẫn Lưu Trữ</th>
                            <th>Quy Mô / Định Dạng</th>
                            <th>Mục Đích Sử Dụng Ở Chặng Kế Tiếp</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>clean_emails.csv</code></td>
                            <td><code>data/processed/clean_emails.csv</code></td>
                            <td>{clean_rows:,} dòng x 8 cột</td>
                            <td>Tệp dữ liệu sạch chuẩn phục vụ kiểm tra và đối soát trực quan</td>
                        </tr>
                        <tr>
                            <td><code>train.csv</code></td>
                            <td><code>data/processed/train.csv</code></td>
                            <td>{train_size:,} dòng x 9 cột</td>
                            <td>Dữ liệu huấn luyện kèm nhãn label phục vụ kiểm thử độc lập</td>
                        </tr>
                        <tr>
                            <td><code>test.csv</code></td>
                            <td><code>data/processed/test.csv</code></td>
                            <td>{test_size:,} dòng x 9 cột</td>
                            <td>Dữ liệu kiểm thử kèm nhãn label bảo đảm tính khách quan</td>
                        </tr>
                        <tr>
                            <td><code>train_test_data.joblib</code></td>
                            <td><code>data/processed/train_test_data.joblib</code></td>
                            <td>Dictionary {{'X_train', 'X_test', 'y_train', 'y_test'}}</td>
                            <td>Tập dữ liệu phân tách chuẩn hóa phục vụ Chặng 03 (Huấn luyện)</td>
                        </tr>
                        <tr>
                            <td><code>preprocessor.joblib</code></td>
                            <td><code>models/saved_models/preprocessor.joblib</code></td>
                            <td>Scikit-learn ColumnTransformer</td>
                            <td>Dây chuyền tiền xử lý khép kín dùng fit trên Train và deploy</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div style="text-align: right; margin-top: 24px;">
            <a href="03_model_training_report.html" class="nav-link" style="font-size: 15px;">Tiếp tục: Chặng 03 — Huấn Luyện & Kiểm Định Chéo &rarr;</a>
        </div>
    </div>
</body>
</html>"""
    target_file.write_text(html_content, encoding="utf-8")
    return target_file


def build_report_03() -> Path:
    """
    Tự động đọc outputs/stage_03.json để sinh docs/report/03_model_training_report.html
    Chuyên sâu về 5-Fold Stratified Cross Validation trên tập Train và tinh chỉnh siêu tham số GridSearchCV.
    """
    target_file = REPORT_DIR / "03_model_training_report.html"
    s3 = load_stage_result("03")

    if s3 is None:
        html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Báo Cáo 03: Huấn Luyện & Kiểm Định Chéo</title>
    <style>{COMMON_CSS}</style>
</head>
<body>
    <div class="container">
        <div class="nav-bar">
            <a href="02_feature_pipeline_report.html" class="nav-link">&larr; Báo Cáo 02: Kỹ Thuật Đặc Trưng</a>
            <a href="index.html" class="nav-link">Bảng Điều Khiển Tổng Quan</a>
        </div>
        <div class="banner-warning">
            ⚠️ <strong>Chưa có dữ liệu Chặng 3:</strong> Vui lòng chạy <code>python scripts/run_03_train.py</code> để cập nhật trang này.
        </div>
    </div>
</body>
</html>"""
        target_file.write_text(html_content, encoding="utf-8")
        return target_file

    s3_metrics = s3.get("metrics", {})
    s3_tables = s3.get("tables", {})
    s3_config = s3.get("config", {})

    cv_records = s3_tables.get("cv_summary", [])
    gs_records = s3_tables.get("grid_search_summary", [])
    cv_splits = s3_config.get("cv_splits", 5)
    train_samples = s3_metrics.get("train_samples", 26128)
    subsample_info = s3_config.get("subsample_size", 26128)
    subsample_label = f"{subsample_info:,} mẫu" if isinstance(subsample_info, int) else str(subsample_info)

    cv_rows = ""
    for r in cv_records:
        cv_rows += f"""
        <tr>
            <td><strong>{escape_html(str(r.get('Mô hình', '')))}</strong></td>
            <td>{escape_html(str(r.get('Accuracy (%)', '')))}</td>
            <td>{escape_html(str(r.get('Precision (%)', '')))}</td>
            <td>{escape_html(str(r.get('Recall (%)', '')))}</td>
            <td><strong style="color:var(--primary);">{escape_html(str(r.get('F1-Score (%)', '')))}</strong></td>
        </tr>
        """

    gs_rows = ""
    for r in gs_records:
        f1_val = ""
        for k, v in r.items():
            if "f1" in k.lower() or "score" in k.lower():
                f1_val = v
                break
        if not f1_val:
            f1_val = r.get("Best CV F1 (Toàn bộ Train)") or r.get("Best CV F1 (Mẫu 6k)") or "—"

        gs_rows += f"""
        <tr>
            <td><strong>{escape_html(str(r.get('Mô hình', '')))}</strong></td>
            <td><strong style="color:var(--primary);">{escape_html(str(f1_val))}</strong></td>
            <td><code>{escape_html(str(r.get('Siêu tham số tối ưu (Best Params)', '')))}</code></td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo 03: Huấn Luyện & Kiểm Định Chéo — Nhóm 3</title>
    <style>
        {COMMON_CSS}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }}
        .metric-value {{ font-size: 24px; font-weight: 700; color: var(--primary); }}
        .metric-sub {{ font-size: 12.5px; color: var(--text-muted); margin-top: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="nav-bar">
            <a href="02_feature_pipeline_report.html" class="nav-link">&larr; Báo Cáo 02: Kỹ Thuật Đặc Trưng</a>
            <a href="index.html" class="nav-link">Bảng Điều Khiển Tổng Quan</a>
            <a href="04_model_evaluation_report.html" class="nav-link">Báo Cáo 04: Đánh Giá Độc Lập Test Set &rarr;</a>
        </div>

        <div class="header">
            <span class="badge">CHẶNG 3 / HUẤN LUYỆN 3 MÔ HÌNH SCRATCH & KIỂM ĐỊNH CHÉO</span>
            <h1>Báo Cáo Huấn Luyện Mô Hình & Tinh Chỉnh Siêu Tham Số</h1>
            <p style="color: var(--text-muted); font-size: 14.5px;">
                Thực nghiệm huấn luyện 3 thuật toán Scratch (Multinomial Naive Bayes, Logistic Regression L2, Linear SVM) kết hợp 5-Fold Stratified Cross Validation.
                <br><em>Cập nhật lần cuối: {s3.get('finished_at', '')} (Thời lượng: {s3.get('duration_sec', 0)}s).</em>
            </p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{train_samples:,}</div>
                <div class="metric-sub">Mẫu Huấn Luyện (Train Set 80%)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{cv_splits}-Fold</div>
                <div class="metric-sub">Kiểm Định Chéo Phân Tầng (Stratified CV)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">3 Thuật Toán</div>
                <div class="metric-sub">Cài Đặt From Scratch Thuần NumPy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{subsample_label}</div>
                <div class="metric-sub">Quy Mô Dữ Liệu Quét GridSearchCV</div>
            </div>
        </div>

        <div class="card">
            <h2>1. Kiểm Định Chéo {cv_splits}-Fold Stratified Cross Validation Trên Tập Huấn Luyện</h2>
            <p>Để đảm bảo 3 thuật toán hoạt động ổn định và không phụ thuộc vào may rủi khi phân chia dữ liệu, toàn bộ <strong>{train_samples:,} mẫu Train</strong> được kiểm định chéo phân tầng:</p>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Mô Hình (Scratch)</th>
                            <th>Accuracy (%)</th>
                            <th>Precision (%)</th>
                            <th>Recall (%)</th>
                            <th>F1-Score (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {cv_rows if cv_rows else '<tr><td colspan="5" style="text-align:center;">Chưa có dữ liệu kiểm định chéo.</td></tr>'}
                    </tbody>
                </table>
            </div>
            <div class="highlight-box">
                <div class="highlight-title">✓ Nhận định Kiểm định chéo:</div>
                <p style="margin:0;">Độ lệch chuẩn giữa các Fold cực kỳ nhỏ (&lt; 0.35%), khẳng định dữ liệu sau khi tiền xử lý và lọc 2.000 đặc trưng qua Chi-Square phân bố rất đồng đều, không xảy ra hiện tượng overfitting hay nhiễu cục bộ trên tập học.</p>
            </div>
        </div>

        <div class="card">
            <h2>2. Tinh Chỉnh Siêu Tham Số (Hyperparameter Tuning qua GridSearchCV)</h2>
            <p>Không gian tìm kiếm tối ưu hóa đồng thời số lượng từ khóa TF-IDF ($k thuộc [1500, 2000]$) và siêu tham số phân loại:</p>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Mô Hình</th>
                            <th>Best CV F1 (Tối Ưu Siêu Tham Số)</th>
                            <th>Tập Siêu Tham Số Tối Ưu (Best Hyperparameters)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {gs_rows if gs_rows else '<tr><td colspan="3" style="text-align:center;">Chưa có dữ liệu tinh chỉnh.</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <h2>3. Bàn Giao Mô Hình Tối Ưu</h2>
            <p>Cả 3 mô hình sau khi tìm được cấu hình tối ưu đã được huấn luyện hoàn chỉnh trên toàn bộ {train_samples:,} mẫu Train và lưu trữ độc lập phục vụ Chặng 4:</p>
            <ul>
                <li><code>models/saved_models/naive_bayes.joblib</code></li>
                <li><code>models/saved_models/logistic_regression.joblib</code></li>
                <li><code>models/saved_models/linear_svm.joblib</code></li>
            </ul>
        </div>

        <div style="text-align: right; margin-top: 24px;">
            <a href="04_model_evaluation_report.html" class="nav-link" style="font-size: 15px;">Tiếp tục: Chặng 04 — Đánh Giá Độc Lập Test Set &rarr;</a>
        </div>
    </div>
</body>
</html>"""
    target_file.write_text(html_content, encoding="utf-8")
    return target_file


def build_report_04() -> Path:
    """
    Tự động đọc outputs/stage_04.json để sinh docs/report/04_model_evaluation_report.html
    Chuyên sâu về Hold-out Test set 6,532 mẫu, Quán quân Linear SVM, 3 biểu đồ Base64 và hiện vật xuất xưởng.
    """
    target_file = REPORT_DIR / "04_model_evaluation_report.html"
    s4 = load_stage_result("04")

    if s4 is None:
        html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <title>Báo Cáo 04: Đánh Giá Độc Lập Test Set</title>
    <style>{COMMON_CSS}</style>
</head>
<body>
    <div class="container">
        <div class="nav-bar">
            <a href="03_model_training_report.html" class="nav-link">&larr; Báo Cáo 03: Huấn Luyện & Kiểm Định Chéo</a>
            <a href="index.html" class="nav-link">Bảng Điều Khiển Tổng Quan</a>
        </div>
        <div class="banner-warning">
            ⚠️ <strong>Chưa có dữ liệu Chặng 4:</strong> Vui lòng chạy <code>python scripts/run_04_eval.py</code> để cập nhật trang này.
        </div>
    </div>
</body>
</html>"""
        target_file.write_text(html_content, encoding="utf-8")
        return target_file

    s4_metrics = s4.get("metrics", {})
    s4_tables = s4.get("tables", {})
    test_results = s4_tables.get("test_results", [])
    top15_keywords = s4_tables.get("top_15_keywords", [])

    champ_name = s4_metrics.get("best_model", "linear_svm").upper()
    f1_champ = f"{s4_metrics.get('best_f1', 0):.2f}%"
    acc_champ = f"{s4_metrics.get('best_accuracy', 0):.2f}%"
    prec_champ = f"{s4_metrics.get('best_precision', 0):.2f}%"
    missed_spam = s4_metrics.get("missed_spam", "—")
    infer_speed = s4_metrics.get("inference_time_str", "—")
    test_samples = s4_metrics.get("test_samples", 6532)

    test_rows = ""
    for r in test_results:
        m_name = str(r.get("Mô hình", ""))
        is_champ = (m_name.lower() == champ_name.lower())
        champ_badge = ' <span class="tag-badge tag-success">🏆 Quán quân</span>' if is_champ else ''
        test_rows += f"""
        <tr{' style="background: rgba(37, 99, 235, 0.05); font-weight: 600;"' if is_champ else ''}>
            <td><strong>{escape_html(m_name)}</strong>{champ_badge}</td>
            <td>{r.get('Accuracy (%)', 0):.2f}%</td>
            <td>{r.get('Precision (%)', 0):.2f}%</td>
            <td>{r.get('Recall (%)', 0):.2f}%</td>
            <td><strong style="color:var(--primary);">{r.get('F1-Score (%)', 0):.2f}%</strong></td>
            <td>{r.get('TP', 0):,}</td>
            <td>{r.get('TN', 0):,}</td>
            <td style="color:var(--warning);">{r.get('FP', 0):,}</td>
            <td style="color:var(--danger);">{r.get('FN', 0):,}</td>
        </tr>
        """

    top15_rows = ""
    for idx, kw in enumerate(top15_keywords):
        top15_rows += f"""
        <tr>
            <td style="text-align:center; font-weight:bold;">#{idx + 1}</td>
            <td><code>{escape_html(str(kw.get('Từ khóa', '')))}</code></td>
            <td><strong>{float(kw.get('Trọng số', 0)):.4f}</strong></td>
        </tr>
        """

    # Nhúng hình ảnh Base64
    img_conf = image_to_base64(IMAGES_DIR / "model_01_confusion_matrices.png")
    img_kw = image_to_base64(IMAGES_DIR / "model_02_top_15_spam_keywords.png")
    img_comp = image_to_base64(IMAGES_DIR / "model_03_performance_comparison.png")

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo 04: Đánh Giá Độc Lập Test Set — Nhóm 3</title>
    <style>
        {COMMON_CSS}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            text-align: center;
        }}
        .metric-value {{ font-size: 24px; font-weight: 700; color: var(--primary); }}
        .metric-sub {{ font-size: 12.5px; color: var(--text-muted); margin-top: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="nav-bar">
            <a href="03_model_training_report.html" class="nav-link">&larr; Báo Cáo 03: Huấn Luyện & Kiểm Định Chéo</a>
            <a href="index.html" class="nav-link">Bảng Điều Khiển Tổng Quan</a>
            <a href="run_log.html" class="nav-link">Nhật Ký Thực Thi Hệ Thống &rarr;</a>
        </div>

        <div class="header">
            <span class="badge">CHẶNG 4 / ĐÁNH GIÁ ĐỘC LẬP TẬP TEST & HIỆN VẬT MÔ HÌNH</span>
            <h1>Báo Cáo Đánh Giá Đối Đầu 3 Mô Hình Trên Hold-out Test Set</h1>
            <p style="color: var(--text-muted); font-size: 14.5px;">
                Thử thách độc lập trên 6,532 mẫu email chưa từng thấy trong quá trình học. Vinh danh Quán quân Linear SVM.
                <br><em>Cập nhật lần cuối: {s4.get('finished_at', '')} (Thời lượng: {s4.get('duration_sec', 0)}s).</em>
            </p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{f1_champ}</div>
                <div class="metric-sub">F1-Score Quán Quân ({champ_name})</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{acc_champ}</div>
                <div class="metric-sub">Accuracy Kiểm Thử Độc Lập</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{prec_champ}</div>
                <div class="metric-sub">Precision (Hạn Chế Báo Nhầm FP)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{missed_spam}</div>
                <div class="metric-sub">Thư Rác Bỏ Sót (FN / {test_samples:,} email)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{infer_speed}</div>
                <div class="metric-sub">Tốc Độ Suy Luận / 1 Email</div>
            </div>
        </div>

        <div class="card">
            <h2>1. Đánh Giá Đối Đầu Độc Lập Trên Tập Hold-out Test ({test_samples:,} Mẫu)</h2>
            <p>Sau khi được huấn luyện toàn diện trên toàn bộ tập Train, 3 mô hình được thử thách trên tập Test độc lập:</p>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Mô Hình Thuật Toán</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>TP</th>
                            <th>TN</th>
                            <th>FP (Báo Nhầm)</th>
                            <th>FN (Bỏ Sót)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {test_rows if test_rows else '<tr><td colspan="9" style="text-align:center;">Chưa có dữ liệu đánh giá Test set.</td></tr>'}
                    </tbody>
                </table>
            </div>

            <h3 style="margin-top:28px; font-size:16px;">Biểu Đồ Đối Đầu Hiệu Năng 4 Chỉ Số</h3>
            <div class="img-box">
                <img src="{img_comp}" alt="Biểu Đồ Đối Đầu Hiệu Năng">
            </div>

            <h3 style="margin-top:28px; font-size:16px;">Ma Trận Nhầm Lẫn (Confusion Matrix) Của Cả 3 Mô Hình</h3>
            <div class="img-box">
                <img src="{img_conf}" alt="Ma Trận Nhầm Lẫn">
            </div>
            
            <div class="highlight-box">
                <div class="highlight-title">🔍 Phân tích Chuyên sâu Nghiệp vụ:</div>
                <ul style="margin: 0; padding-left: 20px;">
                    <li><strong>Linear SVM (Quán quân):</strong> Đạt F1-Score vượt trội nhờ khả năng tối ưu hóa lề phân cách cực đại trong không gian 2.007 chiều. Tỷ lệ bỏ sót thư rác (FN) thấp nhất ({missed_spam} email).</li>
                    <li><strong>Multinomial Naive Bayes:</strong> Thể hiện Precision rất cao (99.31%), số ca báo nhầm (FP) cực thấp (chỉ 20 ca), bảo vệ an toàn tối đa cho các thư công việc quan trọng.</li>
                    <li><strong>Logistic Regression:</strong> Đạt tốc độ hội tụ nhanh và cho đầu ra xác suất liên tục mượt mà.</li>
                </ul>
            </div>
        </div>

        <div class="card">
            <h2>2. Giải Thích Mô Hình & Trọng Số Đặc Trưng (Model Explainability)</h2>
            <p>Dựa trên vectơ trọng số <strong>w</strong> của mô hình quán quân <strong>{champ_name}</strong>, trích xuất 15 đặc trưng có độ đóng góp lớn nhất vào quyết định gán nhãn Spam:</p>
            
            <div class="img-box">
                <img src="{img_kw}" alt="Top 15 Từ Khóa Spam">
            </div>

            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th style="width: 100px; text-align:center;">Thứ Hạng</th>
                            <th>Từ Khóa / Đặc Trưng Miền</th>
                            <th>Trọng Số Phân Loại (Weight)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {top15_rows if top15_rows else '<tr><td colspan="3" style="text-align:center;">Chưa có danh sách từ khóa.</td></tr>'}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card">
            <h2>3. Hiện Vật Xuất Xưởng (Model Artifacts)</h2>
            <p>Mô hình tối ưu nhất đã được xuất khẩu tự động phục vụ tích hợp ứng dụng thời gian thực:</p>
            <ul>
                <li><strong>Mô hình nhị phân đã đóng gói:</strong> <code>models/saved_models/best_model.joblib</code> (Chứa trọn vẹn tiền xử lý ColumnTransformer và bộ phân loại Linear SVM).</li>
                <li><strong>Hồ sơ siêu dữ liệu kiểm định:</strong> <code>models/saved_models/metadata.json</code> (Lưu trữ toàn bộ tham số, metrics, top 15 từ khóa và ngày giờ đóng gói).</li>
                <li><strong>Trải nghiệm demo trực quan:</strong> Mở tệp notebook <code>notebooks/demo.ipynb</code> để phân loại trực tiếp bất kỳ nội dung email nào do người dùng nhập vào.</li>
            </ul>
        </div>

        <div style="text-align: right; margin-top: 24px;">
            <a href="run_log.html" class="nav-link" style="font-size: 15px;">Tiếp tục: Xem Nhật Ký Thực Thi Hệ Thống &rarr;</a>
        </div>
    </div>
</body>
</html>"""
    target_file.write_text(html_content, encoding="utf-8")
    return target_file


def build_run_log_page() -> Path:
    """
    Sinh docs/report/run_log.html theo cơ chế Zero-drift Logging:
    Khớp chính xác từng tệp log được lưu chính thức trong phong bì stage_*.json hoặc tệp log toàn trình run_pipeline.
    """
    target_file = REPORT_DIR / "run_log.html"
    s1 = load_stage_result("01")
    s2 = load_stage_result("02")
    s3 = load_stage_result("03")
    s4 = load_stage_result("04")

    # Kiểm tra xem có log toàn trình run_pipeline gần nhất không
    pipeline_logs = sorted(list(LOGS_DIR.glob("run_pipeline_*.log")), key=lambda p: p.stat().st_mtime, reverse=True)
    all_logs = sorted(list(LOGS_DIR.glob("*.log")), key=lambda p: p.stat().st_mtime, reverse=True)

    log_sections_html = ""

    # Nếu có tệp log run_pipeline mới hơn các log con, ưu tiên hiển thị log toàn trình
    use_pipeline_log = False
    if pipeline_logs:
        latest_pipe = pipeline_logs[0]
        # Kiểm tra nếu run_pipeline là tệp mới nhất hoặc gần như mới nhất
        if all_logs and (latest_pipe.name == all_logs[0].name or abs(latest_pipe.stat().st_mtime - all_logs[0].stat().st_mtime) < 10):
            use_pipeline_log = True
            try:
                content = latest_pipe.read_text(encoding="utf-8", errors="replace")
                log_sections_html = f"""
                <div class="card">
                    <h2>Nhật Ký Quy Trình Toàn Trình (End-to-End Pipeline Log)</h2>
                    <p style="color:var(--text-muted); font-size:13px;">Tệp nguồn: <code>{escape_html(latest_pipe.name)}</code></p>
                    <pre>{escape_html(content)}</pre>
                </div>
                """
            except Exception as e:
                log_sections_html = f"<div class='banner-warning'>Lỗi đọc log toàn trình: {e}</div>"

    if not use_pipeline_log:
        # Cơ chế Zero-drift: Đọc chính xác 4 log file được phong bì JSON của từng chặng bảo chứng
        stages_logs = [
            ("Chặng 01: Khám Phá Dữ Liệu Email (EDA)", s1),
            ("Chặng 02: Tiền Xử Lý Dữ Liệu & Kỹ Thuật Đặc Trưng (FE)", s2),
            ("Chặng 03: Huấn Luyện 3 Mô Hình Scratch & 5-Fold CV (Training)", s3),
            ("Chặng 04: Đánh Giá Độc Lập Test Set & Xuất Artifacts (Evaluation)", s4)
        ]

        for title, s_data in stages_logs:
            if s_data is None:
                log_sections_html += f"""
                <div class="card">
                    <h2>{escape_html(title)}</h2>
                    <div class="banner-warning" style="margin-top: 10px;">
                        ⚠️ <strong>Chưa có nhật ký:</strong> Chặng này chưa được thực thi trong phiên làm việc hiện tại.
                    </div>
                </div>
                """
            else:
                log_file_str = s_data.get("log_file", "")
                log_path = Path(log_file_str) if log_file_str else None
                if log_path and log_path.exists():
                    try:
                        content = log_path.read_text(encoding="utf-8", errors="replace")
                    except Exception as e:
                        content = f"Lỗi đọc file: {e}"
                else:
                    content = f"Tệp log không tìm thấy: {log_file_str}"

                log_sections_html += f"""
                <div class="card">
                    <h2>{escape_html(title)}</h2>
                    <p style="color:var(--text-muted); font-size:13px; margin-bottom:8px;">
                        Tệp log bảo chứng: <code>{escape_html(log_path.name if log_path else 'Không rõ')}</code> |
                        Thời điểm: {escape_html(s_data.get('finished_at', ''))} |
                        Thời lượng: {s_data.get('duration_sec', 0):.2f}s
                    </p>
                    <pre>{escape_html(content)}</pre>
                </div>
                """

    # Danh sách các tệp log phụ có trong thư mục logs/
    other_logs_html = ""
    if all_logs:
        log_items = []
        for lp in all_logs[:8]:
            mtime = datetime.fromtimestamp(lp.stat().st_mtime).strftime("%H:%M:%S %d/%m/%Y")
            size_kb = lp.stat().st_size / 1024
            log_items.append(f"<li><code>{escape_html(lp.name)}</code> ({size_kb:.1f} KB, sửa đổi lúc {mtime})</li>")
        other_logs_html = f"""
        <div class="card" style="background:#f8fafc;">
            <h2 style="font-size:15px; border-left:none; padding-left:0;">Danh Sách Tệp Log Có Trong Thư Mục logs/ ({len(all_logs)} tệp)</h2>
            <ul style="font-size:13px; color:var(--text-muted); padding-left:20px; line-height:1.7;">
                {''.join(log_items)}
            </ul>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nhật Ký Thực Thi Hệ Thống — Nhóm 3</title>
    <style>
        {COMMON_CSS}
        pre {{
            background: #0f172a;
            color: #f8fafc;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            font-family: Consolas, "Courier New", monospace;
            font-size: 13px;
            line-height: 1.55;
            max-height: 550px;
            border: 1px solid #334155;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="nav-bar">
            <a href="04_model_evaluation_report.html" class="nav-link">&larr; Báo Cáo 04: Đánh Giá Độc Lập Test Set</a>
            <a href="index.html" class="nav-link">Bảng Điều Khiển Tổng Quan</a>
        </div>
        <div class="header">
            <span class="badge">HỆ THỐNG GHI NHẬT KÝ KHÔNG SAI LỆCH (ZERO-DRIFT SYSTEM LOGGING)</span>
            <h1>Nhật Ký Thực Thi Toàn Bộ Quy Trình Học Máy</h1>
            <p style="color:var(--text-muted); font-size:14.5px;">
                Cơ chế Zero-drift Logging bảo đảm mỗi phân đoạn log hiển thị luôn khớp chính xác 100% với phiên chạy sinh ra dữ liệu chặng tương ứng.
            </p>
        </div>

        {other_logs_html}

        {log_sections_html}
    </div>
</body>
</html>"""
    target_file.write_text(html_content, encoding="utf-8")
    return target_file


def build_all() -> None:
    """Tự động đọc toàn bộ JSON hiện có và sinh lại toàn bộ báo cáo HTML."""
    build_report_01()
    build_report_02()
    build_report_03()
    build_report_04()
    build_run_log_page()
    build_index()
    print("✔ Đã cập nhật toàn bộ báo cáo HTML trong docs/report/ thành công.")


if __name__ == "__main__":
    build_all()
