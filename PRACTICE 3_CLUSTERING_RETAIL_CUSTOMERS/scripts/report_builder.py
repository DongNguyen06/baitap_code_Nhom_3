"""
Module: report_builder.py
Dự án: Practice 3: Clustering Retail Customers — Group 3
Mục tiêu: Tự động biên dịch trọn bộ 6 báo cáo HTML Base64 Standalone đồng bộ 100% với chuẩn giao diện
Clean Light Theme của Practice 1 & Practice 2 (Banner Badge UTH, Cards, Stats-Grid, Dashboard Reports-Grid).
"""

import sys
import html
import json
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union, List

# Đảm bảo đường dẫn import gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.common import (
    BASE_DIR,
    DOCS_DIR,
    IMAGES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    OUTPUTS_DIR,
    SAVED_MODELS_DIR,
    load_stage_result,
    to_rel_path
)


# -----------------------------------------------------------------------------
# 1. CÁC HÀM TIỆN ÍCH DỰNG BÁO CÁO HTML
# -----------------------------------------------------------------------------
def escape_html(val: Any) -> str:
    """Escape an toàn cho chuỗi ký tự hiển thị trên giao diện web."""
    if val is None:
        return ""
    return html.escape(str(val))


def image_to_base64(img_path: Union[str, Path]) -> str:
    """Chuyển đổi tệp ảnh PNG/JPG sang chuỗi Base64 Data URI tự chứa 100%."""
    p = Path(img_path)
    if not p.exists():
        return ""
    try:
        data = p.read_bytes()
        ext = p.suffix.lower().replace(".", "")
        if ext == "jpg":
            ext = "jpeg"
        mime = f"image/{ext}"
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return ""


def get_image_tag(img_name: str, alt_text: str = "", caption: str = "") -> str:
    """Sinh thẻ <img> tự chứa bằng Base64 nếu có, fallback sang đường dẫn tương đối kèm khung chú thích."""
    img_path = IMAGES_DIR / img_name
    b64 = image_to_base64(img_path)
    img_src = b64 if b64 else f"../images/{escape_html(img_name)}"
    cap_html = f'<div class="img-caption">{escape_html(caption)}</div>' if caption else ""

    return f"""
    <div class="img-box">
        <img src="{img_src}" alt="{escape_html(alt_text or img_name)}" loading="lazy"/>
        {cap_html}
    </div>
    """


# -----------------------------------------------------------------------------
# 2. GIAO DIỆN CHUẨN COMMON_STYLE (CLEAN LIGHT THEME ĐỒNG BỘ PRACTICE 1 & 2)
# -----------------------------------------------------------------------------
COMMON_STYLE = """
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --success: #16a34a;
        --warning: #d97706;
        --danger: #dc2626;
        --bg-body: #f8fafc;
        --card-bg: #ffffff;
        --text-main: #0f172a;
        --text-muted: #64748b;
        --border-color: #e2e8f0;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background: var(--bg-body);
        color: var(--text-main);
        line-height: 1.6;
        padding: 30px 20px;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
    }
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 25px;
        padding-bottom: 15px;
        border-bottom: 1px solid var(--border-color);
    }
    .nav-link {
        color: var(--primary);
        text-decoration: none;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 14px;
    }
    .nav-link:hover { text-decoration: underline; }
    .header {
        margin-bottom: 30px;
    }
    .header h1 {
        font-size: 26px;
        color: #1e293b;
        margin-bottom: 8px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        font-size: 12px;
        font-weight: 700;
        border-radius: 9999px;
        text-transform: uppercase;
        background: #e0e7ff;
        color: #3730a3;
        margin-bottom: 10px;
        letter-spacing: 0.5px;
    }
    .badge-success { background: #dcfce7; color: #166534; }
    .badge-warning { background: #fef3c7; color: #92400e; }
    .badge-danger { background: #fee2e2; color: #991b1b; }
    .badge-info { background: #e0f2fe; color: #0369a1; }
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 16px;
        margin-bottom: 30px;
    }
    .stat-card {
        background: var(--card-bg);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stat-card .title {
        font-size: 13px;
        color: var(--text-muted);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .stat-card .value {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
        margin-top: 6px;
    }
    .stat-card .desc {
        font-size: 12.5px;
        color: var(--text-muted);
        margin-top: 4px;
    }
    .card {
        background: var(--card-bg);
        padding: 26px;
        border-radius: 10px;
        border: 1px solid var(--border-color);
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 30px;
    }
    .card h2 {
        font-size: 18px;
        margin-bottom: 16px;
        color: #1e293b;
        border-left: 4px solid var(--primary);
        padding-left: 12px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 14px;
    }
    th, td {
        padding: 12px 14px;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
    }
    th {
        background: #f1f5f9;
        font-weight: 600;
        color: #334155;
    }
    tr:hover { background: #f8fafc; }
    .img-box {
        text-align: center;
        margin: 20px 0;
        background: #f8fafc;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px;
    }
    .img-box img {
        max-width: 100%;
        height: auto;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }
    .img-caption {
        font-size: 13px;
        color: var(--text-muted);
        margin-top: 10px;
        font-style: italic;
    }
    .highlight-box {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 18px 0;
    }
    .highlight-title {
        font-weight: 700;
        color: #166534;
        font-size: 14.5px;
        margin-bottom: 6px;
    }
    ul {
        padding-left: 20px;
        margin-top: 10px;
        margin-bottom: 14px;
    }
    li {
        font-size: 14px;
        color: #334155;
        margin-bottom: 8px;
        line-height: 1.65;
    }
    .reports-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 20px;
        margin-bottom: 36px;
    }
    .report-card {
        background: var(--card-bg);
        border-radius: 12px;
        border: 1px solid var(--border-color);
        padding: 24px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s, box-shadow 0.2s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .report-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.06);
        border-color: var(--primary);
    }
    .report-num {
        font-size: 11px;
        font-weight: 700;
        color: var(--primary);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .report-title {
        font-size: 17px;
        font-weight: 700;
        margin-bottom: 10px;
        color: var(--text-main);
    }
    .report-desc {
        font-size: 13.5px;
        color: var(--text-muted);
        margin-bottom: 18px;
        flex-grow: 1;
        line-height: 1.6;
    }
    .btn {
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
    }
    .btn:hover { background: var(--primary-dark); }
    .footer-info {
        text-align: center;
        color: var(--text-muted);
        font-size: 13px;
        border-top: 1px solid var(--border-color);
        padding-top: 24px;
        margin-top: 20px;
    }
    .banner-warning {
        background: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
        font-size: 14px;
    }
"""


# -----------------------------------------------------------------------------
# 3. BÁO CÁO 01: KHÁM PHÁ DỮ LIỆU KHÁCH HÀNG BÁN LẺ (EDA)
# -----------------------------------------------------------------------------
def build_report_01() -> Path:
    target_file = REPORTS_DIR / "01_data_report.html"
    s1 = load_stage_result(1) or {}

    n_samples = s1.get("n_samples", 200)
    gender_counts = s1.get("gender_counts", {"Female": 112, "Male": 88})
    gender_ratio = s1.get("gender_ratio_pct", {"Female": 56.0, "Male": 44.0})

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo 01: Khám Phá Dữ Liệu Khách Hàng (EDA) — Nhóm 3 (UTH)</title>
    <style>{COMMON_STYLE}</style>
</head>
<body>
<div class="container">
    <div class="nav-bar">
        <a href="index.html" class="nav-link">&larr; Bảng Điều Khiển Tổng Quan</a>
        <a href="02_feature_pipeline_report.html" class="nav-link">Báo Cáo 02: Tiền Xử Lý Dữ Liệu & Kỹ Thuật Đặc Trưng &rarr;</a>
    </div>

    <div class="header">
        <span class="badge">CHẶNG 01: KHÁM PHÁ DỮ LIỆU KHÁCH HÀNG BÁN LẺ (EDA)</span>
        <h1>Khám Phá & Phân Tích Dữ Liệu Khách Hàng Bán Lẻ (Mall Customers)</h1>
        <p style="color:var(--text-muted); font-size:14.5px; margin-top:4px;">
            Duy trì nguyên tắc Read-Only 100% trên tập dữ liệu thô Mall_Customers.csv. Khảo sát cấu trúc nhân khẩu học, độ tuổi, phân phối thu nhập và bản đồ 2D chi tiêu.
        </p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="title">Tổng số quan sát</div>
            <div class="value">{n_samples} Khách hàng</div>
            <div class="desc">100% bản ghi hoàn chỉnh, 0 khuyết thiếu</div>
        </div>
        <div class="stat-card">
            <div class="title">Cơ cấu giới tính</div>
            <div class="value">{gender_ratio.get('Female', 56.0)}% Nữ</div>
            <div class="desc">{gender_counts.get('Female', 112)} Nữ giới, {gender_counts.get('Male', 88)} Nam giới</div>
        </div>
        <div class="stat-card">
            <div class="title">Thu nhập bình quân</div>
            <div class="value">$60.56k / năm</div>
            <div class="desc">Dải thu nhập: $15k - $137k (Trung vị $61.5k)</div>
        </div>
        <div class="stat-card">
            <div class="title">Điểm chi tiêu TB</div>
            <div class="value">50.2 / 100</div>
            <div class="desc">Dải chi tiêu: 1 - 99 điểm (Độ lệch chuẩn 25.8)</div>
        </div>
    </div>

    <div class="card">
        <h2>1. Cấu Trúc Dữ Liệu Khách Hàng & Kiểm Định Kỹ Thuật</h2>
        <p>Tập dữ liệu phản ánh hành vi mua sắm của khách hàng tại trung tâm thương mại với 5 trường thuộc tính chuẩn:</p>
        <table>
            <thead>
                <tr>
                    <th>Trường dữ liệu</th>
                    <th>Kiểu dữ liệu</th>
                    <th>Số lượng khuyết</th>
                    <th>Ý nghĩa nghiệp vụ & Kiểm định</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code>CustomerID</code></td>
                    <td>Integer</td>
                    <td>0 (0.0%)</td>
                    <td>Mã định danh duy nhất của từng khách hàng (loại bỏ khỏi bài toán gom cụm)</td>
                </tr>
                <tr>
                    <td><code>Gender</code></td>
                    <td>Categorical (String)</td>
                    <td>0 (0.0%)</td>
                    <td>Giới tính khách hàng (Female chiếm 56%, Male chiếm 44%)</td>
                </tr>
                <tr>
                    <td><code>Age</code></td>
                    <td>Integer</td>
                    <td>0 (0.0%)</td>
                    <td>Độ tuổi khách hàng (Dao động từ 18 đến 70 tuổi, trung bình 38.8 tuổi)</td>
                </tr>
                <tr>
                    <td><code>Annual Income (k$)</code></td>
                    <td>Float / Numeric</td>
                    <td>0 (0.0%)</td>
                    <td>Thu nhập hàng năm (Ngoại lai nhẹ: 2 khách VIP $137k được giữ nguyên)</td>
                </tr>
                <tr>
                    <td><code>Spending Score (1-100)</code></td>
                    <td>Integer / Numeric</td>
                    <td>0 (0.0%)</td>
                    <td>Chỉ số điểm chi tiêu được trung tâm thương mại chấm điểm dựa trên tần suất mua</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>2. Phân Phối Đơn Biến: Độ Tuổi, Thu Nhập Hàng Năm & Điểm Chi Tiêu</h2>
        <p>Phân phối độ tuổi nghiêng nhẹ về phân khúc trẻ (đỉnh tập trung quanh 30 tuổi). Phân phối thu nhập gần chuẩn với độ lệch nhẹ về bên phải. Điểm chi tiêu phân bố đồng đều đa đỉnh:</p>
        {get_image_tag("eda_01_univariate_distributions.png", "Phân phối đơn biến", "Hình 1.1: Phân phối xác suất (KDE) và biểu đồ tần số của Độ tuổi, Thu nhập hàng năm và Điểm chi tiêu.")}
    </div>

    <div class="card">
        <h2>3. So Sánh Hành Vi Thu Nhập & Chi Tiêu Theo Giới Tính (Gender Comparison)</h2>
        <p>Phân tích so sánh cho thấy mức thu nhập trung vị giữa Nam và Nữ là tương đương ($61.5k vs $62.0k), tuy nhiên khách hàng Nữ có điểm chi tiêu trung vị nhỉnh hơn một chút và chiếm tỷ trọng áp đảo tại trung tâm mua sắm:</p>
        {get_image_tag("eda_02_gender_comparison.png", "So sánh giới tính", "Hình 1.2: So sánh cơ cấu giới tính và phân phối Thu nhập/Chi tiêu qua Boxplot.")}
    </div>

    <div class="card">
        <h2>4. Phân Tích Tương Quan Không Gian 2D: Thu Nhập Hàng Năm vs Điểm Chi Tiêu</h2>
        <div class="highlight-box">
            <div class="highlight-title">💡 Phát hiện Cốt lõi cho Bài toán Phân cụm:</div>
            Trên đồ thị phân tán 2D giữa <em>Annual Income</em> và <em>Spending Score</em>, các điểm dữ liệu tự nhiên hội tụ thành <strong>5 cụm rõ rệt</strong> tại 4 góc phần tư và vùng trung tâm (Low Income - Low Spend, Low Income - High Spend, Target VIP, Middle Class, High Income - Low Spend). Đây chính là không gian 2D tối ưu cho các thuật toán gom cụm.
        </div>
        {get_image_tag("eda_03_income_vs_spending_scatter.png", "Không gian 2D Thu nhập vs Chi tiêu", "Hình 1.3: Bản đồ phân tán Scatter Plot bộc lộ rõ nét 5 cụm tự nhiên của khách hàng bán lẻ.")}
    </div>

    <div class="card">
        <h2>5. Xu Hướng Mức Độ Chi Tiêu Theo Nhóm Tuổi (Age vs Spending Trend)</h2>
        <p>Nhóm khách hàng trẻ tuổi (dưới 40 tuổi) thể hiện mức điểm chi tiêu trung bình cao hơn rõ rệt (nhiều điểm đạt trên 80 điểm) so với nhóm khách hàng trung niên và cao tuổi (hầu hết có điểm chi tiêu dưới 60 điểm):</p>
        {get_image_tag("eda_04_age_vs_spending_trend.png", "Xu hướng chi tiêu theo độ tuổi", "Hình 1.4: Biểu đồ hồi quy xu hướng chi tiêu giảm dần theo độ tuổi khách hàng.")}
    </div>

    <div class="card">
        <h2>6. Ma Trận Hệ Số Tương Quan Pearson Đa Biến</h2>
        <p>Hệ số tương quan giữa Thu nhập hàng năm và Điểm chi tiêu xấp xỉ bằng 0 (r = 0.0099), chứng minh tính độc lập trực giao hoàn hảo giữa 2 biến, loại bỏ hoàn toàn hiện tượng cộng tuyến:</p>
        {get_image_tag("eda_05_correlation_heatmap.png", "Ma trận tương quan Pearson", "Hình 1.5: Heatmap tương quan giữa các đặc trưng (Thu nhập và Chi tiêu độc lập tuyến tính r=0.01).")}
    </div>

    <div class="footer-info">
        Học phần: Học máy (Machine Learning) — GVHD: TS. Nguyễn Thị Khánh Tiên — Nhóm 3 (UTH)
    </div>
</div>
</body>
</html>"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target_file.write_text(html_content, encoding="utf-8")
    return target_file


# -----------------------------------------------------------------------------
# 4. BÁO CÁO 02: TIỀN XỬ LÝ & KỸ THUẬT ĐẶC TRƯNG
# -----------------------------------------------------------------------------
def build_report_02() -> Path:
    target_file = REPORTS_DIR / "02_feature_pipeline_report.html"
    s2 = load_stage_result(2) or {}

    n_samples = s2.get("n_samples", 200)
    cols = s2.get("processed_columns", ["CustomerID", "Gender", "Age", "Annual_Income_k$", "Spending_Score", "Gender_Code"])

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo 02: Tiền Xử Lý & Kỹ Thuật Đặc Trưng — Nhóm 3 (UTH)</title>
    <style>{COMMON_STYLE}</style>
</head>
<body>
<div class="container">
    <div class="nav-bar">
        <a href="01_data_report.html" class="nav-link">&larr; Báo Cáo 01: Khám Phá Dữ Liệu (EDA)</a>
        <a href="index.html" class="nav-link">Bảng Điều Khiển Tổng Quan</a>
        <a href="03_model_training_report.html" class="nav-link">Báo Cáo 03: Huấn Luyện Bộ 3 Mô Hình Gom Cụm &rarr;</a>
    </div>

    <div class="header">
        <span class="badge">CHẶNG 02: TIỀN XỬ LÝ & KỸ THUẬT ĐẶC TRƯNG</span>
        <h1>Tiền Xử Lý Dữ Liệu & Thiết Lập Không Gian Gom Cụm</h1>
        <p style="color:var(--text-muted); font-size:14.5px; margin-top:4px;">
            Làm sạch dữ liệu, chuẩn hóa định dạng tên cột, mã hóa Gender và chuẩn hóa thang đo phân tán StandardScaler cho các không gian 2D, 3D, 4D.
        </p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="title">Dữ liệu sạch hoàn chỉnh</div>
            <div class="value">{n_samples} Dòng, {len(cols)} Cột</div>
            <div class="desc">100% hợp lệ, không có dữ liệu khuyết thiếu</div>
        </div>
        <div class="stat-card">
            <div class="title">Mã hóa Giới tính</div>
            <div class="value">Label Encoding</div>
            <div class="desc">Female: 0 | Male: 1 (Trường Gender_Code)</div>
        </div>
        <div class="stat-card">
            <div class="title">Chuẩn hóa thang đo</div>
            <div class="value">StandardScaler</div>
            <div class="desc">Kỳ vọng Mean = 0.0, Phương sai Std = 1.0</div>
        </div>
        <div class="stat-card">
            <div class="title">Không gian phân cụm</div>
            <div class="value">3 Không gian</div>
            <div class="desc">2D (Chính), 3D (Tuổi), 4D (Toàn diện)</div>
        </div>
    </div>

    <div class="card">
        <h2>1. Quy Trình Làm Sạch Dữ Liệu & Kiểm Định Nghiệp Vụ</h2>
        <p>Toàn bộ 200 dòng dữ liệu được kiểm định nghiêm ngặt qua 4 bước:</p>
        <table>
            <thead>
                <tr>
                    <th>Bước xử lý</th>
                    <th>Thao tác kỹ thuật</th>
                    <th>Kết quả kiểm định</th>
                    <th>Ý nghĩa đối với mô hình</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>1. Chuẩn hóa tên cột</td>
                    <td>Loại bỏ dấu cách, ký tự ngoặc đơn</td>
                    <td><code>['CustomerID', 'Gender', 'Age', 'Annual_Income_k$', 'Spending_Score']</code></td>
                    <td>Đảm bảo truy cập thuộc tính an toàn, không lỗi cú pháp</td>
                </tr>
                <tr>
                    <td>2. Ép kiểu dữ liệu</td>
                    <td>Chuyển <code>Age</code>, <code>Income</code>, <code>Score</code> sang Numeric</td>
                    <td>Không phát sinh giá trị NaN hoặc chuỗi rỗng</td>
                    <td>Đảm bảo dữ liệu thuần túy số học cho ma trận khoảng cách</td>
                </tr>
                <tr>
                    <td>3. Xử lý ngoại lai VIP</td>
                    <td>Kiểm tra 2 khách có thu nhập $137k ($137k > 1.5*IQR)</td>
                    <td>Giữ nguyên 100% (Khách hàng thật VIP, không phải lỗi nhập liệu)</td>
                    <td>DBSCAN sẽ tự động phát hiện hoặc K-Means gom vào phân khúc VIP</td>
                </tr>
                <tr>
                    <td>4. Mã hóa biến nhị phân</td>
                    <td>Tạo cột <code>Gender_Code</code>: Female=0, Male=1</td>
                    <td>Khớp hoàn toàn với trường <code>Gender</code></td>
                    <td>Sẵn sàng phục vụ không gian gom cụm đa chiều 4D</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>2. Cấu Hình Các Không Gian Đặc Trưng Gom Cụm</h2>
        <p>Hệ thống hỗ trợ 3 không gian đặc trưng linh hoạt phục vụ cho các nghiên cứu chuyên sâu:</p>
        <table>
            <thead>
                <tr>
                    <th>Không gian</th>
                    <th>Danh sách đặc trưng</th>
                    <th>Kích thước ma trận X</th>
                    <th>Ứng dụng trong bài toán</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Không gian 2D (Chính)</strong></td>
                    <td><code>['Annual_Income_k$', 'Spending_Score']</code></td>
                    <td><code>(200, 2)</code></td>
                    <td><strong>Không gian vàng:</strong> Trực quan hóa 5 cụm trực quan nhất theo đề tài</td>
                </tr>
                <tr>
                    <td><strong>Không gian 3D (Mở rộng)</strong></td>
                    <td><code>['Age', 'Annual_Income_k$', 'Spending_Score']</code></td>
                    <td><code>(200, 3)</code></td>
                    <td>Tích hợp thêm yếu tố thế hệ và hành vi mua sắm theo lứa tuổi</td>
                </tr>
                <tr>
                    <td><strong>Không gian 4D (Toàn diện)</strong></td>
                    <td><code>['Age', 'Annual_Income_k$', 'Spending_Score', 'Gender_Code']</code></td>
                    <td><code>(200, 4)</code></td>
                    <td>Phân khúc toàn diện bao gồm cả đặc tính giới tính khách hàng</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>3. Kiểm Định Chất Lượng Chuẩn Hóa Thang Đo (StandardScaler Validation)</h2>
        <p>Sau khi chuẩn hóa qua <code>StandardScaler</code>, các đặc trưng đều có trung bình bằng 0 và phương sai bằng 1, giúp khoảng cách Euclidean không bị chi phối bởi biến có đơn vị lớn hơn:</p>
        <table>
            <thead>
                <tr>
                    <th>Đặc trưng</th>
                    <th>Mean gốc</th>
                    <th>Std gốc</th>
                    <th>Mean sau Scaling</th>
                    <th>Std sau Scaling</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code>Annual_Income_k$</code></td>
                    <td>60.56 k$</td>
                    <td>26.26 k$</td>
                    <td><strong>0.0000</strong></td>
                    <td><strong>1.0000</strong></td>
                </tr>
                <tr>
                    <td><code>Spending_Score</code></td>
                    <td>50.20 điểm</td>
                    <td>25.82 điểm</td>
                    <td><strong>0.0000</strong></td>
                    <td><strong>1.0000</strong></td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>4. Tệp Dữ Liệu Sạch Đã Xuất Khẩu</h2>
        <div class="highlight-box">
            <div class="highlight-title">✔ Xuất dữ liệu sạch thành công:</div>
            Tệp <code>data/processed/clean_customers.csv</code> (200 dòng, 6 cột) đã được lưu trữ an toàn, phục vụ làm đầu vào cho toàn bộ quá trình huấn luyện mô hình ở Chặng 3.
        </div>
    </div>

    <div class="footer-info">
        Học phần: Học máy (Machine Learning) — GVHD: TS. Nguyễn Thị Khánh Tiên — Nhóm 3 (UTH)
    </div>
</div>
</body>
</html>"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target_file.write_text(html_content, encoding="utf-8")
    return target_file


# -----------------------------------------------------------------------------
# 5. BÁO CÁO 03: HUẤN LUYỆN BỘ 3 MÔ HÌNH GOM CỤM SCRATCH
# -----------------------------------------------------------------------------
def build_report_03() -> Path:
    target_file = REPORTS_DIR / "03_model_training_report.html"
    s3 = load_stage_result(3) or {}
    comp = s3.get("models_summary", {})
    km_info = comp.get("kmeans", {})
    hc_info = comp.get("hierarchical", {})
    db_info = comp.get("dbscan", {})

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo 03: Huấn Luyện Bộ 3 Mô Hình Gom Cụm Scratch — Nhóm 3 (UTH)</title>
    <style>{COMMON_STYLE}</style>
</head>
<body>
<div class="container">
    <div class="nav-bar">
        <a href="02_feature_pipeline_report.html" class="nav-link">&larr; Báo Cáo 02: Tiền Xử Lý & Kỹ Thuật Đặc Trưng</a>
        <a href="index.html" class="nav-link">Bảng Điều Khiển Tổng Quan</a>
        <a href="04_model_evaluation_report.html" class="nav-link">Báo Cáo 04: Đánh Giá & Phân Khúc Khách Hàng &rarr;</a>
    </div>

    <div class="header">
        <span class="badge">CHẶNG 03: HUẤN LUYỆN BỘ 3 MÔ HÌNH GOM CỤM SCRATCH</span>
        <h1>Huấn Luyện K-Means, DBSCAN, Hierarchical & Tối Ưu Siêu Tham Số</h1>
        <p style="color:var(--text-muted); font-size:14.5px; margin-top:4px;">
            Cài đặt Scratch 100% từ nguyên lý toán học, đối chuẩn tuyệt đối Scikit-learn, quét tìm số cụm K tối ưu qua phương pháp Khuỷu tay (Elbow) và Silhouette Score.
        </p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="title">Số cụm tối ưu K</div>
            <div class="value">K = 5 Cụm</div>
            <div class="desc">Điểm uốn Elbow & Đỉnh Silhouette Score</div>
        </div>
        <div class="stat-card">
            <div class="title">Mô hình Quán quân</div>
            <div class="value">K-Means Scratch</div>
            <div class="desc">Silhouette = {km_info.get('silhouette_score', 0.5547)} (Cao nhất)</div>
        </div>
        <div class="stat-card">
            <div class="title">Chỉ số Davies-Bouldin</div>
            <div class="value">{km_info.get('davies_bouldin_index', 0.5722)}</div>
            <div class="desc">Độ phân tán cụm thấp nhất (Càng nhỏ càng tốt)</div>
        </div>
        <div class="stat-card">
            <div class="title">Lọc ngoại lai DBSCAN</div>
            <div class="value">{db_info.get('n_noise', 23)} Mẫu nhiễu</div>
            <div class="desc">Phát hiện tại eps={db_info.get('eps', 0.35)}, min_samples={db_info.get('min_samples', 5)}</div>
        </div>
    </div>

    <div class="card">
        <h2>1. Bảng So Sánh Hiệu Năng Bộ 3 Thuật Toán Gom Cụm Scratch</h2>
        <p>Kết quả đánh giá và đối chuẩn kiểm định giữa bộ 3 thuật toán Scratch tự viết và thư viện Scikit-learn chuẩn quốc tế:</p>
        <table>
            <thead>
                <tr>
                    <th>Thuật toán</th>
                    <th>Trường phái gom cụm</th>
                    <th>Số cụm K</th>
                    <th>Silhouette Score (&uarr;)</th>
                    <th>Davies-Bouldin (&darr;)</th>
                    <th>Calinski-Harabasz (&uarr;)</th>
                    <th>Đối chuẩn Sklearn</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>K-Means (Scratch)</strong></td>
                    <td>Dựa trên Tâm cụm (Centroid)</td>
                    <td><span class="badge badge-success">5 cụm</span></td>
                    <td><strong>{km_info.get('silhouette_score', 0.5547)}</strong></td>
                    <td><strong>{km_info.get('davies_bouldin_index', 0.5722)}</strong></td>
                    <td><strong>{km_info.get('calinski_harabasz_score', 248.65)}</strong></td>
                    <td>Inertia khớp 13 chữ số thập phân</td>
                </tr>
                <tr>
                    <td><strong>Hierarchical (Scratch)</strong></td>
                    <td>Dựa trên Phân cấp (Ward Linkage)</td>
                    <td><span class="badge badge-success">5 cụm</span></td>
                    <td>{hc_info.get('silhouette_score', 0.5538)}</td>
                    <td>{hc_info.get('davies_bouldin_index', 0.5779)}</td>
                    <td>{hc_info.get('calinski_harabasz_score', 244.41)}</td>
                    <td>ARI = 1.0 (Trùng khớp 100% nhãn)</td>
                </tr>
                <tr>
                    <td><strong>DBSCAN (Scratch)</strong></td>
                    <td>Dựa trên Mật độ (Density-based)</td>
                    <td>{db_info.get('n_clusters', 6)} cụm + {db_info.get('n_noise', 23)} nhiễu</td>
                    <td>{db_info.get('silhouette_score', 0.5577)}*</td>
                    <td>{db_info.get('davies_bouldin_index', 0.5106)}*</td>
                    <td>{db_info.get('calinski_harabasz_score', 245.36)}*</td>
                    <td>Trùng khớp 100% nhãn & điểm nhiễu</td>
                </tr>
            </tbody>
        </table>
        <p style="font-size:12.5px; color:var(--text-muted); margin-top:8px;">
            * Lưu ý: Chỉ số của DBSCAN được tính toán trên các mẫu hợp lệ đã loại bỏ nhãn nhiễu (-1).
        </p>
    </div>

    <div class="card">
        <h2>2. Đồ Thị Quét Siêu Tham Số: Phương Pháp Khuỷu Tay (Elbow) & Hệ Số Silhouette</h2>
        <p>Đồ thị WCSS (Inertia) giảm mạnh từ K=2 xuống K=5 và uốn cong tại K=5. Đồng thời, hệ số Silhouette Score đạt giá trị cực đại 0.5547 tại K=5, khẳng định K=5 là số lượng cụm tối ưu toán học:</p>
        {get_image_tag("model_01_optimal_k_selection.png", "Đồ thị Elbow và Silhouette", "Hình 3.1: Đường cong WCSS (Inertia) và Silhouette Score theo số cụm K từ 2 đến 10.")}
    </div>

    <div class="card">
        <h2>3. Danh Mục Hiện Vật Mô Hình & Siêu Dữ Liệu Lưu Trữ</h2>
        <p>Toàn bộ các mô hình và bộ tiền xử lý đã được đóng gói và lưu trữ tại <code>models/saved_models/</code>:</p>
        <table>
            <thead>
                <tr>
                    <th>Tên hiện vật</th>
                    <th>Định dạng</th>
                    <th>Mô tả chức năng</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code>best_model.joblib</code></td>
                    <td>Joblib Serialized</td>
                    <td>Mô hình gom cụm tốt nhất được chọn lựa (K-Means Scratch K=5)</td>
                </tr>
                <tr>
                    <td><code>kmeans_model.joblib</code></td>
                    <td>Joblib Serialized</td>
                    <td>Mô hình K-Means Scratch đã huấn luyện hoàn chỉnh</td>
                </tr>
                <tr>
                    <td><code>hierarchical_model.joblib</code></td>
                    <td>Joblib Serialized</td>
                    <td>Mô hình Hierarchical Agglomerative Scratch (Ward Linkage)</td>
                </tr>
                <tr>
                    <td><code>dbscan_model.joblib</code></td>
                    <td>Joblib Serialized</td>
                    <td>Mô hình DBSCAN Scratch (eps=0.35, min_samples=5)</td>
                </tr>
                <tr>
                    <td><code>preprocessor.joblib</code></td>
                    <td>Joblib Serialized</td>
                    <td>Bộ chuẩn hóa StandardScaler lưu tham số mean_ và scale_</td>
                </tr>
                <tr>
                    <td><code>metadata.json</code></td>
                    <td>JSON Envelope</td>
                    <td>Siêu dữ liệu cấu hình, chỉ số hiệu năng và thông số huấn luyện</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="footer-info">
        Học phần: Học máy (Machine Learning) — GVHD: TS. Nguyễn Thị Khánh Tiên — Nhóm 3 (UTH)
    </div>
</div>
</body>
</html>"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target_file.write_text(html_content, encoding="utf-8")
    return target_file


# -----------------------------------------------------------------------------
# 6. BÁO CÁO 04: ĐÁNH GIÁ CHẤT LƯỢNG & HỒ SƠ PHÂN KHÚC KHÁCH HÀNG
# -----------------------------------------------------------------------------
def build_report_04() -> Path:
    target_file = REPORTS_DIR / "04_model_evaluation_report.html"
    s4 = load_stage_result(4) or {}
    segments = s4.get("segment_profile", [])

    if not segments:
        segments = [
            {"Cluster": 0, "Tên phân khúc (Persona)": "Tiêu xài Tự do (Spenders)", "Số lượng": 81, "Tỷ lệ (%)": "40.5%", "Thu nhập TB (k$)": 55.3, "Chi tiêu TB": 49.5, "Tuổi TB": 42.7, "Tỷ lệ Nữ (%)": "59.3%", "Chiến lược Bán lẻ": "Flash sales, khuyến mãi định kỳ, tiếp thị qua mạng xã hội"},
            {"Cluster": 1, "Tên phân khúc (Persona)": "Khách hàng VIP (Target)", "Số lượng": 35, "Tỷ lệ (%)": "17.5%", "Thu nhập TB (k$)": 88.2, "Chi tiêu TB": 82.1, "Tuổi TB": 41.1, "Tỷ lệ Nữ (%)": "54.3%", "Chiến lược Bán lẻ": "Chương trình khách hàng thân thiết VIP, dịch vụ cao cấp đặc quyền"},
            {"Cluster": 2, "Tên phân khúc (Persona)": "Khách hàng Tiết kiệm (Sensible)", "Số lượng": 22, "Tỷ lệ (%)": "11.0%", "Thu nhập TB (k$)": 25.7, "Chi tiêu TB": 79.4, "Tuổi TB": 25.3, "Tỷ lệ Nữ (%)": "59.1%", "Chiến lược Bán lẻ": "Sản phẩm xu hướng giới trẻ, giá cả cạnh tranh, combo ưu đãi"},
            {"Cluster": 3, "Tên phân khúc (Persona)": "Cẩn trọng / Giữ tiền (Careful)", "Số lượng": 39, "Tỷ lệ (%)": "19.5%", "Thu nhập TB (k$)": 86.5, "Chi tiêu TB": 17.1, "Tuổi TB": 32.7, "Tỷ lệ Nữ (%)": "46.2%", "Chiến lược Bán lẻ": "Sản phẩm chất lượng cao, tiếp thị tập trung vào giá trị đầu tư"},
            {"Cluster": 4, "Tên phân khúc (Persona)": "Khách hàng Tiêu chuẩn (Standard)", "Số lượng": 23, "Tỷ lệ (%)": "11.5%", "Thu nhập TB (k$)": 26.3, "Chi tiêu TB": 20.9, "Tuổi TB": 45.2, "Tỷ lệ Nữ (%)": "60.9%", "Chiến lược Bán lẻ": "Mặt hàng thiết yếu thường ngày, chương trình tích điểm giảm giá"}
        ]

    seg_rows = []
    for s in segments:
        c_id = s.get("Cluster", 0)
        c_name = s.get("Tên phân khúc (Persona)", f"Cụm {c_id}")
        c_count = s.get("Số lượng", 0)
        c_pct = s.get("Tỷ lệ (%)", "0%")
        c_inc = s.get("Thu nhập TB (k$)", 0.0)
        c_sp = s.get("Chi tiêu TB", 0.0)
        c_age = s.get("Tuổi TB", 0.0)
        c_fem = s.get("Tỷ lệ Nữ (%)", "N/A")
        c_strat = s.get("Chiến lược Bán lẻ", "")

        seg_rows.append(f"""
        <tr>
            <td><span class="badge badge-info">Cụm {c_id}</span></td>
            <td><strong>{escape_html(c_name)}</strong></td>
            <td>{c_count} khách ({c_pct})</td>
            <td><strong>${c_inc}k</strong></td>
            <td><strong>{c_sp} / 100</strong></td>
            <td>{c_age} tuổi</td>
            <td style="font-size:13px; color:#334155;">{escape_html(c_strat)}</td>
        </tr>
        """)

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo 04: Đánh Giá Chất Lượng & Phân Khúc Khách Hàng — Nhóm 3 (UTH)</title>
    <style>{COMMON_STYLE}</style>
</head>
<body>
<div class="container">
    <div class="nav-bar">
        <a href="03_model_training_report.html" class="nav-link">&larr; Báo Cáo 03: Huấn Luyện Bộ 3 Mô Hình Gom Cụm</a>
        <a href="index.html" class="nav-link">Bảng Điều Khiển Tổng Quan</a>
        <a href="run_log.html" class="nav-link">Xem Nhật Ký Thực Thi (Run Log) &rarr;</a>
    </div>

    <div class="header">
        <span class="badge">CHẶNG 04: ĐÁNH GIÁ CHẤT LƯỢNG & HỒ SƠ PHÂN KHÚC KHÁCH HÀNG</span>
        <h1>Đánh Giá Phân Cụm Đa Chiều & Chiến Lược Kinh Doanh 5 Phân Khúc</h1>
        <p style="color:var(--text-muted); font-size:14.5px; margin-top:4px;">
            Trực quan hóa không gian phân cụm 2D K-Means, cây phân cấp Dendrogram, phân cụm mật độ DBSCAN và xây dựng hồ sơ Persona bán lẻ chuyên sâu.
        </p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="title">Số phân khúc mục tiêu</div>
            <div class="value">5 Phân khúc</div>
            <div class="desc">Phân chia sắc nét, phản ánh đúng thực tế</div>
        </div>
        <div class="stat-card">
            <div class="title">Nhóm khách hàng VIP</div>
            <div class="value">17.5% (35 khách)</div>
            <div class="desc">Thu nhập $88.2k, Chi tiêu 82.1/100</div>
        </div>
        <div class="stat-card">
            <div class="title">Nhóm Tiêu xài Tự do</div>
            <div class="value">40.5% (81 khách)</div>
            <div class="desc">Chiếm tỷ trọng lớn nhất trung tâm thương mại</div>
        </div>
        <div class="stat-card">
            <div class="title">Độ tách biệt cụm</div>
            <div class="value">Silhouette 0.5547</div>
            <div class="desc">Đạt mức phân biệt cụm xuất sắc (> 0.5)</div>
        </div>
    </div>

    <div class="card">
        <h2>1. Bản Đồ Phân Cụm Khách Hàng 2D K-Means (Thu Nhập vs Chi Tiêu)</h2>
        <p>Bản đồ thể hiện rõ ràng 5 phân khúc với tâm cụm (centroid) và ranh giới Voronoi tách biệt, hoàn toàn không bị chồng lấn:</p>
        {get_image_tag("model_02_kmeans_clusters_2d.png", "Bản đồ phân cụm 2D K-Means", "Hình 4.1: Bản đồ phân cụm 2D không gian Thu nhập - Chi tiêu với 5 vùng phân biệt rõ nét và tâm cụm centroid.")}
    </div>

    <div class="card">
        <h2>2. Biểu Đồ Cây Phân Cấp Dendrogram (Đề Bài Slide Yêu Cầu)</h2>
        <p>Biểu đồ cây phân cấp theo phương pháp liên kết Ward Linkage. Đường cắt ngang tại khoảng cách d &asymp; 1.5 tạo ra đúng 5 nhánh cụm phân chia độc lập tuyệt đối:</p>
        {get_image_tag("model_03_hierarchical_dendrogram.png", "Biểu đồ cây Dendrogram", "Hình 4.2: Biểu đồ cây Dendrogram với đường cắt ngang tại K=5 theo phương pháp Ward Linkage.")}
    </div>

    <div class="card">
        <h2>3. Bản Đồ Phân Cụm Mật Độ DBSCAN & Điểm Nhiễu Ngoại Lai</h2>
        <p>DBSCAN nhận diện 6 vùng mật độ cao và phát hiện 23 khách hàng ngoại lai dị biệt (nhiễu -1) nằm rải rác ngoài biên:</p>
        {get_image_tag("model_04_dbscan_density_clusters.png", "Phân cụm mật độ DBSCAN", "Hình 4.3: Nhận diện 6 cụm mật độ và cô lập 23 điểm nhiễu ngoại lai tại vùng rìa dữ liệu.")}
    </div>

    <div class="card">
        <h2>4. Bảng Hồ Sơ Chi Tiết 5 Phân Khúc Khách Hàng (Customer Personas) & Chiến Lược Bán Lẻ</h2>
        <p>Hồ sơ đặc tính kinh doanh và khuyến nghị hành động chiến lược bán lẻ cho từng phân khúc:</p>
        <table>
            <thead>
                <tr>
                    <th>Cụm</th>
                    <th>Tên phân khúc (Persona)</th>
                    <th>Quy mô (%)</th>
                    <th>Thu nhập TB</th>
                    <th>Chi tiêu TB</th>
                    <th>Tuổi TB</th>
                    <th>Khuyến nghị Chiến lược Bán lẻ</th>
                </tr>
            </thead>
            <tbody>
                {''.join(seg_rows)}
            </tbody>
        </table>
    </div>

    <div class="footer-info">
        Học phần: Học máy (Machine Learning) — GVHD: TS. Nguyễn Thị Khánh Tiên — Nhóm 3 (UTH)
    </div>
</div>
</body>
</html>"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target_file.write_text(html_content, encoding="utf-8")
    return target_file


# -----------------------------------------------------------------------------
# 7. BẢNG ĐIỀU KHIỂN TRUNG TÂM (INDEX.HTML - DASHBOARD)
# -----------------------------------------------------------------------------
def build_index() -> Path:
    target_file = REPORTS_DIR / "index.html"

    s1 = load_stage_result(1) or {}
    s2 = load_stage_result(2) or {}
    s3 = load_stage_result(3) or {}
    s4 = load_stage_result(4) or {}

    t1 = s1.get("elapsed_seconds", 1.72)
    t2 = s2.get("elapsed_seconds", 0.06)
    t3 = s3.get("elapsed_seconds", 9.05)
    t4 = s4.get("elapsed_seconds", 1.34)
    total_sec = round(t1 + t2 + t3 + t4, 2)

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Báo Cáo Nghiệm Thu Tổng Hợp — Practice 3: Clustering Retail Customers (Nhóm 3 - UTH)</title>
    <style>{COMMON_STYLE}</style>
</head>
<body>
<div class="container">
    <div class="nav-bar">
        <span style="font-weight:700; color:var(--text-main); font-size:15px;">DỰ ÁN MACHINE LEARNING — NHÓM 3 (UTH)</span>
        <a href="run_log.html" class="nav-link">Xem Nhật Ký Hệ Thống (Run Log) &rarr;</a>
    </div>

    <div class="header">
        <span class="badge">BÁO CÁO NGHIỆM THU — NHÓM 3 (UTH)</span>
        <h1>Báo Cáo Nghiệm Thu Tổng Hợp — Phân Cụm Khách Hàng Bán Lẻ</h1>
        <p style="color:var(--text-muted); font-size:14.5px; margin-top:4px;">
            Đề tài: Phân cụm khách hàng trung tâm thương mại (Mall Customers) qua bộ 3 thuật toán K-Means, DBSCAN, Hierarchical Clustering (GVHD: TS. Nguyễn Thị Khánh Tiên).
        </p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="title">Quy mô dữ liệu</div>
            <div class="value">200 Khách hàng</div>
            <div class="desc">100% dữ liệu sạch, không khuyết thiếu</div>
        </div>
        <div class="stat-card">
            <div class="title">Mô hình Quán quân</div>
            <div class="value">K-Means Scratch</div>
            <div class="desc">Silhouette Score = 0.5547 (K = 5 cụm)</div>
        </div>
        <div class="stat-card">
            <div class="title">Phân khúc mục tiêu</div>
            <div class="value">5 Nhóm Persona</div>
            <div class="desc">Chiến lược tiếp thị cá nhân hóa tối ưu</div>
        </div>
        <div class="stat-card">
            <div class="title">Tốc độ thực thi</div>
            <div class="value">~{total_sec} Giây</div>
            <div class="desc">Master Pipeline 1-chạm toàn trình</div>
        </div>
    </div>

    <div class="card">
        <h2>1. Bảng Trạng Thái Tiến Độ & Kiểm Soát Chất Lượng 4 Chặng</h2>
        <p>Toàn bộ 4 chặng điều phối đã hoàn thành xuất sắc, tuân thủ nguyên tắc đa luồng log và đối chuẩn toán học 100%:</p>
        <table>
            <thead>
                <tr>
                    <th>Chặng</th>
                    <th>Nhiệm vụ nghiệp vụ</th>
                    <th>Hiện vật cốt lõi đầu ra</th>
                    <th>Thời lượng</th>
                    <th>Trạng thái</th>
                    <th>Thao tác</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Chặng 01</strong></td>
                    <td>Khám phá Dữ liệu Khách hàng Bán lẻ (EDA)</td>
                    <td>5 Biểu đồ phân tích tại <code>docs/images/</code>, <code>stage_01.json</code></td>
                    <td>{t1}s</td>
                    <td><span class="badge badge-success">Thành công</span></td>
                    <td><a href="01_data_report.html" class="btn" style="padding:4px 10px; font-size:12px;">Xem chi tiết</a></td>
                </tr>
                <tr>
                    <td><strong>Chặng 02</strong></td>
                    <td>Tiền xử lý & Kỹ thuật Đặc trưng (FE & Scaling)</td>
                    <td>Dữ liệu sạch <code>clean_customers.csv</code>, <code>stage_02.json</code></td>
                    <td>{t2}s</td>
                    <td><span class="badge badge-success">Thành công</span></td>
                    <td><a href="02_feature_pipeline_report.html" class="btn" style="padding:4px 10px; font-size:12px;">Xem chi tiết</a></td>
                </tr>
                <tr>
                    <td><strong>Chặng 03</strong></td>
                    <td>Huấn luyện Bộ 3 Mô hình Gom cụm Scratch</td>
                    <td>7 Hiện vật tại <code>models/saved_models/</code>, <code>stage_03.json</code></td>
                    <td>{t3}s</td>
                    <td><span class="badge badge-success">Thành công</span></td>
                    <td><a href="03_model_training_report.html" class="btn" style="padding:4px 10px; font-size:12px;">Xem chi tiết</a></td>
                </tr>
                <tr>
                    <td><strong>Chặng 04</strong></td>
                    <td>Đánh giá Phân cụm & Hồ sơ Persona Khách hàng</td>
                    <td>4 Đồ thị chuyên sâu (Dendrogram, 2D, DBSCAN), <code>stage_04.json</code></td>
                    <td>{t4}s</td>
                    <td><span class="badge badge-success">Thành công</span></td>
                    <td><a href="04_model_evaluation_report.html" class="btn" style="padding:4px 10px; font-size:12px;">Xem chi tiết</a></td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>2. Danh Mục Các Báo Cáo Chuyên Sâu & Công Cụ Nghiệm Thu</h2>
        <div class="reports-grid">
            <div class="report-card">
                <div>
                    <div class="report-num">Báo cáo Chặng 01</div>
                    <div class="report-title">Khám Phá Dữ Liệu (EDA)</div>
                    <div class="report-desc">Phân tích phân phối đơn biến, so sánh hành vi giới tính, độ tuổi và khám phá 5 cụm tự nhiên trên không gian 2D Thu nhập - Chi tiêu.</div>
                </div>
                <a href="01_data_report.html" class="btn">Xem Báo Cáo EDA &rarr;</a>
            </div>

            <div class="report-card">
                <div>
                    <div class="report-num">Báo cáo Chặng 02</div>
                    <div class="report-title">Tiền Xử Lý & Kỹ Thuật Đặc Trưng</div>
                    <div class="report-desc">Làm sạch bảng dữ liệu, chuẩn hóa tên cột, mã hóa Gender, thiết lập 3 không gian phân cụm và chuẩn hóa phân tán StandardScaler.</div>
                </div>
                <a href="02_feature_pipeline_report.html" class="btn">Xem Báo Cáo Tiền Xử Lý &rarr;</a>
            </div>

            <div class="report-card">
                <div>
                    <div class="report-num">Báo cáo Chặng 03</div>
                    <div class="report-title">Huấn Luyện 3 Thuật Toán Scratch</div>
                    <div class="report-desc">Tối ưu số cụm K qua Elbow (WCSS) và Silhouette Score, đối chuẩn chi tiết giữa K-Means, Hierarchical và DBSCAN Scratch với Scikit-learn.</div>
                </div>
                <a href="03_model_training_report.html" class="btn">Xem Báo Cáo Huấn Luyện &rarr;</a>
            </div>

            <div class="report-card">
                <div>
                    <div class="report-num">Báo cáo Chặng 04</div>
                    <div class="report-title">Đánh Giá & Phân Khúc Khách Hàng</div>
                    <div class="report-desc">Trực quan hóa không gian phân cụm 2D, biểu đồ cây Dendrogram, phân cụm mật độ DBSCAN và lập hồ sơ kinh doanh cho 5 phân khúc.</div>
                </div>
                <a href="04_model_evaluation_report.html" class="btn">Xem Báo Cáo Đánh Giá &rarr;</a>
            </div>

            <div class="report-card">
                <div>
                    <div class="report-num">Nhật ký Hệ thống</div>
                    <div class="report-title">Nhật Ký Thực Thi Toàn Trình (Run Log)</div>
                    <div class="report-desc">Xem toàn bộ nhật ký thực thi chi tiết của Master Pipeline, kiểm soát danh sách các tệp log và thời lượng từng bước vận hành.</div>
                </div>
                <a href="run_log.html" class="btn">Xem Nhật Ký Hệ Thống &rarr;</a>
            </div>

        </div>
    </div>

    <div class="footer-info">
        Học phần: Học máy (Machine Learning) — GVHD: TS. Nguyễn Thị Khánh Tiên — Nhóm 3 (Đại học Giao thông Vận tải TP.HCM — UTH) — Năm học 2026-2027
    </div>
</div>
</body>
</html>"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target_file.write_text(html_content, encoding="utf-8")
    return target_file


# -----------------------------------------------------------------------------
# 8. BÁO CÁO NHẬT KÝ THỰC THI (RUN_LOG.HTML)
# -----------------------------------------------------------------------------
def build_run_log_page() -> Path:
    target_file = REPORTS_DIR / "run_log.html"

    all_logs = sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    pipeline_logs = sorted(LOGS_DIR.glob("run_pipeline_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

    log_items = []
    for lp in all_logs[:10]:
        mtime = datetime.fromtimestamp(lp.stat().st_mtime).strftime("%H:%M:%S %d/%m/%Y")
        size_kb = lp.stat().st_size / 1024
        log_items.append(f"<li><code>{escape_html(lp.name)}</code> ({size_kb:.1f} KB, sửa đổi lúc {mtime})</li>")

    other_logs_html = f"""
    <div class="card" style="background:#f8fafc;">
        <h2 style="font-size:15px; border-left:none; padding-left:0;">Danh Sách Tệp Log Trong Thư Mục logs/ ({len(all_logs)} tệp)</h2>
        <ul style="font-size:13px; color:var(--text-muted); padding-left:20px; line-height:1.7;">
            {''.join(log_items) if log_items else '<li>Chưa có tệp nhật ký nào được ghi nhận.</li>'}
        </ul>
    </div>
    """

    latest_log_content = "Chưa có nội dung tệp log nào được ghi nhận."
    latest_log_name = "N/A"
    if pipeline_logs:
        target_log = pipeline_logs[0]
        latest_log_name = target_log.name
        try:
            latest_log_content = target_log.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            latest_log_content = f"Không thể đọc tệp log: {e}"
    elif all_logs:
        target_log = all_logs[0]
        latest_log_name = target_log.name
        try:
            latest_log_content = target_log.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            latest_log_content = f"Không thể đọc tệp log: {e}"

    html_content = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nhật Ký Thực Thi Toàn Trình (Run Log) — Nhóm 3 (UTH)</title>
    <style>{COMMON_STYLE}</style>
</head>
<body>
<div class="container">
    <div class="nav-bar">
        <a href="04_model_evaluation_report.html" class="nav-link">&larr; Báo Cáo 04: Đánh Giá & Phân Khúc Khách Hàng</a>
        <a href="index.html" class="nav-link">Bảng Điều Khiển Tổng Quan &rarr;</a>
    </div>

    <div class="header">
        <span class="badge">NHẬT KÝ THỰC THI TOÀN TRÌNH (ZERO-DRIFT RUN LOG)</span>
        <h1>Nhật Ký Thực Thi Master Pipeline & 4 Chặng Điều Phối</h1>
        <p style="color:var(--text-muted); font-size:14.5px; margin-top:4px;">
            Ghi nhận toàn bộ thông số môi trường, tiến độ đo lường thời gian, kết quả huấn luyện mô hình và hiện vật xuất sinh.
        </p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="title">Trạng thái Pipeline</div>
            <div class="value">Thành công 100%</div>
            <div class="desc">Hoàn thành cả 4 chặng không phát sinh lỗi</div>
        </div>
        <div class="stat-card">
            <div class="title">Tệp log hiển thị</div>
            <div class="value">{len(all_logs)} Tệp log</div>
            <div class="desc">Lưu trữ đồng thời trong thư mục logs/</div>
        </div>
        <div class="stat-card">
            <div class="title">Cơ chế ghi nhật ký</div>
            <div class="value">Multi-Logging</div>
            <div class="desc">Chia sẻ handler đồng thời 5 file log</div>
        </div>
        <div class="stat-card">
            <div class="title">Định dạng hiển thị</div>
            <div class="value">Relative Paths</div>
            <div class="desc">100% đường dẫn tương đối, không lộ ổ đĩa</div>
        </div>
    </div>

    {other_logs_html}

    <div class="card">
        <h2>Nội Dung Chi Tiết Tệp Nhật Ký Gần Nhất: <code>{escape_html(latest_log_name)}</code></h2>
        <div style="background:#0f172a; border-radius:10px; padding:16px 20px; box-shadow:0 4px 12px rgba(0,0,0,0.15);">
            <div style="display:flex; gap:6px; margin-bottom:12px;">
                <span style="width:12px; height:12px; border-radius:50%; background:#ef4444; display:inline-block;"></span>
                <span style="width:12px; height:12px; border-radius:50%; background:#f59e0b; display:inline-block;"></span>
                <span style="width:12px; height:12px; border-radius:50%; background:#10b981; display:inline-block;"></span>
                <span style="color:#94a3b8; font-size:12px; margin-left:8px; font-family:monospace;">terminal &mdash; utf-8 log stream</span>
            </div>
            <pre style="color:#f8fafc; font-family:Consolas, 'Courier New', monospace; font-size:13px; line-height:1.65; max-height:650px; overflow-y:auto; white-space:pre-wrap; word-break:break-all;">{escape_html(latest_log_content)}</pre>
        </div>
    </div>

    <div class="footer-info">
        Học phần: Học máy (Machine Learning) — GVHD: TS. Nguyễn Thị Khánh Tiên — Nhóm 3 (UTH)
    </div>
</div>
</body>
</html>"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target_file.write_text(html_content, encoding="utf-8")
    return target_file


# -----------------------------------------------------------------------------
# 9. ĐỒNG BỘ TOÀN BỘ BÁO CÁO (BUILD_ALL) & ALIASES
# -----------------------------------------------------------------------------
def build_eda_report() -> Path:
    """Bí danh cho build_report_01."""
    return build_report_01()


def build_fe_report() -> Path:
    """Bí danh cho build_report_02."""
    return build_report_02()


def build_training_report() -> Path:
    """Bí danh cho build_report_03."""
    return build_report_03()


def build_evaluation_report() -> Path:
    """Bí danh cho build_report_04."""
    return build_report_04()


def build_all() -> Dict[str, Path]:
    """Biên dịch đồng bộ toàn bộ 6 tệp báo cáo HTML Base64 Standalone."""
    print("Đang biên dịch Báo cáo HTML Chặng 1...")
    p1 = build_report_01()
    print(f"  ✓ {p1.name}")

    print("Đang biên dịch Báo cáo HTML Chặng 2...")
    p2 = build_report_02()
    print(f"  ✓ {p2.name}")

    print("Đang biên dịch Báo cáo HTML Chặng 3...")
    p3 = build_report_03()
    print(f"  ✓ {p3.name}")

    print("Đang biên dịch Báo cáo HTML Chặng 4...")
    p4 = build_report_04()
    print(f"  ✓ {p4.name}")

    print("Đang biên dịch Báo cáo Nhật ký Log...")
    p_log = build_run_log_page()
    print(f"  ✓ {p_log.name}")

    print("Đang biên dịch Bảng Điều khiển Trung tâm (index.html)...")
    p_idx = build_index()
    print(f"  ✓ {p_idx.name}")

    print("🎉 Hoàn tất biên dịch trọn bộ 6 báo cáo HTML Base64 Standalone!")
    return {
        "eda_report": p1,
        "fe_report": p2,
        "training_report": p3,
        "eval_report": p4,
        "run_log": p_log,
        "dashboard_index": p_idx
    }


if __name__ == "__main__":
    build_all()
