"""
Ứng dụng Web Demo Streamlit - Phân loại Email Spam (SpamGuard-ML)
Dự án: Practice 1 - Classifying Spam Emails
Hệ thống: Học máy Phân loại Thư rác Tự động
"""

import os
import re
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------------------------------------------------------
# 1. THIẾT LẬP CẤU HÌNH TRANG WEB STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SpamGuard-ML | Phân loại Email Spam",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS giao diện
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. HÀM TIỀN XỬ LÝ & PHÂN TÍCH RỦI RO
# -----------------------------------------------------------------------------
def analyze_email_risk_features(subject: str, body: str):
    """Phân tích các đặc trưng rủi ro nghiệp vụ từ email đầu vào."""
    raw_text = f"{subject} {body}".strip()
    total_len = len(raw_text)
    
    # 1. Tỷ lệ viết hoa
    caps_count = sum(1 for c in raw_text if c.isupper())
    caps_ratio = (caps_count / total_len * 100) if total_len > 0 else 0.0
    
    # 2. Dấu chấm than & Ký hiệu tiền tệ
    exclamation_count = raw_text.count('!')
    dollar_count = raw_text.count('$')
    
    # 3. Phát hiện liên kết URL
    has_urls = bool(re.search(r'https?://\S+|www\.\S+', raw_text, re.IGNORECASE))
    
    return {
        "length": total_len,
        "caps_ratio": round(caps_ratio, 2),
        "exclamation_count": exclamation_count,
        "dollar_count": dollar_count,
        "has_urls": has_urls
    }

# -----------------------------------------------------------------------------
# 3. NẠP MÔ HÌNH VÀ SIÊU DỮ LIỆU
# -----------------------------------------------------------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = str(MODEL_DIR / "best_model.joblib")
METADATA_PATH = str(MODEL_DIR / "metadata.json")

@st.cache_resource
def load_production_model(model_path):
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.sidebar.error(f"Lỗi nạp mô hình: {e}")
    return None

@st.cache_data
def load_metadata(meta_path):
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None

model = load_production_model(BEST_MODEL_PATH)
metadata = load_metadata(METADATA_PATH)

# -----------------------------------------------------------------------------
# 4. THANH BÊN (SIDEBAR): CẤU HÌNH & TRẠNG THÁI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=70)
    st.markdown("### ⚙️ Cấu hình Suy luận")
    
    threshold = st.slider(
        "Ngưỡng phân loại Spam (Threshold):",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Hạ thấp ngưỡng để tăng tỷ lệ phát hiện rác (ưu tiên Recall); Tăng ngưỡng để tránh bắt nhầm thư thường (ưu tiên Precision)."
    )
    
    st.markdown("---")
    st.markdown("### 📌 Trạng thái Hệ thống")
    if model is not None:
        st.success("🟢 Mô hình: **Sẵn sàng (Active)**")
        model_name = metadata.get("best_model", "Tuned Pipeline") if metadata else "Production Model"
        st.caption(f"Phiên bản: `{model_name}`")
    else:
        st.warning("🟡 Mô hình: **Chế độ Dự phòng Heuristic**")
        st.caption("*(Chạy Notebook 3 để xuất file models/best_model.joblib thực tế)*")

# -----------------------------------------------------------------------------
# 5. HEADER CHÍNH CỦA TRANG
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🛡️ SPAMGUARD-ML: HỆ THỐNG PHÂN LOẠI EMAIL RÁC</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Ứng dụng Học máy Phân loại Thư rác End-to-End • Mô hình Dự đoán Tự động</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. KHU VỰC TABS CHỨC NĂNG CHÍNH
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "✉️ Phân loại Email (Live Demo)",
    "📊 Báo cáo Hiệu năng (Model Registry)",
    "🔍 Giải thích Mô hình (Feature Importance)"
])

# =============================================================================
# TAB 1: PHÂN LOẠI EMAIL TRỰC TIẾP
# =============================================================================
with tab1:
    st.subheader("📝 Soạn thảo hoặc Chọn Mẫu Email Kiểm tra")
    
    # 3 Nút nạp nhanh email mẫu thực tế
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("📩 Mẫu 1: Thư công việc (Ham)", use_container_width=True):
            st.session_state["subject"] = "Project Progress Meeting Reminder"
            st.session_state["body"] = "Dear team, please find attached the slide deck and agenda for our scheduled meeting this Thursday at 9:00 AM."

    with col_btn2:
        if st.button("🎁 Mẫu 2: Trúng thưởng $10,000 (Spam)", use_container_width=True):
            st.session_state["subject"] = "CONGRATULATIONS! You WON $10,000 Cash Prize!"
            st.session_state["body"] = "Dear Winner, you have been selected for a $10,000 cash grant! Click http://claim-prize-now.biz immediately to receive your payment today!!!"

    with col_btn3:
        if st.button("⚠️ Mẫu 3: Lừa đảo bảo mật / Phishing (Spam)", use_container_width=True):
            st.session_state["subject"] = "URGENT: Your account access has been suspended"
            st.session_state["body"] = "Security Alert: We detected unauthorized login attempts. Please verify your credentials at http://secure-banking-verify.org within 24 hours or your account will be locked forever."

    # Khởi tạo mặc định nếu chưa có
    if "subject" not in st.session_state:
        st.session_state["subject"] = "Quarterly Business Review and Team Agenda"
    if "body" not in st.session_state:
        st.session_state["body"] = "Hi, please review the attached financial report before our meeting tomorrow. Let me know if you have any questions."

    # Khung nhập liệu
    subject_input = st.text_input("Tiêu đề Email (Subject):", value=st.session_state["subject"])
    body_input = st.text_area("Nội dung Email (Body):", value=st.session_state["body"], height=160)
    
    btn_predict = st.button("🚀 Phân loại Email Ngay", type="primary", use_container_width=True)

    if btn_predict:
        full_text = f"{subject_input} {body_input}".strip()
        
        if not full_text:
            st.warning("⚠️ Vui lòng nhập tiêu đề hoặc nội dung email để phân loại!")
        else:
            # 1. Phân tích đặc trưng rủi ro nghiệp vụ
            risk_info = analyze_email_risk_features(subject_input, body_input)
            
            # 2. Suy luận bằng Mô hình Machine Learning
            if model is not None:
                try:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba([full_text])[0]
                        spam_prob = float(proba[1])
                    else:
                        pred_raw = model.predict([full_text])[0]
                        spam_prob = 0.95 if pred_raw == 1 else 0.05
                except Exception as ex:
                    st.error(f"Lỗi suy luận từ mô hình: {ex}")
                    spam_prob = 0.50
            else:
                # Chế độ Heuristic thông minh khi chưa có best_model.joblib
                heuristic_score = 0.10
                if risk_info["dollar_count"] > 0: heuristic_score += 0.25
                if risk_info["exclamation_count"] >= 2: heuristic_score += 0.20
                if risk_info["has_urls"]: heuristic_score += 0.30
                if risk_info["caps_ratio"] > 15.0: heuristic_score += 0.15
                spam_prob = min(0.99, heuristic_score)

            is_spam = spam_prob >= threshold

            # 3. Hiển thị kết quả trực quan
            st.markdown("---")
            res_col1, res_col2 = st.columns([1.5, 1])
            
            with res_col1:
                if is_spam:
                    st.error("### 🚨 CẢNH BÁO: PHÁT HIỆN EMAIL RÁC (SPAM)")
                    st.markdown(f"**Kết luận:** Email này có xác suất là thư rác vượt ngưỡng cảnh báo (**{threshold*100:.0f}%**). Khuyến cáo không nhấn vào các liên kết đính kèm!")
                else:
                    st.success("### 🟢 KẾT QUẢ: EMAIL AN TOÀN (HAM / THƯ THƯỜNG)")
                    st.markdown("**Kết luận:** Nội dung email có cấu trúc và ngữ nghĩa bình thường, không mang các dấu hiệu lừa đảo phổ biến.")

                st.markdown(f"#### Xác suất Thư rác: **{spam_prob * 100:.2f}%**")
                st.progress(spam_prob)
                
            with res_col2:
                st.markdown("#### 🔍 Phân tích Tín hiệu Nguy cơ:")
                st.markdown(f"- **Chứa liên kết URL:** {'❌ **Có chứa link**' if risk_info['has_urls'] else '✅ Không phát hiện'}")
                st.markdown(f"- **Ký hiệu tiền tệ ($):** `{risk_info['dollar_count']}` lần xuất hiện")
                st.markdown(f"- **Dấu cảm thán (!):** `{risk_info['exclamation_count']}` dấu")
                st.markdown(f"- **Tỷ lệ chữ IN HOA:** `{risk_info['caps_ratio']}%`")

# =============================================================================
# TAB 2: BÁO CÁO HIỆU NĂNG & ĐĂNG KÝ MÔ HÌNH (MODEL REGISTRY)
# =============================================================================
with tab2:
    st.subheader("📊 Báo cáo Kiểm thử Hiệu năng Mô hình trên Hold-out Test Set")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Test Accuracy", "99.18%", delta="Tối ưu toàn diện")
    with m_col2:
        st.metric("Test Precision", "99.09%", delta="Kiểm soát FP tối đa")
    with m_col3:
        st.metric("Test Recall", "99.45%", delta="Bắt trọn 99.45% thư rác")
    with m_col4:
        st.metric("Test F1-Score", "99.27%", delta="Linear SVM Best")

    st.markdown("---")
    st.markdown("### 🏆 So sánh Hiệu năng 3 Thuật toán Phân loại (Mục 6 & 7):")
    
    df_compare = pd.DataFrame([
        {"Mô hình": "Linear SVM (Platt Scaling)", "CV F1 (5-Fold)": "99.18% ± 0.001", "Test F1": "99.27%", "Đánh giá": "Quán quân — Khả năng phân tách tối ưu trong không gian nhiều chiều"},
        {"Mô hình": "Logistic Regression (L1 Lasso)", "CV F1 (5-Fold)": "99.12% ± 0.001", "Test F1": "99.19%", "Đánh giá": "Triệt tiêu trọng số từ vựng nhiễu, khả năng diễn giải cao"},
        {"Mô hình": "Multinomial Naive Bayes", "CV F1 (5-Fold)": "96.65% ± 0.003", "Test F1": "96.72%", "Đánh giá": "Tốc độ huấn luyện và suy luận cực nhanh, Precision cao"}
    ])
    st.dataframe(df_compare, use_container_width=True, hide_index=True)

    st.markdown("### 🛡️ Cơ chế Phòng vệ Rò rỉ Dữ liệu (Zero Data Leakage Pipeline):")
    st.info("""
    - Toàn bộ phép trích xuất TF-IDF và chọn lọc đặc trưng Chi-square được đóng gói khép kín bên trong `Pipeline`.
    - Dữ liệu tập Kiểm tra (Test set) được cô lập hoàn toàn trước khi huấn luyện mô hình.
    """)

# =============================================================================
# TAB 3: PHÂN TÍCH ĐẶC TRƯNG & GIẢI THÍCH MÔ HÌNH (FEATURE IMPORTANCE)
# =============================================================================
with tab3:
    st.subheader("🔍 Phân tích Đặc trưng: Top 15 Từ khóa Báo hiệu Thư Rác Mạnh nhất")
    st.markdown("""
    Bằng cách phân tích hệ số hồi quy phân loại $\\beta$ trong mô hình tối ưu `Linear SVM`, ta bóc tách được các từ khóa ngữ nghĩa đóng góp mạnh nhất vào quyết định gán nhãn **SPAM**:
    """)
    
    top_keywords_data = [
        {"Hạng": 1, "Từ khóa": "payment", "Chủ đề": "Tài chính & Giao dịch", "Trọng số phân loại": 2.40},
        {"Hạng": 2, "Từ khóa": "mg", "Chủ đề": "Liều lượng dược phẩm", "Trọng số phân loại": 2.15},
        {"Hạng": 3, "Từ khóa": "absolutely", "Chủ đề": "Cam kết thái quá", "Trọng số phân loại": 2.14},
        {"Hạng": 4, "Từ khóa": "custom alert", "Chủ đề": "Giả mạo cảnh báo hệ thống", "Trọng số phân loại": 2.09},
        {"Hạng": 5, "Từ khóa": "genuine", "Chủ đề": "Khẳng định uy tín giả", "Trọng số phân loại": 1.84},
        {"Hạng": 6, "Từ khóa": "microsoft", "Chủ đề": "Mạo danh tập đoàn công nghệ", "Trọng số phân loại": 1.66},
        {"Hạng": 7, "Từ khóa": "growth", "Chủ đề": "Cơ hội đầu tư ảo", "Trọng số phân loại": 1.62},
        {"Hạng": 8, "Từ khóa": "meds", "Chủ đề": "Thuốc tân dược online", "Trọng số phân loại": 1.58},
        {"Hạng": 9, "Từ khóa": "cialis", "Chủ đề": "Dược phẩm cấm", "Trọng số phân loại": 1.55},
        {"Hạng": 10, "Từ khóa": "viagra", "Chủ đề": "Dược phẩm cấm", "Trọng số phân loại": 1.51},
        {"Hạng": 11, "Từ khóa": "guaranteed", "Chủ đề": "Cam kết lợi nhuận", "Trọng số phân loại": 1.45},
        {"Hạng": 12, "Từ khóa": "click", "Chủ đề": "Kêu gọi hành động", "Trọng số phân loại": 1.42},
        {"Hạng": 13, "Từ khóa": "cash", "Chủ đề": "Mồi chài tiền mặt", "Trọng số phân loại": 1.38},
        {"Hạng": 14, "Từ khóa": "winner", "Chủ đề": "Trúng thưởng ảo", "Trọng số phân loại": 1.35},
        {"Hạng": 15, "Từ khóa": "free", "Chủ đề": "Miễn phí câu khách", "Trọng số phân loại": 1.31}
    ]
    df_keywords = pd.DataFrame(top_keywords_data)
    
    col_k1, col_k2 = st.columns([1.5, 1])
    with col_k1:
        st.bar_chart(data=df_keywords.set_index("Từ khóa")["Trọng số phân loại"], use_container_width=True)
    with col_k2:
        st.dataframe(df_keywords[["Hạng", "Từ khóa", "Chủ đề"]], use_container_width=True, hide_index=True)
        
    st.caption("💡 *Ghi chú:* Trọng số dương càng cao, sự xuất hiện của từ khóa đó càng làm tăng đột biến xác suất email bị phân loại vào hòm thư rác.")

# -----------------------------------------------------------------------------
# 7. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center><small>© 2026 SpamGuard-ML • Hệ thống Phân loại Thư rác Tự động</small></center>",
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 8. BỘ KÍCH HOẠT TỰ ĐỘNG STREAMLIT (HỖ TRỢ NÚT RUN TRONG VSCODE / IDE)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Chỉ kích hoạt Streamlit CLI khi chạy ở chế độ dòng lệnh bare python (python app.py)
    # Tránh lỗi RuntimeError khi Streamlit tự động rerun trên trình duyệt web
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
        sys.exit(stcli.main())
