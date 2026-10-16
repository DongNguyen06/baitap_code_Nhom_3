"""
Ứng dụng Web Demo Streamlit - Phân loại Email Spam (SpamGuard-ML)
Nhóm 3 - Học máy (Đại học Giao thông Vận tải TP.HCM)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.set_page_config(
    page_title="Phân loại Email Spam - Nhóm 3",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ HỆ THỐNG PHÂN LOẠI EMAIL SPAM (SPAMGUARD-ML)")
st.markdown("""
**Học phần:** Học máy (Machine Learning) — UTH  
**Giảng viên hướng dẫn:** TS. Nguyễn Thị Khánh Tiên  
**Thực hiện:** Nhóm 3
""")

MODEL_PATH = "models/best_model.joblib"

st.sidebar.header("⚙️ Cấu hình suy luận")
threshold = st.sidebar.slider("Ngưỡng phân loại Spam (Threshold):", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✉️ Soạn thảo email kiểm tra")
    subject_input = st.text_input("Tiêu đề email (Subject):", value="Urgent: Claim your prize now!")
    body_input = st.text_area("Nội dung email (Body):", value="Congratulations! You have won $1,000,000. Click http://claim-prize.fake to verify your account.", height=180)

    if st.button("🚀 Phân loại Email ngay", type="primary"):
        full_text = f"{subject_input} {body_input}".strip()
        if not full_text:
            st.warning("Vui lòng nhập tiêu đề hoặc nội dung email để phân loại!")
        else:
            if os.path.exists(MODEL_PATH):
                try:
                    model = joblib.load(MODEL_PATH)
                    proba = model.predict_proba([full_text])[0]
                    spam_prob = float(proba[1])
                    is_spam = spam_prob >= threshold

                    if is_spam:
                        st.error(f"### 🚨 CẢNH BÁO: EMAIL RÁC (SPAM)")
                    else:
                        st.success(f"### 🟢 EMAIL HỢP LỆ (HAM)")
                    
                    st.metric("Xác suất là Thư rác (Spam Probability)", f"{spam_prob * 100:.2f}%")
                    st.progress(spam_prob)
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán: {e}")
            else:
                # Chế độ mô phỏng khi chưa chạy train notebook
                simulated_prob = 0.88 if ("$" in full_text or "prize" in full_text.lower() or "http" in full_text.lower()) else 0.12
                st.info("💡 *Chế độ mô phỏng (Chưa tìm thấy models/best_model.joblib — hãy chạy Notebook 3 để sinh model thực tế)*:")
                if simulated_prob >= threshold:
                    st.error(f"### 🚨 CẢNH BÁO: EMAIL RÁC (SPAM)")
                else:
                    st.success(f"### 🟢 EMAIL HỢP LỆ (HAM)")
                st.metric("Xác suất dự đoán ước tính", f"{simulated_prob * 100:.2f}%")
                st.progress(simulated_prob)

with col2:
    st.subheader("ℹ️ Thông tin mô hình")
    st.markdown("""
    - **Thuật toán so sánh:**
      - Multinomial Naive Bayes
      - Logistic Regression (L1)
      - Linear SVM (Platt Scaling)
    - **Kỹ thuật trích xuất:** TF-IDF (1-2 gram) + Chi-square Feature Selection
    - **Chỉ số đánh giá:** Accuracy, Precision, Recall, F1-Score
    - **Đặc trưng phòng thủ:** Pipeline khép kín chống rò rỉ dữ liệu (Data Leakage).
    """)
