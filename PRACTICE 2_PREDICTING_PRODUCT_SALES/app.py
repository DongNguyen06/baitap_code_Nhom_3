"""
Ứng dụng Web Demo Streamlit - Dự đoán Doanh số Bán hàng Sản phẩm (Product Sales)
Nhóm 3 - Học máy (Đại học Giao thông Vận tải)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.set_page_config(
    page_title="Dự đoán Doanh số Sản phẩm - Nhóm 3",
    page_icon="📈",
    layout="wide"
)

st.title("📈 HỆ THỐNG DỰ ĐOÁN DOANH SỐ BÁN HÀNG SẢN PHẨM")
st.markdown("""
**Học phần:** Học máy (Machine Learning) — UTH  
**Giảng viên hướng dẫn:** TS. Nguyễn Thị Khánh Tiên  
**Thực hiện:** Nhóm 3
""")

MODEL_PATH = "models/best_model.joblib"

st.sidebar.header("⚙️ Nhập thông số sản phẩm")

# Demo inputs (có thể tùy chỉnh khớp với tập dữ liệu thực tế)
price = st.sidebar.number_input("Giá sản phẩm (Price):", min_value=1.0, max_value=10000.0, value=120.0, step=5.0)
marketing = st.sidebar.number_input("Ngân sách quảng cáo (Marketing Spend):", min_value=0.0, max_value=50000.0, value=500.0, step=50.0)
category = st.sidebar.selectbox("Ngành hàng (Category):", ["Electronics", "Fashion", "Home & Living", "Beauty", "Sports"])

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📋 Bảng tham số nhập liệu")
    input_data = {
        "Price": price,
        "Marketing_Spend": marketing,
        "Category": category
    }
    input_df = pd.DataFrame([input_data])
    st.dataframe(input_df, use_container_width=True)

    if st.button("🚀 Dự đoán Doanh số ngay", type="primary"):
        if os.path.exists(MODEL_PATH):
            try:
                model = joblib.load(MODEL_PATH)
                prediction = model.predict(input_df)[0]
                st.success(f"### 🎉 Dự đoán Doanh số: **{prediction:,.2f}** đơn vị")
            except Exception as e:
                st.error(f"Lỗi khi dự đoán: {e}")
        else:
            # Mô phỏng nếu chưa train model thực tế
            simulated_sales = max(0, price * 1.8 + marketing * 0.45 + np.random.normal(50, 10))
            st.info("💡 *Chế độ mô phỏng (Chưa tìm thấy best_model.joblib)*:")
            st.success(f"### 🎉 Dự đoán Doanh số ước tính: **{simulated_sales:,.2f}** đơn vị")

with col2:
    st.subheader("ℹ️ Giới thiệu mô hình")
    st.markdown("""
    - **Thuật toán áp dụng:**
      - Tuyến tính: Linear Regression
      - Cây: Decision Tree Regressor
      - Tập hợp: Random Forest Regressor
    - **Chỉ số đánh giá:** MSE, RMSE, MAE, R²
    - **Đặc trưng:** Chuẩn hóa dữ liệu trong Scikit-Learn Pipeline chống rò rỉ dữ liệu (Data Leakage).
    """)
