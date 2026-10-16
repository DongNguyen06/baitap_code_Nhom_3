# THIẾT KẾ HỆ THỐNG THEO TÍNH NĂNG (SPAMGUARD-ML)

## 1. Khám phá & Làm sạch dữ liệu (Notebook 1)
- EDA: Phân tích phân phối nhãn, đo tương quan và phát hiện ngoại lai IQR/Boxplot.
- Làm sạch: Khử trùng lặp, điền rỗng tiêu đề khuyết, bóc tách HTML, lọc stopwords.

## 2. Kỹ thuật đặc trưng & Pipeline (Notebook 2)
- Biến đổi: Biến đổi log, mã hóa Ordinal/One-Hot, tạo biến tỷ lệ viết hoa, ký tự lạ.
- Trích xuất & Lọc: TF-IDF (1-2 gram), lọc biến qua Chi-square và phạt L1.
- Pipeline: Đóng gói toàn bộ luồng chống rò rỉ dữ liệu (Data Leakage).

## 3. Huấn luyện mô hình (Notebook 3)
- Huấn luyện: Naive Bayes, Logistic Regression, Linear SVM với 5-Fold Stratified Cross-Validation.
- Hiệu chuẩn: Platt Scaling cho Linear SVM và tối ưu tham số C.

## 4. Đánh giá & Triển khai (Notebook 3 & Web Demo)
- Đánh giá: Accuracy, Precision, Recall, F1-score, vẽ Confusion Matrix và phân tích lỗi.
- Triển khai: Tối ưu ngưỡng phân loại, lưu best model và dựng Web Demo Streamlit.