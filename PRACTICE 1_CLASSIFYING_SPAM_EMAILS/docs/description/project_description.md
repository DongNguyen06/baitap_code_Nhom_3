# ĐẶC TẢ BÀI THỰC HÀNH 1: PHÂN LOẠI EMAIL SPAM — NHÓM 3

## 1. Project Name
Phân loại Thư rác Điện tử (Spam vs Ham Email Classification)

## 2. Problem
Phân loại nhị phân trên văn bản: xác định một email là Spam (1) hay Ham (0).
- **Dữ liệu thực nghiệm:** `data/raw/CEAS_08.csv` (nguồn Kaggle / CEAS dataset). Gốc gồm 39.154 email (Ham: 44,2%, Spam: 55,8%); sau làm sạch và khử trùng lặp còn 32.660 email sạch (Ham: 52,3%, Spam: 47,7%).
- **Phạm vi bài lab:** Tập trung vào tiền xử lý, tự cài đặt thuật toán, huấn luyện, tinh chỉnh siêu tham số, so sánh đối chứng mô hình và demo dự đoán;, chỉ xử lý email tiếng Anh.

## 3. Target User
- Người học và giảng viên môn Học máy (đánh giá năng lực thực hành và cài đặt thuật toán).
- Về ý tưởng ứng dụng: Người dùng cá nhân cần bộ lọc thư rác tự động, an toàn cho hộp thư điện tử.

## 4. Input / Output
- **Input:** Chuỗi ký tự văn bản thô gồm Tiêu đề (`Subject`) và Nội dung (`Body`) email.
- **Output:** Nhãn phân loại (`Spam` hoặc `Ham`) kèm xác suất dự đoán và độ tin cậy thời gian thực.

## 5. Core Features
- **Khám phá dữ liệu (EDA):** Phân tích phân bố nhãn, độ dài email (ký tự, từ vựng); phân tích tương quan và phát hiện ngoại lai (IQR) trên các đặc trưng miền.
- **Tiền xử lý:** Xử lý tiêu đề khuyết, bóc tách thẻ HTML ngụy trang, URL và ký tự rác bằng Regex, chuyển chữ thường, lọc stopwords và dấu câu; khử triệt để 6.494 mẫu thư rác trùng lặp đa hình.
- **Kỹ thuật đặc trưng:** Trích xuất 7 đặc trưng miền số (tỷ lệ chữ HOA, số dấu `!`, ký hiệu tiền tệ `$`, số link URL, tiền tố `Re:`, nén log độ dài body); kết hợp vector hóa TF-IDF (1–2 gram, chọn lọc 5.000 từ vựng quan trọng qua Chi-square) và ghép nhất quán bằng `ColumnTransformer`.
- **Demo tương tác:** Thử nghiệm dự đoán email tương tác thời gian thực tại `notebooks/demo.ipynb`, hỗ trợ linh hoạt lựa chọn giữa các mô hình (Linear SVM, Logistic Regression, Naive Bayes).

## 6. Technology Constraints
- **Môi trường & Thư viện:** Python 3.10+, NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, Joblib.
- **Thực thi:** Tổ chức qua 4 Jupyter Notebooks tuần tự (EDA & Tiền xử lý $\rightarrow$ Kỹ thuật đặc trưng $\rightarrow$ Huấn luyện & Đánh giá $\rightarrow$ Demo tương tác) hoặc chạy tự động toàn bộ qua `scripts/run_pipeline.py`.

## 7. Algorithm / Evaluation
- **Thuật toán:** Tự cài đặt 3 thuật toán From Scratch bằng NumPy:
  1. *Multinomial Naive Bayes* (làm mượt Laplace $\alpha$).
  2. *Logistic Regression* (tối ưu Gradient Descent với chuẩn hóa L2 hệ số $C$).
  3. *Linear SVM* (tối ưu Pegasos Hinge Loss với hệ số phạt $C$, kết hợp hiệu chuẩn khoảng cách ra xác suất).
  * Tinh chỉnh siêu tham số tối ưu bằng GridSearchCV kết hợp 5-Fold Stratified Cross-Validation.
- **Chia dữ liệu:** Phân tách Train/Test theo tỷ lệ 80/20 có phân tầng (`stratify=y`, `random_state=42`) bảo toàn cấu trúc nhãn, lưu cả dạng trực quan `.csv` và nạp nhanh `.joblib`.
- **Đánh giá:** Accuracy, Precision, Recall, F1-Score và Ma trận nhầm lẫn (Confusion Matrix); đặt trọng tâm kiểm soát Precision và F1-Score để giảm thiểu tối đa số ca bắt nhầm thư thường (False Positive $\le 50$).
- **Tiêu chí nghiệm thu:** Cả 3 mô hình đạt $\text{Accuracy, Precision, F1} \ge 95\%$ trên tập test
