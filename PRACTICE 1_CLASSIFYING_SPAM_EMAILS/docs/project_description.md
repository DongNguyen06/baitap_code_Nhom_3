# ĐẶC TẢ DỰ ÁN (SPAMGUARD-ML)

## 1. Project Name
Hệ thống phân loại Email Spam (SpamGuard-ML)

## 2. Problem
Phân loại nhị phân xác định email là Spam (1) hay Ham (0).

## 3. Target User
Người dùng cá nhân cần tự động lọc thư rác trong hộp thư điện tử.

## 4. Input / Output
- **Input:** Tiêu đề (subject) và nội dung (body) email.
- **Output:** Nhãn (Spam / Ham) kèm xác suất dự đoán thời gian thực.

## 5. Core Features
- Khám phá dữ liệu (EDA): Phân tích phân phối, tương quan biến và ngoại lai IQR.
- Tiền xử lý: Khử trùng lặp, xử lý khuyết, tách HTML, lọc stopwords.
- Kỹ thuật đặc trưng (FE): Biến đổi log, mã hóa, trích xuất TF-IDF, lọc Chi-square và L1.
- Dự đoán email thời gian thực qua giao diện Web trực quan.

## 6. Technology Constraints
Python 3.10+, Scikit-learn, Pandas; tích hợp toàn bộ trong 1 file Jupyter Notebook.

## 7. Algorithm / Evaluation
- **Thuật toán:** Naive Bayes, Logistic Regression, Linear SVM.
- **Đánh giá:** Accuracy, Precision, Recall, F1-score và Confusion Matrix.