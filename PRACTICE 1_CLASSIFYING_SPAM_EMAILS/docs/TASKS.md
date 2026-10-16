# KẾ HOẠCH PHÂN RÃ CÔNG VIỆC (TASKS BREAKDOWN) — SPAMGUARD-ML
## PRACTICE 1: CLASSIFYING SPAM EMAILS — NHÓM 3 (UTH)

> **Nguyên tắc:** Phân công bám sát dòng chảy khoa học dữ liệu thẳng tắp cho bài toán Xử lý ngôn ngữ tự nhiên (EDA toàn diện $\rightarrow$ Làm sạch văn bản $\rightarrow$ Kỹ thuật đặc trưng $\rightarrow$ Huấn luyện $\rightarrow$ Đánh giá & Triển khai) và thực hiện qua **3 Notebooks chuyên sâu tuần tự** theo chuẩn *Machine Learning Walkthrough* của TS. Nguyễn Thị Khánh Tiên.

---

## SƠ ĐỒ LUỒNG THỰC HIỆN TUẦN TỰ (3 NOTEBOOKS PIPELINE)

```text
[CEAS_08.csv]
      │
      ▼
📓 NOTEBOOK 1: 01_data_cleaning_and_eda.ipynb (Khám phá & Làm sạch văn bản)
   ├── TV 1 (15%): Nguyễn Xuân Đông (Trưởng nhóm) — EDA 1 (Cấu trúc), EDA 2 (Chất lượng), EDA 3 (Đơn biến 4 thuộc tính)
   ├── TV 2 (14%): *[Chưa chỉ định]* — EDA 4 (Đa biến tương quan, Ngoại lai IQR) & EDA 5 (Bảng đúc kết EDA)
   └── TV 3 (14%): *[Chưa chỉ định]* — Tiền xử lý (Khử trùng, điền missing, clean_text) ──> [clean_emails.csv]
      │
      ▼
📓 NOTEBOOK 2: 02_feature_engineering_and_pipeline.ipynb (Kỹ thuật đặc trưng & Pipeline)
   ├── TV 4 (14%): *[Chưa chỉ định]* — FE Biến đổi (Log, Binning) & Tạo biến Domain (caps_ratio, !, $)
   └── TV 5 (14%): *[Chưa chỉ định]* — Chia Train/Test phân tầng chống Leak, TF-IDF Vectorizer, Lọc Chi-square
      │                             └──> [train_test_data.joblib & preprocessor.joblib]
      ▼
📓 NOTEBOOK 3: 03_model_training_evaluation_and_explain.ipynb (Huấn luyện & Đánh giá chuyên sâu)
   ├── TV 6 (14%): *[Chưa chỉ định]* — Huấn luyện 3 mô hình (Naive Bayes, Logistic L1, Linear SVM) + 5-Fold Stratified CV
   └── TV 7 (15%): *[Chưa chỉ định]* — Đánh giá Test, Confusion Matrix, Error Analysis & Top từ khóa Spam
      │                             └──> [best_model.joblib & metadata.json]
      ▼
🚀 TV 1: NGUYỄN XUÂN ĐÔNG TRIỂN KHAI WEB DEMO STREAMLIT (app.py) & CLI TEST (scripts/run_predict.py)
```

---

## BẢNG PHÂN CÔNG NHIỆM VỤ & MỨC ĐỘ ĐÓNG GÓP (TỔNG: 100%)

| STT | Thành viên | Vai trò | Đóng góp | Tính năng (Design) | Vị trí thực hiện | Tóm tắt nhiệm vụ chính |
| :---: | :--- | :---: | :---: | :--- | :---: | :--- |
| **TV 1** | **Nguyễn Xuân Đông** | **Trưởng nhóm** | **15%** | **Tính năng 1 & 4:**<br>EDA Nền tảng & Web Demo | **Notebook 1 — Mục 1**<br>kèm `app.py` & CLI | • Khởi tạo dự án, thiết lập môi trường và cấu hình `RANDOM_SEED = 42`.<br>• Khảo sát cấu trúc tổng quan dữ liệu gốc (`shape`, `info`, `describe`).<br>• Đánh giá chất lượng dữ liệu: Thống kê tỷ lệ khuyết thiếu (28 subject) và dữ liệu trùng lặp.<br>• Phân tích đơn biến toàn diện 4 thuộc tính: Biến mục tiêu `label`, liên kết `urls`, độ dài `subject` và tên miền `sender`.<br>• Quản trị phân nhánh Git, rà soát mã nguồn và điều phối tiến độ toàn nhóm.<br>• **Xây dựng và hoàn thiện ứng dụng Web Demo Streamlit (`app.py`) và công cụ dòng lệnh CLI (`scripts/run_predict.py`) phục vụ người dùng cuối.** |
| **TV 2** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 1:**<br>EDA Chuyên sâu | **Notebook 1 — Mục 2** | • Phân tích đa biến: Khảo sát mối quan hệ giữa các thuộc tính với nhãn mục tiêu `label`.<br>• Phân tích phân phối độ dài văn bản nội dung và nhận diện hiện tượng lệch phải đuôi dài.<br>• Xác định giá trị ngoại lai theo công thức khoảng tứ phân vị IQR ($1.5 \times IQR$) và vẽ Boxplot theo nhãn.<br>• Đúc kết các bằng chứng EDA thành Bảng kế hoạch hành động định hướng cho tiền xử lý và đặc trưng. |
| **TV 3** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 1:**<br>Làm sạch dữ liệu | **Notebook 1 — Mục 3** | • Thực thi xử lý chất lượng: Khử trùng lặp hoàn toàn và xử lý các email thiếu tiêu đề.<br>• Viết hàm `clean_text` chuyên sâu: Bóc tách mã HTML, loại bỏ URL và ký tự đặc biệt, lowercase và lọc từ dừng.<br>• Gộp tiêu đề và nội dung thành chuỗi văn bản sạch `text_clean`.<br>• Xuất dữ liệu sạch sang `data/processed/clean_emails.csv` bàn giao cho Notebook 2. |
| **TV 4** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 2:**<br>FE Biến đổi & Domain | **Notebook 2 — Mục 4** | • Kỹ thuật biến đổi đặc trưng: Áp dụng biến đổi log `np.log1p` để nén độ lệch phân phối của độ dài email.<br>• Rời rạc hóa (Binning): Gom nhóm độ dài văn bản liên tục thành các khoảng thứ bậc.<br>• Tạo đặc trưng mới theo ngữ cảnh: Tỷ lệ viết hoa (`caps_ratio`), tần suất ký tự cảm thán (`!`) và ký tự tiền tệ (`$`).<br>• Viết nhận xét đối sánh phân phối giữa thư Spam và Ham. |
| **TV 5** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 2:**<br>FE Trích xuất & Lọc | **Notebook 2 — Mục 5** | • Thiết lập hàng rào phân chia Train/Test 80/20 phân tầng (`stratify=y`) chống rò rỉ dữ liệu.<br>• Trích xuất đặc trưng văn bản: `TfidfVectorizer(max_features=3000, ngram_range=(1, 2))`.<br>• Chọn lọc đặc trưng: Sử dụng `SelectKBest(chi2, k=1000)` để lọc các từ khóa mang tính phân biệt cao nhất.<br>• Đóng gói Pipeline và lưu trữ dữ liệu sang `train_test_data.joblib` và `preprocessor.joblib`. |
| **TV 6** | *[Chưa chỉ định]* | Thành viên | **14%** | **Tính năng 3:**<br>Huấn luyện mô hình | **Notebook 3 — Mục 6** | • Huấn luyện mô hình Naive Bayes kết hợp đánh giá chéo phân tầng trên tập huấn luyện.<br>• Huấn luyện mô hình Logistic Regression kết hợp cơ chế tự động triệt tiêu trọng số từ vô dụng (L1 penalty).<br>• Huấn luyện mô hình Linear SVM và hiệu chuẩn xác suất dự đoán (Platt Scaling).<br>• Áp dụng 5-Fold Stratified CV, tinh chỉnh siêu tham số và lưu 3 mô hình vào `models/`. |
| **TV 7** | *[Chưa chỉ định]* | Thành viên | **15%** | **Tính năng 4:**<br>Đánh giá & Phân tích lỗi | **Notebook 3 — Mục 7** | • Đánh giá độc lập trên tập kiểm thử: Accuracy, Precision, Recall, F1-Score và vẽ Confusion Matrix.<br>• Phân tích lỗi chuyên sâu (Error Analysis): Mổ xẻ các trường hợp False Positives và False Negatives.<br>• **Kỹ thuật Cô Tiên:** Trích xuất và trực quan hóa Top 15 từ khóa có trọng số báo hiệu Spam mạnh nhất.<br>• Tối ưu ngưỡng phân loại (Threshold Tuning), xuất mô hình tối ưu `best_model.joblib` và file `metadata.json` bàn giao cho TV 1 tích hợp ứng dụng. |
| **TỔNG** | **7 thành viên** | | **100%** | | | **Hoàn thành toàn diện 4 tính năng hệ thống qua 3 Notebooks** |
