# THIẾT KẾ TỔNG THỂ HỆ THỐNG PHÂN LOẠI EMAIL RÁC (SPAMGUARD-ML)
> Mô hình thiết kế dự án Học máy chuẩn 5W1H — Nhóm 3 (Khoa CNTT - UTH)

---

### 1. WHY — TẠI SAO LÀM DỰ ÁN NÀY?
* **Vấn đề thực tế:** Vấn nạn thư rác quảng cáo, email giả mạo (phishing) đánh cắp thông tin tài khoản gây nghẽn hòm thư và mất an toàn số.
* **Mục tiêu:** Xây dựng một hệ thống phân loại tự động có khả năng phát hiện chính xác email rác, đồng thời không được chặn nhầm các email công việc quan trọng .

---

### 2. WHAT — HỆ THỐNG LÀM ĐƯỢC NHỮNG GÌ?
* **Đầu vào (Input):** Văn bản tiêu đề (Subject) và nội dung (Body) của email.
* **Đầu ra (Output):** Nhãn phân loại nhị phân (`Ham - 0`: Thư thường; `Spam - 1`: Thư rác) kèm xác suất rủi ro.
* **4 Tầng tính năng chính:**
  1. **Khám phá & Làm sạch dữ liệu (EDA & Clean):** Phân tích phân phối, bóc tách mã HTML, loại bỏ stopwords, xử lý khuyết.
  2. **Kỹ thuật đặc trưng (Feature Engineering):** Trích xuất TF-IDF từ vựng, lọc đặc trưng bằng Chi-square, đóng gói Pipeline chống rò rỉ dữ liệu (Data Leakage).
  3. **Huấn luyện mô hình (Model Training):** Huấn luyện đối đầu 3 thuật toán (Multinomial Naive Bayes, Logistic Regression, Linear SVM) với kiểm định chéo 5-Fold Stratified CV.
  4. **Đánh giá & Triển khai (Evaluation & App):** Đánh giá trên tập Test độc lập (F1, Confusion Matrix, Error Analysis) và tích hợp vào Web Demo Streamlit.

---

### 3. WHO — AI SỬ DỤNG VÀ AI XÂY DỰNG?
* **Người sử dụng:** Người dùng email cá nhân và nhân viên văn phòng cần bộ lọc thông minh, tự động cảnh báo thư khả nghi.
* **Người xây dựng:** Nhóm 3 gồm 7 thành viên phụ trách theo từng khâu chuyên môn từ EDA, Xử lý đặc trưng, Cài đặt thuật toán đến Xây dựng ứng dụng Demo.

---

### 4. WHERE — CẤU TRÚC VÀ TỔ CHỨC DỰ ÁN Ở ĐÂU?
* Toàn bộ mã nguồn, dữ liệu, mô hình và tài liệu được tổ chức chuẩn chỉnh trong thư mục dự án:

```text
PRACTICE 1_CLASSIFYING_SPAM_EMAILS/
├── data/
│   ├── raw/
│   │   └── CEAS_08.csv                       # Dữ liệu gốc 39.154 email thực tế
│   └── processed/
│       └── clean_emails.csv                  # Dữ liệu đã làm sạch HTML và khử trùng lặp
├── docs/
│   ├── project_description.md                # Bản đặc tả dự án ngắn gọn
│   ├── project_design.md                     # Bản thiết kế hệ thống theo 5W1H (file này)
│   ├── TASKS.md                              # Bảng phân rã công việc chi tiết 7 thành viên
│   ├── visualization_tutorial.md             # Hướng dẫn vẽ và đọc biểu đồ
│   └── algorithms/                           # Thư mục tài liệu chi tiết 3 thuật toán
│       ├── logistic_regression.md
│       ├── naive_bayes_classifier.md
│       └── support_vector_machine_classifier.md
├── models/
│   ├── preprocessor.joblib                   # Pipeline tiền xử lý văn bản + TF-IDF
│   ├── best_model.joblib                     # Mô hình tối ưu nhất sau tinh chỉnh
│   └── metadata.json                         # Chỉ số đánh giá và top từ khóa rác
├── notebooks/                                # Quy trình Walkthrough 3 bước
│   ├── 01_data_cleaning_and_eda.ipynb        # Khám phá & Làm sạch văn bản
│   ├── 02_feature_engineering_and_pipeline.ipynb # Kỹ thuật đặc trưng & Chống rò rỉ
│   └── 03_model_training_evaluation_and_explain.ipynb # Huấn luyện đa mô hình & Đánh giá
├── scripts/
│   ├── run_predict.py                        # Giao diện dòng lệnh CLI phân loại nhanh
│   ├── demo_algorithms.py                    # Runner kiểm thử 3 thuật toán Scratch trên dataset thật
│   └── algorithms/                           # Thư viện thuật toán tự cài đặt From Scratch
│       ├── logistic_regression.py
│       ├── naive_bayes.py
│       └── support_vector_machine.py
├── app.py                                    # Giao diện Web tương tác trực quan (Streamlit)
├── requirements.txt                          # Danh sách thư viện cần thiết
└── README.md                                 # Hướng dẫn tổng quan chạy toàn bộ dự án
```

---

### 5. WHEN — TIẾN TRÌNH & THỜI GIAN THỰC HIỆN KHI NÀO?
* **Quy trình thực thi:** Thực hiện tuần tự qua 3 chặng Notebooks (Khám phá $\rightarrow$ Xử lý đặc trưng $\rightarrow$ Huấn luyện & Đánh giá).
* **Thời gian phản hồi suy luận:** Khi đưa vào ứng dụng thực tế (Web Demo / CLI), xử lý và đưa ra kết quả phân loại cho 1 email mới.

---

### 6. HOW — HỆ THỐNG VẬN HÀNH NHƯ THẾ NÀO?
* **Dòng chảy dữ liệu 1 chiều:**
```text
Email thô (Raw Text) 
  └──> Làm sạch văn bản (Khử HTML, ký tự đặc biệt) 
  └──> Vector hóa TF-IDF (Tần suất & Trọng số từ khóa) 
  └──> Tuyển chọn đặc trưng (Chi-Square) 
  └──> Mô hình dự đoán (Naive Bayes / Logistic / SVM) 
  └──> Xuất nhãn (Spam/Ham) và Cảnh báo mức độ rủi ro.
```
* **Cách vận hành thử nghiệm:** Chạy ứng dụng Web Demo bằng 1 lệnh duy nhất: `streamlit run app.py` hoặc chạy runner kiểm thử thuật toán Scratch: `python scripts/demo_algorithms.py`.
