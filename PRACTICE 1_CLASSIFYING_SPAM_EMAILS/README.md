# SPAMGUARD-ML: PHÂN LOẠI EMAIL RÁC BẰNG HỌC MÁY
## PRACTICE 1: CLASSIFYING SPAM EMAILS — NHÓM 3 (UTH)

> **Môn học:** Học máy (Machine Learning) — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Bài toán:** Phân loại nhị phân Email Spam / Ham (Spam Email Classification)  
> **Kiến trúc triển khai:** Hệ thống 4 tầng kết hợp 3 Notebooks Walkthrough chuyên sâu, Thư viện thuật toán tự cài đặt From Scratch và Web Demo Streamlit tương tác thời gian thực.

---

## 📌 1. GIỚI THIỆU DỰ ÁN
Hệ thống giải quyết bài toán phân loại email nhị phân: Xác định email là thư hợp lệ (`Ham - 0`) hay thư rác (`Spam - 1`) dựa trên tiêu đề, nội dung và các thuộc tính liên kết.

Toàn bộ quy trình ML được tổ chức theo **kiến trúc 3 Notebooks tuần tự (Modular Walkthrough)**:
1. `01_data_cleaning_and_eda.ipynb`: Khám phá dữ liệu toàn diện 4 thuộc tính (EDA) và làm sạch văn bản email chuyên sâu (`clean_text`).
2. `02_feature_engineering_and_pipeline.ipynb`: Tạo biến Domain, phân chia Train/Test phân tầng chống rò rỉ dữ liệu, trích xuất TF-IDF và chọn lọc đặc trưng Chi-square.
3. `03_model_training_evaluation_and_explain.ipynb`: Huấn luyện 3 mô hình phân loại (Naive Bayes, Logistic Regression, Linear SVM), kiểm định chéo 5-Fold Stratified CV, đánh giá trên Test set và trích xuất Top từ khóa spam quan trọng.

> **Phân công nhiệm vụ:** Chi tiết phân rã công việc của 7 thành viên được quản lý tập trung tại [docs/TASKS.md](docs/TASKS.md).

---

## 🚀 2. HƯỚNG DẪN KHỞI CHẠY (QUICK START)

### Bước 1: Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
pip install -r requirements.txt
```

### Bước 2: Chạy tuần tự 3 Notebooks
Mở và thực thi các Notebook theo đúng thứ tự chuyển giao dữ liệu:
1. **Chạy Notebook 1:** Khám phá dữ liệu và sinh ra tệp `data/processed/clean_emails.csv`.
   ```bash
   jupyter notebook notebooks/01_data_cleaning_and_eda.ipynb
   ```
2. **Chạy Notebook 2:** Tạo đặc trưng Domain, chia Train/Test, trích xuất TF-IDF và lưu Pipeline.
   ```bash
   jupyter notebook notebooks/02_feature_engineering_and_pipeline.ipynb
   ```
3. **Chạy Notebook 3:** Huấn luyện 3 mô hình, kiểm định chéo 5-Fold, đánh giá Test và xuất `best_model.joblib`.
   ```bash
   jupyter notebook notebooks/03_model_training_evaluation_and_explain.ipynb
   ```

### Bước 3: Chạy độc lập các Thuật toán Tự cài đặt (From Scratch)
```bash
# Kiểm thử Logistic Regression Scratch
python scripts/algorithms/logistic_regression.py

# Kiểm thử Naive Bayes Scratch
python scripts/algorithms/naive_bayes.py

# Kiểm thử Linear SVM Scratch
python scripts/algorithms/support_vector_machine.py
```

### Bước 4: Khởi chạy Ứng dụng Web Demo Streamlit
```bash
streamlit run app.py
```

---

## 📂 3. CẤU TRÚC THƯ MỤC CHUẨN 4 TẦNG

```text
PRACTICE 1_CLASSIFYING_SPAM_EMAILS/
├── data/
│   ├── raw/CEAS_08.csv                       # Dữ liệu nguồn 39.154 email
│   └── processed/clean_emails.csv            # Dữ liệu sạch sau tiền xử lý (Output NB1)
├── docs/                                     # Bộ tài liệu học thuật & đặc tả dự án
│   ├── images/                               # Thư mục chứa biểu đồ trực quan hóa (.png)
│   ├── project_description.md                # Đặc tả bài toán tổng quan
│   ├── project_design.md                     # Thiết kế kiến trúc 4 giai đoạn
│   ├── TASKS.md                              # Bảng phân công nhiệm vụ 7 thành viên
│   ├── visualization_tutorial.md             # Cẩm nang trực quan hóa kết quả 3 thuật toán (42 dòng)
│   └── algorithms/                           # Thư mục chứa đặc tả thuật toán chuẩn 10 mục (< 70 dòng)
│       ├── logistic_regression.md            # Đặc tả Logistic Regression Scratch (59 dòng)
│       ├── naive_bayes.md                    # Đặc tả Multinomial Naive Bayes Scratch (49 dòng)
│       └── support_vector_machine.md         # Đặc tả Linear Support Vector Machine Scratch (50 dòng)
├── models/                                   # Nơi lưu trữ pipeline và mô hình đã huấn luyện
│   ├── preprocessor.joblib                   # Pipeline TF-IDF + Chi-square (Output NB2)
│   ├── best_model.joblib                     # Mô hình Linear SVM hiệu chuẩn Platt Scaling (Output NB3)
│   ├── naive_bayes.joblib                    # Mô hình 1: Multinomial Naive Bayes
│   ├── logistic_regression.joblib            # Mô hình 2: Logistic Regression (L1)
│   ├── linear_svm.joblib                     # Mô hình 3: Linear SVM
│   └── metadata.json                         # Metrics F1, Accuracy và Top 15 từ khóa spam thực tế
├── notebooks/                                # KHÔNG GIAN 3 NOTEBOOKS WALKTHROUGH
│   ├── 01_data_cleaning_and_eda.ipynb        # Khám phá & Làm sạch văn bản (Chuẩn sạch outputs)
│   ├── 02_feature_engineering_and_pipeline.ipynb # Tạo biến Domain & Pipeline Zero Leakage (Chuẩn sạch outputs)
│   └── 03_model_training_evaluation_and_explain.ipynb # Huấn luyện đa mô hình, Đánh giá & Explain (Chuẩn sạch outputs)
├── scripts/
│   ├── run_predict.py                        # CLI tương tác suy luận nhanh cho người dùng
│   └── algorithms/                           # THƯ VIỆN THUẬT TOÁN TỰ CÀI ĐẶT (FROM SCRATCH OOP)
│       ├── logistic_regression.py            # Class LogisticRegressionScratch thuần NumPy
│       ├── naive_bayes.py                    # Class MultinomialNaiveBayesScratch thuần NumPy
│       └── support_vector_machine.py         # Class LinearSVMScratch thuần NumPy
├── app.py                                    # Web Demo Streamlit trực quan phân loại email
├── requirements.txt                          # Danh sách thư viện phụ thuộc
└── README.md                                 # Hướng dẫn tổng quan bài thực hành (file này)
```
