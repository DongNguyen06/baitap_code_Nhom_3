# SPAMGUARD-ML: PHÂN LOẠI EMAIL RÁC BẰNG HỌC MÁY
## PRACTICE 1: CLASSIFYING SPAM EMAILS — NHÓM 3 (UTH)

> **Môn học:** Học máy (Machine Learning) — Trường Đại học Giao thông Vận tải TP.HCM (UTH)  
> **Giảng viên hướng dẫn:** Tiến sĩ Nguyễn Thị Khánh Tiên  
> **Bài toán:** Phân loại nhị phân Email Spam / Ham (Spam Email Classification)  
> **Kiến trúc triển khai:** 3 Notebooks Walkthrough chuyên sâu theo chuẩn của Will Koehrsen & TS. Nguyễn Thị Khánh Tiên

---

## 📌 1. GIỚI THIỆU DỰ ÁN
Hệ thống giải quyết bài toán phân loại email nhị phân: Xác định email là thư hợp lệ (`Ham - 0`) hay thư rác (`Spam - 1`) dựa trên tiêu đề, nội dung và các thuộc tính liên kết.

Toàn bộ quy trình ML được tổ chức theo **kiến trúc 3 Notebooks tuần tự (Modular Walkthrough)**:
1. `01_data_cleaning_and_eda.ipynb`: Khám phá dữ liệu toàn diện 4 thuộc tính (EDA) và làm sạch văn bản email chuyên sâu (`clean_text`).
2. `02_feature_engineering_and_pipeline.ipynb`: Tạo biến Domain, phân chia Train/Test phân tầng chống rò rỉ dữ liệu, trích xuất TF-IDF và chọn lọc đặc trưng Chi-square.
3. `03_model_training_evaluation_and_explain.ipynb`: Huấn luyện 3 mô hình phân loại (Naive Bayes, Logistic Regression, Linear SVM), kiểm định chéo 5-Fold Stratified CV, đánh giá trên Test set và trích xuất Top từ khóa spam quan trọng.

---

## 👥 2. THÀNH VIÊN VÀ PHÂN CÔNG NHIỆM VỤ (TỔNG: 100%)

| STT | Họ và tên | Vai trò | Đóng góp | Vị trí phụ trách trong dự án / Repo |
| :---: | :--- | :---: | :---: | :--- |
| 1 | **Nguyễn Xuân Đông** | **Trưởng nhóm** | **18%** | **Kiến trúc tổng thể, Quản trị Git, Notebook 1 — Mục 1 (EDA Nền tảng), Web Demo Streamlit (`app.py`) & CLI** |
| 2 | **Mai An Thịnh** | Thành viên | **14%** | **Notebook 1 — Mục 2:** EDA Chuyên sâu (Đa biến tương quan, Ngoại lai IQR & Bảng đúc kết định hướng) |
| 3 | **Nguyễn Minh Nhật** | Thành viên | **14%** | **Notebook 1 — Mục 3:** Tiền xử lý dữ liệu & Làm sạch văn bản NLP (`clean_text`, khử trùng lặp, missing) |
| 4 | **Trương Thành Công** | Thành viên | **14%** | **Notebook 2 — Mục 4:** FE Biến đổi (Log, Binning) & Tạo biến Domain (`caps_ratio`, `!`, `$`) |
| 5 | **Võ Thanh Phú** | Thành viên | **14%** | **Notebook 2 — Mục 5:** FE Trích xuất TF-IDF, Lọc Chi-square & Pipeline phân chia Train/Test chống Leak |
| 6 | **Võ Khôi Nguyên** | Thành viên | **14%** | **Notebook 3 — Mục 6:** Huấn luyện 3 mô hình (Naive Bayes, Logistic L1, Linear SVM) + 5-Fold Stratified CV |
| 7 | **Mai Hoàng Danh** | Thành viên | **12%** | **Notebook 3 — Mục 7:** Đánh giá Test, Confusion Matrix, Error Analysis & Feature Importance |
| | **Tổng cộng** | | **100%** | **Hoàn thành toàn diện toàn bộ yêu cầu đồ án môn học** |

### Sơ đồ luồng thực hiện tuần tự qua 3 Notebooks:
```text
[CEAS_08.csv]
      │
      ▼
📓 NOTEBOOK 1: 01_data_cleaning_and_eda.ipynb (Khám phá & Làm sạch văn bản)
   ├── TV 1 (18%): Nguyễn Xuân Đông (Trưởng nhóm) — EDA 1 (Cấu trúc), EDA 2 (Chất lượng), EDA 3 (Đơn biến 4 thuộc tính)
   ├── TV 2 (14%): Mai An Thịnh — EDA 4 (Đa biến tương quan, Ngoại lai IQR) & EDA 5 (Bảng đúc kết EDA)
   └── TV 3 (14%): Nguyễn Minh Nhật — Tiền xử lý (Khử trùng, điền missing, clean_text) ──> [clean_emails.csv]
      │
      ▼
📓 NOTEBOOK 2: 02_feature_engineering_and_pipeline.ipynb (Kỹ thuật đặc trưng & Pipeline)
   ├── TV 4 (14%): Trương Thành Công — FE Biến đổi (Log, Binning) & Tạo biến Domain (caps_ratio, !, $)
   └── TV 5 (14%): Võ Thanh Phú — Chia Train/Test phân tầng chống Leak, TF-IDF Vectorizer, Lọc Chi-square
      │                             └──> [train_test_data.joblib & preprocessor.joblib]
      ▼
📓 NOTEBOOK 3: 03_model_training_evaluation_and_explain.ipynb (Huấn luyện & Đánh giá chuyên sâu)
   ├── TV 6 (14%): Võ Khôi Nguyên — Huấn luyện 3 mô hình (Naive Bayes, Logistic L1, Linear SVM) + 5-Fold Stratified CV
   └── TV 7 (12%): Mai Hoàng Danh — Đánh giá Test, Confusion Matrix, Error Analysis & Top từ khóa Spam
      │                             └──> [best_model.joblib & metadata.json]
      ▼
🚀 NGUYỄN XUÂN ĐÔNG TRIỂN KHAI WEB DEMO STREAMLIT (app.py) & CLI TEST (scripts/run_predict.py)
```

---

## 🚀 3. HƯỚNG DẪN KHỞI CHẠY (QUICK START)

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

### Bước 3: Khởi chạy Ứng dụng Web Demo Streamlit
```bash
streamlit run app.py
```

---

## 📂 4. CẤU TRÚC THƯ MỤC CHUẨN
```text
PRACTICE 1_CLASSIFYING_SPAM_EMAILS/
├── data/
│   ├── raw/CEAS_08.csv                       # Dữ liệu nguồn 39.154 email
│   └── processed/clean_emails.csv            # Dữ liệu sạch sau tiền xử lý (Output NB1)
├── docs/                                     # Bộ tài liệu đặc tả dự án
│   ├── project_description.md                # 7 mục đặc tả dự án chuẩn (dưới 200 từ)
│   ├── project_design.md                     # Thiết kế hệ thống theo 4 tính năng (dưới 200 từ)
│   └── TASKS.md                              # Phân công 7 thành viên theo 3 Notebooks
├── models/                                   # Nơi lưu trữ pipeline và mô hình
│   ├── preprocessor.joblib                   # Pipeline TF-IDF + Chi-square (Output NB2)
│   ├── best_model.joblib                     # Mô hình tối ưu nhất dùng cho web app (Output NB3)
│   ├── naive_bayes.joblib                    # Mô hình 1: Multinomial Naive Bayes
│   ├── logistic_regression.joblib            # Mô hình 2: Logistic Regression (L1)
│   ├── linear_svm.joblib                     # Mô hình 3: Linear SVM
│   └── metadata.json                         # Kết quả đối sánh Accuracy, Precision, Recall, F1
├── notebooks/                                # KHÔNG GIAN 3 NOTEBOOKS WALKTHROUGH
│   ├── 01_data_cleaning_and_eda.ipynb        # [TV1, TV2, TV3] Khám phá & Làm sạch
│   ├── 02_feature_engineering_and_pipeline.ipynb # [TV4, TV5] Tạo biến & Pipeline
│   └── 03_model_training_evaluation_and_explain.ipynb # [TV6, TV7] Huấn luyện, Đánh giá & Phân tích lỗi
├── scripts/
│   └── run_predict.py                        # [TV1] Ô Chat dòng lệnh tương tác suy luận nhanh
├── app.py                                    # [TV1] Web Demo Streamlit trực quan phân loại email
├── requirements.txt                          # Danh sách thư viện cần thiết
└── README.md                                 # Hướng dẫn tổng quan bài thực hành (file này)
```
