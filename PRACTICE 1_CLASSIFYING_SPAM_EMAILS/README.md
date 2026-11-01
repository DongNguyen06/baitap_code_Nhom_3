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

### Bước 3: Chạy Kiểm thử 3 Thuật toán Scratch trên Dataset Thật
```bash
python scripts/demo_algorithms.py
```

### Bước 4: Khởi chạy Ứng dụng Web Demo Streamlit
```bash
streamlit run app.py
```

---