# BÀI THỰC HÀNH 1: PHÂN LOẠI EMAIL SPAM — NHÓM 3
## PRACTICE 1: CLASSIFYING SPAM EMAILS — GROUP 3

> **Môn học:** Học máy (Machine Learning) — UTH  
> **Bài toán:** Phân loại nhị phân Email Spam / Ham (`Spam = 1`, `Ham = 0`)  
> **Kiến trúc:** 4 Jupyter Notebooks, 3 thuật toán Scratch tự viết bằng NumPy thuần túy và Pipeline tự động hóa khép kín.

---

## 📌 1. GIỚI THIỆU DỰ ÁN
Dự án giải quyết bài toán lọc thư rác điện tử bằng Học máy trên bộ dữ liệu `CEAS_08.csv` (39.154 email gốc $\rightarrow$ 32.660 email sạch sau khử trùng lặp).

Quy trình thực hiện được chia thành 4 Notebooks chuyên sâu:
1. `notebooks/01_data_cleaning_and_eda.ipynb`: Khám phá dữ liệu toàn diện (EDA), quét ngoại lai IQR và làm sạch văn bản (`clean_text`), khử 6.494 mẫu trùng lặp.
2. `notebooks/02_feature_engineering_and_pipeline.ipynb`: Trích xuất 7 đặc trưng miền, vector hóa TF-IDF + Chi-square, chia Train/Test 80/20 phân tầng và xuất `train.csv`, `test.csv`.
3. `notebooks/03_model_training_evaluation_and_explain.ipynb`: Huấn luyện 3 mô hình Scratch, kiểm định 5-Fold Stratified CV, đánh giá trên Test set và trích xuất Top từ khóa spam quan trọng.
4. `notebooks/demo.ipynb`: Giao diện thử nghiệm suy luận email tương tác thời gian thực, hỗ trợ lựa chọn linh hoạt giữa 3 mô hình.

> **Phân công nhiệm vụ:** Chi tiết công việc của 7 thành viên được quản lý tại [docs/description/TASKS.md](docs/description/TASKS.md).  
> **Thiết kế chi tiết 5W1H:** Xem tại [docs/description/project_design.md](docs/description/project_design.md).

---

## 📊 2. BẢNG KẾT QUẢ NGHIỆM THU (TEST SET: 6.532 MẪU)

| Thuật toán (Scratch) | Accuracy | Precision | Recall | F1-Score | Bắt nhầm (FP) | Bỏ sót (FN) | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 **Linear SVM** | **98.73%** | **98.71%** | **98.62%** | **98.67%** | **40** | **43** | **Quán quân (Best Model)** |
| 🥈 **Logistic Regression** | 97.69% | 97.05% | 98.14% | 97.59% | 93 | 58 | Cân bằng tốt, xác suất mượt |
| 🥉 **Multinomial Naive Bayes** | 96.34% | 99.31% | 92.96% | 96.03% | 20 | 219 | Precision cao nhất, tốc độ nhanh |

---

## 🚀 3. HƯỚNG DẪN KHỞI CHẠY (QUICK START)

### Bước 1: Cài đặt thư viện phụ thuộc
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
pip install -r requirements.txt
```

### Bước 2: Thực thi dự án
Có 2 phương thức chạy linh hoạt:

* **Cách A — Chạy tự động toàn bộ Pipeline (Khuyến nghị cho CLI):**
  ```bash
  python scripts/run_pipeline.py
  ```

* **Cách B — Mở và chạy từng Notebook (Khuyến nghị cho nghiên cứu trực quan):**
  1. `jupyter notebook notebooks/01_data_cleaning_and_eda.ipynb`
  2. `jupyter notebook notebooks/02_feature_engineering_and_pipeline.ipynb`
  3. `jupyter notebook notebooks/03_model_training_evaluation_and_explain.ipynb`

### Bước 3: Thử nghiệm tương tác với Demo
Mở notebook demo để dán email bất kỳ và kiểm tra khả năng dự đoán của cả 3 mô hình:
```bash
jupyter notebook notebooks/demo.ipynb
```

### Bước 4: Chạy bộ kiểm thử tự động toàn diện
Hỗ trợ kiểm tra tính toàn vẹn cú pháp và logic của toàn bộ 8 nhóm thành phần trong dự án:

* **Chạy kiểm thử riêng từng bước theo nhiệm vụ từng thành viên (1–8):**
  ```bash
  python scripts/test_all_components.py --step 1   # Tiền xử lý (Nguyễn Minh Nhật)
  python scripts/test_all_components.py --step 4   # 3 thuật toán Scratch (Cả 7 thành viên)
  ```
* **Chạy kiểm thử toàn bộ hệ thống (Kiểm tra tích hợp):**
  ```bash
  python scripts/test_all_components.py
  ```

---

## 📁 4. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
PRACTICE 1_CLASSIFYING_SPAM_EMAILS/
├── data/
│   ├── raw/                      # Dữ liệu gốc (CEAS_08.csv)
│   ├── processed/                # Dữ liệu sạch (clean_emails.csv, train.csv, test.csv, train_test_data.joblib)
│   └── scripts/                  # Module tiền xử lý (data_preprocessing.py, eda.py)
├── docs/
│   ├── description/              # Tài liệu thiết kế (project_description.md, project_design.md, TASKS.md)
│   ├── images/                   # Biểu đồ trực quan hóa EDA và kết quả mô hình
│   └── report/                   # Các trang báo cáo HTML độc lập (index.html, 01_data_report.html, ...)
├── models/
│   ├── algorithms/               # Mã nguồn 3 thuật toán Scratch (logistic_regression.py, naive_bayes_classifier.py, support_vector_machine.py)
│   └── saved_models/             # Mô hình tối ưu đã lưu (best_model.joblib, preprocessor.joblib, metadata.json)
├── notebooks/                    # 4 Notebooks thực hành tuần tự và Demo
├── scripts/                      # Các kịch bản chạy tự động (run_pipeline.py, feature_engineering.py, ...)
├── README.md
└── requirements.txt
```
