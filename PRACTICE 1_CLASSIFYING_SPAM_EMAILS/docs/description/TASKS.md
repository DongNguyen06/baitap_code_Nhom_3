# PHÂN CÔNG NHIỆM VỤ ĐỒ ÁN — NHÓM 3
## PRACTICE 1: CLASSIFYING SPAM EMAILS (KHOA CNTT - UTH)

> **Mục tiêu:** Tổ chức thực hiện bài thực hành Học máy theo chu trình 3 chặng tuần tự (*Walkthrough Pipeline*), phân định rõ trách nhiệm của 7 thành viên từ khâu xử lý dữ liệu đến triển khai ứng dụng.  
> **Lưu ý cốt lõi về cài đặt thuật toán:** Cả **7 thành viên đều cùng tham gia** nghiên cứu cơ sở lý thuyết toán học và cài đặt 3 thuật toán From Scratch bằng NumPy thuần túy (`models/algorithms/`). Các nhiệm vụ trong bảng dưới đây phân định trách nhiệm phụ trách chính cho từng mắt xích trong chu trình Pipeline thực thi.

---

## 1. SƠ ĐỒ DÒNG CHẢY CÔNG VIỆC

```text
[CEAS_08.csv]
      │
      ▼
📓 CHẶNG 1: 01_data_cleaning_and_eda.ipynb (Khám phá & Làm sạch NLP)
   ├── Nguyễn Xuân Đông (Trưởng nhóm - 16%): Khảo sát cấu trúc, chất lượng & Đơn biến (data/scripts/eda.py)
   ├── Mai An Thịnh (14%): Đa biến, Phân phối độ dài & Quét ngoại lai IQR (data/scripts/eda.py)
   └── Nguyễn Minh Nhật (14%): Xử lý missing, lọc regex clean_text & khử trùng lặp (data/scripts/data_preprocessing.py)
      │ └──> [data/processed/clean_emails.csv (32.660 email sạch)]
      ▼
📓 CHẶNG 2: 02_feature_engineering_and_pipeline.ipynb (Kỹ thuật đặc trưng & Pipeline)
   ├── Trương Thành Công (14%): Trích xuất 7 đặc trưng miền & Biến đổi log nén phân phối lệch
   └── Võ Thanh Phú (14%): Phân chia Train/Test 80/20, xuất trực quan & ColumnTransformer (TF-IDF + Chi2 + MinMaxScaler)
      │ └──> [train.csv, test.csv, train_test_data.joblib & preprocessor.joblib]
      ▼
📓 CHẶNG 3: 03_model_training_evaluation_and_explain.ipynb (Huấn luyện & Đánh giá)
   ├── Cả 7 thành viên: Cùng nghiên cứu toán học & cài đặt 3 thuật toán Scratch (models/algorithms/)
   ├── Võ Khôi Nguyên (14%): Đóng gói Pipeline huấn luyện & 5-Fold Stratified CV (scripts/model_training.py)
   └── Mai Hoàng Danh (14%): Đánh giá Test 6.532 mẫu, Confusion Matrix & Top từ khóa (scripts/model_evaluation.py)
      │ └──> [models/saved_models/ (best_model.joblib ~307 KB)]
      ▼
🚀 TỔNG HỢP & TRIỂN KHAI:
   └── Nguyễn Xuân Đông (Trưởng nhóm - 16%): Notebook Demo tương tác đa mô hình (notebooks/demo.ipynb) & Bộ kiểm thử (scripts/test_all_components.py)
```

---

## 2. BẢNG PHÂN CÔNG NHIỆM VỤ CHI TIẾT

| STT | Thành viên | Vai trò | Tỷ lệ | Module / Tệp phụ trách | Sản phẩm nghiệm thu cụ thể |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **1** | **Nguyễn Xuân Đông** | **Trưởng nhóm** | **16%** | `data/scripts/eda.py`<br>`notebooks/demo.ipynb`<br>`scripts/test_all_components.py` | • Thiết lập cấu trúc dự án và quản trị Git.<br>• Viết hàm khảo sát cấu trúc dữ liệu (`inspect_data_structure`, `get_univariate_stats`).<br>• Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch.<br>• Xây dựng Notebook Demo tương tác suy luận thời gian thực hỗ trợ chọn đa mô hình (`notebooks/demo.ipynb`).<br>• Phát triển bộ kiểm thử tự động toàn diện hệ thống mã nguồn (`scripts/test_all_components.py`). |
| **2** | **Mai An Thịnh** | Thành viên | **14%** | `data/scripts/eda.py`<br>`notebooks/01_data_cleaning_and_eda.ipynb` | • Viết hàm phát hiện ngoại lai IQR (`detect_iqr_outliers`) và đối soát từ khóa (`get_top_keywords_comparison`).<br>• Phân tích đa biến, vẽ biểu đồ Boxplot độ dài văn bản theo nhãn.<br>• Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch.<br>• Tổng hợp Bảng kế hoạch hành động định hướng cho tiền xử lý và tạo đặc trưng. |
| **3** | **Nguyễn Minh Nhật** | Thành viên | **14%** | `data/scripts/data_preprocessing.py`<br>`notebooks/01_data_cleaning_and_eda.ipynb` | • Xây dựng module làm sạch dữ liệu (`handle_missing_values`, `clean_text`).<br>• Xóa thẻ HTML, URL, ký tự đặc biệt và lọc từ dừng Stopwords.<br>• Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch.<br>• Thực thi nguyên tắc Zero Duplicate Leakage và xuất tệp `clean_emails.csv`. |
| **4** | **Trương Thành Công** | Thành viên | **14%** | `scripts/feature_engineering.py`<br>`notebooks/02_feature_engineering_and_pipeline.ipynb` | • Xây dựng hàm trích xuất 7 đặc trưng miền nghiệp vụ (`extract_domain_features`).<br>• Thực hiện biến đổi log (`log_text_len`) nén độ lệch phân phối văn bản siêu dài.<br>• Phân tích phân phối chữ viết hoa (`caps_ratio`), dấu cảm thán (`!`) và ký hiệu tiền (`$`).<br>• Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch. |
| **5** | **Võ Thanh Phú** | Thành viên | **14%** | `scripts/feature_engineering.py`<br>`notebooks/02_feature_engineering_and_pipeline.ipynb` | • Xây dựng hàm phân chia Train/Test 80/20 phân tầng (`split_stratified_train_test`) và hàm xuất dữ liệu trực quan `export_train_test_csv` (`train.csv`, `test.csv`).<br>• Xây dựng bộ tiền xử lý đa tầng `build_column_transformer` (TF-IDF + Chi-Square + MinMaxScaler).<br>• Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch.<br>• Đóng gói và lưu trữ an toàn `train_test_data.joblib` và `preprocessor.joblib`. |
| **6** | **Võ Khôi Nguyên** | Thành viên | **14%** | `scripts/model_training.py`<br>`notebooks/03_model_training_evaluation_and_explain.ipynb` | • Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch.<br>• Chịu trách nhiệm chính xây dựng module huấn luyện và đóng gói Pipeline (`create_scratch_models`, `build_training_pipelines`).<br>• Thực hiện kiểm định chéo 5-Fold Stratified CV trên tập Train.<br>• Tinh chỉnh siêu tham số tối ưu qua `GridSearchCV` và lưu mô hình vào `models/saved_models/`. |
| **7** | **Mai Hoàng Danh** | Thành viên | **14%** | `scripts/model_evaluation.py`<br>`notebooks/03_model_training_evaluation_and_explain.ipynb` | • Xây dựng module đánh giá độc lập trên 6.532 mẫu Test set (`evaluate_holdout_test`).<br>• Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch.<br>• Trực quan hóa Ma trận nhầm lẫn (Confusion Matrix) và phân tích lỗi (FP vs FN).<br>• Trích xuất Top 15 từ khóa spam và đóng gói hiện vật nghiệm thu `best_model.joblib`. |
| **TỔNG** | **7 thành viên** | | **100%** | | **Hoàn thành toàn diện bài thực hành Classifying Spam Emails** |

---
