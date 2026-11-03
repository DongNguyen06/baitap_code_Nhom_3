# PHÂN CÔNG NHIỆM VỤ ĐỒ ÁN — NHÓM 3
## PRACTICE 2: PREDICTING PRODUCT SALES (KHOA CNTT - UTH)

> **Mục tiêu:** Tổ chức thực hiện bài thực hành Học máy theo chu trình 3 chặng tuần tự (*Walkthrough Pipeline*), phân định rõ trách nhiệm của 7 thành viên từ khâu xử lý dữ liệu đến triển khai mô hình hồi quy doanh số.  
> **Lưu ý cốt lõi về cài đặt thuật toán:** Cả **7 thành viên đều cùng tham gia** nghiên cứu cơ sở lý thuyết toán học và cài đặt 3 thuật toán From Scratch bằng NumPy thuần túy (`models/algorithms/` gồm Linear Regression, Decision Tree Regressor, Random Forest Regressor). Các nhiệm vụ trong bảng dưới đây phân định trách nhiệm phụ trách chính cho từng mắt xích trong chu trình Pipeline thực thi.

---

## 1. SƠ ĐỒ DÒNG CHẢY CÔNG VIỆC

```text
[Amazon Sale Report.csv (128.975 dòng)]
      │
      ▼
📓 CHẶNG 1: 01_data_cleaning_and_eda.ipynb (Khám phá & Làm sạch dữ liệu thương mại điện tử)
   ├── Nguyễn Xuân Đông (Trưởng nhóm - 16%): Khảo sát cấu trúc, chất lượng & Đơn biến Amount (data/scripts/eda.py)
   ├── Mai An Thịnh (14%): Đa biến (Category, Bang, Mùa vụ) & Quét ngoại lai IQR (data/scripts/eda.py)
   └── Nguyễn Minh Nhật (14%): Lọc đơn hủy, chuẩn hóa text, khử trùng lặp (data/scripts/data_preprocessing.py)
      │ └──> [data/processed/clean_amazon_sales.csv (102.852 dòng sạch)]
      ▼
📓 CHẶNG 2: 02_feature_engineering_and_pipeline.ipynb (Kỹ thuật đặc trưng & Pipeline)
   ├── Trương Thành Công (14%): Trích xuất đặc trưng miền (Mùa vụ, is_promoted, is_b2b, Top 10 bang)
   └── Võ Thanh Phú (14%): Phân chia Train/Test 80/20, lọc đa cộng tuyến (|r| > 0.6) & ColumnTransformer
      │ └──> [train.csv, test.csv, train_test_data.joblib & preprocessor.joblib]
      ▼
📓 CHẶNG 3: 03_model_training_evaluation_and_explain.ipynb (Huấn luyện & Đánh giá)
   ├── Cả 7 thành viên: Cùng nghiên cứu toán học & cài đặt 3 thuật toán Scratch (models/algorithms/)
   ├── Võ Khôi Nguyên (14%): Đóng gói Pipeline hồi quy & 5-Fold Cross Validation (scripts/model_training.py)
   └── Mai Hoàng Danh (14%): Đánh giá Hold-out Test 20.571 mẫu, Residual Plot & Feature Importance (scripts/model_evaluation.py)
      │ └──> [models/saved_models/ (best_model.joblib, metadata.json)]
      ▼
🚀 TỔNG HỢP & TRIỂN KHAI:
   └── Nguyễn Xuân Đông (Trưởng nhóm - 16%): Notebook Demo tương tác đa mô hình (notebooks/demo.ipynb) & Bộ kiểm thử (scripts/test_all_components.py)
```

---

## 2. BẢNG PHÂN CÔNG NHIỆM VỤ CHI TIẾT

| STT | Thành viên | Vai trò | Tỷ lệ | Module / Tệp phụ trách | Sản phẩm nghiệm thu cụ thể |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **1** | **Nguyễn Xuân Đông** | **Trưởng nhóm** | **16%** | `data/scripts/eda.py`<br>`notebooks/demo.ipynb`<br>`scripts/test_all_components.py` | • Thiết lập cấu trúc dự án và quản trị Git.<br>• Viết hàm khảo sát cấu trúc dữ liệu (`inspect_data_structure`, `get_univariate_stats`).<br>• Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch.<br>• Xây dựng Notebook Demo tương tác suy luận doanh số thời gian thực (`notebooks/demo.ipynb`).<br>• Phát triển bộ kiểm thử tự động toàn diện hệ thống mã nguồn (`scripts/test_all_components.py`). |
| **2** | **Mai An Thịnh** | Thành viên | **14%** | `data/scripts/eda.py`<br>`notebooks/01_data_cleaning_and_eda.ipynb` | • Viết hàm phân tích đa biến (`get_bivariate_analysis`) và quét ngoại lai IQR (`detect_iqr_outliers`).<br>• Phân tích tương quan giữa doanh số với ngành hàng, địa lý và kênh phân phối.<br>• Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch.<br>• Đúc kết Bảng kế hoạch hành động định hướng cho tiền xử lý và tạo đặc trưng. |
| **3** | **Nguyễn Minh Nhật** | Thành viên | **14%** | `data/scripts/data_preprocessing.py`<br>`notebooks/01_data_cleaning_and_eda.ipynb` | • Xây dựng module làm sạch dữ liệu (`clean_amazon_sales_data`, `handle_missing_and_duplicates`).<br>• Lọc bỏ đơn hàng hủy, đơn Amount $\le 0$, điền khuyết theo ngữ cảnh nghiệp vụ.<br>• Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch.<br>• Thực thi nguyên tắc Zero Duplicate Leakage và xuất tệp `clean_amazon_sales.csv` (102.852 dòng). |
| **4** | **Trương Thành Công** | Thành viên | **14%** | `scripts/feature_engineering.py`<br>`notebooks/02_feature_engineering_and_pipeline.ipynb` | • Xây dựng hàm trích xuất đặc trưng miền nghiệp vụ thương mại điện tử (`extract_domain_features`).<br>• Trích xuất chuỗi thời gian (`month`, `day_of_week`, `is_weekend`), cờ khuyến mãi `is_promoted` và cờ khách sỉ `is_b2b`.<br>• Gom nhóm địa phương `ship-state` thành Top 10 bang lớn và nhóm `'Other'`.<br>• Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch. |
| **5** | **Võ Thanh Phú** | Thành viên | **14%** | `scripts/feature_engineering.py`<br>`notebooks/02_feature_engineering_and_pipeline.ipynb` | • Xây dựng hàm phân chia Train/Test 80/20 (`split_train_test`) và hàm xuất dữ liệu trực quan `export_train_test_csv`.<br>• Lập trình hàm quét và loại bỏ đa cộng tuyến (`detect_collinear_features` với $|r| > 0.6$).<br>• Xây dựng bộ tiền xử lý `build_column_transformer` (StandardScaler + OneHotEncoder).<br>• Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch.<br>• Đóng gói và lưu trữ an toàn `train_test_data.joblib` và `preprocessor.joblib`. |
| **6** | **Võ Khôi Nguyên** | Thành viên | **14%** | `scripts/model_training.py`<br>`notebooks/03_model_training_evaluation_and_explain.ipynb` | • Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch.<br>• Chịu trách nhiệm chính xây dựng module huấn luyện và đóng gói Pipeline (`create_scratch_models`, `build_training_pipelines`).<br>• Thực hiện kiểm định chéo 5-Fold Cross Validation trên tập Train.<br>• Tinh chỉnh siêu tham số tối ưu qua `GridSearchCV` và lưu mô hình vào `models/saved_models/`. |
| **7** | **Mai Hoàng Danh** | Thành viên | **14%** | `scripts/model_evaluation.py`<br>`notebooks/03_model_training_evaluation_and_explain.ipynb` | • Xây dựng module đánh giá độc lập trên 20.571 mẫu Test set (`evaluate_holdout_test`).<br>• Cùng toàn nhóm nghiên cứu & cài đặt 3 thuật toán Scratch.<br>• Trực quan hóa Biểu đồ Thực tế vs Dự đoán, Residual Plot và kiểm tra phần dư.<br>• Trích xuất Top 15 đặc trưng quan trọng (Feature Importances) và đóng gói `best_model.joblib`. |
| **TỔNG** | **7 thành viên** | | **100%** | | **Hoàn thành toàn diện bài thực hành Predicting Product Sales** |

---
