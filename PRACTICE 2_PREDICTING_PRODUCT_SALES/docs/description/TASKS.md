# PHÂN CÔNG NHIỆM VỤ ĐỒ ÁN — NHÓM 3
## PRACTICE 2: PREDICTING PRODUCT SALES (KHOA CNTT - UTH)

> **Mục tiêu:** Tổ chức thực hiện bài thực hành Học máy theo kiến trúc 4 chặng độc lập tuần tự (*4-Stage Machine Learning Pipeline*), phân định rõ rệt giữa **tầng Module thuật toán cốt lõi** (do các thành viên phụ trách) và **tầng Kiến trúc điều phối & Notebooks demo** (do Trưởng nhóm phụ trách tích hợp).  

---

## 1. SƠ ĐỒ DÒNG CHẢY CÔNG VIỆC

```text
[Amazon Sale Report.csv (128.975 dòng thô)]
      │
      ▼
⚙️ CHẶNG 1: KHÁM PHÁ DỮ LIỆU BÁN HÀNG TMĐT (EDA - READ-ONLY 100%)
   ├── Module cốt lõi: data/scripts/eda.py (Mai An Thịnh)
   │     • Khảo sát cấu trúc thuộc tính, phân tích đơn biến Amount, tính mùa vụ, phân bổ địa lý & ngoại lai IQR
   └── Kịch bản thực thi: scripts/run_01_eda.py ➔ outputs/stage_01.json & docs/report/01_data_report.html
      │
      ▼
⚙️ CHẶNG 2: TIỀN XỬ LÝ DỮ LIỆU & KỸ THUẬT ĐẶC TRƯNG (FE & PIPELINE)
   ├── Module tiền xử lý: data/scripts/data_preprocessing.py (Nguyễn Minh Nhật)
   │     • Lọc đơn hủy, loại đơn Amount <= 0, khử trùng lặp ➔ data/processed/clean_amazon_sales.csv (102.852 dòng sạch)
   ├── Module trích xuất: scripts/feature_engineering.py (Trương Thành Công)
   │     • Kỹ thuật 11 đặc trưng TMĐT: chuỗi thời gian, cờ khuyến mãi is_promoted, cờ khách sỉ is_b2b, Top 10 bang
   ├── Module phân tách & đóng gói: scripts/feature_engineering.py (Võ Thanh Phú)
   │     • Phân chia Train/Test 80/20 (Zero Leakage), lọc đa cộng tuyến (|r| > 0.6), ColumnTransformer (Scaler + OHE)
   │     • Xuất train.csv, test.csv, train_test_data.joblib & preprocessor.joblib
   └── Kịch bản thực thi: scripts/run_02_fe.py ➔ outputs/stage_02.json & docs/report/02_feature_pipeline_report.html
      │
      ▼
⚙️ CHẶNG 3: HUẤN LUYỆN 3 MÔ HÌNH SCRATCH & KIỂM ĐỊNH CHÉO (TRAINING & CV)
   ├── Module huấn luyện: scripts/model_training.py (Võ Khôi Nguyên)
   │     • Tự cài đặt 3 thuật toán Scratch (models/algorithms/): Linear Regression, Decision Tree, Random Forest
   │     • Đóng gói Pipeline chuẩn scikit-learn, kiểm định chéo 5-Fold Cross Validation trên 82.281 mẫu, GridSearchCV
   └── Kịch bản thực thi: scripts/run_03_train.py ➔ outputs/stage_03.json & docs/report/03_model_training_report.html
      │
      ▼
⚙️ CHẶNG 4: ĐÁNH GIÁ ĐỘC LẬP TEST SET & BÓC TÁCH ĐẶC TRƯNG (EVALUATION & ARTIFACTS)
   ├── Module đánh giá: scripts/model_evaluation.py (Mai Hoàng Danh)
   │     • Đánh giá khách quan 20.571 mẫu Test set, nghịch đảo expm1 về thang đo tiền tệ INR, đo độ trễ suy luận
   │     • Trực quan hóa Actual vs Predicted, Residual Plot, Top 15 đặc trưng chi phối doanh số
   │     • Đóng gói hiện vật Quán quân: models/saved_models/best_model.joblib & metadata.json
   └── Kịch bản thực thi: scripts/run_04_eval.py ➔ outputs/stage_04.json & docs/report/04_model_evaluation_report.html
      │
      ▼
🚀 TẦNG KIẾN TRÚC ĐIỀU PHỐI, BÁO CÁO TOÀN TRÌNH & NOTEBOOKS TRÌNH DIỄN
   └── Nguyễn Xuân Đông (Trưởng nhóm):
         • Master Orchestrator: scripts/run_pipeline.py (hỗ trợ cờ --quick, zero-drift run_log.html)
         • Hệ thống tiện ích: scripts/common.py, scripts/logger_utils.py, scripts/report_builder.py
         • Bộ kiểm thử tự động toàn diện: scripts/test_all_components.py (8/8 nhóm kiểm tra)
         • Trọn bộ 4 Jupyter Notebooks trình diễn (Demo Layer):
             1. notebooks/01_exploratory_data_analysis.ipynb
             2. notebooks/02_data_preprocessing_and_feature_engineering.ipynb
             3. notebooks/03_model_training_evaluation_and_explain.ipynb
             4. notebooks/demo.ipynb (Giao diện suy luận doanh số thời gian thực đa mô hình)
```

---

## 2. BẢNG PHÂN CÔNG NHIỆM VỤ CHI TIẾT

| STT | Thành viên | Vai trò | Tỷ lệ | Module / Hạng mục phụ trách | Sản phẩm nghiệm thu cụ thể |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **1** | **Nguyễn Xuân Đông** | **Trưởng nhóm** | **16%** | • `scripts/run_pipeline.py`<br>• `scripts/run_01_eda.py` ➔ `run_04_eval.py`<br>• `scripts/common.py`<br>• `scripts/logger_utils.py`<br>• `scripts/report_builder.py`<br>• `scripts/test_all_components.py`<br>• **Toàn bộ 4 Notebooks (`notebooks/`)** | • **Thiết kế kiến trúc hệ thống 4 chặng:** Xây dựng Master Orchestrator điều phối toàn trình, cơ chế logging Zero-drift và hỗ trợ cờ demo `--quick`.<br>• **Phát triển bộ công cụ nền tảng:** Bộ tiện ích phong bì JSON (`common.py`), hệ thống logging UTF-8 (`logger_utils.py`) và bộ sinh tự động 6 báo cáo HTML Base64 standalone (`report_builder.py`).<br>• **Xây dựng bộ kiểm thử tự động:** Kiểm tra toàn diện 8/8 thành phần mã nguồn và cú pháp Python AST (`scripts/test_all_components.py`).<br>• **Chịu trách nhiệm toàn bộ tầng Notebooks trình diễn:** Tích hợp code từ các module backend thành 4 notebook hoàn chỉnh, bảo đảm mô hình Sandwich và quy chuẩn `outputs: []` (`01_exploratory_data_analysis.ipynb`, `02_data_preprocessing_and_feature_engineering.ipynb`, `03_model_training_evaluation_and_explain.ipynb`, `notebooks/demo.ipynb`). |
| **2** | **Mai An Thịnh** | Thành viên | **14%** | • `data/scripts/eda.py` | • Xây dựng module khám phá dữ liệu Chặng 1 EDA thuần túy (Read-Only 100%).<br>• Viết hàm khảo sát cấu trúc (`inspect_data_structure`, `get_univariate_stats`), phân tích đa biến ngành hàng (`get_bivariate_analysis`) và quét ngoại lai IQR (`detect_iqr_outliers`).<br>• Trực quan hóa phân phối doanh số Amount, tính mùa vụ chu kỳ thời gian và phân bổ doanh số theo bang địa lý. |
| **3** | **Nguyễn Minh Nhật** | Thành viên | **14%** | • `data/scripts/data_preprocessing.py` | • Xây dựng module làm sạch dữ liệu thương mại điện tử Chặng 2 (`clean_amazon_sales_data`, `handle_missing_and_duplicates`).<br>• Lọc bỏ triệt để đơn hàng hủy (`Status` chứa Cancelled), lọc đơn có Amount $\le 0$ hoặc Qty $\le 0$, điền khuyết theo ngữ cảnh nghiệp vụ.<br>• Thực thi nghiêm ngặt nguyên tắc Zero Duplicate Leakage và xuất tệp dữ liệu sạch `data/processed/clean_amazon_sales.csv` (102.852 dòng sạch). |
| **4** | **Trương Thành Công** | Thành viên | **14%** | • `scripts/feature_engineering.py` (Phần đặc trưng miền) | • Xây dựng hàm trích xuất 11 đặc trưng miền nghiệp vụ thương mại điện tử (`extract_domain_features`).<br>• Trích xuất yếu tố mùa vụ từ `Date` (`month`, `day_of_week`, `is_weekend`), cờ khuyến mãi `is_promoted` và cờ khách sỉ `is_b2b`.<br>• Thu gọn không gian chiều địa phương `ship-state` thành Top 10 bang lớn có doanh số cao nhất và nhóm gộp `'Other'`. |
| **5** | **Võ Thanh Phú** | Thành viên | **14%** | • `scripts/feature_engineering.py` (Phần phân tách & Pipeline) | • Xây dựng hàm phân chia Train/Test 80/20 độc lập (`split_train_test`) với `random_state=42`.<br>• Lập trình thuật toán quét và loại bỏ đa cộng tuyến (`detect_collinear_features` với $\|r\| > 0.6$).<br>• Xây dựng bộ tiền xử lý `build_column_transformer` (StandardScaler cho biến số, OneHotEncoder cho biến danh mục).<br>• Xuất các tệp đối soát trực quan `train.csv`, `test.csv` và gói nạp nhị phân hiệu năng cao `train_test_data.joblib`, `preprocessor.joblib`. |
| **6** | **Võ Khôi Nguyên** | Thành viên | **14%** | • `scripts/model_training.py`<br>• `models/algorithms/` | • Chịu trách nhiệm module huấn luyện Chặng 3: Tự cài đặt 3 thuật toán hồi quy Scratch thuần NumPy (Linear Regression OLS + L2, Decision Tree Regressor với Quantile Splits, Random Forest Regressor với Bagging).<br>• Đóng gói Pipeline tích hợp scikit-learn (`build_training_pipelines`).<br>• Thiết lập quy trình kiểm định chéo 5-Fold Cross Validation trên 82.281 mẫu Train và tinh chỉnh siêu tham số `GridSearchCV`. |
| **7** | **Mai Hoàng Danh** | Thành viên | **14%** | • `scripts/model_evaluation.py` | • Chịu trách nhiệm module đánh giá độc lập Chặng 4 (`evaluate_holdout_test`) trên 20.571 mẫu Test set.<br>• Chuyển đổi nghịch đảo `expm1` có chốt chặn chống tràn số, tính toán $R^2$, RMSE, MAE theo thang đo tiền tệ INR thực tế.<br>• Vẽ biểu đồ Thực tế vs Dự đoán, Residual Plot kiểm tra phần dư, bóc tách Top 15 đặc trưng chi phối doanh số và đóng gói `best_model.joblib` kèm `metadata.json`. |
| **TỔNG** | **7 thành viên** | | **100%** | | **Hoàn thành toàn diện bài thực hành Predicting Product Sales** |

---
