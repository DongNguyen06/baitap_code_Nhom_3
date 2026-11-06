# PHÂN CÔNG NHIỆM VỤ ĐỒ ÁN — NHÓM 3
## PRACTICE 1: CLASSIFYING SPAM EMAILS (KHOA CNTT - UTH)

> **Mục tiêu:** Tổ chức thực hiện bài thực hành Học máy theo kiến trúc 4 chặng độc lập tuần tự (*4-Stage Machine Learning Pipeline*), phân định rõ rệt giữa **tầng Module thuật toán cốt lõi** (do các thành viên phụ trách) và **tầng Kiến trúc điều phối & Notebooks demo** (do Trưởng nhóm phụ trách tích hợp).  

---

## 1. SƠ ĐỒ DÒNG CHẢY CÔNG VIỆC

```text
[CEAS_08.csv (39.154 email thô)]
      │
      ▼
⚙️ CHẶNG 1: KHÁM PHÁ DỮ LIỆU EMAIL (EDA - READ-ONLY 100%)
   ├── Module cốt lõi: data/scripts/eda.py (Mai An Thịnh)
   │     • Khảo sát cấu trúc thuộc tính, phân tích đơn biến, đa biến và quét ngoại lai IQR
   └── Kịch bản thực thi: scripts/run_01_eda.py ➔ outputs/stage_01.json & docs/report/01_data_report.html
      │
      ▼
⚙️ CHẶNG 2: TIỀN XỬ LÝ NLP & KỸ THUẬT ĐẶC TRƯNG (FE & PIPELINE)
   ├── Module tiền xử lý: data/scripts/data_preprocessing.py (Nguyễn Minh Nhật)
   │     • Làm sạch Regex, khử trùng lặp Zero Duplicate Leakage ➔ data/processed/clean_emails.csv (32.660 email sạch)
   ├── Module trích xuất: scripts/feature_engineering.py (Trương Thành Công)
   │     • Trích xuất 7 đặc trưng miền nghiệp vụ, nén log độ dài văn bản
   ├── Module phân tách & đóng gói: scripts/feature_engineering.py (Võ Thanh Phú)
   │     • Phân chia Train/Test 80/20 phân tầng, ColumnTransformer (TF-IDF 5.000 + Chi-Square + MinMaxScaler)
   │     • Xuất train.csv, test.csv, train_test_data.joblib & preprocessor.joblib
   └── Kịch bản thực thi: scripts/run_02_fe.py ➔ outputs/stage_02.json & docs/report/02_feature_pipeline_report.html
      │
      ▼
⚙️ CHẶNG 3: HUẤN LUYỆN 3 MÔ HÌNH SCRATCH & KIỂM ĐỊNH CHÉO (TRAINING & CV)
   ├── Module huấn luyện: scripts/model_training.py (Võ Khôi Nguyên)
   │     • Tự cài đặt 3 thuật toán Scratch (models/algorithms/): Naive Bayes, Logistic Regression, Linear SVM
   │     • Đóng gói Pipeline chuẩn scikit-learn, kiểm định chéo 5-Fold Stratified CV, GridSearchCV
   └── Kịch bản thực thi: scripts/run_03_train.py ➔ outputs/stage_03.json & docs/report/03_model_training_report.html
      │
      ▼
⚙️ CHẶNG 4: ĐÁNH GIÁ ĐỘC LẬP TEST SET & XUẤT HIỆN VẬT (EVALUATION & ARTIFACTS)
   ├── Module đánh giá: scripts/model_evaluation.py (Mai Hoàng Danh)
   │     • Đánh giá khách quan 6.532 mẫu Test set, Confusion Matrix, Top 15 từ khóa cảnh báo spam
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
             4. notebooks/demo.ipynb (Giao diện suy luận thời gian thực đa mô hình)
```

---

## 2. BẢNG PHÂN CÔNG NHIỆM VỤ CHI TIẾT

| STT | Thành viên | Vai trò | Tỷ lệ | Module / Hạng mục phụ trách | Sản phẩm nghiệm thu cụ thể |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **1** | **Nguyễn Xuân Đông** | **Trưởng nhóm** | **16%** | • `scripts/run_pipeline.py`<br>• `scripts/run_01_eda.py` ➔ `run_04_eval.py`<br>• `scripts/common.py`<br>• `scripts/logger_utils.py`<br>• `scripts/report_builder.py`<br>• `scripts/test_all_components.py`<br>• **Toàn bộ 4 Notebooks (`notebooks/`)** | • **Thiết kế kiến trúc hệ thống 4 chặng:** Xây dựng Master Orchestrator điều phối toàn trình, cơ chế logging Zero-drift và hỗ trợ cờ demo `--quick`.<br>• **Phát triển bộ công cụ nền tảng:** Bộ tiện ích phong bì JSON (`common.py`), hệ thống logging UTF-8 (`logger_utils.py`) và bộ sinh tự động 6 báo cáo HTML Base64 standalone (`report_builder.py`).<br>• **Xây dựng bộ kiểm thử tự động:** Kiểm tra toàn diện 8/8 thành phần mã nguồn và cú pháp Python AST (`scripts/test_all_components.py`).<br>• **Chịu trách nhiệm toàn bộ tầng Notebooks trình diễn:** Tích hợp code từ các module backend thành 4 notebook hoàn chỉnh, bảo đảm mô hình Sandwich và quy chuẩn `outputs: []` (`01_exploratory_data_analysis.ipynb`, `02_data_preprocessing_and_feature_engineering.ipynb`, `03_model_training_evaluation_and_explain.ipynb`, `notebooks/demo.ipynb`). |
| **2** | **Mai An Thịnh** | Thành viên | **14%** | • `data/scripts/eda.py` | • Xây dựng module khám phá dữ liệu Chặng 1 EDA thuần túy (Read-Only 100%).<br>• Viết hàm khảo sát cấu trúc (`inspect_data_structure`, `get_univariate_stats`), phân tích đa biến và phát hiện ngoại lai IQR (`detect_iqr_outliers`).<br>• Đối soát phân phối độ dài văn bản theo nhãn và trích xuất từ khóa thô ban đầu (`get_top_keywords_comparison`). |
| **3** | **Nguyễn Minh Nhật** | Thành viên | **14%** | • `data/scripts/data_preprocessing.py` | • Xây dựng module tiền xử lý dữ liệu NLP Chặng 2 (`clean_text`, `handle_missing_values`).<br>• Xử lý triệt để thẻ HTML ngụy trang, URL ẩn, ký tự đặc biệt và lọc từ dừng Stopwords.<br>• Thực thi nghiêm ngặt nguyên tắc Zero Duplicate Leakage (loại bỏ 6.494 mẫu thư rác trùng lặp đa hình) và xuất tệp dữ liệu sạch `data/processed/clean_emails.csv` (32.660 email sạch). |
| **4** | **Trương Thành Công** | Thành viên | **14%** | • `scripts/feature_engineering.py` (Phần đặc trưng miền) | • Xây dựng hàm trích xuất 7 đặc trưng miền nghiệp vụ email (`extract_domain_features`).<br>• Lập trình thuật toán nén log phân phối văn bản siêu lệch (`log_text_len`).<br>• Thống kê và chuẩn hóa tỷ lệ ký tự viết hoa (`caps_ratio`), dấu cảm thán (`!`), ký hiệu tiền tệ (`$`), số lượng liên kết URL và tiền tố hồi âm `Re:`. |
| **5** | **Võ Thanh Phú** | Thành viên | **14%** | • `scripts/feature_engineering.py` (Phần phân tách & Pipeline) | • Xây dựng hàm phân chia Train/Test 80/20 phân tầng (`split_stratified_train_test`) bảo toàn tỷ lệ nhãn.<br>• Thiết lập dây chuyền tiền xử lý đa tầng `build_column_transformer` kết hợp TF-IDF (1,2-gram), Chi-Square chọn 5.000 từ vựng và MinMaxScaler.<br>• Xuất các tệp đối soát trực quan `train.csv`, `test.csv` và gói nạp nhị phân hiệu năng cao `train_test_data.joblib`, `preprocessor.joblib`. |
| **6** | **Võ Khôi Nguyên** | Thành viên | **14%** | • `scripts/model_training.py`<br>• `models/algorithms/` | • Chịu trách nhiệm module huấn luyện Chặng 3: Tự cài đặt 3 thuật toán Scratch thuần NumPy (Naive Bayes, Logistic Regression, Linear SVM Pegasos).<br>• Đóng gói Pipeline tích hợp scikit-learn (`build_training_pipelines`).<br>• Thiết lập quy trình kiểm định chéo 5-Fold Stratified Cross-Validation và thuật toán tối ưu siêu tham số `GridSearchCV` trên toàn bộ 26.128 mẫu Train. |
| **7** | **Mai Hoàng Danh** | Thành viên | **14%** | • `scripts/model_evaluation.py` | • Chịu trách nhiệm module đánh giá độc lập Chặng 4 (`evaluate_holdout_test`) trên 6.532 mẫu Test set.<br>• Tính toán ma trận nhầm lẫn (Confusion Matrix), đo đếm chính xác số ca bắt nhầm thư thường (FP) và bỏ lọt thư rác (FN).<br>• Bóc tách Top 15 từ khóa có trọng số cảnh báo spam mạnh nhất và đóng gói hiện vật nghiệm thu `best_model.joblib` kèm `metadata.json`. |
| **TỔNG** | **7 thành viên** | | **100%** | | **Hoàn thành toàn diện bài thực hành Classifying Spam Emails** |

---
