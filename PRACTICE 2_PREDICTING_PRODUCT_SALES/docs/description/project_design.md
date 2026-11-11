# THIẾT KẾ DỰ ÁN — PRACTICE 2: PREDICTING PRODUCT SALES
## BÀI THỰC HÀNH HỌC MÁY (MACHINE LEARNING LAB) — NHÓM 3 (UTH)

---

### 1. WHAT — LÀM CÁI GÌ?
* **Bài toán:** Hồi quy dự đoán giá trị doanh số bán hàng của đơn hàng thương mại điện tử trên nền tảng Amazon Ấn Độ.
* **Đầu vào / Đầu ra:**
  * *Đầu vào:* Thông tin chi tiết một đơn hàng: ngành hàng, kích cỡ trang phục, số lượng, cấp dịch vụ giao, kênh hoàn tất, bang nhận hàng, thời gian giao dịch, khuyến mãi và loại khách mua (sỉ/lẻ).
  * *Đầu ra:* Doanh số dự đoán cụ thể theo thang đo tiền tệ INR.
* **Dữ liệu thực nghiệm (`Amazon Sale Report.csv`):**
  * Gốc: 128.975 dòng, 24 cột.
  * Sau làm sạch & khử trùng lặp: Còn lại 102.852 dòng sạch, 12 cột đặc trưng then chốt.
* **Phạm vi bài Lab:**
  * *Có làm:* Khám phá dữ liệu EDA, làm sạch nghiệp vụ TMĐT, loại đơn hủy/lỗi, tạo 11 đặc trưng miền (thời gian, địa phương, cờ nhị phân), kiểm tra đa cộng tuyến, tự code 3 thuật toán Scratch (Linear Regression, Decision Tree, Random Forest), kiểm định chéo 5-Fold CV, đánh giá trên tập Test độc lập và dựng demo suy luận tương tác.
  * *Không làm:* Không dựng API backend thương mại lớn, không dự báo đa chuỗi thời gian ARIMA/LSTM.
* **Kết quả mong đợi:** Toàn bộ pipeline vận hành trơn tru không lỗi, 3 mô hình tự cài đặt vượt qua kiểm định chéo, mô hình tốt nhất đạt $R^2 \approx 0.49 - 0.50$, sai số RMSE được tối thiểu hóa.

---

### 2. WHY — TẠI SAO LÀM & TẠI SAO CHỌN NHƯ VẬY?
* **Vì sao chọn bài toán dự đoán doanh số:** Đây là bài toán cốt lõi của hồi quy trong Học máy và mang tính thực tiễn cao trong ngành bán lẻ trực tuyến, giúp doanh nghiệp chủ động dự báo thu chi, tối ưu hóa chính sách giá và quản lý tồn kho.
* **Vì sao dùng Học máy thay vì công thức bình quân đơn giản:** Giá bán và doanh số chịu ảnh hưởng phi tuyến tính phức tạp bởi nhiều nhân tố đồng thời (kích cỡ, ngành hàng, số lượng, thời điểm trong tuần, chiến dịch khuyến mãi và vị trí địa lý). Các mô hình học máy tự động học được mối liên hệ đa biến này.
* **Vì sao chọn các giải pháp kỹ thuật cụ thể:**
  * *Biến đổi Log $\log(1+\text{Amount})$:* Nén phân phối lệch phải nặng của doanh số, đưa về dạng gần đối xứng chuẩn giúp các mô hình tuyến tính và cây học ổn định hơn.
  * *Gom nhóm Top 10 bang + 'Other':* Ngăn ngừa bùng nổ số chiều khi One-Hot Encoding trên hơn 50 bang, giữ lại hơn 75% giá trị thông tin địa lý quan trọng nhất.
  * *Lọc đa cộng tuyến ($|r| > 0.6$):* Loại bỏ các biến phụ thuộc tuyến tính cao (như thứ trong tuần và cờ cuối tuần), bảo vệ ma trận hiệp phương sai khỏi nguy cơ suy biến.
  * *Tự cài đặt 3 thuật toán Scratch bằng NumPy:* Thấu hiểu bản chất toán học từ mô hình tuyến tính đơn giản (OLS) đến cây quyết định giảm phương sai (MSE Reduction) và kỹ thuật học kết hợp (Ensemble Bagging).
  * *Hàng rào chống rò rỉ dữ liệu (Anti-Leakage Barrier):* Chia tập Train/Test 80/20 trước khi fit thống kê; `ColumnTransformer` chỉ fit trên `X_train`.

---

### 3. WHICH / WHEN / WHERE — DÙNG CÁI NÀO, KHI NÀO, Ở ĐÂU?

#### WHICH — Công cụ, thuật toán và tiêu chí đánh giá
| Thành phần | Lựa chọn kỹ thuật trong dự án | Lý do lựa chọn |
| :--- | :--- | :--- |
| **Thư viện cốt lõi** | `numpy`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn` | Hệ sinh thái chuẩn mực cho xử lý bảng số, đại số ma trận và trực quan hóa. |
| **3 Thuật toán Scratch** | 1. Linear Regression <br>2. Decision Tree Regressor <br>3. Random Forest Regressor | Tự viết 100% bằng NumPy thuần túy, kế thừa BaseEstimator/RegressorMixin. |
| **Thước đo đánh giá** | MSE, RMSE, MAE, Hệ số xác định $R^2$ & Residual Plot | Đo lường toàn diện độ lệch sai số thực tế và đánh giá tính không thiên lệch của phần dư. |
| **Tiêu chí chọn Best Model** | Mô hình có **$R^2$ cao nhất** và **RMSE nhỏ nhất** trên tập Test | Năng lực giải thích biến thiên doanh số vượt trội và sai số dự báo thực tế thấp nhất. |

#### WHEN — Tiến trình thực hiện
* **Giai đoạn 1:** Khám phá dữ liệu bán hàng TMĐT (EDA thuần túy — Read-Only 100%).
* **Giai đoạn 2:** Tiền xử lý dữ liệu TMĐT, khử trùng lặp (Zero Leakage) & Kỹ thuật 11 đặc trưng doanh số.
* **Giai đoạn 3:** Huấn luyện 3 mô hình hồi quy Scratch thuần NumPy & Kiểm định chéo 5-Fold CV.
* **Giai đoạn 4:** Đánh giá kiểm thử độc lập Test set 20.571 mẫu, phân tích sai số INR & Bóc tách đặc trưng.
* **Giai đoạn 5:** Đóng gói mô hình Quán quân (`best_model.joblib`), sinh báo cáo nghiệm thu & Demo thời gian thực.

#### WHERE — Cây thư mục tổ chức thực tế của dự án
```text
PRACTICE 2_PREDICTING_PRODUCT_SALES/
├── data/
│   ├── raw/                      # Dữ liệu thương mại điện tử gốc
│   │   ├── Amazon Sale Report.csv# Tập dữ liệu thô ban đầu (128.975 dòng)
│   │   └── Amazon Sale Report.zip# Bản nén lưu trữ dữ liệu gốc
│   ├── processed/                # Dữ liệu sạch đã tiền xử lý
│   │   ├── clean_amazon_sales.csv# Dữ liệu sạch sau khử đơn hủy và trùng lặp (102.852 dòng)
│   │   ├── train.csv             # Tập huấn luyện mở được bằng Excel (80% — 82.281 dòng)
│   │   ├── test.csv              # Tập kiểm tra mở được bằng Excel (20% — 20.571 dòng)
│   │   └── train_test_data.joblib# Tệp nhị phân lưu trữ toàn bộ dữ liệu phân tách
│   └── scripts/                  # Kịch bản tiền xử lý và EDA
│       ├── data_preprocessing.py # Module làm sạch khuyết thiếu và khử trùng lặp
│       └── eda.py                # Module phân tích đơn biến, đa biến và quét IQR
├── docs/
│   ├── description/              # Tài liệu phân tích và đặc tả dự án
│   │   ├── algorithms/           # Cơ sở toán học & lý thuyết 3 thuật toán Scratch hồi quy
│   │   │   ├── linear_regression.md
│   │   │   ├── decision_tree_regressor.md
│   │   │   └── random_forest_regressor.md
│   │   ├── project_description.md# Mô tả tổng quan bài toán kinh doanh
│   │   ├── project_design.md     # Bản thiết kế chi tiết 5W1H này
│   ├── images/                   # Biểu đồ trực quan hóa EDA và kết quả mô hình
│   │   ├── eda_01_amount_distribution.png
│   │   ├── eda_02_bivariate_category_fulfilment.png
│   │   ├── eda_03_outliers_boxplot.png
│   │   ├── eda_04_temporal_sales_trend.png
│   │   ├── eda_05_geographic_sales_distribution.png
│   │   ├── eda_06_correlation_heatmap.png
│   │   ├── model_01_actual_vs_predicted.png
│   │   ├── model_02_residuals.png
│   │   ├── model_03_performance_comparison.png
│   │   └── model_04_feature_importances.png
│   └── report/                   # Báo cáo kỹ thuật và bảng điều khiển nghiệm thu
│       ├── index.html            # Bảng điều khiển trung tâm (Dashboard tổng hợp 4 chặng)
│       ├── 01_data_report.html   # Báo cáo khám phá dữ liệu bán hàng TMĐT
│       ├── 02_feature_pipeline_report.html # Báo cáo tiền xử lý & kỹ thuật đặc trưng
│       ├── 03_model_training_report.html   # Báo cáo huấn luyện 3 mô hình Scratch & 5-Fold CV
│       ├── 04_model_evaluation_report.html # Báo cáo đánh giá độc lập Test set & bóc tách đặc trưng
│       └── run_log.html          # Báo cáo nhật ký thực thi Zero-drift của cả 4 chặng
├── logs/                         # Nhật ký thực thi chi tiết (UTF-8 console + file log)
│   ├── run_pipeline_*.log        # Nhật ký chạy toàn bộ pipeline
│   ├── run_01_eda_*.log          # Nhật ký Chặng 1 (EDA)
│   ├── run_02_fe_*.log           # Nhật ký Chặng 2 (Feature Engineering)
│   ├── run_03_train_*.log        # Nhật ký Chặng 3 (Training & CV)
│   └── run_04_eval_*.log         # Nhật ký Chặng 4 (Evaluation & Test)
├── models/
│   ├── algorithms/               # Mã nguồn 3 thuật toán hồi quy From Scratch (NumPy thuần túy)
│   │   ├── linear_regression.py  # Hồi quy tuyến tính OLS + L2 Ridge
│   │   ├── decision_tree.py      # Cây quyết định hồi quy (Quantile Splits + Giảm phương sai)
│   │   └── random_forest.py      # Rừng ngẫu nhiên hồi quy (Bootstrap Aggregating)
│   └── saved_models/             # Mô hình tối ưu đã đóng gói và siêu dữ liệu
│       ├── best_model.joblib     # Mô hình quán quân Random Forest Regressor
│       ├── decision_tree.joblib  # Mô hình Cây quyết định hồi quy độc lập
│       ├── linear_regression.joblib # Mô hình Hồi quy tuyến tính độc lập
│       ├── random_forest.joblib  # Mô hình Rừng ngẫu nhiên hồi quy độc lập
│       ├── preprocessor.joblib   # Pipeline biến đổi ColumnTransformer đã fit
│       └── metadata.json         # Bảng thông số nghiệm thu và Top 15 đặc trưng chi phối
├── notebooks/                    # 4 Notebooks Jupyter thực hành tuần tự và Demo (outputs: [])
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing_and_feature_engineering.ipynb
│   ├── 03_model_training_evaluation_and_explain.ipynb
│   └── demo.ipynb                # Demo suy luận doanh số thời gian thực đa mô hình có Guardrail
├── outputs/                      # Dữ liệu phong bì JSON giao tiếp giữa các chặng
│   ├── stage_01.json             # Kết quả Chặng 1 (EDA thuần túy)
│   ├── stage_02.json             # Kết quả Chặng 2 (Tiền xử lý & Kỹ thuật đặc trưng)
│   ├── stage_03.json             # Kết quả Chặng 3 (Huấn luyện Scratch & 5-Fold CV)
│   └── stage_04.json             # Kết quả Chặng 4 (Đánh giá Hold-out Test & Bóc tách đặc trưng)
├── scripts/                      # Các module tái sử dụng và kịch bản thực thi độc lập
│   ├── common.py                 # Hằng số đường dẫn, quản lý phong bì JSON & lưu ảnh
│   ├── logger_utils.py           # Hệ thống logging UTF-8 console + file log, timed block
│   ├── feature_engineering.py    # Kỹ thuật 11 đặc trưng TMĐT & xây dựng Pipeline
│   ├── model_training.py         # Huấn luyện 3 mô hình Scratch & 5-Fold CV
│   ├── model_evaluation.py       # Đánh giá trên tập test, expm1 & xuất đồ thị
│   ├── report_builder.py         # Trình dựng báo cáo HTML Base64 standalone tự động
│   ├── run_01_eda.py             # Kịch bản chạy độc lập Chặng 1
│   ├── run_02_fe.py              # Kịch bản chạy độc lập Chặng 2
│   ├── run_03_train.py           # Kịch bản chạy độc lập Chặng 3
│   ├── run_04_eval.py            # Kịch bản chạy độc lập Chặng 4
│   ├── run_pipeline.py           # Bộ điều phối chạy toàn bộ 4 chặng xuyên suốt
│   └── test_all_components.py    # Bộ kiểm thử tự động 8 bước đạt chuẩn 100%
├── README.md                     # Tài liệu hướng dẫn khởi chạy dự án
└── requirements.txt              # Danh mục thư viện phụ thuộc
```

---

### 4. HOW — QUY TRÌNH XÂY DỰNG 5 BƯỚC (ĐẦU VÀO -> XỬ LÝ -> ĐẦU RA)

* **Bước 1: Khám phá dữ liệu bán hàng TMĐT (Exploratory Data Analysis - EDA):**
  * *Đầu vào:* Tệp `Amazon Sale Report.csv` gốc (128.975 dòng thô).
  * *Xử lý:* Khảo sát cấu trúc thuộc tính, giá trị khuyết thiếu và chất lượng thô ban đầu theo cơ chế Read-Only 100% (không làm sạch và không ghi đè dữ liệu); phân tích đơn biến phân phối lệch phải của `Amount`, chu kỳ doanh số theo tháng/thứ, phân bổ địa lý theo bang và phát hiện ngoại lai IQR.
  * *Đầu ra:* Phong bì `outputs/stage_01.json`, 6 biểu đồ trực quan hóa EDA (`eda_01` ➔ `eda_06`) và báo cáo `01_data_report.html`.

* **Bước 2: Tiền xử lý dữ liệu & Kỹ thuật đặc trưng (Data Preprocessing & Feature Engineering):**
  * *Đầu vào:* Tệp `Amazon Sale Report.csv` gốc.
  * *Xử lý:* Lọc bỏ đơn hàng hủy (`Status` chứa Cancelled); loại đơn có Amount $\le 0$ hoặc Qty $\le 0$; điền khuyết thiếu cho `promotion-ids` ('No_Promotion'), `Courier Status` ('Unshipped') và `ship-state` (mode); khử 100% dòng trùng lặp đa hình theo nguyên tắc Zero Duplicate Leakage; trích xuất chuỗi thời gian (`month`, `day_of_week`, `is_weekend`); gom nhóm địa lý Top 10 bang + 'Other'; tạo cờ nhị phân `is_promoted` và `is_b2b`; phân chia tập Train (80% — 82.281 mẫu) và Test (20% — 20.571 mẫu) với `random_state=42`; quét đa cộng tuyến trên tập Train và loại bỏ biến tương quan $|r| > 0.6$; đóng gói `ColumnTransformer` (StandardScaler cho biến số, OneHotEncoder cho biến danh mục).
  * *Đầu ra:* Tệp dữ liệu sạch `clean_amazon_sales.csv` (102.852 dòng sạch), các tệp `train.csv`, `test.csv`, bộ nạp nhị phân `train_test_data.joblib`, `preprocessor.joblib`, phong bì `outputs/stage_02.json` và báo cáo `02_feature_pipeline_report.html`.

* **Bước 3: Huấn luyện mô hình & Tinh chỉnh siêu tham số (Model Training & Tuning):**
  * *Đầu vào:* Tập dữ liệu huấn luyện `X_train`, `y_train` (82.281 mẫu) và bộ tiền xử lý.
  * *Xử lý:* Khởi tạo 3 thuật toán Scratch thuần NumPy (Linear Regression OLS + L2, Decision Tree Regressor với Quantile Splits, Random Forest Regressor với Bagging); thực hiện đánh giá chéo 5-Fold Cross Validation trên tập Train; chạy tinh chỉnh siêu tham số tối ưu bộ tham số cây rừng.
  * *Đầu ra:* 3 mô hình đã khớp tham số, tệp mô hình tối ưu `best_model.joblib`, phong bì `outputs/stage_03.json` và báo cáo `03_model_training_report.html`.

* **Bước 4: Đánh giá mô hình & Phân tích sai số (Model Evaluation & Residual Analysis):**
  * *Đầu vào:* 3 mô hình và tập kiểm tra độc lập `X_test`, `y_test` (20.571 mẫu).
  * *Xử lý:* Tính toán 4 chỉ số hồi quy (MSE, RMSE, MAE, $R^2$); chuyển đổi nghịch đảo `expm1` có chốt chặn chống tràn số về thang đo tiền tệ INR; vẽ biểu đồ phân tán Thực tế vs Dự đoán kèm đường chéo chuẩn; vẽ biểu đồ phần dư Residual Plot kiểm tra tính không thiên lệch; trích xuất Top 15 đặc trưng đóng góp lớn nhất vào biến thiên doanh số.
  * *Đầu ra:* 4 biểu đồ đánh giá (`model_01` ➔ `model_04`), tệp siêu dữ liệu nghiệm thu `metadata.json`, phong bì `outputs/stage_04.json` và báo cáo `04_model_evaluation_report.html`.

* **Bước 5: Thử nghiệm tương tác & Suy luận thời gian thực (Interactive Demo):**
  * *Đầu vào:* Thông số một đơn hàng bất kỳ (ngành hàng, số lượng, kích cỡ, bang, thời gian, khuyến mãi...).
  * *Xử lý:* Nạp tự động `preprocessor.joblib` và mô hình tối ưu (hoặc cho phép chọn 1 trong 3 mô hình Scratch); tự động biến đổi đặc trưng và dự đoán số tiền doanh số cụ thể (INR).
  * *Đầu ra:* Giao diện trực quan trong `notebooks/demo.ipynb` hiển thị doanh số dự đoán tức thì kèm đối chiếu giữa các mô hình.
