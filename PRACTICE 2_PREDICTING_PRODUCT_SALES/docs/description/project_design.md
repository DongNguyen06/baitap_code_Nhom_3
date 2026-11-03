# THIẾT KẾ DỰ ÁN — PRACTICE 2: PREDICTING PRODUCT SALES
## BÀI THỰC HÀNH HỌC MÁY (MACHINE LEARNING LAB) — NHÓM 3 (UTH)

---

### 1. WHAT — LÀM CÁI GÌ?
* **Bài toán:** Hồi quy dự đoán giá trị doanh số bán hàng ($y = \text{Amount} \in \mathbb{R}$) của đơn hàng thương mại điện tử trên nền tảng Amazon Ấn Độ.
* **Đầu vào / Đầu ra:**
  * *Đầu vào:* Thông tin chi tiết một đơn hàng: ngành hàng, kích cỡ trang phục, số lượng, cấp dịch vụ giao, kênh hoàn tất, bang nhận hàng, thời gian giao dịch, khuyến mãi và loại khách mua (sỉ/lẻ).
  * *Đầu ra:* Doanh số dự đoán ($\hat{y}$, đơn vị INR).
* **Dữ liệu thực nghiệm (`Amazon Sale Report.csv`):**
  * Gốc: 128.975 dòng, 24 cột.
  * Sau làm sạch & khử trùng lặp: Còn lại 102.852 dòng sạch, 12 cột đặc trưng then chốt.
* **Phạm vi bài Lab:**
  * *Có làm:* Làm sạch dữ liệu, loại đơn hủy/lỗi, tạo đặc trưng miền (thời gian, địa phương, cờ nhị phân), kiểm tra đa cộng tuyến, tự code 3 thuật toán Scratch (Linear Regression, Decision Tree, Random Forest), kiểm định chéo 5-Fold CV, đánh giá trên tập Test độc lập và dựng demo suy luận tương tác.
  * *Không làm:* Không dựng API backend thương mại lớn, không dự báo đa chuỗi thời gian ARIMA/LSTM.
* **Kết quả mong đợi:** Toàn bộ pipeline vận hành trơn tru không lỗi, 3 mô hình tự cài đặt vượt qua kiểm định chéo, mô hình tốt nhất đạt $R^2 > 0.50$, sai số RMSE được tối thiểu hóa.

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
| **3 Thuật toán Scratch** | 1. Linear Regression (Normal Equation + L2)<br>2. Decision Tree Regressor (MSE Variance Reduction, Quantile Splits)<br>3. Random Forest Regressor (Ensemble Bagging & Subspace) | Tự viết 100% bằng NumPy thuần túy, kế thừa BaseEstimator/RegressorMixin. |
| **Thước đo đánh giá** | MSE, RMSE, MAE, Hệ số xác định $R^2$ & Residual Plot | Đo lường toàn diện độ lệch sai số thực tế và đánh giá tính không thiên lệch của phần dư. |
| **Tiêu chí chọn Best Model** | Mô hình có **$R^2$ cao nhất** và **RMSE nhỏ nhất** trên tập Test | Năng lực giải thích biến thiên doanh số vượt trội và sai số dự báo thực tế thấp nhất. |

#### WHEN — Tiến trình thực hiện & Điều kiện dừng
* **Giai đoạn 1:** Khám phá & Làm sạch dữ liệu (`clean_amazon_sales.csv` đạt 102.852 dòng).
* **Giai đoạn 2:** Trích xuất đặc trưng miền, lọc đa cộng tuyến, tách Train/Test 80/20 (`train.csv`, `test.csv`).
* **Giai đoạn 3:** Huấn luyện 3 mô hình Scratch & Tinh chỉnh siêu tham số qua 5-Fold Cross Validation.
* **Giai đoạn 4:** Đánh giá độc lập trên tập Test (20.571 mẫu), vẽ Residual Plot và trích xuất Top 15 đặc trưng quan trọng.
* **Giai đoạn 5:** Đóng gói Pipeline vào `models/saved_models/` và dựng giao diện demo suy luận tương tác.
* **Điều kiện dừng & Nghiệm thu:** Khi cả 3 mô hình hoàn tất kiểm thử, mô hình tốt nhất đạt $R^2 > 0.50$ và toàn bộ 8 bài test tự động đạt 100%.

#### WHERE — Cây thư mục tổ chức thực tế của dự án
```text
PRACTICE 2_PREDICTING_PRODUCT_SALES/
├── data/
│   ├── raw/                      # Dữ liệu gốc (Amazon Sale Report.csv, .zip)
│   ├── processed/                # Dữ liệu sạch (clean_amazon_sales.csv, train.csv, test.csv, train_test_data.joblib)
│   └── scripts/                  # Module tiền xử lý (data_preprocessing.py, eda.py)
├── docs/
│   ├── description/              # Tài liệu thiết kế (project_description.md, project_design.md, TASKS.md)
│   ├── algorithms/               # Lý thuyết 3 thuật toán Scratch hồi quy
│   ├── images/                   # Biểu đồ trực quan hóa EDA và kết quả mô hình
│   └── report/                   # Báo cáo nghiệm thu HTML/Markdown
├── models/
│   ├── algorithms/               # Mã nguồn 3 thuật toán Scratch (linear_regression.py, decision_tree.py, random_forest.py)
│   └── saved_models/             # Mô hình tối ưu đã lưu (best_model.joblib, preprocessor.joblib, metadata.json)
├── notebooks/                    # 4 Notebooks thực hành tuần tự và Demo
├── scripts/                      # Các kịch bản chạy tự động (run_pipeline.py, feature_engineering.py, test_all_components.py)
├── README.md
└── requirements.txt
```

---

### 4. HOW — QUY TRÌNH XÂY DỰNG 5 BƯỚC (ĐẦU VÀO -> XỬ LÝ -> ĐẦU RA)

* **Bước 1: Khám phá & Làm sạch dữ liệu (Data Cleaning & EDA):**
  * *Đầu vào:* Tệp `Amazon Sale Report.csv` gốc (128.975 dòng).
  * *Xử lý:* Lọc bỏ đơn hàng hủy (`Status` chứa Cancelled); lọc đơn có Amount $\le 0$ hoặc Qty $\le 0$; điền khuyết thiếu cho `promotion-ids` ('No_Promotion'), `Courier Status` ('Unshipped') và `ship-state` (mode); khử 100% dòng trùng lặp.
  * *Đầu ra:* Tệp dữ liệu sạch `clean_amazon_sales.csv` (102.852 dòng) và các chỉ số thống kê EDA.

* **Bước 2: Kỹ thuật đặc trưng & Phân tách tập dữ liệu (Feature Engineering & Split):**
  * *Đầu vào:* Dữ liệu sạch `clean_amazon_sales.csv`.
  * *Xử lý:* Trích xuất chuỗi thời gian (`month`, `day_of_week`, `is_weekend`); gom nhóm địa lý Top 10 bang + 'Other'; tạo cờ nhị phân `is_promoted` và `is_b2b`; phân chia tập Train (80% — 82.281 mẫu) và Test (20% — 20.571 mẫu) với `random_state=42`; quét đa cộng tuyến trên tập Train và loại bỏ biến tương quan $|r| > 0.6$; đóng gói `ColumnTransformer` (StandardScaler cho biến số, OneHotEncoder cho biến danh mục).
  * *Đầu ra:* Các tệp `train.csv`, `test.csv`, bộ nạp nhị phân `train_test_data.joblib` và `preprocessor.joblib`.

* **Bước 3: Huấn luyện mô hình & Tinh chỉnh siêu tham số (Model Training & Tuning):**
  * *Đầu vào:* Tập dữ liệu huấn luyện `X_train`, `y_train` (82.281 mẫu) và bộ tiền xử lý.
  * *Xử lý:* Khởi tạo 3 thuật toán Scratch thuần NumPy (Linear Regression OLS + L2, Decision Tree Regressor với Quantile Splits, Random Forest Regressor với Bagging); thực hiện đánh giá chéo 5-Fold Cross Validation trên tập Train; chạy tinh chỉnh siêu tham số tối ưu bộ tham số cây rừng.
  * *Đầu ra:* 3 mô hình đã khớp tham số và tệp mô hình tối ưu `best_model.joblib`.

* **Bước 4: Đánh giá mô hình & Phân tích sai số (Model Evaluation & Residual Analysis):**
  * *Đầu vào:* 3 mô hình và tập kiểm tra độc lập `X_test`, `y_test` (20.571 mẫu).
  * *Xử lý:* Tính toán 4 chỉ số hồi quy (MSE, RMSE, MAE, $R^2$); vẽ biểu đồ phân tán Thực tế vs Dự đoán kèm đường chéo chuẩn; vẽ biểu đồ phần dư Residual Plot kiểm tra tính không thiên lệch; trích xuất Top 15 đặc trưng đóng góp lớn nhất vào biến thiên doanh số.
  * *Đầu ra:* Các hình ảnh đánh giá hiệu năng và tệp siêu dữ liệu nghiệm thu `metadata.json`.

* **Bước 5: Thử nghiệm tương tác & Suy luận thời gian thực (Interactive Demo):**
  * *Đầu vào:* Thông số một đơn hàng bất kỳ (ngành hàng, số lượng, kích cỡ, bang, thời gian, khuyến mãi...).
  * *Xử lý:* Nạp tự động `preprocessor.joblib` và mô hình tối ưu (hoặc cho phép chọn 1 trong 3 mô hình Scratch); tự động biến đổi đặc trưng và dự đoán số tiền doanh số cụ thể (INR).
  * *Đầu ra:* Giao diện trực quan trong `notebooks/demo.ipynb` hiển thị doanh số dự đoán tức thì kèm đối chiếu giữa các mô hình.
