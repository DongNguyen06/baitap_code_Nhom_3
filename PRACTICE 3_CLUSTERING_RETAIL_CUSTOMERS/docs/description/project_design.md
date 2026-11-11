# THIẾT KẾ DỰ ÁN — PRACTICE 3: CLUSTERING RETAIL CUSTOMERS
## BÀI THỰC HÀNH HỌC MÁY (MACHINE LEARNING LAB) — NHÓM 3 (UTH)

---

### 1. WHAT — LÀM CÁI GÌ?
* **Bài toán:** Học không giám sát (Unsupervised Learning) — Phân cụm khách hàng (Customer Clustering & Segmentation) tại trung tâm thương mại / cửa hàng bán lẻ nhằm tự động nhóm 200 khách hàng vào các phân khúc hành vi mua sắm riêng biệt dựa trên độ tương đồng dữ liệu.
* **Đầu vào / Đầu ra:**
  * *Đầu vào:* Thông tin khách hàng gồm: Mã định danh (`CustomerID`), Giới tính (`Gender`), Độ tuổi (`Age`), Thu nhập hàng năm (`Annual Income (k$)`) và Điểm thói quen chi tiêu (`Spending Score (1-100)`).
  * *Đầu ra:* Nhãn phân cụm gán cho từng khách hàng ($0 \dots 4$), hồ sơ định lượng 5 nhóm Personas (Spenders, VIP Target, Sensible, Careful, Standard), 4 đồ thị phân cụm chuyên sâu (Dendrogram, 2D K-Means, DBSCAN), tệp mô hình lưu trữ (`.joblib`), siêu dữ liệu `metadata.json` và hệ thống 6 báo cáo HTML Base64 độc lập.
* **Dữ liệu thực nghiệm (`Mall_Customers.csv`):**
  * Gốc: 200 quan sát khách hàng, 5 trường thuộc tính; 100% dữ liệu sạch, không khuyết thiếu (missing = 0), không trùng lặp.
* **Phạm vi bài Lab:**
  * *Có làm:* Khám phá dữ liệu EDA Read-Only, tiền xử lý và chuẩn hóa đặc trưng (StandardScaler), tự cài đặt 100% bằng NumPy 3 thuật toán phân cụm Scratch (K-Means, DBSCAN, Hierarchical Agglomerative), vẽ biểu đồ cây phân cấp Dendrogram theo yêu cầu đề bài slide, tối ưu siêu tham số ($K=5, \varepsilon=0.35, \text{min\_samples}=5$) qua phương pháp Khuỷu tay (Elbow) và Silhouette Score, đánh giá đa chỉ số (Silhouette, Davies-Bouldin, Calinski-Harabasz), lập hồ sơ kinh doanh cho 5 phân khúc và xây dựng demo suy luận phân cụm tương tác thời gian thực.
  * *Không làm:* Không dựng API server thương mại, không kết nối cơ sở dữ liệu bán lẻ production thời gian thực, không tạo tệp phân công `TASKS.md`.
* **Kết quả mong đợi:** Một pipeline điều phối 4 chặng chạy thông suốt 1-chạm (~13 giây), mô hình K-Means Scratch đạt Silhouette Score $\ge 0.55$, Hierarchical Scratch đạt $ARI = 1.0$ so với Scikit-learn, DBSCAN Scratch tách biệt chính xác 23 điểm ngoại lai, xuất đủ 5 tệp log và 6 báo cáo HTML Base64 standalone.

---

### 2. WHY — TẠI SAO LÀM & TẠI SAO CHỌN NHƯ VẬY?
* **Vì sao chọn bài toán phân cụm khách hàng bán lẻ:** Đây là bài toán kinh điển của Học không giám sát trong kinh doanh thực tế. Không gian 2D giữa Thu nhập hàng năm và Điểm chi tiêu gần như trực giao ($r \approx 0.0099$), hình thành 5 cụm tự nhiên rõ rệt, rất phù hợp để kiểm chứng năng lực thuật toán và chuyển giao giá trị tiếp thị kinh doanh.
* **Vì sao dùng Học không giám sát thay vì phân nhóm thủ công:** Dữ liệu khách hàng không có sẵn nhãn phân loại ("Ground Truth"). Việc chia nhóm thủ công bằng quy tắc cảm tính dễ gây thiên lệch và bỏ sót các mẫu hành vi vi mô. Học không giám sát tự động tìm ra ranh giới toán học dựa trên độ tương đồng khoảng cách và mật độ dữ liệu.
* **Vì sao chọn các giải pháp kỹ thuật cụ thể:**
  * *StandardScaler:* Đưa các đặc trưng về cùng thang đo chuẩn tắc $\mu = 0, \sigma = 1.0$, triệt tiêu sự áp đảo của biến Thu nhập (biên độ $15 - 137$) đối với biến Tuổi ($18 - 70$) và Điểm chi tiêu ($1 - 100$) khi tính khoảng cách Euclidean.
  * *Tự cài đặt 3 thuật toán Scratch (K-Means, DBSCAN, Hierarchical):* Đại diện cho 3 trường phái phân cụm trụ cột trong Học máy (Dựa trên Tâm cụm — Dựa trên Mật độ — Dựa trên Phân cấp thứ bậc).
  * *Phương pháp Khuỷu tay (Elbow) & Silhouette Score:* Xác định số cụm tối ưu $K=5$ khách quan dựa trên điểm gập WCSS Inertia và hệ số phân tách Silhouette đạt cực đại ($0.5547$).
  * *Vẽ cây phân cấp Dendrogram:* Trực quan hóa cấu trúc hợp nhất cụm từ dưới lên theo tiêu chí Ward Linkage, đáp ứng đúng yêu cầu trọng tâm của đề bài giảng viên.
  * *Bộ 3 chỉ số nội tại (Silhouette, Davies-Bouldin, Calinski-Harabasz):* Đánh giá chất lượng phân cụm toàn diện từ độ chặt chẽ nội cụm đến độ phân tách giữa các cụm mà không cần nhãn thực tế.
* **Mục tiêu học tập:** Nắm trọn vẹn chu trình một dự án Học không giám sát: từ khám phá dữ liệu, tự cài đặt thuật toán từ công thức toán học (NumPy), đối chuẩn với Scikit-Learn, đến diễn giải kết quả kinh doanh và tự động hóa toàn diện.

---

### 3. WHICH / WHEN / WHERE — DÙNG CÁI NÀO, KHI NÀO, Ở ĐÂU?

#### WHICH — Công cụ, thuật toán và tiêu chí đánh giá
| Thành phần | Lựa chọn kỹ thuật | Lý do lựa chọn |
| :--- | :--- | :--- |
| **Ngôn ngữ & Nền tảng** | Python 3.10+, NumPy | Tự cài đặt thuật toán Scratch thuần ma trận |
| **Xử lý bảng & Thống kê** | Pandas, SciPy | Thao tác DataFrame, tính ma trận liên kết Linkage |
| **Thư viện Đối chuẩn** | Scikit-learn (`KMeans`, `DBSCAN`, `AgglomerativeClustering`) | Đối soát độ chính xác toán học của mô hình Scratch |
| **Trực quan hóa** | Matplotlib, Seaborn (`Agg` headless) | Xuất 9 biểu đồ độ nét cao (DPI 200), không xung đột GUI |
| **Mô hình Quán quân** | K-Means Scratch ($K=5$, init=`k-means++`) | Silhouette đạt đỉnh 0.5547, phân tách 5 phân khúc cân đối |
| **Mô hình Phân cấp** | Hierarchical Agglomerative Scratch (Ward Linkage) | Cắt tại $K=5$, ARI = 1.0 so với Scikit-learn, vẽ Dendrogram |
| **Mô hình Mật độ** | DBSCAN Scratch ($\varepsilon=0.35, \text{min\_samples}=5$) | Tự động phát hiện cụm mật độ, cô lập 23 điểm ngoại lai |
| **Tiêu chí Đánh giá** | Silhouette Score ($\uparrow$), Davies-Bouldin ($\downarrow$), Calinski-Harabasz ($\uparrow$) | Bộ ba chỉ số đánh giá phân cụm không giám sát tiêu chuẩn |
| **Đóng gói & Báo cáo** | Joblib, Base64 HTML Standalone | Lưu trữ mô hình nạp nhanh và sinh 6 báo cáo độc lập |

#### WHEN — Tiến trình thực hiện
| Chặng | Nhiệm vụ nghiệp vụ | Hiện vật cốt lõi đầu ra | Thời lượng | Trạng thái |
| :---: | :--- | :--- | :---: | :---: |
| **Chặng 01** | Khám phá Dữ liệu Khách hàng Bán lẻ (EDA Read-Only) | 5 Biểu đồ tại `docs/images/`, `stage_01.json`, `01_data_report.html` | ~1.7s | ✅ Hoàn thành |
| **Chặng 02** | Tiền xử lý & Kỹ thuật Đặc trưng (FE & Scaling) | `clean_customers.csv`, `preprocessor.joblib`, `stage_02.json` | ~0.05s | ✅ Hoàn thành |
| **Chặng 03** | Huấn luyện Bộ 3 Mô hình Gom cụm Scratch & Tối ưu K | 3 Mô hình `.joblib`, `best_model.joblib`, `stage_03.json` | ~9.5s | ✅ Hoàn thành |
| **Chặng 04** | Đánh giá Phân cụm & Lập hồ sơ 5 Personas Khách hàng | 4 Đồ thị (Dendrogram, 2D, DBSCAN), `metadata.json`, `stage_04.json` | ~1.6s | ✅ Hoàn thành |
| **Chặng 05** | Trực quan hóa Báo cáo HTML & Demo Tương tác | 6 Báo cáo HTML Base64 tại `docs/report/`, `notebooks/demo.ipynb` | ~0.1s | ✅ Hoàn thành |

#### WHERE — Cây thư mục tổ chức thực tế của dự án
```text
PRACTICE 3_CLUSTERING_RETAIL_CUSTOMERS/
├── data/
│   ├── raw/
│   │   └── Mall_Customers.csv             # Dữ liệu gốc 200 khách hàng từ Kaggle
│   ├── processed/
│   │   └── clean_customers.csv            # Dữ liệu sạch sau tiền xử lý và mã hóa
│   └── scripts/
│       ├── data_preprocessing.py          # Module chuẩn hóa cột và mã hóa Gender
│       └── eda.py                         # Module phân tích thống kê mô tả và xuất 5 biểu đồ EDA
├── docs/
│   ├── description/
│   │   ├── algorithms/                    # Tài liệu toán học 3 thuật toán Scratch
│   │   │   └── thuat_toan.md              # Lý thuyết K-Means, DBSCAN, Hierarchical
│   │   ├── project_description.md         # Đặc tả tổng quan bài lab
│   │   └── project_design.md              # Bản thiết kế chi tiết 5W1H này
│   ├── images/                            # 9 Biểu đồ trực quan hóa EDA và Đánh giá phân cụm
│   │   ├── eda_01_distributions.png
│   │   ├── eda_02_gender_comparison.png
│   │   ├── eda_03_spending_vs_income.png
│   │   ├── eda_04_age_vs_spending.png
│   │   ├── eda_05_correlation_heatmap.png
│   │   ├── eval_01_dendrogram.png
│   │   ├── eval_02_clusters_2d.png
│   │   ├── eval_03_radar_profiles.png
│   │   └── eval_04_dbscan_clusters.png
│   └── report/                            # 6 Báo cáo HTML Base64 standalone nộp giảng viên
│       ├── index.html                     # Bảng điều khiển trung tâm (Dashboard tổng hợp)
│       ├── 01_data_report.html            # Báo cáo khám phá dữ liệu EDA
│       ├── 02_feature_pipeline_report.html# Báo cáo tiền xử lý & kỹ thuật đặc trưng
│       ├── 03_model_training_report.html  # Báo cáo huấn luyện 3 thuật toán Scratch
│       ├── 04_model_evaluation_report.html# Báo cáo đánh giá phân cụm & hồ sơ 5 Personas
│       └── run_log.html                   # Nhật ký thực thi toàn trình Zero-drift
├── logs/                                  # Nhật ký thực thi chi tiết (UTF-8 console + file log)
│   ├── run_pipeline_*.log                 # Nhật ký chạy toàn bộ pipeline
│   ├── run_01_eda_*.log                   # Nhật ký Chặng 1 (EDA)
│   ├── run_02_fe_*.log                    # Nhật ký Chặng 2 (Feature Engineering)
│   ├── run_03_train_*.log                 # Nhật ký Chặng 3 (Training & Tuning)
│   └── run_04_eval_*.log                  # Nhật ký Chặng 4 (Evaluation & Profiling)
├── models/
│   ├── algorithms/                        # Mã nguồn 3 thuật toán Scratch thuần NumPy
│   │   ├── kmeans.py                      # K-Means Scratch (K-Means++, WCSS Inertia)
│   │   ├── dbscan.py                      # DBSCAN Scratch (BFS queue, Density-based)
│   │   └── hierarchical.py                # Hierarchical Agglomerative Scratch (Ward Linkage)
│   └── saved_models/                      # Lưu trữ mô hình tối ưu và siêu dữ liệu
│       ├── best_model.joblib              # Mô hình Quán quân K-Means Scratch (K=5)
│       ├── kmeans_model.joblib
│       ├── hierarchical_model.joblib
│       ├── dbscan_model.joblib
│       ├── preprocessor.joblib            # Bộ chuẩn hóa StandardScaler đã khớp
│       └── metadata.json                  # Siêu dữ liệu chỉ số nghiệm thu
├── notebooks/                             # 4 Jupyter Notebooks trình diễn phẳng
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing_and_feature_engineering.ipynb
│   ├── 03_model_training_evaluation_and_explain.ipynb
│   └── demo.ipynb                         # Notebook demo tương tác phân cụm khách hàng mới
├── outputs/                               # Phong bì kết quả trung gian dạng JSON
│   ├── stage_01.json
│   ├── stage_02.json
│   ├── stage_03.json
│   └── stage_04.json
├── scripts/                               # Bộ kịch bản điều phối và tiện ích hệ thống
│   ├── common.py                          # Tiện ích đường dẫn tương đối và hàm dùng chung
│   ├── feature_engineering.py             # Module không gian đặc trưng và StandardScaler
│   ├── logger_utils.py                    # Cấu hình logging đa luồng
│   ├── model_evaluation.py                # Module đánh giá phân cụm và lập hồ sơ Personas
│   ├── model_training.py                  # Module huấn luyện và đối chuẩn Scikit-learn
│   ├── report_builder.py                  # Trình biên dịch 6 báo cáo HTML Base64 standalone
│   ├── run_01_eda.py                      # Điều phối Chặng 1
│   ├── run_02_fe.py                       # Điều phối Chặng 2
│   ├── run_03_train.py                    # Điều phối Chặng 3
│   ├── run_04_eval.py                     # Điều phối Chặng 4
│   ├── run_pipeline.py                    # Master Pipeline 1-chạm toàn trình
│   └── test_all_components.py             # Bộ kiểm định tự động toàn diện 8 bước
├── .gitignore                             # Loại trừ cache, venv, whitelist logs và outputs
└── requirements.txt                       # Danh mục thư viện phụ thuộc
```

---

### 4. HOW — QUY TRÌNH XÂY DỰNG 5 BƯỚC (ĐẦU VÀO -> XỬ LÝ -> ĐẦU RA)

* **Bước 1: Khám phá dữ liệu bán lẻ (Exploratory Data Analysis - EDA):**
  * *Đầu vào:* Tệp `data/raw/Mall_Customers.csv` gốc (200 khách hàng thô).
  * *Xử lý:* Khảo sát cấu trúc thuộc tính, kiểm tra dữ liệu khuyết thiếu và trùng lặp theo cơ chế Read-Only 100% (không làm sạch và không ghi đè dữ liệu); phân tích thống kê mô tả (Mean, Median, Skewness, Kurtosis); phân tích đơn biến phân phối Age, Annual Income, Spending Score; phân tích tương quan Pearson phát hiện tính trực giao giữa Thu nhập và Chi tiêu ($r \approx 0.0099$); quét ngoại lai IQR phát hiện 2 điểm thu nhập cao ($137\text{ k}\$$).
  * *Đầu ra:* Phong bì `outputs/stage_01.json`, 5 biểu đồ trực quan hóa EDA (`eda_01` $\rightarrow$ `eda_05`) tại `docs/images/`, nhật ký `logs/run_01_eda_*.log` và báo cáo `docs/report/01_data_report.html`.

* **Bước 2: Tiền xử lý dữ liệu & Kỹ thuật đặc trưng (Data Preprocessing & Feature Engineering):**
  * *Đầu vào:* Tệp `data/raw/Mall_Customers.csv` gốc.
  * *Xử lý:* Chuẩn hóa tên cột thống nhất (`CustomerID`, `Gender`, `Age`, `Annual_Income_k$`, `Spending_Score`); mã hóa biến danh mục `Gender` thành `Gender_Code` (Female: 0, Male: 1); thiết lập 3 không gian đặc trưng (`2D_Income_Spending`, `3D_Demographic`, `Full_Features`); áp dụng `StandardScaler` chuẩn hóa thang đo đưa dữ liệu về $\mu = 0, \sigma = 1.0$ chống thiên lệch khoảng cách Euclidean.
  * *Đầu ra:* Tệp dữ liệu sạch `data/processed/clean_customers.csv` (200 dòng sạch), bộ nạp nhị phân `models/saved_models/preprocessor.joblib`, phong bì `outputs/stage_02.json`, nhật ký `logs/run_02_fe_*.log` và báo cáo `docs/report/02_feature_pipeline_report.html`.

* **Bước 3: Huấn luyện 3 mô hình Scratch & Tinh chỉnh siêu tham số (Model Training & Hyperparameter Tuning):**
  * *Đầu vào:* Dữ liệu đặc trưng chuẩn hóa trong không gian 2D Thu nhập - Chi tiêu (200 mẫu).
  * *Xử lý:* Khởi tạo và huấn luyện 3 thuật toán Scratch thuần NumPy:
    1. *K-Means Scratch:* Khởi tạo tâm cụm $K$-Means++, lặp Expectation-Maximization hội tụ sau 2–5 vòng lặp; quét $K \in [2, 10]$ tính toán WCSS Inertia và Silhouette để chọn $K=5$ tối ưu.
    2. *Hierarchical Agglomerative Scratch:* Xây dựng ma trận khoảng cách, tính toán ma trận liên kết theo tiêu chí Ward Linkage từ dưới lên và cắt tại $K=5$.
    3. *DBSCAN Scratch:* Quét lân cận bán kính $\varepsilon=0.35$ và $\text{min\_samples}=5$, lan truyền cụm qua hàng đợi BFS, tự động phát hiện cụm mật độ và cô lập 23 điểm nhiễu ngoại lai (-1).
    Thực hiện đối chuẩn chi tiết từng thuật toán Scratch với Scikit-learn (K-Means khớp Inertia tới 13 chữ số thập phân, Hierarchical đạt $ARI = 1.0$).
  * *Đầu ra:* 3 tệp mô hình đã huấn luyện (`kmeans_model.joblib`, `hierarchical_model.joblib`, `dbscan_model.joblib`), tệp mô hình Quán quân `best_model.joblib`, siêu dữ liệu `metadata.json`, phong bì `outputs/stage_03.json`, nhật ký `logs/run_03_train_*.log` và báo cáo `docs/report/03_model_training_report.html`.

* **Bước 4: Đánh giá phân cụm & Lập hồ sơ 5 phân khúc khách hàng (Clustering Evaluation & Persona Profiling):**
  * *Đầu vào:* 3 mô hình đã khớp tham số và dữ liệu khách hàng sạch.
  * *Xử lý:* Đánh giá và so sánh chất lượng phân cụm qua bộ 3 chỉ số nội tại (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index); vẽ biểu đồ cây phân cấp Dendrogram cắt tại khoảng cách tối ưu $K=5$; vẽ biểu đồ phân tán 2D hiển thị 5 cụm và 5 tâm cụm; vẽ biểu đồ phân cụm mật độ DBSCAN tách biệt 23 điểm nhiễu; tổng hợp số liệu thống kê trung bình của từng cụm để xây dựng hồ sơ định lượng và chiến lược kinh doanh cho 5 Personas.
  * *Đầu ra:* 4 biểu đồ chuyên sâu (`eval_01` $\rightarrow$ `eval_04`) tại `docs/images/`, phong bì `outputs/stage_04.json`, nhật ký `logs/run_04_eval_*.log` và báo cáo `docs/report/04_model_evaluation_report.html`.

#### Bảng Hồ sơ 5 Phân khúc Khách hàng Bán lẻ (Customer Personas)
| Phân khúc | Tên Persona | Số lượng (Tỷ lệ) | Thu nhập TB | Chi tiêu TB | Tuổi TB | Đặc điểm Hành vi | Khuyến nghị Chiến lược Bán lẻ |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **0** | **Tiêu xài Tự do (Spenders)** | 81 khách (40.5%) | 55.3 k$ | 49.5 | 42.7 | Thu nhập trung bình, chi tiêu đều đặn, chuộng xu hướng | Chiến dịch Flash Sales, ưu đãi tích điểm, hỗ trợ trả góp 0% |
| **1** | **Khách hàng VIP (Target)** | 35 khách (17.5%) | 88.2 k$ | 82.1 | 41.1 | Thu nhập cao, chi tiêu rất mạnh, ưu tiên chất lượng và thương hiệu | Dịch vụ chăm sóc khách VIP đặc quyền, giới thiệu sản phẩm cao cấp độc quyền |
| **2** | **Khách hàng Tiết kiệm (Sensible)** | 22 khách (11.0%) | 25.7 k$ | 79.4 | 25.3 | Khách hàng trẻ tuổi, thu nhập thấp nhưng sẵn sàng chi tiêu cho sở thích | Tiếp thị qua mạng xã hội (TikTok, Instagram), các gói combo giá rẻ |
| **3** | **Cẩn trọng / Giữ tiền (Careful)** | 39 khách (19.5%) | 86.5 k$ | 17.1 | 32.7 | Thu nhập cao nhưng cực kỳ khắt khe, chi tiêu rất thấp | Tiếp thị dựa trên giá trị cốt lõi, bảo hành lâu dài, cam kết chất lượng |
| **4** | **Khách hàng Tiêu chuẩn (Standard)**| 23 khách (11.5%) | 26.3 k$ | 20.9 | 45.2 | Thu nhập thấp và chi tiêu thấp, tập trung vào mặt hàng thiết yếu | Khuyến mãi giảm giá hàng thiết yếu, tích điểm đổi nhu yếu phẩm |

* **Bước 5: Thử nghiệm tương tác & Phân cụm thời gian thực (Interactive Demo):**
  * *Đầu vào:* Thông số một khách hàng mới bất kỳ (Thu nhập hàng năm `Annual Income` và Điểm chi tiêu `Spending Score`).
  * *Xử lý:* Nạp tự động bộ chuẩn hóa `preprocessor.joblib` và mô hình Quán quân `best_model.joblib` (K-Means Scratch); chuẩn hóa dữ liệu đầu vào theo đúng thang đo đã học và tính toán khoảng cách Euclidean tới 5 tâm cụm để suy luận phân khúc tức thì; hiển thị diễn giải Persona và gợi ý chiến lược chăm sóc tương ứng.
  * *Đầu ra:* Giao diện trực quan trong `notebooks/demo.ipynb` trả về phân khúc khách hàng mục tiêu trong thời gian thực.
